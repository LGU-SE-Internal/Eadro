import os
import time
import copy
from typing import Dict, Any, Optional, Tuple, List

import torch
from torch import nn
import numpy as np
from sklearn.metrics import ndcg_score
from loguru import logger

from .model import MainModel
from .experiment_manager import ExperimentManager
from .config import Config


class BaseModel(nn.Module):
    """Base training class for the EADRO model"""

    def __init__(
        self,
        event_num: int,
        metric_num: int,
        node_num: int,
        device: str,
        config: Config,
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the base model for EADRO

        Args:
            event_num: Number of event types
            metric_num: Number of metrics
            node_num: Number of nodes
            device: Device to run on ("cpu" or "cuda")
            config: Configuration object
            experiment_name: Custom experiment name for tracking
        """
        super(BaseModel, self).__init__()

        # Get training parameters from config
        self.epochs = config.get("training.epochs")
        self.lr = config.get("training.lr")
        self.patience = config.get("training.patience")
        self.device = device

        # Learning rate scheduler parameters
        lr_scheduler = config.get("training.lr_scheduler.type")
        self.lr_scheduler_type = lr_scheduler.lower()
        self.lr_step_size = config.get("training.lr_scheduler.step_size")
        self.lr_gamma = config.get("training.lr_scheduler.gamma")
        self.lr_warmup_epochs = config.get("training.lr_scheduler.warmup_epochs")
        self.lr_min = config.get("training.lr_scheduler.min_lr")

        # Model save directory
        result_dir = config.get("paths.result_dir")
        hash_id = config.get("model.base.hash_id")

        self.model_save_dir = (
            os.path.join(result_dir, hash_id) if hash_id else result_dir
        )

        # Initialize experiment manager
        self.exp_manager = ExperimentManager(config, experiment_name)

        # Create main model with config
        self.model = MainModel(event_num, metric_num, node_num, device, config)
        self.model.to(device)

    def _create_lr_scheduler(self, optimizer) -> Optional[Any]:
        """
        Create learning rate scheduler based on configuration

        Args:
            optimizer: The optimizer to attach scheduler to

        Returns:
            Learning rate scheduler or None if no scheduler is used
        """
        if self.lr_scheduler_type == "none":
            return None
        elif self.lr_scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
            )
        elif self.lr_scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.lr_gamma
            )
        elif self.lr_scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
        elif self.lr_scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.lr_gamma, patience=5
            )
        else:
            logger.warning(
                f"Unknown scheduler type: {self.lr_scheduler_type}, using no scheduler"
            )
            return None

    def _warmup_lr(self, optimizer, epoch: int) -> None:
        """
        Apply learning rate warmup

        Args:
            optimizer: The optimizer to modify
            epoch: Current epoch (1-indexed)
        """
        if self.lr_warmup_epochs > 0 and epoch <= self.lr_warmup_epochs:
            warmup_lr = self.lr * (epoch / self.lr_warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

    def evaluate(self, test_loader: Any, datatype: str = "Test") -> Dict[str, float]:
        self.model.eval()
        hrs, ndcgs = np.zeros(5), np.zeros(5)
        TP, FP, FN = 0, 0, 0  # True Positive, False Positive, False Negative
        batch_cnt, epoch_loss = 0, 0.0

        with torch.no_grad():
            for graph, ground_truths in test_loader:
                res = self.model.forward(graph.to(self.device), ground_truths)
                for idx, faulty_nodes in enumerate(res["y_pred"]):
                    culprit = ground_truths[idx].item()
                    if culprit == -1:  # Normal sample
                        if faulty_nodes[0] == -1:
                            TP += 1  # Correctly predicted as normal
                        else:
                            FP += 1  # Incorrectly predicted as abnormal
                    else:  # Abnormal sample
                        if faulty_nodes[0] == -1:
                            FN += 1  # Incorrectly predicted as normal
                        else:
                            TP += 1  # Correctly predicted as abnormal
                            rank = list(faulty_nodes).index(culprit)
                            for j in range(5):
                                hrs[j] += int(rank <= j)  # Calculate Hit Rate
                                ndcgs[j] += ndcg_score(
                                    np.array([res["y_prob"][idx]]).reshape(1, -1),
                                    np.array([res["pred_prob"][idx]]).reshape(1, -1),
                                    k=j + 1,
                                )
                epoch_loss += res["loss"].item()
                batch_cnt += 1

        pos = TP + FN  # Total number of positive samples
        eval_results = {
            "F1": TP * 2.0 / (TP + FP + pos) if (TP + FP + pos) > 0 else 0,
            "Rec": TP * 1.0 / pos if pos > 0 else 0,  # Recall
            "Pre": TP * 1.0 / (TP + FP) if (TP + FP) > 0 else 0,  # Precision
        }

        # Calculate Hit Rate and NDCG metrics
        for j in [1, 3, 5]:
            eval_results["HR@" + str(j)] = hrs[j - 1] * 1.0 / pos
            eval_results["ndcg@" + str(j)] = ndcgs[j - 1] * 1.0 / pos

        logger.info(
            "{} -- {}".format(
                datatype,
                ", ".join(
                    [k + ": " + str(f"{v:.4f}") for k, v in eval_results.items()]
                ),
            )
        )

        return eval_results

    def fit(
        self,
        train_loader: Any,
        test_loader: Optional[Any] = None,
        evaluation_epoch: int = 10,
        checkpoint_frequency: int = 5,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, float]], Optional[int]]:
        """
        Train the model with enhanced experiment management

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            evaluation_epoch: Number of epochs between evaluations
            checkpoint_frequency: Save checkpoint every N epochs
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Tuple containing the best evaluation results and the epoch of convergence
        """
        best_hr1, coverage, eval_res = -1, None, None
        pre_loss, worse_count = float("inf"), 0
        start_epoch = 1

        from torch.optim.adam import Adam

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = self._create_lr_scheduler(optimizer)

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            try:
                checkpoint = self.exp_manager.load_checkpoint(resume_from_checkpoint)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_hr1 = checkpoint.get("metrics", {}).get("HR@1", -1)
                logger.info(
                    f"Resumed training from epoch {start_epoch}, best HR@1: {best_hr1:.4f}"
                )
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                raise

        train_losses = []
        test_metrics_history = []

        for epoch in range(start_epoch, self.epochs + 1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()

            # Apply learning rate warmup
            self._warmup_lr(optimizer, epoch)

            for graph, groundtruth in train_loader:
                optimizer.zero_grad()
                result = self.model.forward(graph.to(self.device), groundtruth)
                loss = result["loss"]
                loss.backward()

                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1

            # Step scheduler after warmup period (except for ReduceLROnPlateau)
            if (
                lr_scheduler is not None
                and epoch > self.lr_warmup_epochs
                and not isinstance(
                    lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                )
            ):
                lr_scheduler.step()

            epoch_time_elapsed = time.time() - epoch_time_start
            avg_epoch_loss = epoch_loss / batch_cnt
            train_losses.append(avg_epoch_loss)

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Log training metrics
            training_metrics = {
                "loss": avg_epoch_loss,
                "lr": current_lr,
                "time": epoch_time_elapsed,
            }
            self.exp_manager.log_training_step(epoch, training_metrics)

            logger.info(
                "Epoch {}/{}, training loss: {:.5f}, lr: {:.6f} [{:.2f}s]".format(
                    epoch, self.epochs, avg_epoch_loss, current_lr, epoch_time_elapsed
                )
            )

            # Early stopping mechanism
            if avg_epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logger.info("Early stop at epoch: {}".format(epoch))
                    break
            else:
                worse_count = 0
            pre_loss = avg_epoch_loss

            # Periodic evaluation on the test set
            if test_loader is not None and (epoch % evaluation_epoch == 0):
                test_results = self.evaluate(test_loader, datatype="Test")
                test_metrics_history.append(
                    {"epoch": epoch, "metrics": test_results.copy()}
                )

                # Step ReduceLROnPlateau scheduler based on test loss
                if lr_scheduler is not None and isinstance(
                    lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    lr_scheduler.step(avg_epoch_loss)

                is_best = test_results["HR@1"] > best_hr1
                if is_best:
                    best_hr1, eval_res, coverage = (
                        test_results["HR@1"],
                        test_results,
                        epoch,
                    )

                # Save checkpoint with experiment manager
                self.exp_manager.save_checkpoint(
                    model_state=copy.deepcopy(self.model.state_dict()),
                    epoch=epoch,
                    metrics=test_results,
                    optimizer_state=copy.deepcopy(optimizer.state_dict()),
                    is_best=is_best,
                    save_frequency=checkpoint_frequency,
                )

            # Save regular checkpoint even without evaluation
            elif epoch % checkpoint_frequency == 0:
                self.exp_manager.save_checkpoint(
                    model_state=copy.deepcopy(self.model.state_dict()),
                    epoch=epoch,
                    metrics={"loss": avg_epoch_loss},
                    optimizer_state=copy.deepcopy(optimizer.state_dict()),
                    is_best=False,
                    save_frequency=checkpoint_frequency,
                )

        # Final summary
        if coverage and coverage > 5:
            logger.info(
                "* Best result got at epoch {} with HR@1: {:.4f}".format(
                    coverage, best_hr1
                )
            )
        else:
            logger.info("Unable to convergence!")

        # Print experiment summary
        summary = self.exp_manager.get_experiment_summary()
        logger.info("Experiment Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        return eval_res, coverage

    def load_model(self, model_save_file: str = "") -> None:
        """
        Load a pre-trained model

        Args:
            model_save_file: Path to the model file
        """
        self.model.load_state_dict(
            torch.load(model_save_file, map_location=self.device)
        )

    def save_model(self, state: Dict[str, Any], file: Optional[str] = None) -> None:
        """
        Save the model state

        Args:
            state: Model state dictionary
            file: Path to save the file, defaults to None to use the default path
        """
        if file is None:
            file = os.path.join(self.model_save_dir, "model.ckpt")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=False)
        except Exception:
            torch.save(state, file)

    def load_from_checkpoint(
        self, checkpoint_path: Optional[str] = None, load_best: bool = True
    ) -> None:
        """
        Load model from checkpoint using experiment manager

        Args:
            checkpoint_path: Specific checkpoint path, if None will load best or latest
            load_best: If True and no specific path, load best checkpoint, otherwise latest
        """
        checkpoint = self.exp_manager.load_checkpoint(checkpoint_path, load_best)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Model loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}"
        )

    def inference(self, data_loader: Any, save_results: bool = True) -> Dict[str, Any]:
        """
        Run inference on data and optionally save results

        Args:
            data_loader: Data loader for inference
            save_results: Whether to save inference results to file

        Returns:
            Dictionary containing inference results
        """
        self.model.eval()
        all_predictions = []
        all_ground_truths = []
        all_probabilities = []
        inference_metrics = {}

        with torch.no_grad():
            for graph, ground_truths in data_loader:
                result = self.model.forward(graph.to(self.device), ground_truths)

                for idx, faulty_nodes in enumerate(result["y_pred"]):
                    all_predictions.append(faulty_nodes)
                    all_ground_truths.append(ground_truths[idx].item())
                    all_probabilities.append(result["y_prob"][idx])

        # Calculate inference metrics
        inference_metrics = self._calculate_inference_metrics(
            all_predictions, all_ground_truths, all_probabilities
        )

        inference_results = {
            "predictions": all_predictions,
            "ground_truths": all_ground_truths,
            "probabilities": all_probabilities,
            "metrics": inference_metrics,
            "total_samples": len(all_predictions),
        }

        if save_results:
            # Save results using experiment manager
            results_path = self.exp_manager.save_inference_results(inference_results)
            logger.info(f"Inference results saved to: {results_path}")

        return inference_results

    def _calculate_inference_metrics(
        self, predictions: list, ground_truths: list, probabilities: list
    ) -> Dict[str, float]:
        """Calculate metrics for inference results"""
        hrs, ndcgs = np.zeros(5), np.zeros(5)
        TP, FP, FN = 0, 0, 0

        for idx, (faulty_nodes, culprit, probs) in enumerate(
            zip(predictions, ground_truths, probabilities)
        ):
            if culprit == -1:  # Normal sample
                if faulty_nodes[0] == -1:
                    TP += 1  # Correctly predicted as normal
                else:
                    FP += 1  # Incorrectly predicted as abnormal
            else:  # Abnormal sample
                if faulty_nodes[0] == -1:
                    FN += 1  # Incorrectly predicted as normal
                else:
                    TP += 1  # Correctly predicted as abnormal
                    if culprit in faulty_nodes:
                        rank = list(faulty_nodes).index(culprit)
                        for j in range(5):
                            hrs[j] += int(rank <= j)
                            ndcgs[j] += ndcg_score(
                                np.array([probs]).reshape(1, -1),
                                np.array([faulty_nodes]).reshape(1, -1),
                                k=j + 1,
                            )

        pos = TP + FN  # Total number of positive samples
        metrics = {
            "F1": TP * 2.0 / (TP + FP + pos) if (TP + FP + pos) > 0 else 0,
            "Recall": TP * 1.0 / pos if pos > 0 else 0,
            "Precision": TP * 1.0 / (TP + FP) if (TP + FP) > 0 else 0,
        }

        # Calculate Hit Rate and NDCG metrics
        for j in [1, 3, 5]:
            metrics[f"HR@{j}"] = hrs[j - 1] * 1.0 / pos if pos > 0 else 0
            metrics[f"NDCG@{j}"] = ndcgs[j - 1] * 1.0 / pos if pos > 0 else 0

        return metrics

    def get_experiment_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of all checkpoints in current experiment"""
        return self.exp_manager.list_checkpoints()

    def cleanup_experiment(self, keep_best: int = 3, keep_recent: int = 5):
        """Clean up old checkpoints to save disk space"""
        self.exp_manager.cleanup_old_checkpoints(keep_best, keep_recent)

    @classmethod
    def from_experiment(
        cls,
        experiment_name: str,
        event_num: int,
        metric_num: int,
        node_num: int,
        device: str,
        config: Any,
        checkpoint_path: Optional[str] = None,
        load_best: bool = True,
    ) -> "BaseModel":
        exp_manager = ExperimentManager.load_experiment(
            experiment_name, config.get("paths.result_dir", "result")
        )

        # Create model instance
        model = cls(event_num, metric_num, node_num, device, config, experiment_name)
        model.exp_manager = exp_manager

        # Load checkpoint
        model.load_from_checkpoint(checkpoint_path, load_best)

        return model
