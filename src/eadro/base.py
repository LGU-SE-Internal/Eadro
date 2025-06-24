import os
import time
import copy
from typing import Dict, Any, Optional, Tuple

import torch
from torch import nn
import logging
import numpy as np
from sklearn.metrics import ndcg_score
import wandb

from .model import MainModel


class BaseModel(nn.Module):
    """Base training class for the EADRO model"""

    def __init__(
        self,
        event_num: int,
        metric_num: int,
        node_num: int,
        device: str,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 5,
        result_dir: str = "./",
        hash_id: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "eadro-training",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the base model for EADRO

        Args:
            event_num: Number of event types
            metric_num: Number of metrics
            node_num: Number of nodes
            device: Device to run on ("cpu" or "cuda")
            lr: Learning rate
            epochs: Number of training epochs
            patience: Early stopping patience
            result_dir: Directory to save results
            hash_id: Experiment hash ID
            use_wandb: Whether to use wandb for logging
            wandb_project: Wandb project name
        """
        super(BaseModel, self).__init__()

        self.epochs = epochs
        self.lr = lr
        self.patience = patience  # > 0: use early stop
        self.device = device
        self.use_wandb = use_wandb

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=f"run_{hash_id}" if hash_id else None,
                config={
                    "lr": lr,
                    "epochs": epochs,
                    "batch_size": kwargs.get("batch_size", "unknown"),
                    "patience": patience,
                    "alpha": kwargs.get("alpha", "unknown"),
                    "event_num": event_num,
                    "metric_num": metric_num,
                    "node_num": node_num,
                    **kwargs,
                },
            )

        self.model_save_dir = (
            os.path.join(result_dir, hash_id) if hash_id else result_dir
        )
        self.model = MainModel(event_num, metric_num, node_num, device, **kwargs)
        self.model.to(device)

    def evaluate(
        self, test_loader: Any, datatype: str = "Test", log_to_wandb: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            test_loader: Test data loader
            datatype: Data type identifier ("Test", "Train", "Val")
            log_to_wandb: Whether to log metrics to wandb

        Returns:
            Dictionary containing various evaluation metrics
        """
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

        logging.info(
            "{} -- {}".format(
                datatype,
                ", ".join(
                    [k + ": " + str(f"{v:.4f}") for k, v in eval_results.items()]
                ),
            )
        )

        if self.use_wandb and log_to_wandb:
            wandb_metrics = {}
            for k, v in eval_results.items():
                wandb_metrics[f"{datatype.lower()}_{k}"] = v
            wandb_metrics[f"{datatype.lower()}_loss"] = epoch_loss / batch_cnt
            wandb.log(wandb_metrics)

        return eval_results

    def fit(
        self,
        train_loader: Any,
        test_loader: Optional[Any] = None,
        evaluation_epoch: int = 10,
    ) -> Tuple[Optional[Dict[str, float]], Optional[int]]:
        """
        Train the model

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            evaluation_epoch: Number of epochs between evaluations

        Returns:
            Tuple containing the best evaluation results and the epoch of convergence
        """
        best_hr1, coverage, best_state, eval_res = -1, None, None, None  # evaluation
        pre_loss, worse_count = float("inf"), 0

        from torch.optim.adam import Adam

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99)

        train_losses = []
        train_metrics_history = []
        test_metrics_history = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()

            for graph, label in train_loader:
                optimizer.zero_grad()
                result = self.model.forward(graph.to(self.device), label)
                loss = result["loss"]
                loss.backward()

                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1

            epoch_time_elapsed = time.time() - epoch_time_start
            avg_epoch_loss = epoch_loss / batch_cnt
            train_losses.append(avg_epoch_loss)

            train_metrics = None
            if epoch % 5 == 0 or epoch == 1:
                train_metrics = self.evaluate(
                    train_loader, datatype="Train", log_to_wandb=False
                )
                train_metrics_history.append(
                    {"epoch": epoch, "metrics": train_metrics.copy()}
                )

            logging.info(
                "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(
                    epoch, self.epochs, avg_epoch_loss, epoch_time_elapsed
                )
            )

            if self.use_wandb:
                wandb_data = {
                    "epoch": epoch,
                    "train_loss": avg_epoch_loss,
                    "epoch_time": epoch_time_elapsed,
                    "worse_count": worse_count,
                }
                if train_metrics:
                    for metric, value in train_metrics.items():
                        wandb_data[f"train_{metric}"] = value

                wandb.log(wandb_data)

            # Early stopping mechanism
            if avg_epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
            else:
                worse_count = 0
            pre_loss = avg_epoch_loss

            # Periodic evaluation on the test set
            if (epoch + 1) % evaluation_epoch == 0:
                test_results = self.evaluate(test_loader, datatype="Test")
                test_metrics_history.append(
                    {"epoch": epoch, "metrics": test_results.copy()}
                )

                if test_results["HR@1"] > best_hr1:
                    best_hr1, eval_res, coverage = (
                        test_results["HR@1"],
                        test_results,
                        epoch,
                    )
                    best_state = copy.deepcopy(self.model.state_dict())

                if best_state is not None:
                    self.save_model(best_state)

        if coverage and coverage > 5:
            logging.info(
                "* Best result got at epoch {} with HR@1: {:.4f}".format(
                    coverage, best_hr1
                )
            )
        else:
            logging.info("Unable to convergence!")

        # Finish wandb run
        if self.use_wandb:
            wandb.finish()

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
