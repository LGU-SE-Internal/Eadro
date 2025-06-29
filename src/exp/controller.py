import json
import shutil
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    Optional,
    List,
    Callable,
    TypeVar,
    Generic,
    Protocol,
    Union,
    Literal,
    TypedDict,
)

from loguru import logger

from .config import Config

# Type aliases - removing Any usage
PredictionResult = Dict[str, Union[str, int, float, bool, List, Dict]]
MetricsDict = Dict[str, float]
ConfigDict = Dict[str, Union[str, int, float, bool, List, Dict]]
ExperimentStatus = Literal["initialized", "training", "completed", "failed", "paused"]


# More specific checkpoint data type
class CheckpointData(TypedDict):
    epoch: int
    model_state_dict: Dict[str, torch.Tensor]
    metrics: MetricsDict
    timestamp: str
    experiment_name: str
    optimizer_state_dict: Optional[Dict[str, torch.Tensor]]
    additional_data: Optional[ConfigDict]


# Typed dictionary for better type safety
class TrainingInfo(TypedDict):
    total_epochs: int
    training_time: float
    best_epoch: int


class CheckpointInfo(TypedDict):
    epoch: int
    timestamp: str
    metrics: MetricsDict
    is_best: bool
    paths: List[str]


class MetricsLogEntry(TypedDict):
    epoch: int
    phase: str
    timestamp: str
    metrics: MetricsDict


class ExperimentMetadata(TypedDict):
    experiment_name: str
    created_at: str
    updated_at: str
    description: str
    tags: List[str]
    config: ConfigDict
    status: ExperimentStatus
    metrics_history: List[MetricsLogEntry]
    checkpoints: List[CheckpointInfo]
    best_metrics: MetricsDict
    training_info: TrainingInfo


# Generic type variables
ModelT = TypeVar("ModelT", bound="HasStateDict")
DataT = TypeVar("DataT")
OptimizerT = TypeVar("OptimizerT", bound="HasStateDict")
SchedulerT = TypeVar("SchedulerT", bound="HasStep")


# Protocol for objects with state_dict
class HasStateDict(Protocol):
    def state_dict(self) -> Dict[str, torch.Tensor]: ...
    def load_state_dict(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ) -> object: ...


# Protocol for schedulers with step method
class HasStep(Protocol):
    def step(self) -> None: ...


class ModelHandler(ABC, Generic[ModelT]):
    @abstractmethod
    def save_model(self, model: ModelT, path: str) -> None:
        pass

    @abstractmethod
    def load_model(self, path: str) -> ModelT:
        pass

    @abstractmethod
    def get_model_state(self, model: ModelT) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def load_model_state(
        self, model: ModelT, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Load state dictionary into model"""
        pass


class PyTorchModelHandler(ModelHandler[torch.nn.Module]):
    def __init__(self, device: str = "cpu"):
        self.device = device

    def save_model(self, model: torch.nn.Module, path: str) -> None:
        torch.save(model, path)

    def load_model(self, path: str) -> torch.nn.Module:
        return torch.load(path, map_location=self.device)

    def get_model_state(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return model.state_dict()

    def load_model_state(
        self, model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        model.load_state_dict(state_dict)


class DataHandler(ABC, Generic[DataT]):
    @abstractmethod
    def prepare_data(self, config: Config) -> DataT:
        pass

    @abstractmethod
    def get_data_info(self) -> ConfigDict:
        """Get data information and statistics"""
        pass


class TrainingHandler(ABC, Generic[ModelT, DataT, OptimizerT]):
    @abstractmethod
    def train_epoch(
        self, model: ModelT, data: DataT, optimizer: OptimizerT, **kwargs
    ) -> MetricsDict:
        pass

    @abstractmethod
    def validate(self, model: ModelT, data: DataT, **kwargs) -> MetricsDict:
        pass

    @abstractmethod
    def setup_optimizer(self, model: ModelT, config: Config) -> OptimizerT:
        pass


class InferenceHandler(ABC, Generic[ModelT, DataT]):
    @abstractmethod
    def predict(self, model: ModelT, data: DataT, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def postprocess_results(
        self, predictions: torch.Tensor, **kwargs
    ) -> PredictionResult:
        pass


class UniversalExperimentManager(Generic[ModelT, DataT, OptimizerT]):
    def __init__(
        self,
        config: Config,
        experiment_name: Optional[str] = None,
        model_handler: Optional[ModelHandler[ModelT]] = None,
        data_handler: Optional[DataHandler[DataT]] = None,
        training_handler: Optional[TrainingHandler[ModelT, DataT, OptimizerT]] = None,
        inference_handler: Optional[InferenceHandler[ModelT, DataT]] = None,
    ):
        # Setup configuration
        self.config = config

        # Setup experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = self.config.get("dataset")
            name_prefix = self.config.get("experiment.name_prefix")
            experiment_name = f"{name_prefix}_{dataset_name}_{timestamp}"

        self.experiment_name: str = experiment_name

        # Setup handlers with defaults
        device = (
            self.config.get("training.device")
            if self.config.get("training.gpu")
            else "cpu"
        )
        self.model_handler: ModelHandler[ModelT] = model_handler or PyTorchModelHandler(
            device=device
        )  # type: ignore
        self.data_handler: Optional[DataHandler[DataT]] = data_handler
        self.training_handler: Optional[TrainingHandler[ModelT, DataT, OptimizerT]] = (
            training_handler
        )
        self.inference_handler: Optional[InferenceHandler[ModelT, DataT]] = (
            inference_handler
        )

        # Setup directories
        self._setup_directories()

        # Initialize metadata
        self._initialize_metadata()

        # Setup logging
        self._setup_logging()

        logger.info(f"Initialized experiment: {experiment_name}")
        logger.info(f"Experiment directory: {self.experiment_dir}")

    def _setup_directories(self) -> None:
        base_dir = Path(self.config.get("paths.result_dir"))
        self.experiment_dir: Path = base_dir / self.experiment_name

        self.checkpoint_dir: Path = self.experiment_dir / self.config.get(
            "paths.checkpoint_dir"
        )
        self.logs_dir: Path = self.experiment_dir / self.config.get("paths.logs_dir")
        self.inference_dir: Path = self.experiment_dir / self.config.get(
            "paths.inference_dir"
        )

        # Create directories
        for dir_path in [
            self.experiment_dir,
            self.checkpoint_dir,
            self.logs_dir,
            self.inference_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _serialize_config(self) -> ConfigDict:
        """Serialize configuration to dictionary"""
        # Return config as dict for serialization
        return {"config_serialized": True}  # Simplified for now

    def _initialize_metadata(self) -> None:
        """Initialize experiment metadata"""
        self.metadata: ExperimentMetadata = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "description": self.config.get("experiment.description"),
            "tags": self.config.get("experiment.tags"),
            "config": self._serialize_config(),
            "status": "initialized",
            "metrics_history": [],
            "checkpoints": [],
            "best_metrics": {},
            "training_info": {
                "total_epochs": 0,
                "training_time": 0.0,
                "best_epoch": -1,
            },
        }

        self._save_metadata()

    def _deep_convert_for_json(
        self, obj: object
    ) -> Union[Dict, List, str, int, float, bool, None]:
        """Recursively convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {
                key: self._deep_convert_for_json(value) for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._deep_convert_for_json(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "item") and callable(getattr(obj, "item")):  # numpy scalars
            return obj.item()  # type: ignore
        elif hasattr(obj, "tolist") and callable(
            getattr(obj, "tolist")
        ):  # numpy arrays
            return obj.tolist()  # type: ignore
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            return str(obj)

    def _save_metadata(self) -> None:
        """Save metadata to file"""
        self.metadata["updated_at"] = datetime.now().isoformat()
        metadata_path = self.experiment_dir / "experiment_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _setup_logging(self) -> None:
        """Setup experiment logging"""
        log_level = self.config.get("experiment.log_level")
        log_rotation = self.config.get("experiment.log_rotation")
        log_file = self.logs_dir / f"{self.experiment_name}.log"

        # Configure loguru logger
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation=log_rotation,
        )

    def save_checkpoint(
        self,
        model: ModelT,
        epoch: int,
        metrics: MetricsDict,
        optimizer: Optional[OptimizerT] = None,
        is_best: bool = False,
        additional_data: Optional[ConfigDict] = None,
    ) -> str:
        """Save model checkpoint"""
        timestamp = datetime.now().isoformat()

        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model_handler.get_model_state(model),
            "metrics": metrics,
            "timestamp": timestamp,
            "experiment_name": self.experiment_name,
        }

        if optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()

        if additional_data:
            checkpoint_data["additional_data"] = additional_data

        # Save checkpoint files
        checkpoint_paths = []

        # Regular checkpoint (based on frequency)
        frequency = self.config.get("experiment.checkpoint_frequency")
        assert frequency is not None, (
            "experiment.checkpoint_frequency must be configured"
        )
        frequency = int(frequency)
        if epoch % frequency == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.ckpt"
            torch.save(checkpoint_data, checkpoint_path)
            checkpoint_paths.append(str(checkpoint_path))
            logger.debug(f"Saved checkpoint at epoch {epoch}: {checkpoint_path}")

        # Best checkpoint
        if is_best:
            best_checkpoint_path = self.checkpoint_dir / "best_model.ckpt"
            torch.save(checkpoint_data, best_checkpoint_path)

            # Timestamped best checkpoint
            best_timestamped_path = (
                self.checkpoint_dir / f"best_model_epoch_{epoch:04d}.ckpt"
            )
            torch.save(checkpoint_data, best_timestamped_path)

            checkpoint_paths.extend(
                [str(best_checkpoint_path), str(best_timestamped_path)]
            )

            # Update best metrics
            self.metadata["best_metrics"] = metrics.copy()
            self.metadata["training_info"]["best_epoch"] = epoch

            logger.debug(
                f"Saved best checkpoint at epoch {epoch}: {best_checkpoint_path}"
            )

        # Latest checkpoint (always save)
        latest_checkpoint_path = self.checkpoint_dir / "latest_model.ckpt"
        torch.save(checkpoint_data, latest_checkpoint_path)
        checkpoint_paths.append(str(latest_checkpoint_path))

        # Update checkpoint registry
        checkpoint_info: CheckpointInfo = {
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": metrics.copy(),
            "is_best": is_best,
            "paths": checkpoint_paths,
        }
        self.metadata["checkpoints"].append(checkpoint_info)

        # Update training info
        self.metadata["training_info"]["total_epochs"] = max(
            self.metadata["training_info"]["total_epochs"], epoch
        )

        # Cleanup old checkpoints if needed
        auto_cleanup = self.config.get("experiment.auto_cleanup")
        assert auto_cleanup is not None, "experiment.auto_cleanup must be configured"
        if auto_cleanup:
            self._cleanup_checkpoints()

        self._save_metadata()

        return str(latest_checkpoint_path)

    def log_metrics(
        self, epoch: int, metrics: MetricsDict, phase: str = "train"
    ) -> None:
        """Log metrics for the current epoch"""
        entry: MetricsLogEntry = {
            "epoch": epoch,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy(),
        }

        self.metadata["metrics_history"].append(entry)

        # Save metadata periodically
        frequency = self.config.get("experiment.metadata_save_frequency")
        if epoch % frequency == 0:
            self._save_metadata()

    def save_inference_results(
        self, results: PredictionResult, filename: Optional[str] = None
    ) -> str:
        """Save inference results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_results_{timestamp}.json"

        results_path = self.inference_dir / filename

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved inference results to: {results_path}")
        return str(results_path)

    def get_experiment_summary(self) -> ConfigDict:
        """Get experiment summary"""
        return {
            "experiment_name": self.experiment_name,
            "status": self.metadata["status"],
            "total_epochs": self.metadata["training_info"]["total_epochs"],
            "best_epoch": self.metadata["training_info"]["best_epoch"],
            "best_metrics": self.metadata["best_metrics"],
            "total_checkpoints": len(self.metadata["checkpoints"]),
            "metrics_entries": len(self.metadata["metrics_history"]),
        }

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = True,
        map_location: Optional[str] = None,
    ) -> CheckpointData:
        """Load model checkpoint"""
        if checkpoint_path is None:
            if load_best and (self.checkpoint_dir / "best_model.ckpt").exists():
                checkpoint_path_obj = self.checkpoint_dir / "best_model.ckpt"
            else:
                checkpoint_path_obj = self.checkpoint_dir / "latest_model.ckpt"
        else:
            checkpoint_path_obj = Path(checkpoint_path)

        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")

        if map_location is None:
            map_location = "cuda" if self.config.get("training.gpu") else "cpu"

        checkpoint = torch.load(checkpoint_path_obj, map_location=map_location)

        logger.info(f"Loaded checkpoint from: {checkpoint_path_obj}")
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

        return checkpoint

    def load_model_from_checkpoint(
        self,
        model: ModelT,
        checkpoint_path: Optional[str] = None,
        load_best: bool = True,
        load_optimizer: bool = False,
        optimizer: Optional[OptimizerT] = None,
    ) -> CheckpointData:
        """Load model (and optionally optimizer) from checkpoint"""
        checkpoint = self.load_checkpoint(checkpoint_path, load_best)

        # Load model state
        if "model_state_dict" in checkpoint:
            self.model_handler.load_model_state(model, checkpoint["model_state_dict"])

        # Load optimizer state if requested
        if (
            load_optimizer
            and optimizer is not None
            and "optimizer_state_dict" in checkpoint
            and checkpoint["optimizer_state_dict"] is not None
        ):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    def run_training(
        self,
        model: ModelT,
        train_data: DataT,
        val_data: Optional[DataT] = None,
        optimizer: Optional[OptimizerT] = None,
        scheduler: Optional[SchedulerT] = None,
        num_epochs: Optional[int] = None,
        callbacks: Optional[
            List[
                Callable[
                    [
                        "UniversalExperimentManager[ModelT, DataT, OptimizerT]",
                        int,
                        MetricsDict,
                    ],
                    None,
                ]
            ]
        ] = None,
    ) -> ConfigDict:
        """Run training loop"""
        assert self.training_handler is not None, "Training handler not provided"

        if num_epochs is None:
            num_epochs = self.config.get("training.epochs")

        # Ensure num_epochs is an integer and not None
        assert num_epochs is not None, "num_epochs cannot be None"
        num_epochs = int(num_epochs)

        if optimizer is None:
            optimizer = self.training_handler.setup_optimizer(model, self.config)

        self.metadata["status"] = "training"
        self._save_metadata()

        best_metric = float("-inf")
        patience_counter = 0
        patience = self.config.get("training.patience")
        assert patience is not None, "training.patience must be configured"
        patience = int(patience)

        training_start_time = datetime.now()

        try:
            for epoch in range(num_epochs):
                epoch_start_time = datetime.now()

                # Training phase
                train_metrics = self.training_handler.train_epoch(
                    model, train_data, optimizer, epoch=epoch
                )
                self.log_metrics(epoch, train_metrics, "train")

                # Validation phase
                val_metrics = {}
                if val_data is not None:
                    val_metrics = self.training_handler.validate(
                        model, val_data, epoch=epoch
                    )
                    self.log_metrics(epoch, val_metrics, "validation")

                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}

                # Check if best model
                primary_metric = self.config.get("experiment.primary_metric")
                assert primary_metric is not None, (
                    "experiment.primary_metric must be configured"
                )
                is_best = False

                if primary_metric in all_metrics:
                    current_metric = all_metrics[primary_metric]
                    if current_metric > best_metric:
                        best_metric = current_metric
                        is_best = True
                        patience_counter = 0
                    else:
                        patience_counter += 1

                # Save checkpoint
                self.save_checkpoint(
                    model, epoch, all_metrics, optimizer, is_best=is_best
                )

                # Learning rate scheduling
                if scheduler is not None:
                    scheduler.step()

                # Run callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(self, epoch, all_metrics)

                # Early stopping check
                if patience_counter >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} (patience: {patience})"
                    )
                    break

                epoch_time = (datetime.now() - epoch_start_time).total_seconds()
                logger.debug(f"Epoch {epoch} completed in {epoch_time:.2f}s")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            self.metadata["status"] = "failed"
            self._save_metadata()
            raise

        training_time = (datetime.now() - training_start_time).total_seconds()
        self.metadata["training_info"]["training_time"] = training_time
        self.metadata["status"] = "completed"
        self._save_metadata()

        logger.info(f"Training completed in {training_time:.2f}s")

        return self.get_experiment_summary()

    def run_inference(
        self,
        model: ModelT,
        data: DataT,
        checkpoint_path: Optional[str] = None,
        load_best: bool = True,
        save_results: bool = True,
        **inference_kwargs,
    ) -> PredictionResult:
        assert self.inference_handler is not None, "Inference handler not provided"

        if checkpoint_path is not None or load_best:
            self.load_model_from_checkpoint(model, checkpoint_path, load_best)

        predictions = self.inference_handler.predict(model, data, **inference_kwargs)

        # Postprocess results
        results = self.inference_handler.postprocess_results(
            predictions, **inference_kwargs
        )

        # Save results if requested
        if save_results:
            self.save_inference_results(results)

        return results

    def _cleanup_checkpoints(self) -> None:
        """Cleanup old checkpoints based on configuration"""
        max_checkpoints = self.config.get("experiment.max_checkpoints")
        assert max_checkpoints is not None, (
            "experiment.max_checkpoints must be configured"
        )
        max_checkpoints = int(max_checkpoints)

        if len(self.metadata["checkpoints"]) <= max_checkpoints:
            return

        # Sort checkpoints by epoch (newest first)
        checkpoints = sorted(
            self.metadata["checkpoints"], key=lambda x: x["epoch"], reverse=True
        )

        # Keep the most recent checkpoints and best checkpoints
        keep_checkpoints = checkpoints[:max_checkpoints]

        # Add best checkpoints to keep list
        best_checkpoints = [cp for cp in checkpoints if cp.get("is_best", False)]
        for best_cp in best_checkpoints:
            if best_cp not in keep_checkpoints:
                keep_checkpoints.append(best_cp)

        # Remove old checkpoint files
        keep_epochs = {cp["epoch"] for cp in keep_checkpoints}
        removed_count = 0

        for checkpoint in checkpoints:
            if checkpoint["epoch"] not in keep_epochs:
                for path in checkpoint.get("paths", []):
                    checkpoint_path = Path(path)
                    if (
                        checkpoint_path.exists()
                        and "best_model" not in checkpoint_path.name
                    ):
                        checkpoint_path.unlink()
                        removed_count += 1

        # Update metadata
        self.metadata["checkpoints"] = keep_checkpoints

        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} old checkpoint files")

    @classmethod
    def load_experiment(
        cls,
        experiment_name: str,
        config: Config,
        model_handler: Optional[ModelHandler] = None,
        data_handler: Optional[DataHandler] = None,
        training_handler: Optional[TrainingHandler] = None,
        inference_handler: Optional[InferenceHandler] = None,
    ) -> "UniversalExperimentManager":
        """Load existing experiment"""
        base_dir = Path(config.get("paths.result_dir"))
        experiment_dir = base_dir / experiment_name
        metadata_path = experiment_dir / "experiment_metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Experiment metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Create experiment manager
        exp_manager = cls.__new__(cls)
        exp_manager.config = config
        exp_manager.experiment_name = experiment_name
        exp_manager.experiment_dir = experiment_dir
        exp_manager.checkpoint_dir = experiment_dir / config.get("paths.checkpoint_dir")
        exp_manager.logs_dir = experiment_dir / config.get("paths.logs_dir")
        exp_manager.inference_dir = experiment_dir / config.get("paths.inference_dir")
        exp_manager.metadata = metadata

        # Setup handlers
        device = config.get("training.device") if config.get("training.gpu") else "cpu"
        exp_manager.model_handler = model_handler or PyTorchModelHandler(device=device)
        exp_manager.data_handler = data_handler
        exp_manager.training_handler = training_handler
        exp_manager.inference_handler = inference_handler

        # Setup logging
        exp_manager._setup_logging()

        logger.info(f"Loaded existing experiment: {experiment_name}")
        return exp_manager
