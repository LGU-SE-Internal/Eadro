import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from loguru import logger
from .config import Config


class ExperimentManager:
    def __init__(self, config: Config, experiment_name: Optional[str] = None):
        self.config = config

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = config.get("dataset")
            experiment_name = f"exp_{dataset_name}_{timestamp}"

        self.experiment_name = experiment_name

        self.base_result_dir = Path(config.get("paths.result_dir"))
        self.experiment_dir = self.base_result_dir / experiment_name
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.inference_dir = self.experiment_dir / "inference"

        # Create directories
        for dir_path in [
            self.experiment_dir,
            self.checkpoint_dir,
            self.logs_dir,
            self.inference_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize experiment metadata
        self.experiment_metadata = {
            "experiment_name": experiment_name,
            "created_at": datetime.now().isoformat(),
            "dataset": config.get("dataset"),
            "config": self._serialize_config(config),
            "best_metrics": {},
            "checkpoints": [],
            "training_history": [],
        }

        # Save configuration
        self._save_config()

        logger.info(f"Initialized experiment: {experiment_name}")
        logger.info(f"Experiment directory: {self.experiment_dir}")

    def _serialize_config(self, config: Config) -> Dict[str, Any]:
        return self._extract_important_config_keys(config)

    def _extract_important_config_keys(self, config: Config) -> Dict[str, Any]:
        important_keys = [
            "dataset",
            "training.epochs",
            "training.batch_size",
            "training.lr",
            "training.patience",
            "training.train_ratio",
            "training.gpu",
            "training.random_seed",
            "training.evaluation_epoch",
            "training.lr_scheduler.type",
            "training.lr_scheduler.step_size",
            "training.lr_scheduler.gamma",
            "model.self_attn",
            "model.fuse_dim",
            "model.alpha",
            "paths.result_dir",
            "paths.data_root",
            "experiment.checkpoint_frequency",
            "experiment.auto_cleanup",
        ]

        config_dict: Dict[str, Any] = {}
        for key in important_keys:
            try:
                value = config.get(key)
                if value is not None:
                    config_dict[key] = self._convert_value_for_json(value)
            except Exception:
                # If key doesn't exist, skip it
                pass

        return config_dict

    def _deep_convert_for_json(self, obj) -> Any:
        """Recursively convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {
                key: self._deep_convert_for_json(value) for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._deep_convert_for_json(item) for item in obj]
        else:
            return self._convert_value_for_json(obj)

    def _convert_value_for_json(self, value) -> Any:
        """Convert individual value to JSON-serializable format"""
        if isinstance(value, (bool, int, float, str, type(None))):
            return value
        elif hasattr(value, "item"):  # numpy scalars
            return value.item()
        elif hasattr(value, "tolist"):  # numpy arrays
            return value.tolist()
        elif isinstance(value, Path):
            return str(value)
        else:
            return str(value)

    def _save_config(self):
        """Save current configuration to experiment directory"""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                self.experiment_metadata, f, indent=2, default=self._json_serializer
            )

    def _json_serializer(self, obj):
        """Custom JSON serializer for handling non-standard types"""
        if isinstance(obj, (bool, int, float, str)):
            return obj
        elif hasattr(obj, "item"):  # numpy scalars
            return obj.item()
        elif hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return str(obj)

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        epoch: int,
        metrics: Dict[str, float],
        optimizer_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        save_frequency: int = 5,
    ) -> str:
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "experiment_name": self.experiment_name,
        }

        if optimizer_state is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer_state

        if epoch % save_frequency == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.ckpt"
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch}: {checkpoint_path}")

        # Best checkpoint
        if is_best:
            best_checkpoint_path = self.checkpoint_dir / "best_model.ckpt"
            torch.save(checkpoint_data, best_checkpoint_path)

            # Also save as timestamped best checkpoint
            best_timestamped_path = (
                self.checkpoint_dir / f"best_model_epoch_{epoch:04d}.ckpt"
            )
            torch.save(checkpoint_data, best_timestamped_path)

            logger.info(
                f"Saved best checkpoint at epoch {epoch}: {best_checkpoint_path}"
            )

            # Update best metrics
            self.experiment_metadata["best_metrics"] = metrics.copy()
            self.experiment_metadata["best_epoch"] = epoch

        # Latest checkpoint (always overwrite)
        latest_checkpoint_path = self.checkpoint_dir / "latest_model.ckpt"
        torch.save(checkpoint_data, latest_checkpoint_path)

        # Update checkpoint registry
        checkpoint_info = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy(),
            "is_best": is_best,
            "path": str(
                checkpoint_path
                if epoch % save_frequency == 0
                else latest_checkpoint_path
            ),
        }
        self.experiment_metadata["checkpoints"].append(checkpoint_info)

        # Update metadata
        self._save_config()

        return str(latest_checkpoint_path)

    def load_checkpoint(
        self, checkpoint_path: Optional[str] = None, load_best: bool = True
    ) -> Dict[str, Any]:
        if checkpoint_path is None:
            if load_best:
                checkpoint_path_obj = self.checkpoint_dir / "best_model.ckpt"
                if not checkpoint_path_obj.exists():
                    checkpoint_path_obj = self.checkpoint_dir / "latest_model.ckpt"
            else:
                checkpoint_path_obj = self.checkpoint_dir / "latest_model.ckpt"
        else:
            checkpoint_path_obj = Path(checkpoint_path)

        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")

        checkpoint = torch.load(checkpoint_path_obj, map_location="cpu")
        logger.info(f"Loaded checkpoint from: {checkpoint_path_obj}")
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

        return checkpoint

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        return self.experiment_metadata.get("checkpoints", [])

    def log_training_step(self, epoch: int, metrics: Dict[str, Any]):
        log_entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        self.experiment_metadata["training_history"].append(log_entry)

        if epoch % 10 == 0:
            self._save_config()

    def save_inference_results(
        self, results: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_results_{timestamp}.json"

        results_path = self.inference_dir / filename

        results_with_metadata = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }

        with open(results_path, "w") as f:
            json.dump(results_with_metadata, f, indent=2, default=self._json_serializer)

        logger.info(f"Saved inference results: {results_path}")
        return str(results_path)

    def get_experiment_summary(self) -> Dict[str, Any]:
        summary = {
            "experiment_name": self.experiment_name,
            "total_checkpoints": len(self.experiment_metadata.get("checkpoints", [])),
            "best_metrics": self.experiment_metadata.get("best_metrics", {}),
            "best_epoch": self.experiment_metadata.get("best_epoch", None),
            "training_duration": self._calculate_training_duration(),
            "experiment_dir": str(self.experiment_dir),
        }
        return summary

    def _calculate_training_duration(self) -> Optional[str]:
        history = self.experiment_metadata.get("training_history", [])
        if len(history) < 2:
            return None

        start_time = datetime.fromisoformat(history[0]["timestamp"])
        end_time = datetime.fromisoformat(history[-1]["timestamp"])
        duration = end_time - start_time

        return str(duration)

    def cleanup_old_checkpoints(self, keep_best: int = 3, keep_recent: int = 5):
        checkpoints = self.experiment_metadata.get("checkpoints", [])

        # Sort by metrics (assuming HR@1 as primary metric)
        best_checkpoints = sorted(
            checkpoints, key=lambda x: x.get("metrics", {}).get("HR@1", 0), reverse=True
        )[:keep_best]

        # Get recent checkpoints
        recent_checkpoints = sorted(
            checkpoints, key=lambda x: x.get("epoch", 0), reverse=True
        )[:keep_recent]

        # Combine and deduplicate
        keep_paths = set()
        for cp in best_checkpoints + recent_checkpoints:
            keep_paths.add(cp.get("path"))

        # Remove checkpoints not in keep list
        removed_count = 0
        for checkpoint in checkpoints:
            path = checkpoint.get("path")
            if path and path not in keep_paths:
                checkpoint_path = Path(path)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    removed_count += 1

        logger.info(f"Cleaned up {removed_count} old checkpoints")

    @classmethod
    def load_experiment(
        cls, experiment_name: str, base_result_dir: str = "result"
    ) -> "ExperimentManager":
        experiment_dir = Path(base_result_dir) / experiment_name
        config_path = experiment_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {config_path}")

        with open(config_path, "r") as f:
            metadata = json.load(f)

        # Create a dummy config object - in practice you'd want to restore the actual config
        class DummyConfig:
            def __init__(self, config_dict):
                self._config = config_dict

            def get(self, key, default=None):
                keys = key.split(".")
                value = self._config
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return value

        config = DummyConfig(metadata.get("config", {}))

        exp_manager = cls.__new__(cls)
        exp_manager.config = config
        exp_manager.experiment_name = experiment_name
        exp_manager.base_result_dir = Path(base_result_dir)
        exp_manager.experiment_dir = experiment_dir
        exp_manager.checkpoint_dir = experiment_dir / "checkpoints"
        exp_manager.logs_dir = experiment_dir / "logs"
        exp_manager.inference_dir = experiment_dir / "inference"
        exp_manager.experiment_metadata = metadata

        logger.info(f"Loaded existing experiment: {experiment_name}")
        return exp_manager
