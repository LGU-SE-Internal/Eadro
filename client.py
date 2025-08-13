#!/usr/bin/env -S uv run -s
"""
Unified Eadro Client - Data preprocessing and training in one tool
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from src.eadro.handlers import (
    EadroDataHandler,
    EadroInferenceHandler,
    EadroModelHandler,
    EadroTrainingHandler,
    OptimizerAdapter,
    create_eadro_model,
)
from src.eadro.model import MainModel
from src.eadro.utils import seed_everything
from src.exp.config import Config
from src.exp.controller import UniversalExperimentManager
from src.preprocessing.processor import Processor

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def create_dataset(
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    n_workers: int = typer.Option(
        16, "--workers", help="Number of parallel workers for dataset creation"
    ),
    dataset_folder: str = typer.Option(
        ...,
        "--dataset-folder",
        help="Specific dataset folder name (e.g., __dev__rcabench_test_r1)",
    ),
    label: str = typer.Option(
        ...,
        "--label",
        help="Dataset label: train or test (all datapacks in this dataset will have this label)",
    ),
) -> None:
    """Create and preprocess dataset with specified label for all datapacks"""
    logger.info("Starting dataset creation...")

    if config_file is None:
        config_file = "settings.toml"

    try:
        if label not in ["train", "test"]:
            raise ValueError(f"Invalid label '{label}'. Must be 'train' or 'test'.")

        processor = Processor({}, conf=config_file)

        logger.info(
            f"Processing dataset folder: {dataset_folder} with label: {label} (all datapacks)"
        )
        processor.process_dataset_folder(
            dataset_folder=dataset_folder,
            label=label,
            use_parallel=True,
            n_workers=n_workers,
        )
        logger.info(
            f"Dataset creation completed for {dataset_folder} with label {label}!"
        )

    except Exception as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        raise


@app.command()
def train(
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    dataset_folder: str = typer.Option(
        ...,
        "--dataset-folder",
        help="Dataset folder name to use for training (contains only train data)",
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
) -> None:
    """Train the model using dataset with only train data"""
    if config_file is None:
        config_file = "settings.toml"
    config = Config(Path(config_file))

    logger.info("Starting model training...")
    logger.info(f"Training dataset folder: {dataset_folder} (pre-split train data)")

    try:
        random_seed = config.get("training.random_seed")
        seed_everything(random_seed)
        device = "cuda" if config.get("training.gpu") else "cpu"
        model_handler = EadroModelHandler(device)
        data_handler = EadroDataHandler()
        training_handler = EadroTrainingHandler(device)
        inference_handler = EadroInferenceHandler(device)

        # Use pre-split train data directly - no internal splitting needed
        data = data_handler.prepare_data(
            config,
            dataset_folder,
            train_split="train",
            test_split="train",  # Use train data for validation too since it's pre-split
            test_dataset_folder=None,
        )
        train_loader, val_loader, metadata = data

        config.set("node_num", len(metadata.services))
        config.set("event_num", len(metadata.log_templates) + 1)
        config.set("metric_num", len(metadata.metrics))

        model = create_eadro_model(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=device,
            config=config,
        )

        model = model.to(device)

        experiment_manager = UniversalExperimentManager[
            MainModel, tuple, OptimizerAdapter
        ](
            config=config,
            experiment_name=experiment_name,
            model_handler=model_handler,
            data_handler=data_handler,
            training_handler=training_handler,
            inference_handler=inference_handler,
        )

        optimizer = training_handler.setup_optimizer(model, config)
        num_epochs = config.get("training.epochs")

        training_results = experiment_manager.run_training(
            model=model,
            train_data=data,
            val_data=data,  # Using same data for validation since pre-split
            optimizer=optimizer,
            num_epochs=num_epochs,
        )

        logger.info(f"Training completed. Results: {training_results}")
        summary = experiment_manager.get_experiment_summary()
        logger.info(f"Experiment summary: {summary}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


@app.command()
def test(
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    checkpoint_path: str = typer.Option(
        ..., "--checkpoint", help="Path to trained model checkpoint"
    ),
    test_dataset_folder: str = typer.Option(
        ...,
        "--test-dataset-folder",
        help="Test dataset folder name (contains only test data)",
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Test experiment name"),
) -> None:
    """Test the trained model on test dataset"""
    if config_file is None:
        config_file = "settings.toml"
    config = Config(Path(config_file))

    logger.info("Starting model testing...")
    logger.info(f"Test dataset folder: {test_dataset_folder} (pre-split test data)")
    logger.info(f"Using checkpoint: {checkpoint_path}")

    try:
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        random_seed = config.get("training.random_seed")
        seed_everything(random_seed)
        device = "cuda" if config.get("training.gpu") else "cpu"

        model_handler = EadroModelHandler(device)
        data_handler = EadroDataHandler()
        training_handler = EadroTrainingHandler(device)
        inference_handler = EadroInferenceHandler(device)

        # Use pre-split test data directly
        data = data_handler.prepare_data(
            config,
            test_dataset_folder,
            train_split="test",  # Load test data as train split for consistency
            test_split="test",
            test_dataset_folder=None,
        )
        _, test_loader, metadata = data

        config.set("node_num", len(metadata.services))
        config.set("event_num", len(metadata.log_templates) + 1)
        config.set("metric_num", len(metadata.metrics))

        model = create_eadro_model(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=device,
            config=config,
        )

        model = model.to(device)

        test_experiment_name = (
            experiment_name
            or f"test_{test_dataset_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        experiment_manager = UniversalExperimentManager[
            MainModel, tuple, OptimizerAdapter
        ](
            config=config,
            experiment_name=test_experiment_name,
            model_handler=model_handler,
            data_handler=data_handler,
            training_handler=training_handler,
            inference_handler=inference_handler,
        )

        # Run inference on pre-split test data
        test_results = experiment_manager.run_inference(
            model=model,
            data=data,
            checkpoint_path=checkpoint_path,
            load_best=False,
            save_results=True,
        )

        logger.info(f"Testing completed. Results: {test_results}")

    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise


@app.command()
def train_test_pipeline(
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    train_dataset_folder: str = typer.Option(
        ...,
        "--train-dataset-folder",
        help="Training dataset folder name (all datapacks are train data)",
    ),
    test_dataset_folder: str = typer.Option(
        ...,
        "--test-dataset-folder",
        help="Test dataset folder name (all datapacks are test data)",
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
    skip_train_dataset: bool = typer.Option(
        False, "--skip-train-dataset", help="Skip training dataset creation step"
    ),
    skip_test_dataset: bool = typer.Option(
        False, "--skip-test-dataset", help="Skip test dataset creation step"
    ),
    skip_training: bool = typer.Option(
        False, "--skip-training", help="Skip training step"
    ),
    checkpoint_path: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="Checkpoint path for testing (required if skipping training)",
    ),
) -> None:
    """Run complete two-step pipeline: train dataset creation + training + test dataset creation + testing"""
    logger.info("Starting two-step train-test pipeline...")

    if config_file is None:
        config_file = "settings.toml"

    # Step 1: Create training dataset
    if not skip_train_dataset:
        logger.info("Step 1: Creating training dataset...")
        try:
            processor = Processor({}, conf=config_file)
            logger.info(
                f"Processing training dataset folder: {train_dataset_folder} with label: train"
            )
            processor.process_dataset_folder(
                dataset_folder=train_dataset_folder,
                label="train",
                use_parallel=True,
                n_workers=16,
            )
            logger.info("Training dataset creation completed!")
        except Exception as e:
            logger.error(f"Training dataset creation failed: {str(e)}")
            raise
    else:
        logger.info("Skipping training dataset creation step...")

    # Step 2: Train model
    if not skip_training:
        logger.info("Step 2: Training model...")
        train(config_file, train_dataset_folder, experiment_name)
        logger.info("Training completed!")
    else:
        logger.info("Skipping training step...")
        if checkpoint_path is None:
            raise ValueError("Checkpoint path is required when skipping training step")

    # Step 3: Create test dataset
    if not skip_test_dataset:
        logger.info("Step 3: Creating test dataset...")
        try:
            processor = Processor({}, conf=config_file)
            logger.info(
                f"Processing test dataset folder: {test_dataset_folder} with label: test"
            )
            processor.process_dataset_folder(
                dataset_folder=test_dataset_folder,
                label="test",
                use_parallel=True,
                n_workers=16,
            )
            logger.info("Test dataset creation completed!")
        except Exception as e:
            logger.error(f"Test dataset creation failed: {str(e)}")
            raise
    else:
        logger.info("Skipping test dataset creation step...")

    # Step 4: Test model
    logger.info("Step 4: Testing model...")
    if checkpoint_path is None:
        # Find the latest checkpoint from the training experiment
        # This would need to be implemented based on your experiment manager's checkpoint saving logic
        raise ValueError("Please specify checkpoint path for testing")

    test(
        config_file,
        checkpoint_path,
        test_dataset_folder,
        f"{experiment_name}_test" if experiment_name else None,
    )
    logger.info("Two-step pipeline completed successfully!")


@app.command()
def inference(
    checkpoint_path: str,
    datapack_path: Path,
    config_file: Optional[str] = "settings.toml",
) -> list:
    logger.info("Starting model inference...")

    if config_file is None:
        config_file = "settings.toml"
    config = Config(Path(config_file))

    try:
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        datapack_path_obj = datapack_path
        if not datapack_path_obj.exists():
            raise FileNotFoundError(f"Datapack directory not found: {datapack_path}")

        random_seed = config.get("training.random_seed")
        seed_everything(random_seed)

        processor = Processor({}, conf=config_file)

        dataset_name = datapack_path_obj.name
        config.set("dataset", dataset_name)

        processor.datapack_path = datapack_path_obj.parent
        processor.datapacks = [datapack_path_obj]
        processor.derive_files()

        logger.info("Creating metadata...")
        metadata = processor.create_metadata()

        logger.info("Processing datapack to generate samples...")
        from src.preprocessing.processor import process_datapack_parallel

        dataset_config = processor.dataset_config
        sample_interval = int(dataset_config.sample_interval)  # type: ignore
        sample_step = int(dataset_config.sample_step)  # type: ignore

        samples = process_datapack_parallel(
            datapack_path_obj,
            dataset_name,
            sample_interval,
            sample_step,
            metadata,
            processor.log_files,
            processor.metric_files,
            processor.trace_files,
            "drain.ini",
            "cache/drain/temp",
        )

        logger.info(f"Generated {len(samples)} samples for inference")

        if not samples:
            raise ValueError("No samples generated from the datapack")

        inference_handler = EadroInferenceHandler("cpu")

        from torch.utils.data import DataLoader

        from src.eadro.handlers import ChunkDataset, collate_fn

        inference_dataset = ChunkDataset(samples, metadata)
        batch_size = config.get("training.batch_size")

        inference_loader = DataLoader(
            inference_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        device = "cuda" if config.get("training.gpu") else "cpu"

        config.set("node_num", len(metadata.services))
        config.set("event_num", len(metadata.log_templates) + 1)
        config.set("metric_num", len(metadata.metrics))

        logger.info("Creating model...")
        model = create_eadro_model(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=device,
            config=config,
        )

        model = model.to(device)

        inference_experiment_name = f"inference_{datapack_path_obj.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_manager = UniversalExperimentManager[
            MainModel, tuple, OptimizerAdapter
        ](
            config=config,
            experiment_name=inference_experiment_name,
            model_handler=EadroModelHandler(device),
            data_handler=EadroDataHandler(),
            training_handler=EadroTrainingHandler(device),
            inference_handler=inference_handler,
        )

        data = (inference_loader, inference_loader, metadata)

        results = experiment_manager.run_inference(
            model=model,
            data=data,
            checkpoint_path=checkpoint_path,
            load_best=False,
            save_results=False,
        )

        logger.info(results)
        assert "predictions" in results and isinstance(results["predictions"], list)
        return results["predictions"]

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise


@app.command()
def create_dataset_by_id(
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    dataset_id: int = typer.Option(
        ..., "--dataset-id", help="Dataset ID from RCABench platform"
    ),
    label: str = typer.Option(
        ...,
        "--label",
        help="Dataset label: train or test",
    ),
) -> None:
    """Create and preprocess dataset using dataset ID from RCABench platform"""
    logger.info(f"Starting dataset creation for dataset ID: {dataset_id}")

    if config_file is None:
        config_file = "settings.toml"

    try:
        if label not in ["train", "test"]:
            raise ValueError(f"Invalid label '{label}'. Must be 'train' or 'test'.")

        # This will use the RCABench platform to get datapacks and process them
        # The actual processing will be handled in the data handler
        logger.info(
            f"Dataset creation setup completed for dataset ID {dataset_id} with label {label}"
        )
        logger.info("Data will be processed dynamically when training/testing begins")

    except Exception as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        raise


@app.command()
def train_by_id(
    dataset_id: int = typer.Option(
        ..., "--dataset-id", help="Dataset ID for training from RCABench platform"
    ),
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
) -> None:
    """Train the model using dataset ID (automatically labeled as training data)"""
    if config_file is None:
        config_file = "settings.toml"
    config = Config(Path(config_file))

    logger.info("Starting model training...")
    logger.info(f"Training dataset ID: {dataset_id}")

    try:
        random_seed = config.get("training.random_seed")
        seed_everything(random_seed)
        device = "cuda" if config.get("training.gpu") else "cpu"

        model_handler = EadroModelHandler(device)
        data_handler = EadroDataHandler()
        training_handler = EadroTrainingHandler(device)
        inference_handler = EadroInferenceHandler(device)

        # Load training data and split into train/validation
        train_loader, val_loader, metadata = (
            data_handler.prepare_train_data_by_dataset_id(config, dataset_id)
        )

        config.set("node_num", len(metadata.services))
        config.set("event_num", len(metadata.log_templates) + 1)
        config.set("metric_num", len(metadata.metrics))

        model = create_eadro_model(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=device,
            config=config,
        )

        model = model.to(device)

        # 使用带dataset_id的实验名称
        if experiment_name is None:
            experiment_name = f"train_dataset_{dataset_id}"

        # Create experiment manager
        experiment_manager = UniversalExperimentManager[
            MainModel, tuple, OptimizerAdapter
        ](
            config=config,
            experiment_name=experiment_name,
            model_handler=model_handler,
            data_handler=data_handler,
            training_handler=training_handler,
            inference_handler=inference_handler,
        )

        optimizer = training_handler.setup_optimizer(model, config)
        num_epochs = config.get("training.epochs")

        # Create SEPARATE data tuples for training and validation
        train_data = (
            train_loader,
            val_loader,
            metadata,
        )  # Training handler will use train_loader
        val_data = (
            train_loader,
            val_loader,
            metadata,
        )  # Training handler will use val_loader

        training_results = experiment_manager.run_training(
            model=model,
            train_data=train_data,
            val_data=val_data,  # Now validation will be performed correctly
            optimizer=optimizer,
            num_epochs=num_epochs,
        )

        logger.info(f"Training completed. Results: {training_results}")

        # 获取实验总结
        summary = experiment_manager.get_experiment_summary()
        logger.info(f"Experiment summary: {summary}")

        # 获取保存的checkpoint路径
        best_checkpoint_path = experiment_manager.checkpoint_dir / "best_model.ckpt"
        latest_checkpoint_path = experiment_manager.checkpoint_dir / "latest_model.ckpt"

        logger.info(f"Best checkpoint saved at: {best_checkpoint_path}")
        logger.info(f"Latest checkpoint saved at: {latest_checkpoint_path}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


@app.command()
def test_by_id(
    test_dataset_id: int = typer.Option(
        ..., "--test-dataset-id", help="Dataset ID for test data from RCABench platform"
    ),
    train_dataset_id: int = typer.Option(
        ...,
        "--train-dataset-id",
        help="Dataset ID that was used for training (for metadata and best checkpoint)",
    ),
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Test experiment name"),
) -> None:
    """Test the model using dataset IDs - automatically uses best checkpoint from training"""
    if config_file is None:
        config_file = "settings.toml"
    config = Config(Path(config_file))

    logger.info("Starting model testing...")
    logger.info(f"Test dataset ID: {test_dataset_id}")
    logger.info(f"Train dataset ID (for metadata): {train_dataset_id}")

    try:
        random_seed = config.get("training.random_seed")
        seed_everything(random_seed)
        device = "cuda" if config.get("training.gpu") else "cpu"

        model_handler = EadroModelHandler(device)
        data_handler = EadroDataHandler()
        training_handler = EadroTrainingHandler(device)
        inference_handler = EadroInferenceHandler(device)

        # Load test data and metadata from training phase
        logger.info("Loading test data and training metadata...")
        test_loader, metadata = data_handler.prepare_test_data_by_dataset_id(
            config, test_dataset_id, train_dataset_id
        )

        config.set("node_num", len(metadata.services))
        config.set("event_num", len(metadata.log_templates) + 1)
        config.set("metric_num", len(metadata.metrics))

        model = create_eadro_model(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=device,
            config=config,
        )

        model = model.to(device)

        test_experiment_name = (
            experiment_name
            or f"test_{test_dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        experiment_manager = UniversalExperimentManager[
            MainModel, tuple, OptimizerAdapter
        ](
            config=config,
            experiment_name=test_experiment_name,
            model_handler=model_handler,
            data_handler=data_handler,
            training_handler=training_handler,
            inference_handler=inference_handler,
        )

        # Create data tuple for testing
        data = (None, test_loader, metadata)

        # Run inference on test data
        test_results = experiment_manager.run_inference(
            model=model,
            data=data,
            load_best=False,
            save_results=True,
        )

        logger.info(f"Testing completed. Results: {test_results}")

    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise


@app.command()
def get_best_checkpoint(
    dataset_id: int = typer.Option(
        ..., "--dataset-id", help="Dataset ID that was used for training"
    ),
    experiment_name: Optional[str] = typer.Option(
        None,
        "--experiment-name",
        help="Custom experiment name (if used during training)",
    ),
) -> None:
    """Get the best checkpoint path for a trained dataset"""
    # 构造实验名称
    if experiment_name is None:
        experiment_name = f"train_dataset_{dataset_id}"

    # 从实验管理器的checkpoint目录查找
    from src.exp.config import Config

    config = Config(Path("settings.toml"))

    # 构造checkpoint目录路径
    base_checkpoint_dir = Path(config.get("paths.ckpt"))
    experiment_checkpoint_dir = base_checkpoint_dir / experiment_name

    best_checkpoint_path = experiment_checkpoint_dir / "best_model.ckpt"

    if not best_checkpoint_path.exists():
        logger.error(f"No best checkpoint found for dataset {dataset_id}")
        logger.info(f"Expected path: {best_checkpoint_path}")
        logger.info(f"Experiment name: {experiment_name}")
        return

    logger.info(f"Best checkpoint for dataset {dataset_id}: {best_checkpoint_path}")
    print(str(best_checkpoint_path))  # For easy script usage


@app.command()
def pipeline_by_id(
    train_dataset_id: int = typer.Option(
        ..., "--train-dataset-id", help="Dataset ID for training data"
    ),
    test_dataset_id: int = typer.Option(
        ..., "--test-dataset-id", help="Dataset ID for test data"
    ),
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
) -> None:
    """Run complete pipeline: train with one dataset ID, then test with another dataset ID"""
    logger.info("Starting complete train-test pipeline...")
    logger.info(f"Training dataset ID: {train_dataset_id}")
    logger.info(f"Test dataset ID: {test_dataset_id}")

    if config_file is None:
        config_file = "settings.toml"

    # Step 1: Train model
    logger.info(f"Step 1: Training model using dataset ID {train_dataset_id}...")
    train_by_id(train_dataset_id, config_file, experiment_name)
    logger.info("Training completed!")

    # Step 2: Test model (automatically uses best checkpoint)
    logger.info(f"Step 2: Testing model using dataset ID {test_dataset_id}...")
    test_by_id(
        test_dataset_id,
        train_dataset_id,  # Only used for metadata lookup
        config_file,
        f"{experiment_name}_test" if experiment_name else None,
        None,  # No checkpoint_path - will automatically use best model
    )
    logger.info("Complete pipeline finished successfully!")


@app.command()
def inference_from_converted_datapack(
    checkpoint_path: str = typer.Option(
        ..., "--checkpoint", help="Path to trained model checkpoint"
    ),
    datapack_path: Path = typer.Option(
        ..., "--datapack-path", help="Path to converted datapack directory"
    ),
) -> list:
    """Run inference on a single converted datapack (for RCABench platform integration)"""
    logger.info(f"Starting inference from converted datapack: {datapack_path}")

    config_file = "settings.toml"
    config = Config(Path(config_file))

    try:
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        if not datapack_path.exists():
            raise FileNotFoundError(f"Datapack directory not found: {datapack_path}")

        random_seed = config.get("training.random_seed")
        seed_everything(random_seed)

        device = "cuda" if config.get("training.gpu") else "cpu"

        # Try to load metadata from unified metadata directory
        dataset_name = config.get("dataset")
        metadata_dir = Path(config.get("paths.metadata"))
        metadata_path = metadata_dir / f"{dataset_name}_metadata.pkl"
        metadata = None

        if metadata_path.exists():
            from src.preprocessing.base import DatasetMetadata

            metadata = DatasetMetadata.from_pkl(str(metadata_path))
            logger.info(f"Loaded metadata from {metadata_path}")

        # Create model architecture (metadata will be created if needed during data loading)
        inference_handler = EadroInferenceHandler(device)

        # Load checkpoint properly - handle both checkpoint format and direct state_dict format
        from torch import load as torch_load

        checkpoint = torch_load(checkpoint_path_obj, map_location=device)

        # Check if checkpoint contains metadata (our format) or is direct state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Our checkpoint format with metadata
            model_state_dict = checkpoint["model_state_dict"]
            logger.info(
                f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
            )
        else:
            # Direct state_dict format
            model_state_dict = checkpoint
            logger.info("Loaded checkpoint as direct state_dict")

        # We need to create the model with the right dimensions
        # This will be handled by loading the data first
        logger.info("Loading data and creating model...")

        data_handler = EadroDataHandler()
        samples, metadata = data_handler.load_single_datapack_for_inference(
            config, datapack_path, metadata
        )

        config.set("node_num", len(metadata.services))
        config.set("event_num", len(metadata.log_templates) + 1)
        config.set("metric_num", len(metadata.metrics))

        model = create_eadro_model(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=device,
            config=config,
        )

        # Load the correct state_dict into model
        model.load_state_dict(model_state_dict)
        model = model.to(device)

        # Run inference
        results = inference_handler.predict_from_single_datapack(
            model=model,
            config=config,
            datapack_path=datapack_path,
            metadata=metadata,
        )

        # Extract service names from predictions - 修复数组判断问题
        predictions = results["predictions"]
        service_names = []

        for pred_list in predictions:
            # 确保pred_list是list或array，并检查长度和第一个元素
            if len(pred_list) > 0 and pred_list[0] != -1:
                service_id = pred_list[0]
                # 确保service_id在有效范围内
                if 0 <= service_id < len(metadata.services):
                    service_name = metadata.services[service_id].name
                    service_names.append(service_name)
                else:
                    logger.warning(f"Invalid service_id: {service_id}, skipping")

        # Return unique service names (preserve order)
        unique_service_names = list(dict.fromkeys(service_names))

        logger.info(f"Inference completed. Predicted services: {unique_service_names}")
        return unique_service_names

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    app()
