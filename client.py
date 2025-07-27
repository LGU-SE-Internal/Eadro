#!/usr/bin/env -S uv run -s
"""
Unified Eadro Client - Data preprocessing and training in one tool
"""

import typer
from pathlib import Path
from typing import Optional
from datetime import datetime
from loguru import logger
from src.preprocessing.processor import Processor
from src.exp.controller import UniversalExperimentManager
from src.exp.config import Config
from src.eadro.handlers import (
    EadroModelHandler,
    EadroDataHandler,
    EadroTrainingHandler,
    EadroInferenceHandler,
    create_eadro_model,
    OptimizerAdapter,
)
from src.eadro.model import MainModel
from src.eadro.utils import seed_everything


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
        ..., "--dataset-folder", help="Specific dataset folder name (e.g., __dev__rcabench_test_r1)"
    ),
    label: Optional[str] = typer.Option(
        None, "--label", help="Dataset label: train or test (auto-detect from folder name if not specified)"
    ),
) -> None:
    """Create and preprocess dataset"""
    logger.info("Starting dataset creation...")

    if config_file is None:
        config_file = "settings.toml"

    try:
        processor = Processor({}, conf=config_file)
        
        if label is None:
            if "train" in dataset_folder.lower():
                inferred_label = "train"
            elif "test" in dataset_folder.lower():
                inferred_label = "test"
            else:
                raise ValueError(f"Cannot auto-detect label from folder name '{dataset_folder}'. Please specify --label train or --label test")
        else:
            if label not in ["train", "test"]:
                raise ValueError(f"Invalid label '{label}'. Must be 'train' or 'test'.")
            inferred_label = label
        
        logger.info(f"Processing dataset folder: {dataset_folder} with label: {inferred_label}")
        processor.process_dataset_folder(
            dataset_folder=dataset_folder, 
            label=inferred_label,
            use_parallel=True, 
            n_workers=n_workers
        )
        logger.info(f"Dataset creation completed for {dataset_folder}!")
            
    except Exception as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        raise


@app.command()
def train(
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    dataset_folder: str = typer.Option(
        ..., "--dataset-folder", help="Dataset folder name to use for training"
    ),
    test_dataset_folder: Optional[str] = typer.Option(
        None, "--test-dataset-folder", help="Test dataset folder name (optional, defaults to same as train)"
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
) -> None:
    """Train the model"""
    if config_file is None:
        config_file = "settings.toml"
    config = Config(Path(config_file))

    logger.info("Starting model training...")

    # 显示数据集信息
    actual_test_folder = test_dataset_folder if test_dataset_folder is not None else dataset_folder
    logger.info(f"Training dataset folder: {dataset_folder}")
    logger.info(f"Test dataset folder: {actual_test_folder}")

    try:
        random_seed = config.get("training.random_seed")
        seed_everything(random_seed)
        device = "cuda" if config.get("training.gpu") else "cpu"
        model_handler = EadroModelHandler(device)
        data_handler = EadroDataHandler()
        training_handler = EadroTrainingHandler(device)
        inference_handler = EadroInferenceHandler(device)

        # 指定数据集文件夹和使用训练集、测试集数据
        data = data_handler.prepare_data(config, dataset_folder, train_split="train", test_split="test", test_dataset_folder=test_dataset_folder)
        train_loader, test_loader, metadata = data

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

        # Ensure the entire model is on the correct device
        model = model.to(device)

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

        training_results = experiment_manager.run_training(
            model=model,
            train_data=data,
            val_data=data,
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
def pipeline(
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    dataset_folder: str = typer.Option(
        ..., "--dataset-folder", help="Dataset folder name to use for the pipeline"
    ),
    test_dataset_folder: Optional[str] = typer.Option(
        None, "--test-dataset-folder", help="Test dataset folder name (optional, defaults to same as train)"
    ),
    label: Optional[str] = typer.Option(
        None, "--label", help="Dataset label: train or test (auto-detect from folder name if not specified)"
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
    skip_dataset: bool = typer.Option(
        False, "--skip-dataset", help="Skip dataset creation step"
    ),
) -> None:
    """Run the complete pipeline: dataset creation + training"""
    logger.info("Starting complete pipeline...")

    if not skip_dataset:
        logger.info("Step 1: Creating dataset...")
        try:
            if config_file is None:
                config_file = "settings.toml"
            processor = Processor({}, conf=config_file)
            
            if label is None:
                if "train" in dataset_folder.lower():
                    inferred_label = "train"
                elif "test" in dataset_folder.lower():
                    inferred_label = "test"
                else:
                    raise ValueError(f"Cannot auto-detect label from folder name '{dataset_folder}'. Please specify --label train or --label test")
            else:
                if label not in ["train", "test"]:
                    raise ValueError(f"Invalid label '{label}'. Must be 'train' or 'test'.")
                inferred_label = label
            
            logger.info(f"Processing dataset folder: {dataset_folder} with label: {inferred_label}")
            processor.process_dataset_folder(
                dataset_folder=dataset_folder, 
                label=inferred_label,
                use_parallel=True, 
                n_workers=16
            )
            logger.info("Dataset creation completed!")
        except Exception as e:
            logger.error(f"Dataset creation failed: {str(e)}")
            raise
    else:
        logger.info("Skipping dataset creation step...")

    logger.info("Step 2: Training model...")
    train(config_file, dataset_folder, test_dataset_folder, experiment_name)
    logger.info("Pipeline completed successfully!")


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
            "rcabench",
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

        from src.eadro.handlers import ChunkDataset, collate_fn
        from torch.utils.data import DataLoader

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


if __name__ == "__main__":
    app()
