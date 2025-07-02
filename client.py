#!/usr/bin/env -S uv run -s
"""
Unified Eadro Client - Data preprocessing and training in one tool
"""

import typer
from pathlib import Path
from typing import Optional
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
import torch


def get_device(config: Config) -> str:
    """Get device based on configuration"""
    use_gpu = config.get("training.gpu")
    if use_gpu and torch.cuda.is_available():
        logger.info("Using GPU...")
        return "cuda"
    logger.info("Using CPU...")
    return "cpu"


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def create_dataset(
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
) -> None:
    """Create and preprocess dataset"""
    logger.info("Starting dataset creation...")

    if config_file is None:
        config_file = "settings.toml"

    try:
        processor = Processor({}, conf=config_file)
        processor.process_dataset()
        logger.info("Dataset creation completed successfully!")
    except Exception as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        raise


@app.command()
def train(
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
) -> None:
    """Train the model"""
    if config_file is None:
        config_file = "settings.toml"
    config = Config(Path(config_file))

    logger.info("Starting model training...")

    try:
        random_seed = config.get("training.random_seed")
        seed_everything(random_seed)

        device = get_device(config)

        model_handler = EadroModelHandler(device)
        data_handler = EadroDataHandler()
        training_handler = EadroTrainingHandler(device)
        inference_handler = EadroInferenceHandler(device)

        data = data_handler.prepare_data(config)
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
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name"),
    skip_dataset: bool = typer.Option(
        False, "--skip-dataset", help="Skip dataset creation step"
    ),
) -> None:
    """Run the complete pipeline: dataset creation + training"""
    logger.info("Starting complete pipeline...")

    if not skip_dataset:
        logger.info("Step 1: Creating dataset...")
        create_dataset(config_file)
        logger.info("Dataset creation completed!")
    else:
        logger.info("Skipping dataset creation step...")

    logger.info("Step 2: Training model...")
    train(config_file, experiment_name)
    logger.info("Pipeline completed successfully!")


@app.command()
def inference(
    checkpoint_path: str = typer.Argument(
        ..., help="Path to the model checkpoint file"
    ),
    datapack_path: str = typer.Argument(..., help="Path to the datapack directory"),
    config_file: Optional[str] = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", help="Output file path for inference results"
    ),
) -> None:
    logger.info("Starting model inference...")

    if config_file is None:
        config_file = "settings.toml"
    config = Config(Path(config_file))

    try:
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        datapack_path_obj = Path(datapack_path)
        if not datapack_path_obj.exists():
            raise FileNotFoundError(f"Datapack directory not found: {datapack_path}")

        random_seed = config.get("training.random_seed")
        seed_everything(random_seed)

        device = get_device(config)

        data_handler = EadroDataHandler()
        inference_handler = EadroInferenceHandler(device)

        dataset_name = datapack_path_obj.name
        config.set("dataset", dataset_name)

        import shutil

        try:
            samples_src = datapack_path_obj / f"{dataset_name}_samples.pkl"
            metadata_src = datapack_path_obj / f"{dataset_name}_metadata.pkl"

            if not samples_src.exists():
                raise FileNotFoundError(
                    f"Samples file not found in datapack: {samples_src}"
                )
            if not metadata_src.exists():
                raise FileNotFoundError(
                    f"Metadata file not found in datapack: {metadata_src}"
                )

            cache_dir = Path(".cache")
            cache_dir.mkdir(exist_ok=True)

            samples_dst = cache_dir / f"{dataset_name}_samples.pkl"
            metadata_dst = cache_dir / f"{dataset_name}_metadata.pkl"

            shutil.copy2(samples_src, samples_dst)
            shutil.copy2(metadata_src, metadata_dst)

            logger.info("Loading inference data...")
            data = data_handler.prepare_data(config)
            train_loader, test_loader, metadata = data

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

            logger.info(f"Loading model weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)

            logger.info("Running inference...")
            predictions = inference_handler.predict(model, data)
            results = inference_handler.postprocess_results(predictions)

            logger.info("Inference completed!")
            logger.info(f"Total samples: {results['total_samples']}")
            logger.info(f"Normal samples: {results['normal_samples']}")
            logger.info(f"Anomaly samples: {results['anomaly_samples']}")
            logger.info(f"Anomaly rate: {results['anomaly_rate']:.4f}")

            if output_file:
                import json

                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to: {output_file}")

        finally:
            if samples_dst.exists():
                samples_dst.unlink()
            if metadata_dst.exists():
                metadata_dst.unlink()

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    app()
