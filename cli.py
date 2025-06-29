from pathlib import Path
from typing import Optional
import typer
from loguru import logger
from src.preprocessing.processor import Processor
from src.eadro.utils import seed_everything
from src.eadro.config import Config
from src.eadro.experiment_sdk import UniversalExperimentManager
from src.eadro.eadro_handlers import (
    EadroModelHandler,
    EadroDataHandler,
    EadroTrainingHandler,
    EadroInferenceHandler,
    load_timeseries_data,
    setup_model_config,
    get_device,
)

app = typer.Typer()


@app.command()
def train_sdk(
    config_file: str = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Custom experiment name"),
) -> None:
    config = Config(config_file)

    try:
        seed_everything(config.get("training.random_seed"))
        device = get_device(config.get("training.gpu"))

        train_samples, test_samples, metadata = load_timeseries_data(config)
        setup_model_config(config, metadata)

        model_handler = EadroModelHandler()
        data_handler = EadroDataHandler(train_samples, test_samples, metadata)
        training_handler = EadroTrainingHandler(str(device), config)
        inference_handler = EadroInferenceHandler(str(device))

        exp_manager = UniversalExperimentManager(
            config=config,
            experiment_name=experiment_name,
            model_handler=model_handler,
            data_handler=data_handler,
            training_handler=training_handler,
            inference_handler=inference_handler,
        )

        logger.info(f"Created experiment: {exp_manager.experiment_name}")

        # 准备数据
        train_loader, test_loader = data_handler.prepare_data(config)

        # 创建模型
        model = model_handler.create_model(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=str(device),
            config=config,
        )

        logger.info("Starting training with SDK...")

        # 使用SDK的训练功能
        result = exp_manager.run_training(
            model=model,
            train_data=train_loader,
            val_data=test_loader,
            num_epochs=config.get("training.epochs"),
        )

        logger.info("Training completed successfully!")
        logger.info("Final summary:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


@app.command()
def train_manual(
    config_file: str = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    experiment_name: Optional[str] = typer.Option(None, help="Custom experiment name"),
    checkpoint_freq: int = typer.Option(5, help="Save checkpoint every N epochs"),
) -> None:
    """使用手动训练循环"""
    config = Config(config_file)

    try:
        # 基本设置
        seed_everything(config.get("training.random_seed"))
        device = get_device(config.get("training.gpu"))

        # 使用handler中的函数加载数据
        train_samples, test_samples, metadata = load_timeseries_data(config)
        setup_model_config(config, metadata)

        # 创建处理器
        model_handler = EadroModelHandler()
        data_handler = EadroDataHandler(train_samples, test_samples, metadata)
        training_handler = EadroTrainingHandler(str(device), config)

        # 创建实验管理器
        exp_manager = UniversalExperimentManager(
            config=config,
            experiment_name=experiment_name,
            model_handler=model_handler,
            data_handler=data_handler,
            training_handler=training_handler,
        )

        # 准备数据和模型
        train_loader, test_loader = data_handler.prepare_data(config)
        model = model_handler.create_model(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=str(device),
            config=config,
        )

        # 设置优化器和调度器
        optimizer = training_handler.setup_optimizer(model, config)
        scheduler = training_handler._create_lr_scheduler(
            optimizer, config.get("training.epochs")
        )

        # 手动训练循环
        best_hr1 = -1
        patience = config.get("training.patience")
        evaluation_epoch = config.get("training.evaluation_epoch")

        for epoch in range(1, config.get("training.epochs") + 1):
            # 训练一个epoch
            train_metrics = training_handler.train_epoch(
                model, train_loader, optimizer, epoch=epoch, scheduler=scheduler
            )

            # 记录训练指标
            exp_manager.log_metrics(epoch, train_metrics, "train")

            logger.info(
                f"Epoch {epoch}/{config.get('training.epochs')}, "
                f"loss: {train_metrics['loss']:.5f}, "
                f"lr: {train_metrics['lr']:.6f}, "
                f"time: {train_metrics['time']:.2f}s"
            )

            # 早停检查
            worse_count = train_metrics.get("worse_count", 0)
            if worse_count >= patience:
                logger.info(f"Early stop at epoch: {epoch}")
                break

            # 验证和保存检查点
            if epoch % evaluation_epoch == 0:
                val_metrics = training_handler.validate(model, test_loader, epoch=epoch)
                exp_manager.log_metrics(epoch, val_metrics, "validation")

                # 检查是否是最佳模型
                is_best = val_metrics["HR@1"] > best_hr1
                if is_best:
                    best_hr1 = val_metrics["HR@1"]

                # 保存检查点
                exp_manager.save_checkpoint(
                    model,
                    epoch,
                    {**train_metrics, **val_metrics},
                    optimizer,
                    is_best=is_best,
                )

                logger.info(f"Validation - HR@1: {val_metrics['HR@1']:.4f}")
                if is_best:
                    logger.info(f"New best HR@1: {best_hr1:.4f}")

            # 定期保存检查点
            elif epoch % checkpoint_freq == 0:
                exp_manager.save_checkpoint(
                    model, epoch, train_metrics, optimizer, is_best=False
                )

        # 打印最终总结
        summary = exp_manager.get_experiment_summary()
        logger.info("Training completed!")
        logger.info("Experiment Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


@app.command()
def inference_sdk(
    experiment_name: str = typer.Argument(..., help="Name of the experiment"),
    config_file: str = typer.Option(
        "settings.toml", "--config", help="Config file path"
    ),
    checkpoint_path: Optional[str] = typer.Option(
        None, "--checkpoint", help="Specific checkpoint path"
    ),
    load_best: bool = typer.Option(
        True, "--best/--latest", help="Load best vs latest checkpoint"
    ),
    dataset: Optional[str] = typer.Option(None, help="Dataset name override"),
    gpu: Optional[bool] = typer.Option(None, help="Use GPU override"),
    save_results: bool = typer.Option(
        True, "--save/--no-save", help="Save inference results"
    ),
    use_test_data: bool = typer.Option(
        True, "--test/--train", help="Use test vs train data"
    ),
) -> None:
    """使用SDK运行推理"""
    config = Config(config_file)
    if dataset:
        config.set("dataset", dataset)
    if gpu is not None:
        config.set("training.gpu", gpu)

    try:
        seed_everything(config.get("training.random_seed"))
        device = get_device(config.get("training.gpu"))

        train_samples, test_samples, metadata = load_timeseries_data(config)
        setup_model_config(config, metadata)

        result_dir = config.get("paths.result_dir")

        data_handler = EadroDataHandler(train_samples, test_samples, metadata)
        model_handler = EadroModelHandler()

        exp_manager = UniversalExperimentManager.load_experiment(
            experiment_name=experiment_name,
            base_dir=result_dir,
            model_handler=model_handler,
            data_handler=data_handler,
            training_handler=EadroTrainingHandler(str(device), config),
            inference_handler=EadroInferenceHandler(str(device)),
        )

        train_loader, test_loader = data_handler.prepare_data(config)
        data_loader = test_loader if use_test_data else train_loader
        data_type = "test" if use_test_data else "train"

        model = model_handler.create_model(
            event_num=config.get("event_num"),
            metric_num=config.get("metric_num"),
            node_num=config.get("node_num"),
            device=str(device),
            config=config,
        )

        logger.info(f"Running inference on {data_type} data...")

        # 运行推理
        results = exp_manager.run_inference(
            model, data_loader, checkpoint_path, load_best, save_results
        )

        # 显示结果
        logger.info("Inference Results:")
        logger.info(f"Total samples: {results['total_samples']}")
        logger.info("Metrics:")
        for metric_name, value in results["metrics"].items():
            logger.info(f"  {metric_name}: {value:.4f}")

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise


@app.command()
def list_experiments_sdk(
    result_dir: str = typer.Option("result", help="Result directory"),
) -> None:
    """列出所有SDK管理的实验"""
    result_path = Path(result_dir)
    if not result_path.exists():
        logger.error(f"Result directory not found: {result_dir}")
        return

    experiments = [
        exp_dir.name
        for exp_dir in result_path.iterdir()
        if exp_dir.is_dir() and (exp_dir / "experiment_metadata.json").exists()
    ]

    if experiments:
        logger.info(f"Found {len(experiments)} SDK experiments:")
        for exp_name in sorted(experiments):
            try:
                exp_manager = UniversalExperimentManager.load_experiment(
                    experiment_name=exp_name, base_dir=result_dir
                )
                summary = exp_manager.get_experiment_summary()
                status = summary.get("status", "unknown")
                best_hr1 = summary.get("best_metrics", {}).get("HR@1", "N/A")
                if isinstance(best_hr1, (int, float)):
                    best_hr1_str = f"{best_hr1:.4f}"
                else:
                    best_hr1_str = str(best_hr1)
                logger.info(
                    f"  - {exp_name} [Status: {status}] [Best HR@1: {best_hr1_str}]"
                )
            except Exception:
                logger.info(f"  - {exp_name} [Status: unknown]")
    else:
        logger.info("No SDK experiments found")


@app.command()
def list_checkpoints_sdk(
    experiment_name: str = typer.Argument(..., help="Name of the experiment"),
    result_dir: str = typer.Option("result", help="Result directory"),
) -> None:
    """列出SDK实验的所有检查点"""
    try:
        exp_manager = UniversalExperimentManager.load_experiment(
            experiment_name=experiment_name, base_dir=result_dir
        )

        checkpoints = exp_manager.list_checkpoints()

        if checkpoints:
            logger.info(
                f"Found {len(checkpoints)} checkpoints for experiment '{experiment_name}':"
            )
            for cp in sorted(checkpoints, key=lambda x: x["epoch"]):
                epoch = cp["epoch"]
                timestamp = cp["timestamp"]
                is_best = " (BEST)" if cp.get("is_best", False) else ""
                metrics = cp.get("metrics", {})
                hr1 = metrics.get("HR@1", "N/A")
                if isinstance(hr1, (int, float)):
                    hr1_str = f"{hr1:.4f}"
                else:
                    hr1_str = str(hr1)
                logger.info(
                    f"  Epoch {epoch:4d}: HR@1={hr1_str} [{timestamp}]{is_best}"
                )
        else:
            logger.info(f"No checkpoints found for experiment '{experiment_name}'")

    except Exception as e:
        logger.error(f"Failed to list checkpoints: {str(e)}")


@app.command()
def dataset():
    processor = Processor({}, conf="settings.toml")
    processor.process_dataset()


if __name__ == "__main__":
    app()
