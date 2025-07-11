from rcabench_platform.v2.cli.main import main
from rcabench_platform.v2.algorithms.spec import (
    global_algorithm_registry,
    Algorithm,
    AlgorithmArgs,
    AlgorithmAnswer,
)
from client import inference
import os
from pathlib import Path


class Eadro(Algorithm):
    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> list[AlgorithmAnswer]:
        ckpt_path = os.environ["CHECKPOINT_PATH"]
        results = inference(
            checkpoint_path=ckpt_path,
            datapack_path=args.input_folder,
        )
        answers = [
            AlgorithmAnswer(level="service", name=name, rank=i + 1)
            for i, name in enumerate(results)
        ]
        return answers


if __name__ == "__main__":
    registry = global_algorithm_registry()
    registry["eadro"] = Eadro
    main(enable_builtin_algorithms=False)
