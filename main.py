import os
from pprint import pprint

from rcabench_platform.v2.algorithms.spec import (
    Algorithm,
    AlgorithmAnswer,
    AlgorithmArgs,
    global_algorithm_registry,
)
from rcabench_platform.v2.cli.main import main

from client import inference_from_converted_datapack


class Eadro(Algorithm):
    def needs_cpu_count(self) -> int | None:
        return 30

    def __call__(self, args: AlgorithmArgs) -> list[AlgorithmAnswer]:
        ckpt_path = os.environ["CHECKPOINT_PATH"]
        pprint(args)

        # Use the inference function for converted datapacks
        results = inference_from_converted_datapack(
            checkpoint_path=ckpt_path,
            datapack_path=args.input_folder,  # This is the converted datapack path
        )

        # Convert results to AlgorithmAnswer format
        answers = [
            AlgorithmAnswer(level="service", name=name, rank=i + 1)
            for i, name in enumerate(results)
        ]
        return answers


if __name__ == "__main__":
    registry = global_algorithm_registry()
    registry["eadro"] = Eadro
    main(enable_builtin_algorithms=False)

