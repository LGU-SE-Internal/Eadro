import concurrent.futures
import functools
import multiprocessing
import sys
import time
from pathlib import Path

import typer
from rcabench.openapi import (
    AlgorithmsApi,
    DatasetsApi,
    DtoGranularityResultEnhancedRequest,
    DtoGranularityResultItem,
)
from rcabench_platform.v2.algorithms.spec import (
    AlgorithmArgs,
)
from rcabench_platform.v2.clients.rcabench_ import RCABenchClient
from rcabench_platform.v2.datasets.rcabench import valid
from rcabench_platform.v2.logging import logger, timeit
from rcabench_platform.v2.sources.convert import convert_datapack
from rcabench_platform.v2.sources.rcabench import RcabenchDatapackLoader

from main import Eadro

app = typer.Typer(pretty_exceptions_show_locals=False)


# Top-level helper so it’s picklable
def _run_callable(fn):
    return fn()


@timeit()
def get_dataset(id: int):
    with RCABenchClient() as client:
        api = DatasetsApi(client)
        resp = api.api_v2_datasets_id_get(id=id, include_injections=True)
        assert resp.data is not None

    assert resp.data.injections is not None
    return resp.data.injections


@timeit()
def run_job(
    algorithm,
    algorithm_id: int,
    injection_id: int,
    injection_name: str,
    label: str | None = None,
):
    """
    Plain, top-level worker function (no decorators, no Typer annotations).
    Safe to use from multiprocessing pools.
    """
    input_path = Path("data") / "rcabench_dataset" / injection_name
    converted_input_path = input_path / "converted"
    _, _valid = valid(input_path)
    if not _valid:
        return
    convert_datapack(
        loader=RcabenchDatapackLoader(src_folder=input_path, datapack=injection_name),
        dst_folder=converted_input_path,
        skip_finished=True,
    )

    a = algorithm()
    t0 = time.time()
    answers = a(
        AlgorithmArgs(
            dataset="rcabench",
            datapack=injection_name,
            input_folder=converted_input_path,
            output_folder=Path("/tmp/baro_output") / injection_name,
        )
    )
    t1 = time.time()
    runtime = t1 - t0

    result_rows = [
        {"level": ans.level, "result": ans.name, "rank": ans.rank, "confidence": 0}
        for ans in answers
    ]
    if len(answers) == 0:
        logger.warning(f"No answers from algorithm `{algorithm}`")
        return
    with RCABenchClient() as client:
        algo_api = AlgorithmsApi(client)

        resp = algo_api.api_v2_algorithms_algorithm_id_results_post(
            algorithm_id=algorithm_id,
            execution_id=None,
            label=label,
            request=DtoGranularityResultEnhancedRequest(
                results=[
                    DtoGranularityResultItem(
                        level=row["level"],
                        result=row["result"],
                        rank=row["rank"],
                        confidence=row["confidence"],
                    )
                    for row in result_rows
                ],
                datapack_id=injection_id,
                duration=runtime,
            ),
        )
        logger.info(
            f"Submit detector result: response code: {resp.code}, message: {resp.message}"
        )


def run_batch(
    algorithm,
    algorithm_id: int,
    datasets: list[int],
    label: str | None = None,
    *,
    use_cpus: int | None = None,
    ignore_exceptions: bool = True,
):
    logger.debug(f"algorithms=`{algorithm}`")

    for dataset in datasets:
        datapacks_injections = get_dataset(dataset)

        alg = algorithm()
        alg_cpu_count = alg.needs_cpu_count()

        if alg_cpu_count is None:
            parallel = 0
        else:
            assert alg_cpu_count > 0
            usable_cpu_count = use_cpus or max(multiprocessing.cpu_count() - 4, 0)
            parallel = usable_cpu_count // alg_cpu_count

        del alg

        tasks = []
        for datapack in datapacks_injections:
            tasks.append(
                functools.partial(
                    run_job,
                    algorithm,
                    algorithm_id,
                    datapack.id,
                    datapack.injection_name,
                    label=label,
                )
            )

        t0 = time.time()
        failed_tasks = []

        if parallel and parallel > 1:
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=parallel, mp_context=ctx
            ) as ex:
                future_to_task = {
                    ex.submit(_run_callable, task): task for task in tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.warning(f"Task failed: {exc}, will retry later")
                        failed_tasks.append(task)
        else:
            for task in tasks:
                try:
                    task()
                except Exception as exc:
                    logger.warning(f"Task failed: {exc}, will retry later")
                    failed_tasks.append(task)

        # Retry failed tasks
        if failed_tasks:
            logger.info(f"Retrying {len(failed_tasks)} failed tasks...")
            final_failed_tasks = []

            if parallel and parallel > 1:
                ctx = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=parallel, mp_context=ctx
                ) as ex:
                    future_to_task = {
                        ex.submit(_run_callable, task): task for task in failed_tasks
                    }
                    for future in concurrent.futures.as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            future.result()
                            logger.info("Retry succeeded")
                        except Exception as exc:
                            logger.error(f"Retry failed: {exc}")
                            final_failed_tasks.append(task)
            else:
                for task in failed_tasks:
                    try:
                        task()
                        logger.info("Retry succeeded")
                    except Exception as exc:
                        logger.error(f"Retry failed: {exc}")
                        final_failed_tasks.append(task)

            if final_failed_tasks:
                logger.error(f"{len(final_failed_tasks)} tasks failed even after retry")
            else:
                logger.info("All failed tasks succeeded on retry")

        t1 = time.time()

        total_walltime = t1 - t0
        avg_walltime = total_walltime / len(tasks)

        logger.debug(f"Total   walltime: {total_walltime:.3f} seconds")
        logger.debug(f"Average walltime: {avg_walltime:.3f} seconds")

        logger.debug(f"Finished running algorithm `{algorithm}` on dataset `{dataset}`")

        sys.stdout.flush()


@app.command()
def single_test(name: str = "ts3-ts-basic-service-request-replace-method-4css6n",label: str | None = None):
    run_job(
        algorithm=Eadro,
        algorithm_id=49,
        injection_id=2287,
        injection_name=name,
        label=label,
    )


@app.command()
def batch_test(label: str | None = None):
    run_batch(
        algorithm=Eadro,
        algorithm_id=76,
        datasets=[18],
        label=label
    )


if __name__ == "__main__":
    app()
