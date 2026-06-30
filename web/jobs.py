import logging
import os
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger("inspectra.jobs")

MAX_WORKERS = int(os.environ.get("PIPELINE_MAX_WORKERS", "2"))

_executor: ProcessPoolExecutor | None = None


def get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        logger.info("Pipeline process pool started (max_workers=%d)", MAX_WORKERS)
    return _executor


def shutdown_executor() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=False)
        _executor = None


def submit_analysis(analysis_id: int) -> None:
    from web.pipeline import run_pipeline  

    executor = get_executor()
    future = executor.submit(run_pipeline, analysis_id)
    future.add_done_callback(_log_result(analysis_id))


def _log_result(analysis_id: int):
    def _cb(future):
        exc = future.exception()
        if exc is not None:
            logger.error("Analysis %s raised in worker process: %s", analysis_id, exc)
        else:
            logger.info("Analysis %s finished in worker process", analysis_id)
    return _cb