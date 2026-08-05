from __future__ import annotations

import functools
import inspect
import logging
import time
from typing import Any, Callable, TypeVar, cast

from .logger import get_logger

F = TypeVar("F", bound=Callable[..., Any])


def trace_execution(
    level: int = logging.DEBUG,
    name: str | None = None,
    logger: logging.Logger | None = None,
    logger_name: str | None = None,
    log_args: bool = False,
    log_return: bool = False,
) -> Callable[[F], F]:
    """Decorator to trace function entry, execution duration, and exit status.

    Logs execution start, execution time, and any exceptions raised.
    Works for both functions and class methods.
    """

    def decorator(func: F) -> F:
        func_name = name or func.__name__
        module_name = logger_name or func.__module__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target_logger = logger or get_logger(module_name)

            if log_args:
                arg_summary = f"args={args}, kwargs={kwargs}"
                target_logger.log(level, "[TRACE START] Entering %s (%s)", func_name, arg_summary)
            else:
                target_logger.log(level, "[TRACE START] Entering %s", func_name)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                if log_return:
                    target_logger.log(
                        level,
                        "[TRACE SUCCESS] Exited %s in %.4fs | return=%s",
                        func_name,
                        elapsed,
                        type(result).__name__,
                    )
                else:
                    target_logger.log(
                        level,
                        "[TRACE SUCCESS] Exited %s in %.4fs",
                        func_name,
                        elapsed,
                    )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                target_logger.error(
                    "[TRACE FAILED] %s failed after %.4fs: %s (%s)",
                    func_name,
                    elapsed,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            target_logger = logger or get_logger(module_name)

            if log_args:
                arg_summary = f"args={args}, kwargs={kwargs}"
                target_logger.log(level, "[TRACE START] Entering async %s (%s)", func_name, arg_summary)
            else:
                target_logger.log(level, "[TRACE START] Entering async %s", func_name)

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                target_logger.log(
                    level,
                    "[TRACE SUCCESS] Exited async %s in %.4fs",
                    func_name,
                    elapsed,
                )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                target_logger.error(
                    "[TRACE FAILED] Async %s failed after %.4fs: %s (%s)",
                    func_name,
                    elapsed,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                raise

        if inspect.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, wrapper)

    return decorator


class trace_pipeline_stage:
    """Context manager and decorator for tracing major pipeline execution stages.

    Example
    -------
    >>> with trace_pipeline_stage("Ingestion Stage"):
    ...     loader.process()
    """

    def __init__(
        self,
        stage_name: str,
        logger_name: str = "docureason.pipeline",
        level: int = logging.INFO,
    ) -> None:
        self.stage_name = stage_name
        self.logger = get_logger(logger_name)
        self.level = level
        self.start_time: float = 0.0

    def __enter__(self) -> trace_pipeline_stage:
        self.logger.log(self.level, "==================================================")
        self.logger.log(self.level, ">>> STAGE START: %s", self.stage_name)
        self.logger.log(self.level, "==================================================")
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        elapsed = time.perf_counter() - self.start_time
        if exc_type is None:
            self.logger.log(
                self.level,
                ">>> STAGE COMPLETED: %s in %.4fs",
                self.stage_name,
                elapsed,
            )
            self.logger.log(self.level, "==================================================")
        else:
            self.logger.error(
                ">>> STAGE FAILED: %s after %.4fs due to %s: %s",
                self.stage_name,
                elapsed,
                exc_type.__name__,
                exc_val,
            )
            self.logger.log(self.level, "==================================================")

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_pipeline_stage(
                self.stage_name or func.__qualname__,
                logger_name=func.__module__,
                level=self.level,
            ):
                return func(*args, **kwargs)

        return cast(F, wrapper)
