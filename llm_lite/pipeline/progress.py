from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def console_log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M')}] {message}", flush=True)


@contextmanager
def progress_bar(
    *,
    description: str,
    total: int | None = None,
    unit: str = "it",
) -> Iterator[tqdm]:
    bar = tqdm(
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
    )
    try:
        yield bar
    finally:
        bar.close()


def track_progress(
    iterable: Iterable[T],
    *,
    description: str,
    total: int | None = None,
    unit: str = "it",
) -> Iterator[T]:
    with progress_bar(description=description, total=total, unit=unit) as bar:
        for item in iterable:
            yield item
            bar.update(1)
