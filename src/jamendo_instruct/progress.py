from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional


class _NullProgress:
    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update(self, n: int = 1) -> None:
        return None


def _progress_enabled(cfg: Any) -> bool:
    try:
        return bool(cfg.stage.progress.enabled)
    except Exception:
        try:
            return bool(cfg.pipeline.log_stage_boundaries)
        except Exception:
            return False


def _show_bar(cfg: Any) -> bool:
    try:
        return bool(cfg.stage.progress.show_bar)
    except Exception:
        return False


def _bar_width(cfg: Any) -> Optional[int]:
    try:
        width = getattr(cfg.stage.progress, "bar_width", None)
        if width in (None, "", 0):
            return None
        return int(width)
    except Exception:
        return None


def _console() -> Any:
    try:
        from rich.console import Console

        return Console()
    except Exception:
        return None


@dataclass
class _RichProgressHandle:
    progress: Any
    task_id: Any

    def __enter__(self) -> "_RichProgressHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update(self, n: int = 1) -> None:
        self.progress.update(self.task_id, advance=n)


@dataclass
class _TqdmProgressHandle:
    progress: Any

    def __enter__(self) -> "_TqdmProgressHandle":
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return bool(self.progress.__exit__(exc_type, exc, tb))

    def update(self, n: int = 1) -> None:
        self.progress.update(n)


def _build_rich_progress(cfg: Any, *, total: Optional[int]) -> Any:
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except Exception:
        return None

    console = _console()
    if console is None:
        return None

    description = TextColumn("[bold cyan]{task.description}[/bold cyan]", justify="left")
    bar = BarColumn(
        bar_width=_bar_width(cfg),
        style="grey35",
        complete_style="bright_cyan",
        finished_style="green",
        pulse_style="cyan",
    )

    if total is None:
        columns = (
            SpinnerColumn(style="cyan"),
            description,
            TextColumn("[dim]{task.completed:,.0f} {task.fields[unit]}[/dim]", justify="right"),
            TimeElapsedColumn(),
        )
    else:
        columns = (
            SpinnerColumn(style="cyan"),
            description,
            bar,
            TaskProgressColumn(text_format="[bold]{task.percentage:>3.0f}%[/bold]"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
        )

    return Progress(
        *columns,
        console=console,
        transient=True,
        expand=True,
    )


def _load_tqdm(*, total: Optional[int]) -> Any:
    # tqdm.rich currently assumes a numeric total when rendering tasks.
    # tqdm.auto may also resolve to the rich backend, so keep unknown totals
    # on the plain std implementation.
    if total is None:
        try:
            from tqdm.std import tqdm

            return tqdm
        except Exception:
            pass
    else:
        try:
            from tqdm.rich import tqdm

            return tqdm
        except Exception:
            pass

    try:
        from tqdm.auto import tqdm

        return tqdm
    except Exception:
        return None


def rich_tqdm(
    cfg: Any,
    *,
    total: Optional[int],
    desc: str,
    unit: str = "item",
) -> Any:
    if not _progress_enabled(cfg) or not _show_bar(cfg):
        return _NullProgress()

    rich_progress = _build_rich_progress(cfg, total=total)
    if rich_progress is not None:
        @contextmanager
        def _manager() -> Any:
            with rich_progress as progress:
                task_id = progress.add_task(desc, total=total, unit=unit)
                yield _RichProgressHandle(progress=progress, task_id=task_id)

        return _manager()

    tqdm = _load_tqdm(total=total)
    if tqdm is None:
        return _NullProgress()

    return _TqdmProgressHandle(
        tqdm(
            total=total,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            leave=False,
        )
    )


def rich_tqdm_iter(
    cfg: Any,
    items: Iterable[Any],
    *,
    total: Optional[int],
    desc: str,
    unit: str = "item",
) -> Iterable[Any]:
    if not _progress_enabled(cfg) or not _show_bar(cfg):
        yield from items
        return

    with rich_tqdm(cfg, total=total, desc=desc, unit=unit) as progress:
        for item in items:
            progress.update(1)
            yield item


class StageTracker:
    def __init__(
        self,
        cfg: Any,
        stage_name: str,
        *,
        title: str | None = None,
        subtitle: str | None = None,
        total_steps: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.stage_name = stage_name
        self.total_steps = total_steps
        self.current_step = 0
        if title:
            self.start(title=title, subtitle=subtitle)

    def _emit(self, message: str) -> None:
        if not _progress_enabled(self.cfg):
            return
        console = _console()
        if console is not None:
            console.print(message)
        else:
            print(f"[{self.stage_name}] {message}", flush=True)

    def start(self, *, title: str, subtitle: str | None = None) -> None:
        if not _progress_enabled(self.cfg):
            return
        console = _console()
        heading = f"{self.stage_name.upper()} :: {title}"
        if console is not None:
            console.rule(f"[bold bright_cyan]{heading}[/bold bright_cyan]")
            if subtitle:
                console.print(f"[dim]{subtitle}[/dim]")
        else:
            print(f"[{self.stage_name}] === {title} ===", flush=True)
            if subtitle:
                print(f"[{self.stage_name}] {subtitle}", flush=True)

    def step(self, label: str, detail: str | None = None) -> None:
        self.current_step += 1
        if self.total_steps:
            prefix = f"[bold cyan][{self.stage_name} {self.current_step}/{self.total_steps}][/bold cyan]"
        else:
            prefix = f"[bold cyan][{self.stage_name}][/bold cyan]"
        message = f"{prefix} [bold]{label}[/bold]"
        if detail:
            message += f" [dim]- {detail}[/dim]"
        self._emit(message)

    def note(self, message: str) -> None:
        self._emit(f"[dim][{self.stage_name}] {message}[/dim]")

    def finish(self, summary: str | None = None) -> None:
        message = f"[bold green][{self.stage_name}] complete[/bold green]"
        if summary:
            message += f" [dim]- {summary}[/dim]"
        self._emit(message)
