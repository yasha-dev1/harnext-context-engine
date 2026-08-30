"""Charts and self-contained HTML reporting for evaluation runs."""

from pathlib import Path


def build_report(run_dir: str | Path) -> Path:
    """Lazily import the renderer so ``python -m ...report`` stays warning-free."""

    from harnext_eval.report.report import build_report as render

    return render(run_dir)

__all__ = ["build_report"]
