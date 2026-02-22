from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from frs.explainability import explain_scored_csv
from frs.logging_setup import setup_logging

app = typer.Typer(add_completion=False, help="Explainability utilities (SHAP).")


@app.command("explain")
def explain_cmd(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, help="Training run directory (contains bundle.joblib)"),
    input_path: Path = typer.Option(..., "--input", exists=True, help="Scored CSV to explain (ideally from `frs score`)"),
    out_csv: Path = typer.Option("outputs/explain.csv", "--out", help="Path to write explanations CSV"),
    out_json: Path = typer.Option("outputs/explain_summary.json", "--summary", help="Path to write explanation summary JSON"),
    figures_dir: Path = typer.Option("outputs/figures/explain", "--figures", help="Directory to write explainability figures"),
    top_k: int = typer.Option(5, "--top-k", min=1, max=20, help="Top-k SHAP features per row"),
    max_rows: int = typer.Option(2000, "--max-rows", min=50, max=50000, help="Max rows to explain (speed guardrail)"),
    background_size: int = typer.Option(200, "--background-size", min=20, max=5000, help="Background size for permutation SHAP"),
    method: str = typer.Option("auto", "--method", help="auto|tree|permutation"),
    no_beeswarm: bool = typer.Option(False, "--no-beeswarm", help="Disable beeswarm figure (faster)"),
):
    log = setup_logging()
    log.info("Explain: run_dir=%s input=%s method=%s", run_dir, input_path, method)

    if method not in ("auto", "tree", "permutation"):
        raise ValueError("--method must be one of: auto, tree, permutation")

    res = explain_scored_csv(
        run_dir=run_dir,
        scored_csv=input_path,
        out_csv=out_csv,
        out_json=out_json,
        figures_dir=figures_dir,
        top_k=top_k,
        max_rows=max_rows,
        background_size=background_size,
        method=method,  # type: ignore[arg-type]
        make_beeswarm=not no_beeswarm,
    )

    rprint("[green]Explainability CSV saved:[/green]", str(res.out_csv))
    rprint("[green]Explainability summary saved:[/green]", str(res.out_json))
    rprint("[green]Figures saved in:[/green]", str(res.figures_dir))
