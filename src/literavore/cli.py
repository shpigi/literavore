"""Typer CLI for Literavore — conference paper processing pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from literavore.config import load_config
from literavore.db import Database

app = typer.Typer(name="literavore", help="Conference paper processing pipeline")


def _get_db(config_path: Optional[Path]) -> Database:
    config = load_config(config_path)
    db_path = Path(config.storage.data_dir) / "literavore.db"
    return Database(db_path)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
    stage: Optional[str] = typer.Option(None, "--stage", "-s", help="Run only this stage"),
    from_stage: Optional[str] = typer.Option(
        None, "--from-stage", "-f", help="Run from this stage onwards"
    ),
    force: bool = typer.Option(False, "--force", help="Re-process already completed items"),
    dev: bool = typer.Option(False, "--dev", help="Enable dev mode (sets LITERAVORE_DEV_MODE=1)"),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b",
        help="Process papers in batches of N through download/extract/summarize",
    ),
) -> None:
    """Run the pipeline."""
    if dev:
        os.environ["LITERAVORE_DEV_MODE"] = "1"

    try:
        cfg = load_config(config)
    except Exception as exc:
        typer.echo(f"Error loading config: {exc}", err=True)
        raise typer.Exit(code=1)

    if batch_size is not None:
        cfg.pipeline.batch_size = batch_size

    try:
        from literavore.pipeline import Pipeline  # noqa: PLC0415
    except ImportError as exc:
        typer.echo(f"Error importing pipeline: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        pipeline = Pipeline(cfg)
        stages = [stage] if stage else None
        pipeline.run(stages=stages, from_stage=from_stage, force=force)
    except Exception as exc:
        typer.echo(f"Pipeline error: {exc}", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Host to bind (default from config)"),
    port: Optional[int] = typer.Option(
        None, "--port", "-p", help="Port to listen on (default from config)"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Start the API server."""
    try:
        cfg = load_config(config)
    except Exception as exc:
        typer.echo(f"Error loading config: {exc}", err=True)
        raise typer.Exit(code=1)

    bind_host = host or cfg.serve.api_host
    bind_port = port or cfg.serve.api_port

    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        typer.echo(
            "uvicorn is required to run the API server. Install it with: pip install uvicorn",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Starting API server on {bind_host}:{bind_port}")
    uvicorn.run("literavore.serve.api:app", host=bind_host, port=bind_port)


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------


@app.command()
def ui(
    port: Optional[int] = typer.Option(
        None, "--port", "-p", help="Port to listen on (default from config)"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Start Streamlit UI."""
    try:
        cfg = load_config(config)
    except Exception as exc:
        typer.echo(f"Error loading config: {exc}", err=True)
        raise typer.Exit(code=1)

    bind_port = port or cfg.serve.streamlit_port

    ui_script = Path(__file__).parent / "serve" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_script),
        "--server.port",
        str(bind_port),
        "--server.fileWatcherType",
        "none",
    ]

    typer.echo(f"Starting Streamlit UI on port {bind_port}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        typer.echo(
            "streamlit is required to run the UI. Install it with: pip install streamlit", err=True
        )
        raise typer.Exit(code=1)
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Streamlit exited with error: {exc}", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------


@app.command()
def mcp() -> None:
    """Start MCP server (stdio transport)."""
    from literavore.serve.mcp_server import run as run_mcp  # noqa: PLC0415

    run_mcp()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Show pipeline status."""
    try:
        cfg = load_config(config)
    except Exception as exc:
        typer.echo(f"Error loading config: {exc}", err=True)
        raise typer.Exit(code=1)

    db_path = Path(cfg.storage.data_dir) / "literavore.db"

    try:
        db = Database(db_path)
    except Exception as exc:
        typer.echo(f"Error opening database: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        stats = db.get_run_stats()
        papers = db.get_papers()
    except Exception as exc:
        typer.echo(f"Error reading database: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Total papers: {len(papers)}")
    typer.echo("")

    if not stats:
        typer.echo("No processing state recorded yet.")
        return

    # Collect all statuses seen across all stages for column headers
    all_statuses: list[str] = []
    for stage_stats in stats.values():
        for s in stage_stats:
            if s not in all_statuses:
                all_statuses.append(s)
    all_statuses.sort()

    # Header
    col_width = 12
    stage_col_width = 16
    header = f"{'Stage':<{stage_col_width}}" + "".join(
        f"{s:<{col_width}}" for s in all_statuses
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    for stage in sorted(stats):
        row = f"{stage:<{stage_col_width}}"
        for s in all_statuses:
            count = stats[stage].get(s, 0)
            row += f"{count:<{col_width}}"
        typer.echo(row)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


@app.command()
def reset(
    stage: Optional[str] = typer.Option(None, "--stage", help="Reset only this stage"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Reset processing state."""
    if stage:
        prompt = f"This will delete all processing state for stage '{stage}'. Continue?"
    else:
        prompt = "This will delete ALL processing state. Continue?"

    confirmed = typer.confirm(prompt)
    if not confirmed:
        typer.echo("Aborted.")
        raise typer.Exit(code=0)

    try:
        cfg = load_config(config)
    except Exception as exc:
        typer.echo(f"Error loading config: {exc}", err=True)
        raise typer.Exit(code=1)

    db_path = Path(cfg.storage.data_dir) / "literavore.db"

    try:
        db = Database(db_path)
    except Exception as exc:
        typer.echo(f"Error opening database: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        if stage:
            db._conn.execute(
                "DELETE FROM processing_state WHERE stage = ?", (stage,)
            )
            db._conn.commit()
            typer.echo(f"Reset processing state for stage '{stage}'.")
        else:
            db._conn.execute("DELETE FROM processing_state")
            db._conn.commit()
            typer.echo("Reset all processing state.")
    except Exception as exc:
        typer.echo(f"Error resetting state: {exc}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
