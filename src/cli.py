"""
Command-line interface for the document processing pipeline.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core.config import settings
from .workflow import DocumentProcessor
from .utils.logging import setup_logging

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def cli(verbose: bool, log_level: str):
    """
    Agentic Document Processor CLI

    Process documents through classification, extraction, validation, and redaction pipeline.
    """
    if verbose:
        log_level = "DEBUG"
    setup_logging(log_level)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path for results")
@click.option("--format", "-f", "output_format", type=click.Choice(["json", "table"]), default="table")
def process(file_path: str, output: Optional[str], output_format: str):
    """
    Process a single document file.

    FILE_PATH: Path to the document to process
    """
    console.print(f"\n[bold blue]Processing document:[/] {file_path}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing document...", total=None)

        try:
            processor = DocumentProcessor()
            result = processor.process_file(file_path)

            progress.update(task, completed=True)

            if output_format == "json":
                output_data = result.model_dump()

                # Handle datetime serialization
                def json_serializer(obj):
                    from datetime import datetime
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")

                if output:
                    with open(output, "w") as f:
                        json.dump(output_data, f, indent=2, default=json_serializer)
                    console.print(f"\n[green]Results saved to:[/] {output}")
                else:
                    console.print_json(data=output_data, default=json_serializer)
            else:
                _display_result_table(result)

            # Show summary
            status_color = "green" if result.success else "red"
            console.print(f"\n[bold {status_color}]Status:[/] {'Success' if result.success else 'Failed'}")
            console.print(f"[bold]Processing time:[/] {result.processing_time_ms:.2f}ms")

            if result.errors:
                console.print("\n[bold red]Errors:[/]")
                for error in result.errors:
                    console.print(f"  - {error}")

        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"\n[bold red]Error:[/] {e}")
            sys.exit(1)


@cli.command()
@click.argument("file_paths", nargs=-1, type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory for results")
def batch(file_paths: tuple, output_dir: Optional[str]):
    """
    Process multiple documents in batch.

    FILE_PATHS: Paths to documents to process
    """
    if not file_paths:
        console.print("[yellow]No files specified[/]")
        return

    console.print(f"\n[bold blue]Processing {len(file_paths)} documents[/]\n")

    processor = DocumentProcessor()
    results = []

    with Progress(console=console) as progress:
        task = progress.add_task("Processing...", total=len(file_paths))

        for file_path in file_paths:
            try:
                result = processor.process_file(file_path)
                results.append((file_path, result, None))
            except Exception as e:
                results.append((file_path, None, str(e)))

            progress.advance(task)

    # Display summary table
    table = Table(title="Batch Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Type")
    table.add_column("Fields")
    table.add_column("PII Found")
    table.add_column("Time (ms)")

    success_count = 0
    for file_path, result, error in results:
        path = Path(file_path)
        if result:
            success_count += 1
            status = "[green]Success[/]"
            doc_type = result.document_type.value if result.document_type else "unknown"
            fields = str(len(result.extracted_data))
            pii = str(sum(result.pii_summary.values())) if result.pii_summary else "0"
            time_ms = f"{result.processing_time_ms:.1f}"
        else:
            status = "[red]Failed[/]"
            doc_type = "-"
            fields = "-"
            pii = "-"
            time_ms = "-"

        table.add_row(path.name, status, doc_type, fields, pii, time_ms)

    console.print(table)
    console.print(f"\n[bold]Summary:[/] {success_count}/{len(file_paths)} successful")


@cli.command()
def health():
    """
    Check health of the document processor.
    """
    console.print("\n[bold blue]Health Check[/]\n")

    try:
        processor = DocumentProcessor()
        status = processor.health_check()

        table = Table(title="Component Status")
        table.add_column("Component")
        table.add_column("Status")

        for component, healthy in status.items():
            status_str = "[green]Healthy[/]" if healthy else "[red]Unhealthy[/]"
            table.add_row(component, status_str)

        console.print(table)

        overall = all(status.values())
        color = "green" if overall else "red"
        console.print(f"\n[bold {color}]Overall Status:[/] {'Healthy' if overall else 'Unhealthy'}")

    except Exception as e:
        console.print(f"[bold red]Health check failed:[/] {e}")
        sys.exit(1)


@cli.command()
@click.option("--host", default=None, help="Server host")
@click.option("--port", default=None, type=int, help="Server port")
def serve(host: Optional[str], port: Optional[int]):
    """
    Start the FastAPI server.
    """
    from .api import start_server

    console.print(f"\n[bold blue]Starting server[/] on {host or settings.api_host}:{port or settings.api_port}\n")
    start_server(host=host, port=port)


@cli.command()
def document_types():
    """
    List supported document types.
    """
    from .schemas.document import DocumentType, DOCUMENT_SCHEMAS

    console.print("\n[bold blue]Supported Document Types[/]\n")

    table = Table()
    table.add_column("Type")
    table.add_column("Required Fields")
    table.add_column("Optional Fields")

    for doc_type in DocumentType:
        schema = DOCUMENT_SCHEMAS.get(doc_type, {})
        required = ", ".join(schema.get("required_fields", [])[:3])
        if len(schema.get("required_fields", [])) > 3:
            required += "..."
        optional = ", ".join(schema.get("optional_fields", [])[:3])
        if len(schema.get("optional_fields", [])) > 3:
            optional += "..."

        table.add_row(doc_type.value, required or "-", optional or "-")

    console.print(table)


def _display_result_table(result):
    """Display processing result as tables."""
    from datetime import datetime

    # Document info
    console.print("\n[bold]Document Information[/]")
    info_table = Table(show_header=False)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value")

    if result.document_metadata:
        info_table.add_row("File", result.document_metadata.file_name)
        info_table.add_row("Type", result.document_type.value if result.document_type else "unknown")
        info_table.add_row("Size", f"{result.document_metadata.file_size_bytes / 1024:.1f} KB")

    console.print(info_table)

    # Extracted data
    if result.extracted_data:
        console.print("\n[bold]Extracted Data[/]")
        data_table = Table()
        data_table.add_column("Field")
        data_table.add_column("Value")

        for field, value in list(result.extracted_data.items())[:10]:
            val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            data_table.add_row(field, val_str)

        if len(result.extracted_data) > 10:
            data_table.add_row("...", f"({len(result.extracted_data) - 10} more fields)")

        console.print(data_table)

    # PII summary
    if result.pii_summary:
        console.print("\n[bold]PII Detected and Redacted[/]")
        pii_table = Table()
        pii_table.add_column("PII Type")
        pii_table.add_column("Count")

        for pii_type, count in result.pii_summary.items():
            pii_table.add_row(pii_type, str(count))

        console.print(pii_table)


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
