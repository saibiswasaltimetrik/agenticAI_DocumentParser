#!/usr/bin/env python3
"""
Main entry point for the Agentic Document Processor.

This module provides the primary interface for running the document
processing pipeline either via CLI or programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli import main as cli_main
from src.workflow import DocumentProcessor, process_document
from src.utils.logging import setup_logging


def main():
    """Run the CLI interface."""
    setup_logging()
    cli_main()


if __name__ == "__main__":
    main()
