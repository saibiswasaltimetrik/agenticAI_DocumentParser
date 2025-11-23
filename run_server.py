#!/usr/bin/env python3
"""
Script to start the FastAPI server.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api import start_server
from src.utils.logging import setup_logging


def main():
    """Start the API server."""
    setup_logging()
    print("Starting Agentic Document Processor API Server...")
    start_server()


if __name__ == "__main__":
    main()
