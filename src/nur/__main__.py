"""
__main__.py

This file allows running NUR as a module: `python -m nur`

It delegates to the Typer app defined in cli.py.
"""

from .cli import app

if __name__ == "__main__":
    app()
