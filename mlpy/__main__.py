"""Allow running MLPY as a module: python -m mlpy"""

from mlpy.cli.main import cli

if __name__ == '__main__':
    cli()