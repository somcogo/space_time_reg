"""Command-line entrypoint: ``stmr-run --config configs/cmr_soft_con.yaml``."""

from src.config import parse_cli
from src.pipeline import run


def main():
    config = parse_cli()
    run(config)


if __name__ == "__main__":
    main()
