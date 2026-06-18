"""Entry point for `python -m projections`. The driver lives in projection.py so
the worker function is importable there rather than from __main__."""

from .projection import main

if __name__ == '__main__':
    main()
