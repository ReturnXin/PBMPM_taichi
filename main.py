"""Main entry for the shared water, soft-body and rigid-body solver.

Importing this module does not initialize Taichi or open a window.
"""


def main(argv=None):
    from rigid_coupling.cli import main as run
    return run(argv)


if __name__ == "__main__":
    main()
