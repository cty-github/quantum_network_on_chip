# Legacy compatibility wrapper. Prefer:
#   python evaluate/run_mech_routing.py --experiment package_size
try:
    from .run_mech_routing import run_experiment
except ImportError:
    from run_mech_routing import run_experiment


def main() -> None:
    run_experiment("package_size")


if __name__ == "__main__":
    main()
