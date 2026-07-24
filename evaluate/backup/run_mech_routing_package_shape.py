# Legacy compatibility wrapper. Prefer:
#   python evaluate/run_mech_routing.py --experiment package_shape
try:
    from .run_mech_routing import run_experiment
except ImportError:
    from run_mech_routing import run_experiment


def main() -> None:
    run_experiment("package_shape")


if __name__ == "__main__":
    main()
