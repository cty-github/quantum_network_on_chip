# Legacy compatibility wrapper. Prefer:
#   python evaluate/run_mech_routing.py --experiment geometry
try:
    from .run_mech_routing import run_experiment
except ImportError:
    from run_mech_routing import run_experiment


def main() -> None:
    run_experiment("geometry")


if __name__ == "__main__":
    main()
