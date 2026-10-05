from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
RUNS_ROOT = ROOT / "runs_multiseed"

CONDITIONS = {
    "no_protection": 0,
    "protect_25": 25,
    "protect_50": 50,
    "protect_75": 75,
    "protect_100": 100,
}

CONDITION_ORDER = [
    "no_protection",
    "protect_25",
    "protect_50",
    "protect_75",
    "protect_100",
]

DISPLAY_ORDER = [
    "protect_25",
    "protect_50",
    "no_protection",
    "protect_75",
    "protect_100",
]
