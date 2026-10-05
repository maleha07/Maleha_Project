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

# Coordinate-preserving RMT conditions. These are separate from CONDITIONS,
# whose values represent CKKS protected-area percentages.
RMT_TRANSFORM_SEED = 20261005
RMT_KEY_ID = "kvasir-coordinate-rmt-v1-seed-20261005"
RMT_CONDITIONS = {
    "rmt_block16_n0": {"block_size": 16, "noise_level": 0.0},
    "rmt_block16_n005": {"block_size": 16, "noise_level": 0.05},
    "rmt_block16_n010": {"block_size": 16, "noise_level": 0.10},
    "rmt_block8_n005": {"block_size": 8, "noise_level": 0.05},
    "rmt_block32_n005": {"block_size": 32, "noise_level": 0.05},
}
