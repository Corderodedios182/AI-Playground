from pathlib import Path
import pandas as pd
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw(filename: str) -> pd.DataFrame:
    path = DATA_DIR / "raw" / filename
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {path.suffix}")


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    path = DATA_DIR / "processed" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    return path
