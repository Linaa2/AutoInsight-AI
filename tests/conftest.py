"""Shared pytest fixtures for AutoInsight-AI tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small DataFrame covering numeric, categorical, boolean, and missing values."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "age": [
                25, 30, 35, 40, None, 50, 55, 60, 65, 70,
                25, 30, None, 40, 45, 50, 55, 60, 65, 70,
            ],
            "salary": np.random.uniform(30_000, 150_000, n).round(2),
            "department": ["Engineering", "Marketing", "HR", "Finance", None] * 4,
            "is_active": [True, False, True, True, False] * 4,
            "score": [
                1.5, 2.0, None, 4.0, 5.0, 1.5, 2.0, 3.0, None, 5.0,
                1.0, 2.0, 3.0, 4.0, None, 1.5, 2.5, 3.5, 4.5, 5.0,
            ],
        }
    )


@pytest.fixture
def sample_csv_file(tmp_path, sample_df) -> str:
    path = tmp_path / "sample.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_excel_file(tmp_path, sample_df) -> str:
    path = tmp_path / "sample.xlsx"
    sample_df.to_excel(path, index=False)
    return str(path)


@pytest.fixture
def sample_parquet_file(tmp_path, sample_df) -> str:
    path = tmp_path / "sample.parquet"
    sample_df.to_parquet(path, index=False)
    return str(path)
