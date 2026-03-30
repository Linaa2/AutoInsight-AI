"""Tests for tools.data_loader."""

import pytest
import pandas as pd

from tools.data_loader import DataLoader, UnsupportedFormatError


class TestDataLoader:
    # ------------------------------------------------------------------
    # Load from disk
    # ------------------------------------------------------------------

    def test_load_csv(self, sample_csv_file, sample_df):
        df = DataLoader().load(sample_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == sample_df.shape
        assert list(df.columns) == list(sample_df.columns)

    def test_load_excel(self, sample_excel_file, sample_df):
        df = DataLoader().load(sample_excel_file)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == sample_df.shape

    def test_load_parquet(self, sample_parquet_file, sample_df):
        df = DataLoader().load(sample_parquet_file)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == sample_df.shape

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            DataLoader().load("/nonexistent/path/data.csv")

    def test_load_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("{}")
        with pytest.raises(UnsupportedFormatError):
            DataLoader().load(str(path))

    # ------------------------------------------------------------------
    # Load from uploaded bytes
    # ------------------------------------------------------------------

    def test_load_from_upload_csv(self, sample_csv_file, sample_df):
        with open(sample_csv_file, "rb") as fh:
            raw = fh.read()
        df = DataLoader().load_from_upload(raw, "sample.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == sample_df.shape

    def test_load_from_upload_unsupported_raises(self):
        with pytest.raises(UnsupportedFormatError):
            DataLoader().load_from_upload(b"{}", "data.json")

    # ------------------------------------------------------------------
    # Excel sheet selection via env
    # ------------------------------------------------------------------

    def test_excel_sheet_env_integer(self, tmp_path, sample_df, monkeypatch):
        monkeypatch.setenv("DATA_LOADER_EXCEL_SHEET", "0")
        path = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(path) as writer:
            sample_df.to_excel(writer, sheet_name="Sheet1", index=False)
            sample_df.head(5).to_excel(writer, sheet_name="Sheet2", index=False)
        df = DataLoader().load(str(path))
        assert df.shape[0] == sample_df.shape[0]

    def test_excel_sheet_env_name(self, tmp_path, sample_df, monkeypatch):
        monkeypatch.setenv("DATA_LOADER_EXCEL_SHEET", "Sheet2")
        path = tmp_path / "named.xlsx"
        with pd.ExcelWriter(path) as writer:
            sample_df.to_excel(writer, sheet_name="Sheet1", index=False)
            sample_df.head(5).to_excel(writer, sheet_name="Sheet2", index=False)
        df = DataLoader().load(str(path))
        assert df.shape[0] == 5
