"""Data loading utilities supporting CSV, Excel, and Parquet formats."""

import io
import os
from enum import Enum
from pathlib import Path
from typing import Union

import pandas as pd


class FileFormat(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"


SUPPORTED_EXTENSIONS: dict[str, FileFormat] = {
    ".csv": FileFormat.CSV,
    ".xlsx": FileFormat.EXCEL,
    ".xls": FileFormat.EXCEL,
    ".parquet": FileFormat.PARQUET,
}


class UnsupportedFormatError(Exception):
    """Raised when a file extension is not supported."""


class DataLoader:
    """Loads tabular data from CSV, Excel, or Parquet files.

    Environment variables:
        DATA_LOADER_EXCEL_SHEET: Sheet index (int) or name (str) to load from
                                 Excel files. Defaults to 0 (first sheet).
    """

    def __init__(self) -> None:
        sheet_env = os.getenv("DATA_LOADER_EXCEL_SHEET", "0")
        # Use integer index when the env value is purely numeric, else treat as sheet name.
        self._excel_sheet: Union[int, str] = (
            int(sheet_env) if sheet_env.isdigit() else sheet_env
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a DataFrame from a file on disk.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedFormatError: If the file extension is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        fmt = self._detect_format(path)
        return self._dispatch(fmt, path)

    def load_from_upload(self, file_bytes: bytes, file_name: str) -> pd.DataFrame:
        """Load a DataFrame from raw bytes (e.g., a Streamlit file upload).

        Args:
            file_bytes: Raw file contents.
            file_name: Original file name — used to detect the format.

        Raises:
            UnsupportedFormatError: If the file extension is not supported.
        """
        fmt = self._detect_format(Path(file_name))
        return self._dispatch(fmt, io.BytesIO(file_bytes))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_format(self, path: Path) -> FileFormat:
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS.keys())}"
            )
        return SUPPORTED_EXTENSIONS[ext]

    def _dispatch(
        self, fmt: FileFormat, source: Union[Path, io.BytesIO]
    ) -> pd.DataFrame:
        if fmt == FileFormat.CSV:
            return self._load_csv(source)
        if fmt == FileFormat.EXCEL:
            return self._load_excel(source)
        if fmt == FileFormat.PARQUET:
            return self._load_parquet(source)
        raise UnsupportedFormatError(f"No loader registered for format: {fmt}")

    def _load_csv(self, source: Union[Path, io.BytesIO]) -> pd.DataFrame:
        # sep=None + engine="python" enables automatic delimiter detection.
        return pd.read_csv(source, sep=None, engine="python")

    def _load_excel(self, source: Union[Path, io.BytesIO]) -> pd.DataFrame:
        return pd.read_excel(source, sheet_name=self._excel_sheet)

    def _load_parquet(self, source: Union[Path, io.BytesIO]) -> pd.DataFrame:
        return pd.read_parquet(source)
