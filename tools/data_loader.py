"""Data loading utilities supporting CSV, Excel, and Parquet formats."""

import io
from enum import StrEnum
from pathlib import Path

import pandas as pd

from config.settings import settings


class FileFormat(StrEnum):
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

    Args:
        excel_sheet: Sheet index (int) or name (str) to load from Excel files.
                     Defaults to ``settings.DATA_LOADER_EXCEL_SHEET``.
    """

    def __init__(self, excel_sheet: int | str | None = None) -> None:
        if excel_sheet is not None:
            self._excel_sheet: int | str = excel_sheet
        else:
            sheet_env = settings.DATA_LOADER_EXCEL_SHEET
            self._excel_sheet = int(sheet_env) if sheet_env.isdigit() else sheet_env

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, file_path: str | Path) -> pd.DataFrame:
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

    def _dispatch(self, fmt: FileFormat, source: Path | io.BytesIO) -> pd.DataFrame:
        if fmt == FileFormat.CSV:
            return self._load_csv(source)
        if fmt == FileFormat.EXCEL:
            return self._load_excel(source)
        if fmt == FileFormat.PARQUET:
            return self._load_parquet(source)
        raise UnsupportedFormatError(f"No loader registered for format: {fmt}")

    def _load_csv(self, source: Path | io.BytesIO) -> pd.DataFrame:
        # sep=None + engine="python" enables automatic delimiter detection.
        return pd.read_csv(source, sep=None, engine="python")

    def _load_excel(self, source: Path | io.BytesIO) -> pd.DataFrame:
        return pd.read_excel(source, sheet_name=self._excel_sheet)

    def _load_parquet(self, source: Path | io.BytesIO) -> pd.DataFrame:
        return pd.read_parquet(source)
