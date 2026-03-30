"""Tests for visualization.executor.

Covers the four key behaviors:
    1. Valid Plotly express code executes and returns a figure
    2. Valid plotly.graph_objects code executes and returns a figure
    3. Code with a syntax error returns a clean failure (no exception propagated)
    4. Code referencing a non-existent column returns a clean failure
    5. Code that does not assign to `fig` returns a descriptive failure
    6. Forbidden imports fail gracefully (no arbitrary library loading)
"""

from __future__ import annotations

import pandas as pd
import pytest

from visualization.executor import execute_chart

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A minimal DataFrame that covers numeric and categorical columns."""
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "West"],
            "sales": [120.5, 85.0, 200.3, 150.8],
            "units": [10, 7, 20, 15],
            "month": ["Jan", "Feb", "Mar", "Apr"],
        }
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_execute_bar_chart_success(sample_df: pd.DataFrame) -> None:
    """Executor succeeds for a simple plotly.express bar chart."""
    code = "fig = px.bar(df, x='region', y='sales', title='Sales by Region')"
    result = execute_chart(sample_df, code)

    assert result.success is True
    assert result.figure is not None
    assert result.error is None


def test_execute_scatter_chart_success(sample_df: pd.DataFrame) -> None:
    """Executor succeeds for a plotly.express scatter chart."""
    code = "fig = px.scatter(df, x='units', y='sales', title='Units vs Sales')"
    result = execute_chart(sample_df, code)

    assert result.success is True
    assert result.figure is not None


def test_execute_histogram_success(sample_df: pd.DataFrame) -> None:
    """Executor succeeds for a histogram."""
    code = "fig = px.histogram(df, x='sales', title='Sales Distribution')"
    result = execute_chart(sample_df, code)

    assert result.success is True
    assert result.figure is not None


def test_execute_graph_objects_bar_success(sample_df: pd.DataFrame) -> None:
    """Executor succeeds for a plotly.graph_objects chart."""
    code = "fig = go.Figure(data=[go.Bar(x=df['region'], y=df['sales'])])"
    result = execute_chart(sample_df, code)

    assert result.success is True
    assert result.figure is not None


def test_execute_line_chart_success(sample_df: pd.DataFrame) -> None:
    """Executor succeeds for a line chart."""
    code = "fig = px.line(df, x='month', y='sales', title='Sales Over Time')"
    result = execute_chart(sample_df, code)

    assert result.success is True


def test_execute_box_chart_success(sample_df: pd.DataFrame) -> None:
    """Executor succeeds for a box plot."""
    code = "fig = px.box(df, y='sales', title='Sales Box Plot')"
    result = execute_chart(sample_df, code)

    assert result.success is True


# ---------------------------------------------------------------------------
# Failure / graceful-degradation tests
# ---------------------------------------------------------------------------


def test_execute_syntax_error_returns_failure(sample_df: pd.DataFrame) -> None:
    """Syntax errors do not propagate — executor returns a clean failure."""
    code = "fig = px.bar(df, x='region', y='sales'"  # missing closing paren
    result = execute_chart(sample_df, code)

    assert result.success is False
    assert result.figure is None
    assert result.error is not None
    assert len(result.error) > 0


def test_execute_nonexistent_column_returns_failure(sample_df: pd.DataFrame) -> None:
    """References to columns that don't exist in df produce a clean failure."""
    code = "fig = px.bar(df, x='nonexistent_column', y='sales')"
    result = execute_chart(sample_df, code)

    assert result.success is False
    assert result.figure is None
    assert result.error is not None


def test_execute_no_fig_assigned_returns_failure(sample_df: pd.DataFrame) -> None:
    """Code that assigns to a variable other than `fig` is treated as failure."""
    code = "chart = px.bar(df, x='region', y='sales')"  # 'chart' not 'fig'
    result = execute_chart(sample_df, code)

    assert result.success is False
    assert result.figure is None
    assert result.error is not None
    assert "fig" in result.error.lower()


def test_execute_empty_code_returns_failure(sample_df: pd.DataFrame) -> None:
    """Empty code string produces a clean failure (no fig assigned)."""
    result = execute_chart(sample_df, "")

    assert result.success is False
    assert result.figure is None


def test_execute_import_attempt_fails_gracefully(sample_df: pd.DataFrame) -> None:
    """Attempting to import an arbitrary library fails gracefully.

    With __builtins__ = {}, __import__ is not available, so `import` statements
    in generated code will raise and be caught cleanly.
    """
    code = "import os; fig = px.bar(df, x='region', y='sales')"
    result = execute_chart(sample_df, code)

    # The import should fail since __builtins__ is empty
    assert result.success is False
    assert result.error is not None


def test_execute_result_figure_is_plotly_object(sample_df: pd.DataFrame) -> None:
    """The returned figure is a genuine Plotly Figure object."""
    import plotly.graph_objects as go

    code = "fig = px.bar(df, x='region', y='sales')"
    result = execute_chart(sample_df, code)

    assert result.success is True
    assert isinstance(result.figure, go.Figure)


# ---------------------------------------------------------------------------
# Guard: make sure the test module itself imports without error
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_import_executor() -> None:
    """Smoke test: the executor module is importable."""
    from visualization import execute_chart as _fn

    assert callable(_fn)
