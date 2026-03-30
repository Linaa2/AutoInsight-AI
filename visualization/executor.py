"""Safe chart code executor for the visualization pipeline.

The LLM-generated chart code is executed inside a restricted Python namespace
that exposes only the libraries needed for Plotly chart creation:
    - ``df``  — the caller's pandas DataFrame
    - ``pd``  — pandas (for any DataFrame manipulation the code needs)
    - ``px``  — plotly.express
    - ``go``  — plotly.graph_objects
    - ``np``  — numpy (for numerical helpers)

``__builtins__`` is intentionally left empty so that the generated code cannot
call dangerous built-in functions (``open``, ``__import__``, ``exec``, etc.).
Simple Plotly one-liners (``fig = px.bar(df, ...)``) do not need builtins.

The generated code is expected to assign its result to a variable named ``fig``.
If it does not, execution is considered a failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from visualization.schemas import ChartExecutionResult


def execute_chart(df: pd.DataFrame, code: str) -> ChartExecutionResult:
    """Execute *code* against *df* in a restricted namespace.

    Args:
        df:   The dataset as a pandas DataFrame.
        code: Python source code that must assign a Plotly figure to ``fig``.
              Example: ``"fig = px.bar(df, x='region', y='sales')"``

    Returns:
        A :class:`~visualization.schemas.ChartExecutionResult` with:
            - ``success``: True when the code ran and produced a ``fig``.
            - ``figure``:  The Plotly Figure object, or None on failure.
            - ``error``:   A human-readable error string on failure, else None.
    """
    # Restricted execution namespace — only expose what the generated code needs.
    # Setting __builtins__ to {} prevents import and dangerous built-ins while
    # still allowing attribute access, function calls, and list/dict literals.
    namespace: dict[str, object] = {
        "__builtins__": {},
        "df": df,
        "pd": pd,
        "px": px,
        "go": go,
        "np": np,
    }

    try:
        exec(code, namespace)
    except Exception as exc:
        return ChartExecutionResult(
            success=False,
            figure=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    fig = namespace.get("fig")
    if fig is None:
        return ChartExecutionResult(
            success=False,
            figure=None,
            error="Code did not assign a Plotly figure to the variable 'fig'.",
        )

    return ChartExecutionResult(success=True, figure=fig, error=None)
