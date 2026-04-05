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


def _sanitize_code(code: str) -> str:
    """Apply safe, targeted normalization to LLM-generated code.

    LLMs sometimes:
    1. Use literal ``\\n`` (two chars) instead of a real newline inside JSON strings.
    2. Concatenate multiple Python statements on a single line with only a space,
       which produces a ``SyntaxError``.  We fix the most common pattern: a closing
       parenthesis/bracket followed directly by a new assignment statement.

    The function first tries to compile the code as-is.  Only when that fails does
    it apply heuristic fixes and retry — so valid code is never touched.
    """
    import re

    # Normalise line endings
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    # Replace literal \n (backslash + n, two chars) with a real newline.
    # This handles the case where the LLM double-escaped the JSON newline.
    if "\\n" in code:
        code = code.replace("\\n", "\n")

    # Fast path: code is already valid
    try:
        compile(code, "<string>", "exec")
        return code
    except SyntaxError:
        pass

    # Heuristic: insert a newline before any new assignment that immediately
    # follows a closing paren/bracket/quote with only whitespace in between.
    # Targets the common pattern:  ``... .reset_index() fig = px.bar(...)``
    fixed = re.sub(
        r"([)\]\"'])(\s+)([a-zA-Z_]\w*\s*=\s*)",
        lambda m: m.group(1) + "\n" + m.group(3),
        code,
    )

    try:
        compile(fixed, "<string>", "exec")
        return fixed
    except SyntaxError:
        pass

    # If still broken, return the original code and let exec() surface the error.
    return code


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
        exec(_sanitize_code(code), namespace)
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
