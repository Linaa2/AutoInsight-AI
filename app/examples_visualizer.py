"""Predefined test examples for the Visualizer Playground.

Each example provides a small DataFrame, a simulated profiler output string, and
a simulated analyst insights string.  These are used in ``app/pages/visualizer_playground.py``
so the visualizer agent can be tested without running the Profiler or Analyst agents live.

Adding a new example
--------------------
1. Append a dict to the list returned by ``get_examples()``.
2. Required keys: ``name``, ``df``, ``profiler_output``, ``analyst_output``.
3. Optional key: ``columns_info`` (derived from ``df.dtypes`` automatically if absent).
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd


class VisualizerExample(TypedDict, total=False):
    """Shape of each predefined test example.

    Required keys:
        name:            Display name shown in the UI selectbox.
        df:              Small representative DataFrame.
        profiler_output: Simulated profiler agent text output.
        analyst_output:  Simulated analyst agent insights text.

    Optional keys:
        columns_info:    Pre-formatted column description (derived from df otherwise).
    """

    name: str  # required
    df: pd.DataFrame  # required
    profiler_output: str  # required
    analyst_output: str  # required
    columns_info: str  # optional


# ---------------------------------------------------------------------------
# Example 1 — E-commerce sales
# ---------------------------------------------------------------------------

_ECOMMERCE_DF = pd.DataFrame(
    {
        "region": ["North", "South", "East", "West", "North", "South", "East", "West"],
        "category": [
            "Electronics",
            "Clothing",
            "Electronics",
            "Clothing",
            "Home",
            "Home",
            "Clothing",
            "Electronics",
        ],
        "month": ["Jan", "Jan", "Jan", "Jan", "Feb", "Feb", "Feb", "Feb"],
        "sales": [12500.0, 8700.0, 15300.0, 9200.0, 13100.0, 9400.0, 14800.0, 10500.0],
        "units_sold": [125, 87, 153, 92, 131, 94, 148, 105],
        "returns": [5, 12, 8, 3, 7, 10, 6, 4],
    }
)

_ECOMMERCE_PROFILER = """\
Dataset: e-commerce transactions (8 rows, 6 columns)
Columns:
  - region       (object)  : 4 unique values — North, South, East, West
  - category     (object)  : 3 unique values — Electronics, Clothing, Home
  - month        (object)  : 2 unique values — Jan, Feb
  - sales        (float64) : mean=11700.0, min=8700.0, max=15300.0
  - units_sold   (int64)   : mean=117.0, min=87, max=153
  - returns      (int64)   : mean=6.9, min=3, max=12

Data quality: no missing values, no duplicates detected.
"""

_ECOMMERCE_ANALYST = """\
Key insights:
1. Electronics drives the highest average sales across all regions.
2. The East region consistently outperforms other regions in sales volume.
3. Clothing has the highest return rate relative to units sold.
4. Sales grew modestly from January to February across most categories.
5. The North region shows strong performance in Electronics but weak Clothing sales.
"""

# ---------------------------------------------------------------------------
# Example 2 — HR / employee dataset
# ---------------------------------------------------------------------------

_HR_DF = pd.DataFrame(
    {
        "department": [
            "Engineering",
            "Marketing",
            "Sales",
            "HR",
            "Engineering",
            "Marketing",
            "Sales",
            "HR",
            "Engineering",
            "Sales",
        ],
        "gender": ["M", "F", "M", "F", "F", "M", "F", "M", "M", "F"],
        "tenure_years": [3, 1, 5, 2, 7, 4, 1, 6, 2, 3],
        "salary": [85000, 62000, 71000, 54000, 110000, 75000, 68000, 57000, 90000, 74000],
        "satisfaction_score": [4.1, 3.5, 3.8, 4.2, 3.9, 3.3, 4.0, 4.5, 3.7, 3.6],
        "attrition": [0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    }
)

_HR_PROFILER = """\
Dataset: HR employee records (10 rows, 6 columns)
Columns:
  - department        (object)  : 4 unique values — Engineering, Marketing, Sales, HR
  - gender            (object)  : 2 unique values — M, F
  - tenure_years      (int64)   : mean=3.4, min=1, max=7
  - salary            (int64)   : mean=74600.0, min=54000, max=110000
  - satisfaction_score(float64) : mean=3.86, min=3.3, max=4.5
  - attrition         (int64)   : 30% of employees left (attrition=1)

Data quality: no missing values detected.
"""

_HR_ANALYST = """\
Key insights:
1. Engineering has the highest average salary, significantly above other departments.
2. Employees with low satisfaction scores (below 3.5) show higher attrition likelihood.
3. Short-tenure employees (under 2 years) account for most attrition cases.
4. Marketing has the lowest average salary and highest attrition rate.
5. HR employees report the highest satisfaction scores despite mid-range salaries.
"""

# ---------------------------------------------------------------------------
# Example 3 — Climate / weather dataset
# ---------------------------------------------------------------------------

_CLIMATE_DF = pd.DataFrame(
    {
        "city": [
            "Paris",
            "Paris",
            "Paris",
            "London",
            "London",
            "London",
            "Berlin",
            "Berlin",
            "Berlin",
        ],
        "month": ["Jan", "Apr", "Jul", "Jan", "Apr", "Jul", "Jan", "Apr", "Jul"],
        "avg_temp_c": [4.5, 12.0, 24.5, 5.0, 11.5, 21.0, 1.5, 10.0, 20.5],
        "precipitation_mm": [52, 42, 35, 55, 45, 40, 42, 38, 50],
        "sunshine_hours": [2.0, 5.5, 8.5, 1.5, 5.0, 7.0, 2.5, 5.5, 8.0],
        "humidity_pct": [82, 70, 62, 85, 72, 65, 84, 68, 63],
    }
)

_CLIMATE_PROFILER = """\
Dataset: monthly climate observations for European cities (9 rows, 6 columns)
Columns:
  - city              (object)  : 3 unique values — Paris, London, Berlin
  - month             (object)  : 3 unique values — Jan, Apr, Jul
  - avg_temp_c        (float64) : mean=12.3, min=1.5 (Berlin/Jan), max=24.5 (Paris/Jul)
  - precipitation_mm  (int64)   : mean=44.3, min=35, max=55
  - sunshine_hours    (float64) : mean=5.1, min=1.5, max=8.5
  - humidity_pct      (int64)   : mean=72.3, min=62, max=85

Data quality: no missing values. Dataset is balanced (3 months x 3 cities).
"""

_CLIMATE_ANALYST = """\
Key insights:
1. Temperature increases sharply from January to July in all three cities.
2. Paris reaches the highest summer temperatures; Berlin has the coldest winters.
3. Sunshine hours and temperature are positively correlated across all cities.
4. Precipitation is relatively stable year-round but slightly lower in summer.
5. Humidity decreases as temperature rises, most noticeably from April to July.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_examples() -> list[VisualizerExample]:
    """Return the list of all predefined test examples.

    Each element is a :class:`VisualizerExample` dict ready for use in the
    Streamlit visualizer playground.  Add new entries here to extend the registry.
    """
    return [
        {
            "name": "E-commerce Sales",
            "df": _ECOMMERCE_DF,
            "profiler_output": _ECOMMERCE_PROFILER,
            "analyst_output": _ECOMMERCE_ANALYST,
        },
        {
            "name": "HR / Employee Records",
            "df": _HR_DF,
            "profiler_output": _HR_PROFILER,
            "analyst_output": _HR_ANALYST,
        },
        {
            "name": "Climate / Weather Data",
            "df": _CLIMATE_DF,
            "profiler_output": _CLIMATE_PROFILER,
            "analyst_output": _CLIMATE_ANALYST,
        },
    ]
