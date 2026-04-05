"""
agents/mock_profiler.py — Mock data to test Phase 2 without the Profiler.

Simulates the Phase 1 output (profile_data, profiler_output, sample_text)
with a realistic e-commerce dataset.

Usage:
    from agents.mock_profiler import get_mock_state
    state = get_mock_state()
    # → state contains profiler_output, profile_data, sample_text, columns_info
"""


def get_mock_profile_data() -> dict:
    """Return a simulated structured profile_data (output of data_profiler.py)."""
    return {
        "shape": {"rows": 1500, "cols": 8},
        "columns": [
            {
                "name": "order_id",
                "dtype": "int64",
                "missing": 0,
                "missing_pct": 0.0,
                "unique": 1500,
                "stats": {
                    "mean": 750.5,
                    "std": 433.01,
                    "min": 1,
                    "25%": 375.75,
                    "50%": 750.5,
                    "75%": 1125.25,
                    "max": 1500,
                },
            },
            {
                "name": "date",
                "dtype": "object",
                "missing": 0,
                "missing_pct": 0.0,
                "unique": 365,
                "top_values": {
                    "2024-03-15": 8,
                    "2024-06-22": 7,
                    "2024-11-25": 12,
                    "2024-01-10": 6,
                    "2024-09-05": 7,
                },
            },
            {
                "name": "product",
                "dtype": "object",
                "missing": 0,
                "missing_pct": 0.0,
                "unique": 25,
                "top_values": {
                    "Widget Pro": 185,
                    "Gadget X": 162,
                    "Module Z": 148,
                    "Kit Starter": 130,
                    "Cable USB-C": 120,
                },
            },
            {
                "name": "category",
                "dtype": "object",
                "missing": 5,
                "missing_pct": 0.3,
                "unique": 5,
                "top_values": {
                    "Electronics": 520,
                    "Accessories": 380,
                    "Software": 310,
                    "Service": 185,
                    "Training": 100,
                },
            },
            {
                "name": "region",
                "dtype": "object",
                "missing": 0,
                "missing_pct": 0.0,
                "unique": 4,
                "top_values": {
                    "Ile-de-France": 480,
                    "Auvergne-Rhone-Alpes": 350,
                    "Occitanie": 380,
                    "Bretagne": 290,
                },
            },
            {
                "name": "quantity",
                "dtype": "int64",
                "missing": 0,
                "missing_pct": 0.0,
                "unique": 48,
                "stats": {
                    "mean": 3.2,
                    "std": 2.8,
                    "min": 1,
                    "25%": 1,
                    "50%": 2,
                    "75%": 4,
                    "max": 25,
                },
            },
            {
                "name": "unit_price",
                "dtype": "float64",
                "missing": 12,
                "missing_pct": 0.8,
                "unique": 25,
                "stats": {
                    "mean": 49.90,
                    "std": 38.50,
                    "min": 5.99,
                    "25%": 19.99,
                    "50%": 39.99,
                    "75%": 69.99,
                    "max": 199.99,
                },
            },
            {
                "name": "total",
                "dtype": "float64",
                "missing": 12,
                "missing_pct": 0.8,
                "unique": 320,
                "stats": {
                    "mean": 159.68,
                    "std": 185.42,
                    "min": 5.99,
                    "25%": 39.99,
                    "50%": 89.97,
                    "75%": 199.96,
                    "max": 2499.75,
                },
            },
        ],
        "missing_values": {"category": 5, "unit_price": 12, "total": 12},
        "duplicates": 3,
        "memory_mb": 0.09,
        "sample": (
            "   order_id        date      product     category              region  quantity  unit_price    total\n"
            "0        1  2024-01-03   Widget Pro  Electronics      Ile-de-France         2       59.99   119.98\n"
            "1        2  2024-01-03     Gadget X   Accessories           Occitanie         1       29.99    29.99\n"
            "2        3  2024-01-04    Module Z     Software  Auvergne-Rhone-Alpes         5       19.99    99.95\n"
            "3        4  2024-01-05  Kit Starter     Service            Bretagne         1      149.99   149.99\n"
            "4        5  2024-01-05   Cable USB-C  Accessories      Ile-de-France        10        9.99    99.90\n"
        ),
        "describe": (
            "          order_id     quantity   unit_price        total\n"
            "count  1500.000000  1500.000000  1488.000000  1488.000000\n"
            "mean    750.500000     3.200000    49.900000   159.680000\n"
            "std     433.012500     2.800000    38.500000   185.420000\n"
            "min       1.000000     1.000000     5.990000     5.990000\n"
            "25%     375.750000     1.000000    19.990000    39.990000\n"
            "50%     750.500000     2.000000    39.990000    89.970000\n"
            "75%    1125.250000     4.000000    69.990000   199.960000\n"
            "max    1500.000000    25.000000   199.990000  2499.750000"
        ),
    }


def get_mock_profiler_output() -> str:
    """Return the simulated markdown that the Profiler would have generated."""
    return """\
# Dataset Profile — E-commerce Sales 2024

## Overview
| Metric | Value |
|--------|-------|
| Rows | 1,500 |
| Columns | 8 |
| Memory | 0.09 MB |
| Duplicates | 3 |

## Data Types
| Column | Type | Unique Values | Example |
|--------|------|---------------|---------|
| order_id | int64 | 1,500 | 1 |
| date | object | 365 | 2024-01-03 |
| product | object | 25 | Widget Pro |
| category | object | 5 | Electronics |
| region | object | 4 | Ile-de-France |
| quantity | int64 | 48 | 2 |
| unit_price | float64 | 25 | 59.99 |
| total | float64 | 320 | 119.98 |

## Data Quality
- **category**: 5 missing values (0.3%)
- **unit_price**: 12 missing values (0.8%)
- **total**: 12 missing values (0.8%)
- 3 duplicate rows detected

## Key Statistics

### Numerical
| Column | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| quantity | 3.2 | 2.8 | 1 | 25 |
| unit_price | 49.90 | 38.50 | 5.99 | 199.99 |
| total | 159.68 | 185.42 | 5.99 | 2,499.75 |

### Categorical
- **product**: Top 3 → Widget Pro (185), Gadget X (162), Module Z (148)
- **category**: Electronics (520), Accessories (380), Software (310), Service (185), Training (100)
- **region**: Ile-de-France (480), Occitanie (380), Auvergne-Rhone-Alpes (350), Bretagne (290)

## Recommendations
- Fix the 12 missing values on unit_price/total (likely related)
- Investigate the 3 duplicates
- quantity column has a max of 25, verify if this is normal
- total std dev (185.42) exceeds the mean (159.68), high dispersion
"""


def get_mock_sample_text() -> str:
    """Return the simulated sample."""
    return (
        "   order_id        date      product     category              region  quantity  unit_price    total\n"
        "0        1  2024-01-03   Widget Pro  Electronics      Ile-de-France         2       59.99   119.98\n"
        "1        2  2024-01-03     Gadget X   Accessories           Occitanie         1       29.99    29.99\n"
        "2        3  2024-01-04    Module Z     Software  Auvergne-Rhone-Alpes         5       19.99    99.95\n"
        "3        4  2024-01-05  Kit Starter     Service            Bretagne         1      149.99   149.99\n"
        "4        5  2024-01-05   Cable USB-C  Accessories      Ile-de-France        10        9.99    99.90\n"
    )


def get_mock_columns_info() -> str:
    """Return column info for the LLM."""
    return (
        "Available columns:\n"
        "- order_id (int64): order identifier\n"
        "- date (object): order date (YYYY-MM-DD format)\n"
        "- product (object): product name (25 unique values)\n"
        "- category (object): category (5 values: Electronics, Accessories, Software, Service, Training)\n"
        "- region (object): region (4 values: Ile-de-France, Auvergne-Rhone-Alpes, Occitanie, Bretagne)\n"
        "- quantity (int64): ordered quantity (1 to 25)\n"
        "- unit_price (float64): unit price (5.99 to 199.99)\n"
        "- total (float64): total amount (5.99 to 2499.75)\n"
    )


def get_mock_state() -> dict:
    """
    Return a complete simulated LangGraph state, ready to be passed to analyst_node().

    Usage:
        from agents.mock_profiler import get_mock_state
        from agents.analyst import analyst_node

        state = get_mock_state()
        result = analyst_node(state)
        print(result["analyst_output"])
    """
    return {
        "profile_text": get_mock_profiler_output(),
        "profile_data": get_mock_profile_data(),
        "sample_text": get_mock_sample_text(),
        "columns_info": get_mock_columns_info(),
        "profiler_output": get_mock_profiler_output(),
        "analyst_output": None,
        "visualizer_output": None,
        "reporter_output": None,
        "error": None,
    }
