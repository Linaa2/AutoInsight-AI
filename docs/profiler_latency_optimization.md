# Profiler Latency Optimization

## Problem Statement

`ProfilerAgent.describe()` was found to take **200+ seconds** in typical usage.
Investigation revealed the root cause: the agent serialized the full
`DataProfile.to_dict()` result as indented JSON and passed it directly to the LLM.

```python
# BEFORE — high-latency approach
profile_json = json.dumps(profile.to_dict(), indent=2, default=str)
chain.invoke({"profile_json": profile_json})
```

For a dataset with 20 columns, this produces a payload of ~15–25 KB.  At typical
local-LLM token rates the model spends most of its time processing input rather than
generating output.

---

## Two-Layer Architecture

The solution keeps the **full deterministic profile** intact (lossless forensic record)
and introduces a separate **compact LLM context** layer that is purpose-built for the
prompt:

```
CSV / Excel / Parquet
        │
        ▼
 DataProfiler.profile()          ← unchanged; all stats computed deterministically
        │
        ▼
  DataProfile                    ← full record; stored / logged as before
        │
        ▼
 build_profiler_prompt_context() ← NEW compact builder in agents/profiler_context.py
        │
        ▼
  compact context dict           ← sub-10 KB JSON sent to LLM
        │
        ▼
  ProfilerAgent (LLM)            ← reads compact context, writes markdown report
```

`DataProfile` and `DataProfiler` are **not modified**.

---

## Compact Context Builder

**Module:** `agents/profiler_context.py`

### Output Structure

```json
{
  "detail_mode": "fast",
  "dataset_overview": {
    "rows": 1000,
    "columns": 15,
    "memory_mb": 0.45,
    "column_type_counts": {"numeric": 8, "categorical": 5, "datetime": 1, "boolean": 1, "other": 0}
  },
  "data_quality": {
    "total_missing_pct": 3.2,
    "total_missing_count": 480,
    "duplicates_count": 12,
    "duplicates_pct": 1.2,
    "quality_rating": "✅ Good",
    "high_missing_columns": [{"column": "age", "missing_pct": 12.5}]
  },
  "column_summaries": [
    {"name": "salary", "dtype": "float64", "dtype_category": "numeric",
     "missing_pct": 0.0, "unique_count": 950, "unique_pct": 95.0,
     "min": 30000.0, "max": 150000.0, "mean": 72345.1}
  ],
  "highlights": [
    "HIGH_SKEWNESS: 'salary' is strongly right-skewed (skewness=2.34).",
    "HIGH_MISSINGNESS: 'age' is missing 12.5% of values."
  ],
  "sample_rows": [{"id": 1, "salary": 45000.0, ...}]
}
```

---

## Fast vs Full Detail Modes

| Feature | `fast` (default) | `full` |
|---|---|---|
| Numeric fields | min, max, mean, missing_pct, unique_pct, zeros_pct | + median, std, q25, q75, skewness, kurtosis, zeros_count |
| Categorical top-values | top 3 | top 7 |
| Sample rows | 2 | 5 |
| Deterministic flags | All standard flags | + LOW_CARDINALITY, CONSTANT_COLUMN |
| Typical payload size | ~30–40 % of full profile JSON | ~55–65 % of full profile JSON |
| Recommended for | Interactive use, quick exploration | Detailed audit, model preparation |

### Deterministic Flags

All flags are computed **without any LLM involvement**:

| Flag | Condition |
|---|---|
| `HIGH_MISSINGNESS` | `missing_pct > 10 %` |
| `HIGH_SKEWNESS` | `\|skewness\| > 2.0` |
| `HIGH_ZERO_RATE` | `zeros_pct > 30 %` |
| `NEAR_CONSTANT` | `unique_count ≤ 3` |
| `HIGH_CARDINALITY` | `unique_pct > 50 %` (categorical/boolean) |
| `LOW_CARDINALITY` | `unique_pct < 0.5 %` (full mode only) |
| `CONSTANT_COLUMN` | `unique_count == 1` (full mode only) |
| `WIDE_DATE_RANGE` | `date_range_days > 5 years` |
| `ZERO_DATE_RANGE` | `date_range_days == 0` |

---

## Timing Instrumentation

`ProfilerAgent.describe()` now logs `DEBUG`-level timing for both phases:

```
DEBUG agents.profiler: Profiler context built in 0.002s | mode=fast | columns=15 | highlights=3
DEBUG agents.profiler: Profiler LLM generation in 18.4s | total=18.4s
```

Enable with:

```python
import logging
logging.getLogger("agents.profiler").setLevel(logging.DEBUG)
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PROFILER_DETAIL_MODE` | `fast` | LLM context detail level: `fast` or `full` |
| `PROFILER_SAMPLE_ROWS` | `5` | Rows included in `DataProfile.samples` |
| `PROFILER_TOP_VALUES` | `10` | Max top values stored per categorical column |

### Per-call Override

```python
agent = ProfilerAgent()
report = agent.describe(profile, detail_mode="full")   # one-off full report
```

---

## Prompt Changes

The profiler prompt was updated to use `{profile_summary_json}` (replacing
`{profile_json}`).  The system prompt now instructs the LLM to:

1. Use the `highlights` list as authoritative pre-computed signals.
2. Adjust verbosity based on the `detail_mode` field in the context.
3. Never invent data points not present in the summary.

---

## Benchmarking

To measure the improvement locally:

```python
import time, json
from tools.profiler_engine import DataProfiler
from agents.profiler_context import build_profiler_prompt_context

df = ...  # load your dataset
profile = DataProfiler().profile(df)

# Raw payload size
raw = json.dumps(profile.to_dict(), indent=2, default=str)
fast = json.dumps(build_profiler_prompt_context(profile, "fast"), default=str)
full = json.dumps(build_profiler_prompt_context(profile, "full"), default=str)

print(f"Raw:  {len(raw):,} chars")
print(f"Fast: {len(fast):,} chars  ({100*len(fast)//len(raw)}% of raw)")
print(f"Full: {len(full):,} chars  ({100*len(full)//len(raw)}% of raw)")
```

---

## Testing

New test module: `tests/test_profiler_context.py`

Coverage areas:

- **Structure** — all required keys present in both modes
- **Payload size** — fast < full < raw profile JSON
- **Column summaries** — correct dtype-specific fields per mode
- **Deterministic flags** — HIGH_MISSINGNESS, HIGH_SKEWNESS, HIGH_CARDINALITY, WIDE_DATE_RANGE
- **Quality rating** — correct label for clean/poor datasets
- **Dataset overview accuracy** — row/column counts match, type counts sum correctly

Run:

```bash
uv run pytest tests/test_profiler_context.py -v
```
