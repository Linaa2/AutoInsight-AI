# Logger Module Specification

`utils/logger.py` — Modular, rotating-file logger factory for AutoInsight-AI.

---

## Overview

The module provides `get_module_logger(__name__)`, a single call that creates a
**per-module rotating log file** in the `logs/` directory and returns a
`ModuleLogger` instance that wraps Python's standard `logging.Logger`.

Key properties:

| Feature | Detail |
|---|---|
| File naming | `{module_name}_{YYYYMMDD}.log` |
| Rotation | Daily at midnight (`TimedRotatingFileHandler`) |
| Backup retention | 30 rotated files |
| Console output | Optional (toggle via `console=` param) |
| Log format | `%(asctime)s \| %(name)s \| %(levelname)s \| %(message)s` |
| Dependencies | stdlib only (`logging`, `pathlib`, `datetime`) |

---

## Installation

No additional packages required — the module uses Python's standard library only.

```
Python >= 3.11
```

---

## Quick Start

```python
# In any module, e.g. app/main.py
from utils.logger import get_module_logger

logger = get_module_logger(__name__)

logger.debug("Debug detail")
logger.info("App started")
logger.warning("Low memory")
logger.error("Connection failed")
logger.critical("Unrecoverable error")
```

This automatically creates `logs/app_main_20260330.log` (dots converted to
underscores, date injected).

---

## Context Manager

```python
from utils.logger import get_module_logger

with get_module_logger(__name__) as logger:
    logger.info("Processing started")
    # ... do work ...
    logger.info("Processing complete")
# Handlers are flushed and closed on exit; registry entry is removed.
```

Use the context manager in short-lived scripts or batch jobs where you want
explicit resource cleanup.

---

## API Reference

### `get_module_logger`

```python
def get_module_logger(
    name: str,
    level: int = logging.DEBUG,
    log_dir: Path | None = None,
    console: bool = True,
) -> ModuleLogger: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Module name; pass `__name__`. |
| `level` | `int` | `logging.DEBUG` | Minimum level recorded. |
| `log_dir` | `Path \| None` | `logs/` | Directory for log files. |
| `console` | `bool` | `True` | Also write to stdout. |

Returns a cached `ModuleLogger` — repeated calls with the same `name` return
the same instance.

### `ModuleLogger`

Thin wrapper around `logging.Logger` exposing:
`debug`, `info`, `warning`, `error`, `critical`, `exception`, `name`.

Supports the context-manager protocol (`__enter__` / `__exit__`).

---

## Configuration Examples

### Production — INFO level, no console

```python
import logging
from utils.logger import get_module_logger

logger = get_module_logger(__name__, level=logging.INFO, console=False)
```

### Custom log directory

```python
from pathlib import Path
from utils.logger import get_module_logger

logger = get_module_logger(__name__, log_dir=Path("/var/log/autoinsight"))
```

### Streamlit app

```python
# app/main.py
from utils.logger import get_module_logger

logger = get_module_logger(__name__, console=False)  # Streamlit captures stdout

logger.info("Streamlit app initialised")
```

### Agent module

```python
# agents/profiler_agent.py
from utils.logger import get_module_logger

logger = get_module_logger(__name__)

def run_profiler(df):
    logger.info("Profiler started — shape: %s", df.shape)
    # ...
    logger.info("Profiler complete")
```

---

## Example Log Output

```
2026-03-30 14:22:01,453 | app.main | INFO | Streamlit app initialised
2026-03-30 14:22:05,110 | agents.profiler_agent | DEBUG | Profiler started — shape: (1000, 12)
2026-03-30 14:22:07,882 | agents.profiler_agent | INFO | Profiler complete
2026-03-30 14:25:13,001 | tools.data_loader | WARNING | File size exceeds 100 MB
2026-03-30 14:30:00,000 | app.main | ERROR | Connection failed: timeout
```

---

## File Layout

```
logs/
├── app_main_20260330.log
├── agents_profiler_agent_20260330.log
├── tools_data_loader_20260330.log
└── agents_profiler_agent_20260329.log   ← rotated backup
```

Rotated files are kept for **30 days**, then automatically deleted.

---

## Singleton / Caching Behaviour

The first call to `get_module_logger(name)` creates and caches the logger.
Subsequent calls return the same instance.  The cache is keyed by `name`, so
different modules get independent loggers and log files.

```python
a = get_module_logger("my.module")
b = get_module_logger("my.module")
assert a is b  # True
```

Using the context manager removes the entry from the cache, allowing a fresh
logger to be created on the next call.

---

## Running Tests

```bash
# Run logger tests only
uv run pytest tests/test_logger.py -v --tb=short

# Run full test suite
make test

# Run all quality checks
make check
```

---

## Troubleshooting

### Permission denied when creating `logs/`

```
PermissionError: [Errno 13] Permission denied: 'logs'
```

Pass an absolute path in a writable location:

```python
from pathlib import Path
logger = get_module_logger(__name__, log_dir=Path("/tmp/myapp/logs"))
```

### Duplicate log entries

If you see each message printed twice, a root logger handler may be active.
The module sets `propagate = False` on every logger it creates, so duplicate
output usually means an external library is adding handlers to the root logger
after `get_module_logger` is called.  Add the following early in your entry
point to silence the root logger:

```python
import logging
logging.getLogger().handlers.clear()
```

### Log file not created

The file is created when the `TimedRotatingFileHandler` is first attached
(i.e., on the first `get_module_logger` call), not when the first message is
logged.  If the file is missing, check that the process has write permission to
`log_dir`.

### `caplog` fixture doesn't capture records

`ModuleLogger` sets `propagate = False`, so pytest's `caplog` — which attaches
to the root logger — won't see records by default.  For unit tests, manually
attach `caplog.handler` to the specific logger:

```python
def test_something(tmp_path, caplog):
    logger = get_module_logger("my.module", log_dir=tmp_path, console=False)
    logger._logger.addHandler(caplog.handler)
    with caplog.at_level(logging.INFO, logger="my.module"):
        logger.info("captured")
    logger._logger.removeHandler(caplog.handler)
    assert "captured" in caplog.text
```
