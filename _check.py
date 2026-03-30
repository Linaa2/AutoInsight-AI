# ruff: noqa
import ast, sys

files = [
    "diagnostics/__init__.py",
    "diagnostics/telemetry.py",
    "diagnostics/renderer.py",
    "orchestration/graph.py",
    "orchestration/state.py",
    "app/main.py",
]
for f in files:
    try:
        ast.parse(open(f).read())
        print(f"OK: {f}")
    except SyntaxError as e:
        print(f"ERR: {f}: {e}")
        sys.exit(1)
print("ALL SYNTAX OK")
