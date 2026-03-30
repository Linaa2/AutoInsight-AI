def test_import_package() -> None:
    """Smoke test to ensure the test package can be imported."""
    import tests  # noqa: F401


def test_import_orchestration() -> None:
    """Smoke test for the canonical orchestration entry point."""
    from orchestration.graph import build_graph, run_analysis  # noqa: F401
    from orchestration.state import NodeTraceEntry, PipelineState  # noqa: F401
