"""API layer exports (lazy to avoid circular imports during middleware init)."""

__all__ = ["app", "main", "pipeline_app", "pipeline_main"]


def __getattr__(name):
    if name in ("app", "main"):
        from .app import app, main  # noqa: F811
        globals()["app"] = app
        globals()["main"] = main
        return globals()[name]
    if name in ("pipeline_app", "pipeline_main"):
        from .pipeline_app import app as pipeline_app, main as pipeline_main  # noqa: F811
        globals()["pipeline_app"] = pipeline_app
        globals()["pipeline_main"] = pipeline_main
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
