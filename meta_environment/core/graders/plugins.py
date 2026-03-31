"""Plugin system for registering and loading grader implementations."""

from __future__ import annotations

import importlib
from typing import Callable, Dict, Optional

from .framework import AccuracyGrader, SemanticSimilarityGrader
from .interfaces import BaseGrader


GraderFactory = Callable[[], BaseGrader]


class GraderPluginRegistry:
    """Registry for built-in and custom grader plugins."""

    def __init__(self):
        self._factories: Dict[str, GraderFactory] = {}

    def register(self, name: str, factory: GraderFactory) -> None:
        if not name.strip():
            raise ValueError("Plugin name cannot be empty")
        self._factories[name.strip()] = factory

    def available_plugins(self) -> Dict[str, GraderFactory]:
        return dict(self._factories)

    def create(self, name: str) -> BaseGrader:
        if name not in self._factories:
            raise KeyError(f"Unknown grader plugin: {name}")
        grader = self._factories[name]()
        if not isinstance(grader, BaseGrader):
            raise TypeError(f"Plugin '{name}' did not return a BaseGrader")
        return grader

    def load_from_path(self, path: str, *, name: Optional[str] = None) -> str:
        """Load plugin from `module:attribute` path.

        The target attribute can be:
          - a `BaseGrader` subclass
          - a callable returning a `BaseGrader` instance
          - a concrete `BaseGrader` instance
        """
        if ":" not in path:
            raise ValueError("Plugin path must use 'module:attribute' format")

        module_name, attr_name = path.split(":", 1)
        module = importlib.import_module(module_name)
        target = getattr(module, attr_name)

        resolved_name = name or attr_name

        if isinstance(target, BaseGrader):
            self.register(resolved_name, lambda: target)
            return resolved_name

        if isinstance(target, type) and issubclass(target, BaseGrader):
            self.register(resolved_name, target)
            return resolved_name

        if callable(target):
            def _factory() -> BaseGrader:
                instance = target()
                if not isinstance(instance, BaseGrader):
                    raise TypeError("Plugin callable must return BaseGrader")
                return instance

            self.register(resolved_name, _factory)
            return resolved_name

        raise TypeError("Unsupported plugin target type")


def create_default_registry() -> GraderPluginRegistry:
    registry = GraderPluginRegistry()
    registry.register("accuracy", AccuracyGrader)
    registry.register("semantic_similarity", SemanticSimilarityGrader)
    return registry
