"""Small explicit registries used by tasks, recipes, adapters, and metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from threading import RLock
from typing import Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """A deterministic name-to-component registry with duplicate protection."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}
        self._lock = RLock()

    def add(self, name: str, item: T) -> T:
        key = self._normalize(name)
        with self._lock:
            if key in self._items:
                raise KeyError(f"{self.kind} {key!r} is already registered")
            self._items[key] = item
        return item

    def register(self, name: str | None = None) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            key = name or getattr(item, "name", None)
            if not key:
                raise ValueError(f"registered {self.kind} must declare a name")
            return self.add(key, item)

        return decorator

    def get(self, name: str) -> T:
        key = self._normalize(name)
        try:
            return self._items[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown {self.kind} {key!r}; available: {', '.join(self.names()) or '<none>'}"
            ) from exc

    def create(self, name: str, *args: object, **kwargs: object) -> T:
        """Instantiate a registered class, or return a registered instance.

        Adapter classes can require constructor arguments (for example a local
        dataset path). Stateless task, recipe, and metric instances are
        returned unchanged.
        """

        item = self.get(name)
        if isinstance(item, type):
            return item(*args, **kwargs)  # type: ignore[return-value]
        if args or kwargs:
            raise TypeError(f"registered {self.kind} {name!r} is not a factory")
        return item

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._normalize(name) in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    @staticmethod
    def _normalize(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("registry name must be a non-empty string")
        return name.strip().lower()


TASKS: Registry[object] = Registry("task")
RECIPES: Registry[object] = Registry("recipe")
DATASETS: Registry[object] = Registry("dataset")
METRICS: Registry[object] = Registry("metric")


def registry_snapshot() -> dict[str, tuple[str, ...]]:
    return {
        "tasks": TASKS.names(),
        "recipes": RECIPES.names(),
        "datasets": DATASETS.names(),
        "metrics": METRICS.names(),
    }
