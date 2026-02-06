"""Distro handler registry.

Provides a protocol defining the interface each distribution module must
implement, and a registry for looking up handlers by name.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DistroHandler(Protocol):
    """Interface that every distribution module must satisfy."""

    def fetch_packages(self, release: str) -> pd.DataFrame | None: ...

    def filter_github_repos(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def add_local_repo_cache_path_column(
        self, df: pd.DataFrame, cache_dir: str
    ) -> pd.DataFrame: ...

    def add_upstream_version_column(
        self, df: pd.DataFrame, version_column: str, new_column_name: str
    ) -> pd.DataFrame: ...

    def merge_release_packages(
        self, dfs: list[pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class _DebianAdapter:
    """Adapts the debian module's top-level functions to the DistroHandler protocol."""

    def __init__(self) -> None:
        from osslag.distro import debian as _deb

        self._mod = _deb

    def fetch_packages(self, release: str) -> pd.DataFrame | None:
        return self._mod.fetch_packages(release)

    def filter_github_repos(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._mod.filter_github_repos(df)

    def add_local_repo_cache_path_column(
        self, df: pd.DataFrame, cache_dir: str
    ) -> pd.DataFrame:
        return self._mod.add_local_repo_cache_path_column(df, cache_dir=cache_dir)

    def add_upstream_version_column(
        self, df: pd.DataFrame, version_column: str, new_column_name: str
    ) -> pd.DataFrame:
        return self._mod.add_upstream_version_column(
            df, version_column, new_column_name=new_column_name
        )

    def merge_release_packages(
        self, dfs: list[pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._mod.merge_release_packages(dfs)


# --------------- Registry ---------------

_HANDLERS: dict[str, DistroHandler] = {}


def _init_handlers() -> None:
    """Lazily populate the handler registry."""
    if not _HANDLERS:
        _HANDLERS["debian"] = _DebianAdapter()
        # Add new distros here, e.g.:
        # _HANDLERS["fedora"] = _FedoraAdapter()


def get_handler(distro: str) -> DistroHandler:
    """Return the handler for *distro* (case-insensitive).

    Raises ``ValueError`` if the distro is not supported.
    """
    _init_handlers()
    key = distro.lower()
    if key not in _HANDLERS:
        supported = ", ".join(sorted(_HANDLERS))
        raise ValueError(
            f"Unsupported distro '{distro}'. Supported: {supported}"
        )
    return _HANDLERS[key]


def supported_distros() -> list[str]:
    """Return a sorted list of supported distro names."""
    _init_handlers()
    return sorted(_HANDLERS)
