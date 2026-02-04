from importlib.metadata import version

__version__ = version("osslag")
__all__ = ["__version__", "get_version"]


def get_version() -> str:
    """Return the current version of osslag."""
    return __version__
