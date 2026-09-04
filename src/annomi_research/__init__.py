"""Source-disjoint and disagreement-aware AnnoMI research tools."""

from importlib.metadata import PackageNotFoundError, version

from .constants import LABELS

try:
    __version__ = version("annomi-counselling-dialogue-analysis")
except PackageNotFoundError:  # Source tree imported before installation.
    __version__ = "0.3.0.dev0"

__all__ = ["LABELS", "__version__"]
