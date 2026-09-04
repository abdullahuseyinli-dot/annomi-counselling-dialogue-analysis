from importlib.metadata import version

from annomi_research import __version__
from annomi_research.cli import _parser


def test_package_version_comes_from_distribution_metadata() -> None:
    assert __version__ == version("annomi-counselling-dialogue-analysis")


def test_parser_exposes_validation_and_research_commands_without_loading_models() -> None:
    parser = _parser()
    assert parser.parse_args(["validate"]).command == "validate"
    help_text = parser.format_help()
    for command in ("run-baselines", "run-neural", "run-panel", "run-qtrace", "run-safe-mi"):
        assert command in help_text
