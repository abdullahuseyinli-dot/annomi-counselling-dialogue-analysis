from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIMPLE_DATA = ROOT / "data" / "raw" / "AnnoMI" / "dataset.csv"
FULL_DATA = ROOT / "data" / "raw" / "AnnoMI" / "AnnoMI-full.csv"
SIMPLE_MANIFEST = ROOT / "data" / "source_manifest.json"
FULL_MANIFEST = ROOT / "data" / "source_manifest_full.json"
PROTOCOL = ROOT / "configs" / "research" / "protocol_v1.json"
NEURAL_CONFIG = ROOT / "configs" / "research" / "neural_v1.json"
DASH_CONFIG = ROOT / "configs" / "research" / "dash_mi_v1.json"
PANEL_CONFIG = ROOT / "configs" / "research" / "panel_mi_v1.json"
AC_PROTOCOL = ROOT / "configs" / "research" / "protocol_ac_v1.json"
QTRACE_CONFIG = ROOT / "configs" / "research" / "qtrace_mi_v1.json"
SAFE_MI_PROTOCOL = ROOT / "configs" / "research" / "protocol_safe_mi_v2.json"
SAFE_MI_CONFIG = ROOT / "configs" / "research" / "safe_mi_v2.json"
SAFE_MI_EXTENSION_PROTOCOL = ROOT / "configs" / "research" / "protocol_safe_mi_v2_1.json"
MI_TAGS_EXTERNAL_PROTOCOL = ROOT / "configs" / "research" / "protocol_mi_tags_external_v1.json"
LEGACY_SPLIT = ROOT / "results" / "protocol" / "official_split.json"
RESEARCH_RESULTS = ROOT / "results" / "research"
ARTIFACTS = ROOT / "artifacts"

LABELS = ("reflection", "question", "therapist_input", "other")
CLIENT_LABELS = ("change", "neutral", "sustain")
QUALITY_LABELS = ("high", "low")
LABEL_COLUMNS = (
    "therapist_input_exists",
    "therapist_input_subtype",
    "reflection_exists",
    "reflection_subtype",
    "question_exists",
    "question_subtype",
    "main_therapist_behaviour",
    "client_talk_type",
)
KEY_COLUMNS = ("transcript_id", "utterance_id")
