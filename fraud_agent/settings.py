"""Central configuration: .env + YAML files."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else (ROOT / p).resolve()


DATA_DIR = _path("DATA_DIR", "data/HHGOA_IEEE")


def _is_synthetic(d: Path) -> bool:
    readme = d / "README.md"
    return d.name.upper().startswith("SYNTH") or (readme.exists() and "SYNTHETIC" in readme.read_text(encoding="utf-8", errors="ignore")[:500])


IS_SYNTHETIC = _is_synthetic(DATA_DIR)
DATASET_LABEL = ("SYNTHETIC development data (not HHGOA; results are not benchmark results)" if IS_SYNTHETIC
                 else f"HHGOA_IEEE ({DATA_DIR.name})")
# outputs are separated by dataset so synthetic results can never be mistaken for benchmark answers
OUTPUT_DIR = _path("OUTPUT_DIR", "outputs/synthetic" if IS_SYNTHETIC else "outputs/hhgoa")
CACHE_DIR = ROOT / ".cache"
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "tigergraph").lower()
TG_GRAPH = os.getenv("TG_GRAPHNAME", "FraudGraph")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")
EVIDENCE_MODE = os.getenv("EVIDENCE_MODE", "dataset").lower()


def data_tag() -> str:
    """Stable short id of the active dataset (keys caches so SYNTH and real data never mix)."""
    import hashlib
    return hashlib.sha1(str(DATA_DIR).lower().encode()).hexdigest()[:10]


def llm_enabled() -> bool:
    return bool(LLM_MODEL and (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")))


@lru_cache
def datamap() -> dict:
    return yaml.safe_load((ROOT / "config" / "datamap.yaml").read_text(encoding="utf-8"))


@lru_cache
def policy_rules() -> dict:
    return yaml.safe_load((ROOT / "config" / "policy_rules.yaml").read_text(encoding="utf-8"))


for d in (OUTPUT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
