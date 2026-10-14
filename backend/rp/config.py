"""Settings from the environment (.env in backend/ is loaded if present)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://rp:rp@127.0.0.1:5433/researchpulse")
# OFFLINE=1 forces the template narrator even when a key is set (demo escape hatch, tests).
OFFLINE = os.environ.get("OFFLINE") == "1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Checked against https://integrate.api.nvidia.com/v1/models on 2026-10-14 (Llama 3.3 70B is no longer listed).
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
# Nemotron reasoning ("thinking") mode: off by default. Measured 2026-10-14: same JSON answer, 1.0 s vs 7.5 s.
NVIDIA_THINKING = os.environ.get("NVIDIA_THINKING", "0") == "1"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "researchpulse-demo@example.org")

# Versions recorded with every assessment so an old conclusion can be traced to the rules that made it.
POLICY_VERSION = "p1"
EXTRACTOR_VERSION = "claims-r1"

# Bounded-agent budget per assessment.
AGENT_MAX_TOOL_CALLS = 3
AGENT_DEADLINE_S = 25
LLM_MAX_NARRATIONS_PER_RUN = 14
