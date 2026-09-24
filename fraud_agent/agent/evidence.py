"""Controlled evidence gathering (mock channels).

dataset     – responses shipped with the benchmark (if the dataset provides them)
simulated   – seeded simulation from the outcome model in policy_rules.yaml (clearly flagged)
interactive – the UI supplies the response (analyst plays customer / step-up service)
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from .. import settings
from .policy import rules

KEYWORDS = {
    "request_customer_validation": {
        "_keys": ["customer", "validation", "validate", "owner", "cardholder", "confirm"],
        "denied": ["den", "not me", "unauthori", "did not", "didn't", "don't recogn", "not recogn", "fraud", "no"],
        "confirmed": ["confirm", "recogn", "authori", "yes", "genuine", "legit", "made it", "mine"],
        "no_response": ["no response", "no_response", "unreach", "noanswer", "no answer", "timeout", "none"],
    },
    "request_step_up_auth": {
        "_keys": ["step", "auth", "otp", "mfa", "2fa", "biometric"],
        "failed": ["fail", "wrong", "incorrect", "reject"],
        "abandoned": ["abandon", "timeout", "no response", "cancel"],
        "passed": ["pass", "success", "verified", "ok"],
    },
    "request_analyst_info": {
        "_keys": ["analyst", "merchant", "kyc", "additional", "info", "delivery", "chargeback"],
        "adverse": ["adverse", "fraud", "not deliver", "mismatch", "suspicious", "negative", "confirmed fraud"],
        "benign": ["benign", "deliver", "legit", "match", "positive", "genuine", "clean"],
        "inconclusive": ["inconclusive", "unknown", "unclear", "pending", "none"],
    },
}


def normalize_outcome(request: str, raw) -> str | None:
    if raw is None:
        return None
    s = str(raw).lower()
    kw = KEYWORDS[request]
    # longest matching phrase wins ("no response" beats "no")
    best, best_len = None, 0
    for outcome, words in kw.items():
        if outcome == "_keys":
            continue
        for w in words:
            if re.search(rf"\b{re.escape(w)}", s) and len(w) > best_len:
                best, best_len = outcome, len(w)
    return best


class EvidenceProvider:
    mode = "base"

    def get(self, case_id: str, request: str, p_fraud: float) -> dict | None: ...


class DatasetEvidence(EvidenceProvider):
    mode = "dataset"

    def __init__(self, evidence: dict):
        self.ev = evidence

    def get(self, case_id, request, p_fraud):
        rec = self.ev.get(str(case_id))
        if not rec:
            return None
        flat = rec if isinstance(rec, dict) else {"responses": rec}
        cand = []
        for k, v in flat.items():
            kl = k.lower()
            if isinstance(v, (dict, list)):
                v = json.dumps(v)
            if any(x in kl for x in KEYWORDS[request]["_keys"]) or kl == request:
                cand.append(v)
        for v in cand:
            o = normalize_outcome(request, v)
            if o:
                return {"outcome": o, "raw": str(v)[:500], "source": "dataset"}
        return None


class SimulatedEvidence(EvidenceProvider):
    """Samples an outcome from the policy outcome model given the current posterior.
    Deterministic per (case, request). Marked `simulated` everywhere it is recorded."""
    mode = "simulated"

    def get(self, case_id, request, p_fraud):
        spec = rules()["evidence_requests"][request]["outcomes"]
        seed = int(hashlib.sha1(f"{case_id}:{request}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        is_fraud = rng.random() < p_fraud
        r, acc = rng.random(), 0.0
        for o, pr in spec.items():
            acc += pr["pf"] if is_fraud else pr["pl"]
            if r <= acc:
                return {"outcome": o, "raw": f"simulated ({'fraud' if is_fraud else 'legit'} world draw)", "source": "simulated"}
        return {"outcome": list(spec)[-1], "raw": "simulated", "source": "simulated"}


class InteractiveEvidence(EvidenceProvider):
    mode = "interactive"

    def __init__(self):
        self.answers: dict[tuple[str, str], str] = {}

    def set(self, case_id, request, outcome):
        self.answers[(str(case_id), request)] = outcome

    def get(self, case_id, request, p_fraud):
        o = self.answers.get((str(case_id), request))
        return {"outcome": o, "raw": "provided in UI", "source": "interactive"} if o else None


def make_provider(mode: str | None = None, dataset_evidence: dict | None = None) -> EvidenceProvider:
    mode = (mode or settings.EVIDENCE_MODE).lower()
    if mode == "dataset":
        return ChainEvidence([DatasetEvidence(dataset_evidence or {}), SimulatedEvidence()])
    if mode == "interactive":
        return InteractiveEvidence()
    return SimulatedEvidence()


class ChainEvidence(EvidenceProvider):
    mode = "dataset+simulated-fallback"

    def __init__(self, providers):
        self.providers = providers

    def get(self, case_id, request, p_fraud):
        for p in self.providers:
            r = p.get(case_id, request, p_fraud)
            if r:
                return r
        return None
