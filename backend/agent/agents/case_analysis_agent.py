"""
CaseAnalysisAgent: normalizes facts, extracts entities, suggests case_type_hint
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CaseAnalysisAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, facts: str) -> Dict[str, Any]:
        facts = (facts or "").strip()
        result: Dict[str, Any] = {
            "normalized_facts": facts,
            "case_type_hint": None,
            "entities": {}
        }
        if not facts:
            return result

        prompt = (
            "You are a legal case analyzer for Nepali law. Extract a concise normalization of the facts, "
            "a suggested case type (e.g., Inheritance, Property Dispute, Family/Property, Constitutional, Contract, Criminal, Administrative), "
            "and key entities/relations (parties, relationships like co-wife, deceased ancestor, property/land, timeframe).\n\n"
            f"Facts: {facts}\n\n"
            "Return ONLY JSON with keys: normalized_facts, case_type_hint, entities (object)."
        )
        try:
            text = self.llm.generate(prompt, max_tokens=384, temperature=0.2)
            import re, json
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    result.update({
                        "normalized_facts": data.get("normalized_facts", facts) or facts,
                        "case_type_hint": data.get("case_type_hint"),
                        "entities": data.get("entities", {})
                    })
        except Exception:
            # Fallback: simple heuristic
            lower = facts.lower()
            if any(k in lower for k in ["inherit", "co wife", "co-wife", "grandfather", "share", "partition", "land"]):
                result["case_type_hint"] = "Inheritance"

        return result


