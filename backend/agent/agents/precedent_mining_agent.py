"""
PrecedentMiningAgent: extracts winning arguments/principles from retrieved cases
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class PrecedentMiningAgent:
    def __init__(self, strategy_generator):
        self.sg = strategy_generator

    def run(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = "\n".join([
            f"Case {c.get('case_number_english','')}: {c.get('summary','')[:300]}"
            for c in (cases or [])[:10]
        ])
        precedents = []
        try:
            precedents = self.sg._extract_precedents_with_ai(text)
        except Exception:
            precedents = []
        return {"precedents": precedents}


