"""
PrecedentRetrievalAgent: fetches laws (KB) and similar cases
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PrecedentRetrievalAgent:
    def __init__(self, agentic_tools):
        self.tools = agentic_tools

    def run(self, query: str) -> Dict[str, Any]:
        kb = self.tools.execute("search_knowledge_base", query=query, limit=5)
        cases = self.tools.execute("search_cases", query=query, limit=10)
        return {
            "laws": kb.get("results", []) if isinstance(kb, dict) else [],
            "cases": cases.get("results", []) if isinstance(cases, dict) else []
        }


