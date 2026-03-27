"""
ArgumentationAgent: generates the full strategy draft using StrategyGenerator
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ArgumentationAgent:
    def __init__(self, strategy_generator):
        self.sg = strategy_generator

    def run(self, facts: str, laws: List[Dict[str, Any]], cases: List[Dict[str, Any]], case_type_hint: Optional[str] = None, desired_outcome: Optional[str] = None, required_documents: Optional[List[Dict[str, Any]]] = None, case_type_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        laws_text = "\n".join([f"- {l.get('source','Unknown')} (p. {l.get('page','N/A')}): {l.get('text','')[:200]}" for l in (laws or [])[:5]])
        cases_text = "\n".join([f"- {c.get('case_number_english','N/A')} ({c.get('case_type_english','N/A')}): {c.get('summary','')[:200]}" for c in (cases or [])[:10]])
        retrieval_context = f"=== LEGAL DOCUMENTS ===\n{laws_text}\n\n=== COURT CASES ===\n{cases_text}".strip()
        
        # Add documents to retrieval context if provided
        if required_documents:
            docs_text = "\n".join([f"- {d.get('document','Unknown')}: {d.get('purpose','')}" for d in required_documents[:8]])
            retrieval_context += f"\n\n=== REQUIRED DOCUMENTS ===\n{docs_text}"
        
        strategy = self.sg.generate(
            case_facts=facts,
            desired_outcome=desired_outcome,
            case_type_hint=case_type_hint,
            retrieval_context=retrieval_context,
            case_type_info=case_type_info
        )
        return {"strategy": strategy}


