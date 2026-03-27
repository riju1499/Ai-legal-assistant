"""
CaseTypeAnalysisAgent: Deeply analyzes case facts to determine case type, parties, and legal framework
"""

import logging
import json
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


class CaseTypeAnalysisAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, facts: str) -> Dict[str, Any]:
        """
        Analyze case facts to determine case type, prosecution/defense, and legal framework
        
        Args:
            facts: Case facts description
            
        Returns:
            Dictionary with case type analysis
        """
        facts = (facts or "").strip()
        if not facts:
            return {
                "case_type": "Unknown",
                "is_prosecution": False,
                "is_defense": False,
                "legal_domain": "unknown"
            }
        
        prompt = f"""Analyze this legal case description comprehensively.

Case Facts: {facts}

Determine:
1. Case Type: Criminal, Civil, Property, Contract, Constitutional, Administrative, etc.
2. If Criminal: Is this PROSECUTION (seeking conviction) or DEFENSE (defending against charges)?
3. If Criminal: What is the charge (murder, theft, fraud, etc.)?
4. If Civil: What type (property dispute, contract breach, inheritance, etc.)?
5. Parties: Who is the claimant/plaintiff? Who is the defendant/respondent?
6. Core Legal Issue: What is the main legal question?
7. Desired Outcome: What is the client seeking?

Return ONLY a JSON object:
{{
    "case_type": "Criminal" | "Civil" | "Property" | "Contract" | etc.,
    "is_prosecution": true/false (if criminal),
    "is_defense": true/false (if criminal),
    "charge_type": "murder" | "theft" | etc. (if criminal),
    "civil_type": "property" | "contract" | etc. (if civil),
    "claimant": "who is seeking relief",
    "defendant": "who is defending",
    "core_legal_issue": "what is the main legal question",
    "legal_domain": "criminal law" | "civil law" | "property law" | etc.
}}"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=400, temperature=0.2)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group(0))
                if isinstance(result, dict):
                    # Ensure boolean values
                    result["is_prosecution"] = result.get("is_prosecution", False)
                    result["is_defense"] = result.get("is_defense", False)
                    return result
        except Exception as e:
            logger.warning(f"Case type analysis failed: {e}")
        
        # Fallback: keyword-based detection
        facts_lower = facts.lower()
        
        # Check for criminal indicators
        if any(kw in facts_lower for kw in ["murder", "killed", "stab", "accused", "convict", "crime", "victim", "weapon"]):
            case_type = "Criminal"
            is_prosecution = any(kw in facts_lower for kw in ["want to convict", "prosecution", "state", "seeking conviction"])
            is_defense = any(kw in facts_lower for kw in ["was accused", "defending", "my friend", "alibi"])
            charge_type = "murder" if "murder" in facts_lower or "killed" in facts_lower else "unknown"
            legal_domain = "criminal law"
        elif any(kw in facts_lower for kw in ["land", "property", "inherit", "partition", "lalpurja"]):
            case_type = "Property"
            legal_domain = "property law"
            is_prosecution = False
            is_defense = False
            charge_type = None
        elif any(kw in facts_lower for kw in ["contract", "agreement", "breach"]):
            case_type = "Contract"
            legal_domain = "contract law"
            is_prosecution = False
            is_defense = False
            charge_type = None
        else:
            case_type = "Civil"
            legal_domain = "civil law"
            is_prosecution = False
            is_defense = False
            charge_type = None
        
        return {
            "case_type": case_type,
            "is_prosecution": is_prosecution,
            "is_defense": is_defense,
            "charge_type": charge_type,
            "legal_domain": legal_domain,
            "core_legal_issue": "Determined from case facts"
        }

