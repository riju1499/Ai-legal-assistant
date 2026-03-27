"""
LawIdentificationAgent: Identifies relevant laws based on case type and facts
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class LawIdentificationAgent:
    def __init__(self, llm_client, agentic_tools):
        self.llm = llm_client
        self.tools = agentic_tools
    
    def run(self, facts: str, case_type: str, legal_domain: str) -> Dict[str, Any]:
        """
        Identify relevant laws based on case type and facts
        
        Args:
            facts: Case facts description
            case_type: Detected case type
            legal_domain: Legal domain (criminal law, civil law, etc.)
            
        Returns:
            Dictionary with identified laws
        """
        facts_lower = (facts or "").lower()
        case_type_lower = (case_type or "").lower()
        
        # Build case-type-specific search queries
        queries = []
        
        if "criminal" in case_type_lower:
            if "murder" in facts_lower or "killed" in facts_lower:
                queries = [
                    "National Penal Code Act 2017 murder homicide",
                    "National Criminal Procedure Act 2017 evidence prosecution",
                    "Nepal Constitution criminal rights",
                    "National Penal Code Act 2017 Section 182",
                    "National Criminal Procedure Act 2017 Section 40"
                ]
            elif "theft" in facts_lower:
                queries = [
                    "National Penal Code Act 2017 theft property",
                    "National Criminal Procedure Act 2017 theft cases"
                ]
            else:
                queries = [
                    f"National Penal Code Act 2017 {case_type}",
                    "National Criminal Procedure Act 2017",
                    "Nepal Constitution criminal procedure"
                ]
        elif "property" in case_type_lower or "inherit" in case_type_lower:
            queries = [
                "Muluki Civil Code 2017 joint family property inheritance",
                "Muluki Civil Code 2017 Sections 145-280",
                "Muluki Civil Code 2017 Section 259",
                "Muluki Civil Code 2017 Section 124",
                "Nepal Constitution Article 15 property rights"
            ]
        elif "contract" in case_type_lower:
            queries = [
                "Muluki Civil Code 2017 contract",
                "Muluki Civil Code 2017 breach of contract",
                "Nepal contract law"
            ]
        else:
            queries = [
                f"Nepal {case_type} law",
                f"Muluki Civil Code 2017 {case_type}",
                f"National laws {case_type}"
            ]
        
        # Search for each query
        all_laws = []
        seen_sources = set()
        
        for query in queries:
            try:
                results = self.tools.execute("search_knowledge_base", query=query, limit=3)
                if results and isinstance(results, dict):
                    law_results = results.get("results", [])
                    for law in law_results:
                        source = law.get('source', '')
                        # Avoid duplicates
                        if source and source not in seen_sources:
                            seen_sources.add(source)
                            all_laws.append(law)
            except Exception as e:
                logger.warning(f"Law search failed for query '{query}': {e}")
        
        return {
            "laws": all_laws[:15],  # Limit to top 15
            "queries_used": queries,
            "case_type": case_type,
            "legal_domain": legal_domain
        }

