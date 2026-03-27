"""
Causal reasoning module for legal case analysis
"""

import logging
from typing import List, Dict, Any, Optional
from .prompts import CAUSAL_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class CausalReasoning:
    """
    Extracts and analyzes causal relationships in legal cases
    """
    
    def __init__(self, llm_client):
        """
        Initialize causal reasoning module
        
        Args:
            llm_client: LLMClient instance
        """
        self.llm = llm_client
    
    def extract_causal_chain(self, case_summary: str) -> List[Dict[str, str]]:
        """
        Extract cause-effect relationships from a case summary
        
        Args:
            case_summary: Case summary text
            
        Returns:
            List of causal relationships
        """
        if not case_summary or case_summary == "Summary generation failed":
            return []
        
        try:
            prompt = CAUSAL_EXTRACTION_PROMPT.format(case_summary=case_summary)
            response = self.llm.generate(prompt, max_tokens=512, temperature=0.3)
            
            # Parse the LLM response
            causal_chains = self._parse_causal_response(response)
            return causal_chains
            
        except Exception as e:
            logger.error(f"Causal extraction failed: {e}")
            return []
    
    def _parse_causal_response(self, response: str) -> List[Dict[str, str]]:
        """Parse LLM response into structured causal relationships"""
        chains = []
        current_chain = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('CAUSE:'):
                current_chain['cause'] = line.replace('CAUSE:', '').strip()
            elif line.startswith('EFFECT:'):
                current_chain['effect'] = line.replace('EFFECT:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                current_chain['confidence'] = line.replace('CONFIDENCE:', '').strip()
                if current_chain.get('cause') and current_chain.get('effect'):
                    chains.append(current_chain.copy())
                    current_chain = {}
        
        # Add last chain if complete
        if current_chain.get('cause') and current_chain.get('effect'):
            chains.append(current_chain)
        
        return chains
    
    def analyze_multiple_cases(
        self, 
        cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze causal patterns across multiple cases
        
        Args:
            cases: List of case metadata dictionaries
            
        Returns:
            Aggregated causal analysis
        """
        all_chains = []
        
        for case in cases:
            summary = case.get('summary', '')
            if summary and summary != "Summary generation failed":
                chains = self.extract_causal_chain(summary)
                for chain in chains:
                    chain['case_number'] = case.get('case_number_english', 'Unknown')
                    chain['case_type'] = case.get('case_type_english', 'Unknown')
                all_chains.append(chains)
        
        return {
            "total_patterns": sum(len(c) for c in all_chains),
            "patterns": [chain for chains in all_chains for chain in chains]
        }
    
    def identify_common_patterns(
        self, 
        causal_chains: List[Dict[str, str]], 
        min_occurrences: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Identify commonly occurring causal patterns
        
        Args:
            causal_chains: List of causal relationship dictionaries
            min_occurrences: Minimum number of occurrences to be considered common
            
        Returns:
            List of common patterns with occurrence counts
        """
        pattern_counts = {}
        
        for chain in causal_chains:
            # Create a simplified pattern key
            cause_key = self._normalize_text(chain.get('cause', ''))
            effect_key = self._normalize_text(chain.get('effect', ''))
            pattern_key = f"{cause_key} → {effect_key}"
            
            if pattern_key not in pattern_counts:
                pattern_counts[pattern_key] = {
                    'cause': chain.get('cause'),
                    'effect': chain.get('effect'),
                    'count': 0,
                    'examples': []
                }
            
            pattern_counts[pattern_key]['count'] += 1
            if len(pattern_counts[pattern_key]['examples']) < 3:
                pattern_counts[pattern_key]['examples'].append({
                    'case_number': chain.get('case_number'),
                    'case_type': chain.get('case_type')
                })
        
        # Filter by minimum occurrences and sort
        common_patterns = [
            p for p in pattern_counts.values() 
            if p['count'] >= min_occurrences
        ]
        common_patterns.sort(key=lambda x: x['count'], reverse=True)
        
        return common_patterns
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for pattern matching"""
        # Simple normalization: lowercase, remove extra spaces
        return ' '.join(text.lower().split())
    
    def explain_causality(
        self, 
        query: str, 
        relevant_cases: List[Dict[str, Any]]
    ) -> str:
        """
        Generate causal explanation for a query based on relevant cases
        
        Args:
            query: User query
            relevant_cases: List of relevant case metadata
            
        Returns:
            Causal explanation text
        """
        if not relevant_cases:
            return "No causal patterns identified."
        
        # Extract causal chains from relevant cases
        chains = []
        for case in relevant_cases[:5]:  # Limit to top 5
            summary = case.get('summary', '')
            case_chains = self.extract_causal_chain(summary)
            chains.extend(case_chains)
        
        if not chains:
            return "Could not extract clear causal relationships from available cases."
        
        # Build explanation
        explanation_parts = []
        explanation_parts.append("**Causal Analysis:**\n")
        
        for i, chain in enumerate(chains[:3], 1):  # Show top 3
            cause = chain.get('cause', 'Unknown')
            effect = chain.get('effect', 'Unknown')
            explanation_parts.append(
                f"{i}. **When**: {cause}\n   **Then**: {effect}\n"
            )
        
        return '\n'.join(explanation_parts)

