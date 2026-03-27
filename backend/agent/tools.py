"""
Agent tools for legal case analysis
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class AgentTools:
    """
    Tools for the legal AI agent to interact with case database and knowledge base
    """
    
    def __init__(self, search_engine, metadata: List[Dict], glossary_dir: Path, qdrant_kb=None):
        """
        Initialize agent tools
        
        Args:
            search_engine: SemanticSearchEngine instance from main.py
            metadata: Case metadata list
            glossary_dir: Path to glossary directory
            qdrant_kb: QdrantKnowledgeBase instance (optional)
        """
        self.search_engine = search_engine
        self.metadata = metadata
        self.glossary_dir = glossary_dir
        self.qdrant_kb = qdrant_kb
        
        # Load glossaries
        self.glossaries = self._load_glossaries()
    
    def _load_glossaries(self) -> Dict[str, Any]:
        """Load all legal glossaries"""
        glossaries = {}
        glossary_files = [
            'legal_glossary.json',
            'legal_terms.json',
            'case_types.json',
            'courts.json',
            'verdicts.json'
        ]
        
        for filename in glossary_files:
            file_path = self.glossary_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        glossaries[filename.replace('.json', '')] = json.load(f)
                    logger.info(f"Loaded {filename}")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
        
        return glossaries
    
    def search_similar_cases(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar cases using semantic search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar cases with metadata
        """
        try:
            results = self.search_engine.search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Case search failed: {e}")
            return []
    
    def get_case_details(self, case_index: int) -> Optional[Dict[str, Any]]:
        """Get full details of a specific case by index"""
        if 0 <= case_index < len(self.metadata):
            return self.metadata[case_index]
        return None
    
    def analyze_case_outcomes(self, case_indices: List[int]) -> Dict[str, Any]:
        """
        Analyze outcomes from multiple cases
        
        Args:
            case_indices: List of case indices to analyze
            
        Returns:
            Analysis of case outcomes and patterns
        """
        cases = [self.get_case_details(idx) for idx in case_indices if idx is not None]
        cases = [c for c in cases if c is not None]
        
        if not cases:
            return {"error": "No valid cases found"}
        
        # Extract case types, courts, and basic patterns
        case_types = {}
        courts = {}
        
        for case in cases:
            ct = case.get('case_type_english', 'Unknown')
            court = case.get('court_english', 'Unknown')
            
            case_types[ct] = case_types.get(ct, 0) + 1
            courts[court] = courts.get(court, 0) + 1
        
        return {
            "total_cases": len(cases),
            "case_types": case_types,
            "courts": courts,
            "cases": cases
        }
    
    def find_legal_provisions(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search glossary for relevant legal terms and provisions
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            Matching legal terms and definitions
        """
        results = []
        
        # Search in legal glossary
        if 'legal_glossary' in self.glossaries:
            glossary = self.glossaries['legal_glossary']
            for keyword in keywords:
                keyword_lower = keyword.lower()
                for term, definition in glossary.items():
                    if keyword_lower in term.lower():
                        results.append({
                            "term": term,
                            "definition": definition,
                            "source": "legal_glossary"
                        })
        
        return results
    
    def find_causal_patterns(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find relevant causal patterns based on query (deprecated - now AI-driven)
        
        Args:
            query: Search query
            top_k: Number of patterns to return
            
        Returns:
            Empty list - patterns are now extracted dynamically via AI
        """
        # This method is kept for backward compatibility but returns empty
        # Causal patterns are now extracted dynamically using AI from case analysis
        return []
    
    def extract_key_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract key legal entities from text (simple keyword-based)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            "case_types": [],
            "courts": [],
            "articles": []
        }
        
        text_lower = text.lower()
        
        # Extract case types
        if 'case_types' in self.glossaries:
            for case_type in self.glossaries['case_types']:
                if isinstance(case_type, str) and case_type.lower() in text_lower:
                    entities["case_types"].append(case_type)
        
        # Extract courts
        if 'courts' in self.glossaries:
            for court in self.glossaries['courts']:
                if isinstance(court, str) and court.lower() in text_lower:
                    entities["courts"].append(court)
        
        # Extract article references (simple pattern)
        import re
        article_pattern = r'article\s+(\d+)'
        articles = re.findall(article_pattern, text_lower)
        entities["articles"] = [f"Article {a}" for a in articles]
        
        return entities
    
    def search_knowledge_base(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search Qdrant knowledge base for relevant legal information
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant knowledge chunks from PDFs
        """
        if not self.qdrant_kb:
            logger.warning("Qdrant knowledge base not available")
            return []
        
        try:
            results = self.qdrant_kb.search(
                query=query,
                limit=limit,
                score_threshold=0.3  # Lowered from 0.5 for better recall
            )
            logger.info(f"Qdrant search returned {len(results)} results for query: {query[:50]}...")
            if results:
                logger.info(f"  Top result: {results[0]['source']} (Page {results[0]['page']}, Score: {results[0]['score']:.3f})")
            return results
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

