"""
RAG (Retrieval-Augmented Generation) pipeline for legal queries
"""

import logging
from typing import Dict, Any, List, Optional
from .prompts import (
    QUERY_CLASSIFICATION_PROMPT,
    LEGAL_RESPONSE_PROMPT,
    LOOPHOLE_ANALYSIS_PROMPT,
    CASE_SUMMARY_PROMPT
)
from .causal_reasoning import CausalReasoning

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the RAG workflow for legal queries
    """
    
    def __init__(self, llm_client, agent_tools):
        """
        Initialize RAG pipeline
        
        Args:
            llm_client: LLMClient instance
            agent_tools: AgentTools instance
        """
        self.llm = llm_client
        self.tools = agent_tools
        self.causal = CausalReasoning(llm_client)
    
    def process_query(
        self, 
        query: str, 
        include_similar_cases: bool = True
    ) -> Dict[str, Any]:
        """
        Process a legal query end-to-end
        
        Args:
            query: User query
            include_similar_cases: Whether to include similar case retrieval
            
        Returns:
            Response dictionary with answer, cases, and metadata
        """
        try:
            # Step 1: Classify query intent
            query_type = self._classify_query(query)
            logger.info(f"Query classified as: {query_type}")
            
            # Step 2: Retrieve relevant information
            context = self._build_context(query, query_type, include_similar_cases)
            
            # Step 3: Generate response
            response = self._generate_response(query, query_type, context)
            
            # Step 4: Add disclaimer
            response['disclaimer'] = self._get_disclaimer()
            
            return response
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your query. Please try again or rephrase your question.",
                "error": str(e),
                "similar_cases": [],
                "causal_explanation": "",
                "disclaimer": self._get_disclaimer()
            }
    
    def _classify_query(self, query: str) -> str:
        """Classify query intent"""
        try:
            prompt = QUERY_CLASSIFICATION_PROMPT.format(query=query)
            classification = self.llm.generate(prompt, max_tokens=50, temperature=0.1)
            classification = classification.strip().upper()
            
            # Validate classification
            valid_types = [
                'LAW_EXPLANATION', 
                'CASE_RECOMMENDATION', 
                'LEGAL_ADVICE', 
                'LOOPHOLE_ANALYSIS', 
                'GENERAL_INQUIRY'
            ]
            
            for vtype in valid_types:
                if vtype in classification:
                    return vtype
            
            # Default
            return 'GENERAL_INQUIRY'
            
        except Exception as e:
            logger.warning(f"Query classification failed: {e}")
            return 'GENERAL_INQUIRY'
    
    def _build_context(
        self, 
        query: str, 
        query_type: str, 
        include_similar_cases: bool
    ) -> Dict[str, Any]:
        """Build context for LLM response generation"""
        context = {
            'similar_cases': [],
            'causal_patterns': [],
            'legal_provisions': [],
            'knowledge_chunks': [],
            'entities': {}
        }
        
        # Extract entities
        context['entities'] = self.tools.extract_key_entities(query)
        
        # PRIORITY 1: Search Qdrant knowledge base for authoritative legal information
        logger.info(f"🔍 Searching Qdrant for query: {query[:100]}")
        try:
            kb_results = self.tools.search_knowledge_base(query, limit=5)
            logger.info(f"📊 Qdrant returned {len(kb_results)} results")
            if kb_results:
                context['knowledge_chunks'] = kb_results
                logger.info(f"✅ Found {len(kb_results)} relevant knowledge chunks from PDFs")
                for i, chunk in enumerate(kb_results[:2], 1):
                    logger.info(f"   {i}. {chunk.get('source', 'Unknown')} (Page {chunk.get('page', 'N/A')}, Score: {chunk.get('score', 0):.3f})")
            else:
                logger.warning(f"⚠️  No knowledge chunks found for query")
        except Exception as e:
            logger.error(f"❌ Knowledge base search failed: {e}", exc_info=True)
        
        # Retrieve similar cases (except for pure law explanations)
        if include_similar_cases and query_type != 'LAW_EXPLANATION':
            try:
                cases = self.tools.search_similar_cases(query, k=5)
                context['similar_cases'] = cases
                
                # Extract causal patterns from retrieved cases
                if cases:
                    case_indices = [c.get('index') for c in cases if c.get('index') is not None]
                    case_details = [
                        self.tools.get_case_details(idx) 
                        for idx in case_indices
                    ]
                    case_details = [c for c in case_details if c is not None]
                    
                    # Analyze causality
                    causal_analysis = self.causal.analyze_multiple_cases(case_details)
                    if causal_analysis['patterns']:
                        context['causal_patterns'] = causal_analysis['patterns'][:3]
                
            except Exception as e:
                logger.error(f"Case retrieval failed: {e}")
        
        # Find relevant legal provisions from glossary
        keywords = query.split()[:10]  # Use first 10 words as keywords
        context['legal_provisions'] = self.tools.find_legal_provisions(keywords)
        
        # Causal patterns are now extracted dynamically from case analysis above (line 152)
        # No need to load static patterns from knowledge base
        
        return context
    
    def _generate_response(
        self, 
        query: str, 
        query_type: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final response using LLM"""
        
        # Format context for prompt
        knowledge_text = self._format_knowledge_chunks(context.get('knowledge_chunks', []))
        similar_cases_text = self._format_cases(context['similar_cases'])
        causal_patterns_text = self._format_causal_patterns(context['causal_patterns'])
        provisions_text = self._format_provisions(context['legal_provisions'])
        
        # Combine context - prioritize authoritative knowledge from PDFs
        combined_context = f"{knowledge_text}\n{provisions_text}\n{similar_cases_text}"
        
        # Debug: Log what context is being sent to LLM
        logger.info(f"📝 Context for LLM ({len(combined_context)} chars):")
        logger.info(f"   - Knowledge chunks: {len(context.get('knowledge_chunks', []))}")
        logger.info(f"   - Similar cases: {len(context.get('similar_cases', []))}")
        logger.info(f"   - Legal provisions: {len(context.get('legal_provisions', []))}")
        
        # Select appropriate prompt
        if query_type == 'LOOPHOLE_ANALYSIS':
            prompt = LOOPHOLE_ANALYSIS_PROMPT.format(
                situation=query,
                legal_context=combined_context
            )
        else:
            prompt = LEGAL_RESPONSE_PROMPT.format(
                query=query,
                context=combined_context
            )
        
        # Generate response
        try:
            response_text = self.llm.generate(prompt, max_tokens=1024, temperature=0.7)
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            response_text = "I apologize, but I'm unable to generate a response at this time."
        
        # Build response object
        response = {
            "response": response_text,
            "query_type": query_type,
            "similar_cases": self._filter_relevant_cases(
                context['similar_cases'], 
                query_type
            ),
            "causal_explanation": causal_patterns_text,
            "entities": context['entities']
        }
        
        return response
    
    def _format_cases(self, cases: List[Dict[str, Any]]) -> str:
        """Format cases for prompt"""
        if not cases:
            return "No similar cases found."
        
        formatted = ["Similar Cases:"]
        for i, case in enumerate(cases[:5], 1):
            case_num = case.get('case_number_english', 'N/A')
            case_type = case.get('case_type_english', 'N/A')
            court = case.get('court_english', 'N/A')
            summary = case.get('summary', 'No summary available')
            
            # Truncate summary if too long
            if len(summary) > 300:
                summary = summary[:300] + "..."
            
            formatted.append(
                f"\n{i}. Case: {case_num}\n"
                f"   Type: {case_type}\n"
                f"   Court: {court}\n"
                f"   Summary: {summary}"
            )
        
        return '\n'.join(formatted)
    
    def _format_causal_patterns(self, patterns: List[Dict[str, str]]) -> str:
        """Format causal patterns for prompt"""
        if not patterns:
            return "No causal patterns identified."
        
        formatted = ["Causal Patterns:"]
        for i, pattern in enumerate(patterns[:3], 1):
            cause = pattern.get('cause', 'Unknown')
            effect = pattern.get('effect', 'Unknown')
            confidence = pattern.get('confidence', 'unknown')
            
            formatted.append(
                f"\n{i}. When: {cause}\n"
                f"   Then: {effect}\n"
                f"   Confidence: {confidence}"
            )
        
        return '\n'.join(formatted)
    
    def _format_provisions(self, provisions: List[Dict[str, Any]]) -> str:
        """Format legal provisions for prompt"""
        if not provisions:
            return ""
        
        formatted = ["Relevant Legal Terms:"]
        for prov in provisions[:5]:
            term = prov.get('term', '')
            definition = prov.get('definition', '')
            formatted.append(f"- {term}: {definition}")
        
        return '\n'.join(formatted)
    
    def _format_knowledge_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format knowledge base chunks for prompt"""
        if not chunks:
            return "No relevant documents found."
        
        formatted = []
        for i, chunk in enumerate(chunks[:5], 1):
            source = chunk.get('source', 'Unknown')
            page = chunk.get('page', 'N/A')
            text = chunk.get('text', '')
            score = chunk.get('score', 0)
            
            # Truncate if too long
            if len(text) > 800:
                text = text[:800] + "..."
            
            formatted.append(
                f"\nDocument {i}:\n"
                f"Source: {source}, Page {page}\n"
                f"Content: {text}\n"
                f"{'-' * 80}"
            )
        
        return '\n'.join(formatted)
    
    def _filter_relevant_cases(
        self, 
        cases: List[Dict[str, Any]], 
        query_type: str
    ) -> List[Dict[str, Any]]:
        """
        Filter cases to include only if they add value
        """
        # Always show cases for case recommendation queries
        if query_type == 'CASE_RECOMMENDATION':
            return cases
        
        # For legal advice and loophole analysis, show top 3
        if query_type in ['LEGAL_ADVICE', 'LOOPHOLE_ANALYSIS']:
            return cases[:3]
        
        # For law explanations, only show if highly relevant (score > 0.8)
        if query_type == 'LAW_EXPLANATION':
            return [c for c in cases if c.get('score', 0) > 0.8][:2]
        
        # Default: show top 3
        return cases[:3]
    
    def _get_disclaimer(self) -> str:
        """Get legal disclaimer text"""
        return (
            "⚠️ **Legal Disclaimer**: This is AI-generated legal information, "
            "not legal advice. Consult a qualified attorney for your specific situation."
        )

