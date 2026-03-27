"""
Intelligent tool definitions for the legal AI agent
Each tool has a clear description for LLM-based routing
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Available tool types"""
    KNOWLEDGE_BASE = "knowledge_base"  # Laws, constitution, legal documents
    CASE_SEARCH = "case_search"        # Search processed court cases
    WEB_SEARCH = "web_search"          # Internet search for recent info
    HYBRID = "hybrid"                  # Combination of multiple tools
    STRATEGY = "strategy"              # Causal, precedent-backed strategy generation


class IntelligentTools:
    """
    Intelligent tool wrapper with clear descriptions for LLM routing
    """
    
    def __init__(self, agent_tools, qdrant_kb, web_search_enabled=True, strategy_generator=None):
        """
        Initialize intelligent tools
        
        Args:
            agent_tools: AgentTools instance (has case search)
            qdrant_kb: QdrantKnowledgeBase instance
            web_search_enabled: Whether web search is available
        """
        self.agent_tools = agent_tools
        self.qdrant_kb = qdrant_kb
        self.web_search_enabled = web_search_enabled
        self.strategy_generator = strategy_generator
    
    def get_tool_descriptions(self) -> str:
        """Get tool descriptions for LLM routing"""
        return """
Available Tools:

1. **knowledge_base** - Search legal documents (Constitution, laws, acts)
   - Use for: Constitutional articles, legal provisions, statutory law
   - Example: "What does Article 12 say?", "What is the Civil Code?"
   - Returns: Exact text from legal documents with citations

2. **case_search** - Search 10,456 processed court cases
   - Use for: Similar cases, precedents, case law, judicial interpretations
   - Example: "Find cases about property disputes", "Similar citizenship cases"
   - Returns: Relevant court cases with summaries and outcomes

3. **web_search** - Search the internet for recent information
   - Use for: Recent events, current legal developments, context not in database
   - Example: "Recent Supreme Court rulings 2024", "Current citizenship law debates"
   - Returns: Up-to-date information from web sources

4. **hybrid** - Combine multiple tools for comprehensive answers
   - Use for: Complex queries needing multiple sources
   - Example: "What does law say + are there cases + recent developments?"
   - Returns: Synthesized response from all relevant sources

5. **strategy** - Generate legal case strategy (causal, with precedents)
   - Use for: "prepare strategy", "what if", "arguments/counter-arguments", "probability"
   - Returns: Structured plan with factors, laws, precedents, evidence/witness plan, timeline

Tool Selection Guidelines:
- If query asks about specific law/article → knowledge_base
- If query asks about cases/precedents → case_search
- If query asks about recent events → web_search
- If query is complex or asks "what should I do?" → hybrid
- When in doubt, use hybrid for comprehensive answers
"""
    
    def search_knowledge_base(
        self, 
        query: str, 
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Search legal documents in Qdrant
        
        Returns:
            Dict with 'tool', 'results', 'summary'
        """
        logger.info(f"🔍 [KNOWLEDGE_BASE] Searching: {query[:100]}")
        
        try:
            results = self.qdrant_kb.search(
                query=query,
                limit=limit,
                score_threshold=0.3
            )
            
            summary = f"Found {len(results)} relevant legal documents"
            if results:
                top_source = results[0].get('source', 'Unknown')
                summary += f" (top: {top_source})"
            
            logger.info(f"✅ [KNOWLEDGE_BASE] {summary}")
            
            return {
                'tool': 'knowledge_base',
                'query': query,
                'results': results,
                'summary': summary,
                'count': len(results)
            }
        except Exception as e:
            logger.error(f"❌ [KNOWLEDGE_BASE] Error: {e}")
            return {
                'tool': 'knowledge_base',
                'query': query,
                'results': [],
                'summary': f"Error searching knowledge base: {str(e)}",
                'count': 0
            }
    
    def search_cases(
        self, 
        query: str, 
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Search processed court cases
        
        Returns:
            Dict with 'tool', 'results', 'summary'
        """
        logger.info(f"⚖️  [CASE_SEARCH] Searching: {query[:100]}")
        
        try:
            # Use the existing case search from agent_tools
            cases = self.agent_tools.search_similar_cases(query, k=limit)
            
            summary = f"Found {len(cases)} similar court cases"
            if cases:
                top_case = cases[0].get('case_number_english', 'Unknown')
                summary += f" (top: {top_case})"
            
            logger.info(f"✅ [CASE_SEARCH] {summary}")
            
            return {
                'tool': 'case_search',
                'query': query,
                'results': cases,
                'summary': summary,
                'count': len(cases)
            }
        except Exception as e:
            logger.error(f"❌ [CASE_SEARCH] Error: {e}")
            return {
                'tool': 'case_search',
                'query': query,
                'results': [],
                'summary': f"Error searching cases: {str(e)}",
                'count': 0
            }
    
    def search_web(
        self, 
        query: str, 
        limit: int = 3
    ) -> Dict[str, Any]:
        """
        Search the web for recent information
        
        Returns:
            Dict with 'tool', 'results', 'summary'
        """
        logger.info(f"🌐 [WEB_SEARCH] Searching: {query[:100]}")
        
        if not self.web_search_enabled:
            logger.warning("⚠️  [WEB_SEARCH] Web search is disabled")
            return {
                'tool': 'web_search',
                'query': query,
                'results': [],
                'summary': 'Web search is currently disabled',
                'count': 0
            }
        
        try:
            # Note: Web search would be implemented here
            # For now, return placeholder
            logger.info("✅ [WEB_SEARCH] Completed (placeholder)")
            
            return {
                'tool': 'web_search',
                'query': query,
                'results': [],
                'summary': 'Web search results (feature coming soon)',
                'count': 0
            }
        except Exception as e:
            logger.error(f"❌ [WEB_SEARCH] Error: {e}")
            return {
                'tool': 'web_search',
                'query': query,
                'results': [],
                'summary': f"Error searching web: {str(e)}",
                'count': 0
            }
    
    def hybrid_search(
        self, 
        query: str,
        use_knowledge_base: bool = True,
        use_cases: bool = True,
        use_web: bool = False
    ) -> Dict[str, Any]:
        """
        Perform hybrid search across multiple tools
        
        Returns:
            Dict with results from all tools
        """
        logger.info(f"🔄 [HYBRID] Searching across multiple sources: {query[:100]}")
        
        results = {
            'tool': 'hybrid',
            'query': query,
            'sources': {}
        }
        
        # Search knowledge base
        if use_knowledge_base:
            kb_results = self.search_knowledge_base(query, limit=3)
            results['sources']['knowledge_base'] = kb_results
        
        # Search cases
        if use_cases:
            case_results = self.search_cases(query, limit=3)
            results['sources']['cases'] = case_results
        
        # Search web
        if use_web and self.web_search_enabled:
            web_results = self.search_web(query, limit=2)
            results['sources']['web'] = web_results
        
        # Create summary
        summaries = []
        for source_name, source_data in results['sources'].items():
            count = source_data.get('count', 0)
            if count > 0:
                summaries.append(f"{source_name}: {count} results")
        
        results['summary'] = "Hybrid search completed - " + ", ".join(summaries) if summaries else "No results found"
        
        logger.info(f"✅ [HYBRID] {results['summary']}")
        
        return results

    def generate_strategy(
        self,
        query: str,
        desired_outcome: Optional[str] = None,
        case_type_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a causal, precedent-backed strategy for any case type.
        Uses the injected StrategyGenerator and augments with brief retrieval context.
        """
        logger.info(f"🧭  [STRATEGY] Generating strategy for: {query[:120]}")
        if self.strategy_generator is None:
            return {
                'tool': 'strategy',
                'summary': 'Strategy generator not available',
                'strategy': {},
                'error': True
            }

        # Build a light retrieval context from case search to ground the strategy
        retrieval_context = None
        try:
            cases = self.agent_tools.search_similar_cases(query, k=5)
            lines = []
            for h in cases[:5]:
                num = h.get('case_number_english', '')
                ctype = h.get('case_type_english', '')
                court = h.get('court_english', '')
                summary = h.get('summary', '')
                lines.append(f"Case {num} | {ctype} | {court}: {summary[:300]}")
            retrieval_context = "\n".join(lines)
        except Exception:
            retrieval_context = None

        strategy = self.strategy_generator.generate(
            case_facts=query,
            desired_outcome=desired_outcome,
            case_type_hint=case_type_hint,
            retrieval_context=retrieval_context,
        )

        return {
            'tool': 'strategy',
            'query': query,
            'summary': 'Strategy generated',
            'strategy': strategy
        }
    
    def execute_tool(
        self, 
        tool_name: str, 
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by name
        
        Args:
            tool_name: One of 'knowledge_base', 'case_search', 'web_search', 'hybrid'
            query: Search query
            **kwargs: Additional arguments for the tool
        
        Returns:
            Tool execution results
        """
        tool_map = {
            'knowledge_base': self.search_knowledge_base,
            'case_search': self.search_cases,
            'web_search': self.search_web,
            'hybrid': self.hybrid_search,
            'strategy': self.generate_strategy
        }
        
        if tool_name not in tool_map:
            logger.error(f"Unknown tool: {tool_name}")
            return {
                'tool': tool_name,
                'query': query,
                'results': [],
                'summary': f"Unknown tool: {tool_name}",
                'error': True
            }
        
        return tool_map[tool_name](query, **kwargs)

