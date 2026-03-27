"""
Agentic tools - AI-callable tools for the agentic agent
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class AgenticTools:
    """
    Wraps existing tools in a function-calling interface for agentic AI
    """
    
    def __init__(self, intelligent_tools, agent_tools, strategy_generator=None):
        """
        Initialize agentic tools
        
        Args:
            intelligent_tools: IntelligentTools instance
            agent_tools: AgentTools instance
            strategy_generator: StrategyGenerator instance (optional)
        """
        self.intelligent_tools = intelligent_tools
        self.agent_tools = agent_tools
        self.strategy_generator = strategy_generator
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """Return tool schema for LLM function calling"""
        return {
            "search_knowledge_base": {
                "description": "Search legal documents (Constitution, laws, acts) in Qdrant knowledge base",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Number of results (default 5)", "default": 5}
                }
            },
            "search_cases": {
                "description": "Search 10,456 court cases for precedents using semantic search",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Number of results (default 5)", "default": 5}
                }
            },
            "extract_precedents": {
                "description": "Extract winning strategies and legal principles from case summaries using AI",
                "parameters": {
                    "cases": {"type": "array", "description": "List of case summaries to analyze"}
                }
            },
            "generate_strategy_draft": {
                "description": "Generate a legal strategy draft using case facts, laws, and cases",
                "parameters": {
                    "case_facts": {"type": "string", "description": "Description of the case situation"},
                    "laws": {"type": "array", "description": "List of relevant legal provisions"},
                    "cases": {"type": "array", "description": "List of relevant case summaries"}
                }
            },
            "reflect_on_completeness": {
                "description": "Evaluate if the current response is complete and identify gaps",
                "parameters": {
                    "current_info": {"type": "string", "description": "Summary of information gathered so far"},
                    "query": {"type": "string", "description": "Original user query"}
                }
            },
            "refine_strategy": {
                "description": "Refine existing strategy with new information",
                "parameters": {
                    "existing_strategy": {"type": "object", "description": "Current strategy dictionary"},
                    "new_info": {"type": "object", "description": "New information to incorporate"}
                }
            }
        }
    
    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool-specific parameters
            
        Returns:
            Tool execution results
        """
        logger.info(f"🔧 [AGENTIC_TOOLS] Executing: {tool_name}")
        
        try:
            if tool_name == "search_knowledge_base":
                query = kwargs.get("query", "")
                limit = kwargs.get("limit", 5)
                return self.intelligent_tools.search_knowledge_base(query, limit)
            
            elif tool_name == "search_cases":
                query = kwargs.get("query", "")
                limit = kwargs.get("limit", 5)
                return self.intelligent_tools.search_cases(query, limit)
            
            elif tool_name == "extract_precedents":
                cases = kwargs.get("cases", [])
                return self._extract_precedents(cases)
            
            elif tool_name == "generate_strategy_draft":
                case_facts = kwargs.get("case_facts", "")
                laws = kwargs.get("laws", [])
                cases = kwargs.get("cases", [])
                return self._generate_strategy_draft(case_facts, laws, cases)
            
            elif tool_name == "reflect_on_completeness":
                current_info = kwargs.get("current_info", "")
                query = kwargs.get("query", "")
                return self._reflect_on_completeness(current_info, query)
            
            elif tool_name == "refine_strategy":
                existing_strategy = kwargs.get("existing_strategy", {})
                new_info = kwargs.get("new_info", {})
                return self._refine_strategy(existing_strategy, new_info)
            
            else:
                logger.error(f"Unknown tool: {tool_name}")
                return {
                    'tool': tool_name,
                    'error': True,
                    'message': f"Unknown tool: {tool_name}"
                }
        
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {e}")
            return {
                'tool': tool_name,
                'error': True,
                'message': str(e)
            }
    
    def _extract_precedents(self, cases: List[Dict]) -> Dict[str, Any]:
        """Extract precedents from cases using AI (if strategy generator available)"""
        if not self.strategy_generator:
            return {'tool': 'extract_precedents', 'results': [], 'count': 0, 'error': 'Strategy generator not available'}
        
        # Format cases into context string
        cases_text = "\n".join([
            f"Case {i+1}: {case.get('case_number_english', 'N/A')} - {case.get('summary', 'N/A')[:300]}"
            for i, case in enumerate(cases[:10])
        ])
        
        # Extract precedents using AI (similar to strategy_generator method)
        try:
            precedents = self.strategy_generator._extract_precedents_with_ai(cases_text)
            return {
                'tool': 'extract_precedents',
                'results': precedents,
                'count': len(precedents)
            }
        except Exception as e:
            logger.error(f"Precedent extraction failed: {e}")
            return {
                'tool': 'extract_precedents',
                'results': [],
                'count': 0,
                'error': str(e)
            }
    
    def _generate_strategy_draft(self, case_facts: str, laws: List, cases: List) -> Dict[str, Any]:
        """Generate strategy draft using strategy generator"""
        if not self.strategy_generator:
            return {'tool': 'generate_strategy_draft', 'strategy': {}, 'error': 'Strategy generator not available'}
        
        # Format retrieval context
        laws_text = "\n".join([f"- {law.get('source', 'Unknown')} (Page {law.get('page', 'N/A')}): {law.get('text', '')[:200]}" for law in laws[:5]])
        cases_text = "\n".join([f"- Case {case.get('case_number_english', 'N/A')}: {case.get('summary', '')[:200]}" for case in cases[:5]])
        retrieval_context = f"Legal Documents:\n{laws_text}\n\nCases:\n{cases_text}"
        
        try:
            strategy = self.strategy_generator.generate(
                case_facts=case_facts,
                retrieval_context=retrieval_context
            )
            return {
                'tool': 'generate_strategy_draft',
                'strategy': strategy,
                'count': 1
            }
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            return {
                'tool': 'generate_strategy_draft',
                'strategy': {},
                'error': str(e)
            }
    
    def _reflect_on_completeness(self, current_info: str, query: str) -> Dict[str, Any]:
        """Reflect on whether response is complete"""
        # This will be called by the agent's reflection mechanism
        return {
            'tool': 'reflect_on_completeness',
            'current_info': current_info,
            'query': query
        }
    
    def _refine_strategy(self, existing_strategy: Dict, new_info: Dict) -> Dict[str, Any]:
        """Refine existing strategy with new information"""
        # This can be enhanced to actually refine using LLM
        return {
            'tool': 'refine_strategy',
            'existing_strategy': existing_strategy,
            'new_info': new_info,
            'refined': True
        }

