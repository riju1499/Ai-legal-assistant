"""
Intelligent router that decides which tools to use based on query analysis
"""

import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


ROUTER_PROMPT = """You are an intelligent routing system for a legal AI assistant. Analyze the user's query and decide which tools to use.

User Query: {query}

Available Tools:
1. **knowledge_base** - Legal documents (Constitution, laws, acts)
2. **case_search** - 10,456 court cases with precedents
3. **web_search** - Internet search for recent information
4. **hybrid** - Combination of multiple tools
5. **strategy** - Generate causal, precedent-backed case strategy (checklists, arguments, counterfactuals)

Analysis Guidelines:
- Constitutional/law questions → knowledge_base
- Case precedents/similar cases → case_search  
- Recent events/current affairs → web_search
- Complex queries asking for guidance → hybrid
- Strategy planning keywords ("strategy", "prepare case", "arguments", "counter-arguments", "what if", "probability", "success chance", "timeline", "checklist") → strategy
- When uncertain → hybrid

IMPORTANT: Respond with ONLY a valid JSON object. No additional text, explanations, or formatting.

{{
  "primary_tool": "tool_name",
  "reasoning": "why this tool",
  "use_knowledge_base": true,
  "use_cases": false,
  "use_web": false,
  "confidence": "high"
}}"""


class IntelligentRouter:
    """
    Routes queries to appropriate tools using LLM reasoning
    """
    
    def __init__(self, llm_client):
        """
        Initialize router
        
        Args:
            llm_client: LLMClient instance for routing decisions
        """
        self.llm = llm_client
    
    def route_query(self, query: str, conversation_history: list = None) -> Dict[str, Any]:
        """
        Analyze query and determine which tools to use
        
        Args:
            query: User query
            conversation_history: Previous conversation for context
            
        Returns:
            Dict with routing decision
        """
        logger.info(f"🧠 [ROUTER] Analyzing query: {query[:100]}")
        
        try:
            # Build context from conversation history
            context_info = ""
            if conversation_history and len(conversation_history) > 0:
                recent = conversation_history[-4:]  # Last 2 exchanges
                context_info = "\n\nRecent Conversation:\n"
                for msg in recent:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    content = msg["content"][:200]  # Truncate
                    context_info += f"{role}: {content}\n"
            
            # Get LLM routing decision
            prompt = ROUTER_PROMPT.format(query=query) + context_info
            response = self.llm.generate(prompt, max_tokens=256, temperature=0.3)
            
            # Parse JSON response
            routing = self._parse_routing_response(response)
            
            logger.info(f"✅ [ROUTER] Decision: {routing['primary_tool']} (confidence: {routing.get('confidence', 'unknown')})")
            logger.info(f"   Reasoning: {routing.get('reasoning', 'N/A')[:100]}")
            
            return routing
            
        except Exception as e:
            logger.error(f"❌ [ROUTER] Error: {e}, falling back to hybrid")
            return self._fallback_routing(query)
    
    def _parse_routing_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM routing response with robust JSON extraction"""
        try:
            # Clean and extract JSON from response
            response = response.strip()
            
            # Try multiple strategies to extract JSON
            json_str = None
            
            # Strategy 1: Find first complete JSON object
            start = response.find('{')
            if start >= 0:
                # Find the matching closing brace
                brace_count = 0
                end = start
                for i, char in enumerate(response[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                
                if brace_count == 0:
                    json_str = response[start:end]
            
            # Strategy 2: Try to find JSON in lines
            if not json_str:
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        json_str = line
                        break
            
            # Strategy 3: Extract first JSON-like structure
            if not json_str:
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
                if json_match:
                    json_str = json_match.group(0)
            
            if json_str:
                routing = json.loads(json_str)
                
                # Validate required fields
                if 'primary_tool' not in routing:
                    raise ValueError("Missing primary_tool")
                
                # Set defaults
                routing.setdefault('use_knowledge_base', False)
                routing.setdefault('use_cases', False)
                routing.setdefault('use_web', False)
                routing.setdefault('confidence', 'medium')
                routing.setdefault('reasoning', 'No reasoning provided')
                
                # If primary tool is hybrid, enable relevant sources
                if routing['primary_tool'] == 'hybrid':
                    routing['use_knowledge_base'] = True
                    routing['use_cases'] = True
                elif routing['primary_tool'] == 'knowledge_base':
                    routing['use_knowledge_base'] = True
                elif routing['primary_tool'] == 'case_search':
                    routing['use_cases'] = True
                elif routing['primary_tool'] == 'web_search':
                    routing['use_web'] = True
                elif routing['primary_tool'] == 'strategy':
                    routing['use_knowledge_base'] = True
                    routing['use_cases'] = True
                
                return routing
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.warning(f"Failed to parse routing response: {e}")
            logger.warning(f"Response was: {response[:200]}")
            raise
    
    def _fallback_routing(self, query: str) -> Dict[str, Any]:
        """
        Fallback routing using simple heuristics
        """
        query_lower = query.lower()
        
        # Check for keywords
        law_keywords = ['article', 'constitution', 'act', 'law', 'provision', 'section']
        case_keywords = ['case', 'precedent', 'ruling', 'judgment', 'court decision']
        recent_keywords = ['recent', 'latest', 'current', '2024', '2025', 'today']
        action_keywords = ['what should', 'how do i', 'what can i do', 'help me', 'advice']
        strategy_keywords = ['strategy', 'prepare case', 'arguments', 'counter-arguments', 'what if', 'probability', 'success chance', 'timeline', 'checklist', 'witness', 'evidence plan']
        
        has_law = any(kw in query_lower for kw in law_keywords)
        has_case = any(kw in query_lower for kw in case_keywords)
        has_recent = any(kw in query_lower for kw in recent_keywords)
        has_action = any(kw in query_lower for kw in action_keywords)
        has_strategy = any(kw in query_lower for kw in strategy_keywords)
        
        # Determine primary tool
        if has_strategy:
            primary = 'strategy'
            use_kb = True
            use_cases = True
            use_web = False
            reasoning = "User requests case strategy or counterfactual/probability planning"
        elif has_action or (has_law and has_case):
            primary = 'hybrid'
            use_kb = True
            use_cases = True
            use_web = False
            reasoning = "Complex query or action-oriented - using multiple sources"
        elif has_recent:
            primary = 'web_search'
            use_kb = False
            use_cases = False
            use_web = True
            reasoning = "Query about recent events"
        elif has_case:
            primary = 'case_search'
            use_kb = False
            use_cases = True
            use_web = False
            reasoning = "Query about court cases"
        elif has_law:
            primary = 'knowledge_base'
            use_kb = True
            use_cases = False
            use_web = False
            reasoning = "Query about legal documents"
        else:
            # Default to hybrid for ambiguous queries
            primary = 'hybrid'
            use_kb = True
            use_cases = True
            use_web = False
            reasoning = "Ambiguous query - using multiple sources"
        
        logger.info(f"🔄 [ROUTER] Fallback routing: {primary}")
        
        return {
            'primary_tool': primary,
            'use_knowledge_base': use_kb,
            'use_cases': use_cases,
            'use_web': use_web,
            'confidence': 'low',
            'reasoning': reasoning,
            'fallback': True
        }

