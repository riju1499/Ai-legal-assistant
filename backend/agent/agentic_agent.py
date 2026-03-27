"""
Agentic Legal AI Agent with iterative reasoning and autonomous tool selection
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional

from .agent_state import AgentState

logger = logging.getLogger(__name__)


DECISION_PROMPT = """You are an autonomous legal AI agent. Based on the current state, decide what action to take next.

Current Query: {query}

Current State:
- Iteration: {iteration}/{max_iterations}
- Tools used so far: {tools_used}
- Information gathered: {info_summary}

Available Tools:
1. search_knowledge_base(query, limit) - Search legal documents (Constitution, laws)
2. search_cases(query, limit) - Search 10,456 court cases
3. extract_precedents(cases) - Extract winning strategies from cases
4. generate_strategy_draft(case_facts, laws, cases) - Generate legal strategy
5. reflect_on_completeness(current_info, query) - Evaluate completeness
6. refine_strategy(existing_strategy, new_info) - Refine strategy

Think step-by-step:
1. What information do I still need to answer this query?
2. Which tool will get me the most useful information next?
3. Should I continue gathering info or start synthesizing?

Respond with ONLY a valid JSON object:
{{
  "action": "tool_name" or "DONE",
  "tool_name": "search_knowledge_base" | "search_cases" | "extract_precedents" | "generate_strategy_draft" | "reflect_on_completeness" | "refine_strategy",
  "parameters": {{"query": "...", "limit": 5}},
  "reasoning": "Why I'm choosing this action"
}}

If you have enough information to answer, set action to "DONE"."""

REFLECTION_PROMPT = """Evaluate whether you have enough information to provide a complete answer.

Query: {query}

Current Information Gathered:
{info_summary}

Tools Used: {tools_used}
Iterations: {iteration}

Questions to consider:
1. Does the information fully answer the user's query?
2. Are there any obvious gaps or missing pieces?
3. Would one more tool search significantly improve the answer?
4. Is the information sufficient for a comprehensive legal response?

Respond with ONLY a valid JSON object:
{{
  "complete": true or false,
  "gaps": ["gap1", "gap2"],
  "needs_more": true or false,
  "reasoning": "Explanation"
}}"""

SYNTHESIS_PROMPT = """You are Wakalat Sewa, an intelligent legal AI assistant for Nepali law.

User Query: {query}

{conversation_context}

You have gathered the following information through iterative reasoning:

{all_information}

Synthesize a comprehensive, accurate response that:
1. Directly answers the user's query
2. Cites all sources (legal documents with page numbers, cases with case numbers)
3. Provides actionable guidance if applicable
4. Uses markdown for formatting

Agent Reasoning Summary:
- Iterations used: {iterations}
- Tools used: {tools_used}
- Reasoning steps: {reasoning_history}

Provide your comprehensive response:"""


class AgenticLegalAgent:
    """
    Agentic agent with iterative reasoning loop
    """
    
    def __init__(self, agentic_tools, llm_client, max_iterations=5):
        """
        Initialize agentic agent
        
        Args:
            agentic_tools: AgenticTools instance
            llm_client: LLMClient instance
            max_iterations: Maximum number of reasoning iterations
        """
        self.tools = agentic_tools
        self.llm = llm_client
        self.max_iterations = max_iterations
    
    def run(self, query: str, conversation_history: list = None) -> Dict[str, Any]:
        """
        Main agentic reasoning loop
        
        Args:
            query: User query
            conversation_history: Previous conversation messages
            
        Returns:
            Complete response with reasoning details
        """
        logger.info(f"🤖 [AGENTIC] Starting agentic reasoning for: {query[:100]}")
        
        state = AgentState(query=query, conversation_history=conversation_history or [])
        
        # Iterative reasoning loop
        for iteration in range(self.max_iterations):
            logger.info(f"🔄 [AGENTIC] Iteration {iteration + 1}/{self.max_iterations}")
            
            # Think and decide next action
            action = self._think_and_decide(state, iteration)
            
            if action.get("action") == "DONE":
                logger.info("✅ [AGENTIC] Agent decided it has enough information")
                state.is_complete = True
                break
            
            # Execute action
            tool_name = action.get("tool_name", "")
            parameters = action.get("parameters", {})
            reasoning = action.get("reasoning", "")
            
            logger.info(f"🔧 [AGENTIC] Executing: {tool_name} with {parameters}")
            results = self.tools.execute(tool_name, **parameters)
            
            # Update state
            state.update(tool_name, results, reasoning)
            
            # Reflect: Should we stop?
            if self._should_stop(state, iteration):
                logger.info("✅ [AGENTIC] Reflection indicates we should stop")
                break
        
        # Final synthesis
        logger.info(f"📝 [AGENTIC] Synthesizing final response after {state.iteration_count} iterations")
        response = self._synthesize_final_response(state)
        
        return response
    
    def _think_and_decide(self, state: AgentState, iteration: int) -> Dict[str, Any]:
        """AI decides what to do next"""
        # Build info summary
        info_summary = []
        for tool, results in state.gathered_info.items():
            if isinstance(results, dict):
                count = results.get('count', len(results.get('results', [])))
                info_summary.append(f"{tool}: {count} results")
            else:
                info_summary.append(f"{tool}: executed")
        
        info_text = "\n".join(info_summary) if info_summary else "No information gathered yet"
        
        prompt = DECISION_PROMPT.format(
            query=state.query,
            iteration=iteration + 1,
            max_iterations=self.max_iterations,
            tools_used=", ".join(state.tools_used) if state.tools_used else "None",
            info_summary=info_text
        )
        
        try:
            response = self.llm.generate(prompt, max_tokens=512, temperature=0.3)
            decision = self._parse_decision(response)
            return decision
        except Exception as e:
            logger.error(f"Decision parsing failed: {e}")
            # Fallback: default decision based on state
            if not state.has_info_from("search_knowledge_base"):
                return {"action": "continue", "tool_name": "search_knowledge_base", "parameters": {"query": state.query}, "reasoning": "Fallback: search knowledge base first"}
            elif not state.has_info_from("search_cases"):
                return {"action": "continue", "tool_name": "search_cases", "parameters": {"query": state.query}, "reasoning": "Fallback: search cases"}
            else:
                return {"action": "DONE", "reasoning": "Have enough information"}
    
    def _parse_decision(self, response: str) -> Dict[str, Any]:
        """Parse LLM decision response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                decision = json.loads(json_match.group(0))
                
                # Validate
                if decision.get("action") == "DONE":
                    return {"action": "DONE", "reasoning": decision.get("reasoning", "Sufficient information gathered")}
                
                tool_name = decision.get("tool_name", "")
                if not tool_name:
                    raise ValueError("Missing tool_name")
                
                return {
                    "action": "continue",
                    "tool_name": tool_name,
                    "parameters": decision.get("parameters", {}),
                    "reasoning": decision.get("reasoning", "")
                }
        except Exception as e:
            logger.warning(f"Failed to parse decision: {e}")
        
        # Default fallback
        return {"action": "DONE", "reasoning": "Decision parsing failed, defaulting to done"}
    
    def _should_stop(self, state: AgentState, iteration: int) -> bool:
        """Check if agent should stop iterating"""
        # Skip reflection on first iteration
        if iteration == 0:
            return False
        
        # Skip reflection if we've used many iterations already
        if iteration >= self.max_iterations - 1:
            return True
        
        # Build reflection prompt
        info_summary = []
        for tool, results in state.gathered_info.items():
            if isinstance(results, dict):
                if tool == "search_knowledge_base":
                    kb_results = results.get('results', [])
                    info_summary.append(f"Legal documents: {len(kb_results)} found")
                elif tool == "search_cases":
                    case_results = results.get('results', [])
                    info_summary.append(f"Court cases: {len(case_results)} found")
                elif tool == "generate_strategy_draft":
                    info_summary.append("Strategy draft generated")
        
        info_text = "\n".join(info_summary) if info_summary else "No information gathered"
        
        prompt = REFLECTION_PROMPT.format(
            query=state.query,
            info_summary=info_text,
            tools_used=", ".join(state.tools_used),
            iteration=state.iteration_count
        )
        
        try:
            response = self.llm.generate(prompt, max_tokens=256, temperature=0.3)
            result = self._parse_reflection(response)
            return result.get("complete", False) or not result.get("needs_more", True)
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            # Default: continue if we have some info, stop if we've iterated a lot
            return iteration >= 3
    
    def _parse_reflection(self, response: str) -> Dict[str, Any]:
        """Parse reflection response"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.warning(f"Failed to parse reflection: {e}")
        
        return {"complete": False, "needs_more": True, "gaps": [], "reasoning": "Reflection parsing failed"}
    
    def _synthesize_final_response(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize final response from all gathered information"""
        
        # Build conversation context
        conversation_context = ""
        if state.conversation_history:
            recent = state.conversation_history[-6:] if len(state.conversation_history) > 6 else state.conversation_history
            conversation_context = "=== PREVIOUS CONVERSATION ===\n"
            for msg in recent:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")[:300]
                conversation_context += f"{role}: {content}\n"
            conversation_context += "=== END OF PREVIOUS CONVERSATION ===\n"
        
        # Build comprehensive information summary
        all_info = []
        
        # Knowledge base results
        if state.has_info_from("search_knowledge_base"):
            kb_results = state.gathered_info["search_knowledge_base"].get('results', [])
            all_info.append("=== LEGAL DOCUMENTS ===\n")
            for i, chunk in enumerate(kb_results[:5], 1):
                source = chunk.get('source', 'Unknown')
                page = chunk.get('page', 'N/A')
                text = chunk.get('text', '')[:400]
                all_info.append(f"Document {i}: {source}, Page {page}\n{text}\n")
        
        # Case results
        if state.has_info_from("search_cases"):
            case_results = state.gathered_info["search_cases"].get('results', [])
            all_info.append("\n=== COURT CASES ===\n")
            for i, case in enumerate(case_results[:5], 1):
                case_num = case.get('case_number_english', 'N/A')
                case_type = case.get('case_type_english', 'N/A')
                summary = case.get('summary', '')[:300]
                all_info.append(f"Case {i}: {case_num} ({case_type})\n{summary}\n")
        
        # Strategy if generated
        if state.has_info_from("generate_strategy_draft"):
            strategy = state.gathered_info["generate_strategy_draft"].get('strategy', {})
            all_info.append("\n=== GENERATED STRATEGY ===\n")
            all_info.append(str(strategy.get('strategic_paragraph', ''))[:500])
        
        all_info_text = "\n".join(all_info)
        
        # Build synthesis prompt
        prompt = SYNTHESIS_PROMPT.format(
            query=state.query,
            conversation_context=conversation_context,
            all_information=all_info_text,
            iterations=state.iteration_count,
            tools_used=", ".join(state.tools_used),
            reasoning_history="\n".join(state.reasoning_history[-3:]) if state.reasoning_history else "None"
        )
        
        try:
            response_text = self.llm.generate(prompt, max_tokens=1024, temperature=0.7)
            
            # Extract similar cases for display
            similar_cases = []
            if state.has_info_from("search_cases"):
                case_results = state.gathered_info["search_cases"].get('results', [])
                similar_cases = case_results[:3]
            
            return {
                'response': response_text,
                'query_type': 'AGENTIC',
                'similar_cases': similar_cases,
                'causal_explanation': '',
                'disclaimer': self._get_disclaimer(),
                'agentic_info': {
                    'iterations': state.iteration_count,
                    'tools_used': state.tools_used,
                    'reasoning_steps': state.reasoning_history
                }
            }
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                'response': "I apologize, but I encountered an error synthesizing the response. Please try again.",
                'error': str(e),
                'similar_cases': [],
                'causal_explanation': '',
                'disclaimer': self._get_disclaimer()
            }
    
    def _get_disclaimer(self) -> str:
        """Get legal disclaimer"""
        return (
            "⚠️ **Legal Disclaimer**: This is AI-generated legal information, "
            "not legal advice. Consult a qualified attorney for your specific situation."
        )

