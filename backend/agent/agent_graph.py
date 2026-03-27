"""
Intelligent multi-tool agent for legal query processing
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class LegalAgentGraph:
    """
    Intelligent agent that routes queries to appropriate tools and synthesizes responses
    """
    
    def __init__(self, rag_pipeline, intelligent_tools, router, synthesizer):
        """
        Initialize intelligent agent
        
        Args:
            rag_pipeline: RAGPipeline instance (for backward compatibility)
            intelligent_tools: IntelligentTools instance
            router: IntelligentRouter instance
            synthesizer: ResponseSynthesizer instance
        """
        self.rag = rag_pipeline
        self.tools = intelligent_tools
        self.router = router
        self.synthesizer = synthesizer
    
    def run(self, query: str, conversation_history: list = None) -> Dict[str, Any]:
        """
        Run the intelligent agent workflow
        
        Args:
            query: User query
            conversation_history: Previous conversation messages for context
            
        Returns:
            Response dictionary
        """
        logger.info(f"🤖 [AGENT] Processing query: {query[:100]}...")
        
        if conversation_history:
            logger.info(f"💬 [AGENT] Using conversation history: {len(conversation_history)} messages")
        
        try:
            # Step 1: Intelligent routing - decide which tools to use
            logger.info("📍 [AGENT] Step 1: Routing query to appropriate tools")
            routing = self.router.route_query(query, conversation_history)
            
            # Step 2: Execute selected tools
            logger.info(f"🔧 [AGENT] Step 2: Executing {routing['primary_tool']}")
            
            if routing['primary_tool'] == 'hybrid':
                # Use multiple tools
                tool_results = self.tools.hybrid_search(
                    query=query,
                    use_knowledge_base=routing.get('use_knowledge_base', True),
                    use_cases=routing.get('use_cases', True),
                    use_web=routing.get('use_web', False)
                )
            elif routing['primary_tool'] == 'strategy':
                # Generate strategy plan
                tool_results = self.tools.execute_tool(
                    tool_name='strategy',
                    query=query
                )
            else:
                # Use single tool
                tool_results = self.tools.execute_tool(
                    tool_name=routing['primary_tool'],
                    query=query
                )
            
            # Step 3: Synthesize response from all sources
            logger.info("🔨 [AGENT] Step 3: Synthesizing response")
            if tool_results.get('tool') == 'strategy':
                # Format strategy into a readable response
                response_text = self._format_strategy(tool_results.get('strategy', {}))
            else:
                response_text = self.synthesizer.synthesize(
                    query=query,
                    tool_results=tool_results,
                    routing_info=routing,
                    conversation_history=conversation_history
                )
            
            # Step 4: Add disclaimer
            disclaimer = self.rag._get_disclaimer()
            
            # Extract similar cases if available
            similar_cases = []
            if tool_results.get('tool') == 'hybrid':
                case_data = tool_results.get('sources', {}).get('cases', {})
                similar_cases = case_data.get('results', [])[:3]
            elif tool_results.get('tool') == 'case_search':
                similar_cases = tool_results.get('results', [])[:3]
            elif tool_results.get('tool') == 'strategy':
                # Try to surface precedents if present
                strategy = tool_results.get('strategy', {})
                precedents = strategy.get('precedents') or []
                similar_cases = [{'case_number_english': p.get('case_id', ''), 'summary': p.get('holding', '')} for p in precedents[:3]]
            
            logger.info(f"✅ [AGENT] Query processed successfully")
            
            return {
                'response': response_text,
                'query_type': routing['primary_tool'],
                'similar_cases': similar_cases,
                'causal_explanation': '',
                'disclaimer': disclaimer,
                'routing_info': {
                    'primary_tool': routing['primary_tool'],
                    'reasoning': routing.get('reasoning', ''),
                    'confidence': routing.get('confidence', 'unknown')
                },
                'error': ''
            }
            
        except Exception as e:
            logger.error(f"❌ [AGENT] Workflow failed: {e}", exc_info=True)
            return {
                'response': "I apologize, but I encountered an error processing your query. Please try again.",
                'error': str(e),
                'similar_cases': [],
                'causal_explanation': '',
                'disclaimer': self.rag._get_disclaimer()
            }

    def _format_strategy(self, strategy: Dict[str, Any]) -> str:
        """Create a concise, readable summary from strategy JSON."""
        if not isinstance(strategy, dict) or not strategy:
            return "Strategy generation failed."
        ct = strategy.get('case_type', 'Unknown')
        goal = strategy.get('desired_outcome', '')
        para = strategy.get('strategic_paragraph') or ''
        # Fallback: try to extract paragraph from raw if present
        if not para and isinstance(strategy.get('raw'), str):
            import re
            m = re.search(r'"strategic_paragraph"\s*:\s*"([^"]+)"', strategy['raw'])
            if m:
                para = m.group(1)
        strengths = strategy.get('strengths') or []
        weaknesses = strategy.get('weaknesses') or []
        prob = strategy.get('success_probability') or {}
        sp = prob.get('point')
        ci = prob.get('ci') or []
        args = strategy.get('arguments') or []
        ctr = strategy.get('counter_arguments') or []
        laws = strategy.get('applicable_laws') or []

        lines = []
        lines.append(f"Strategy ({ct}){f' – Goal: {goal}' if goal else ''}")
        if para:
            lines.append(para)
        if sp is not None:
            if isinstance(ci, list) and len(ci) == 2:
                lines.append(f"Estimated success probability: {sp:.2f} (CI: {ci[0]:.2f}-{ci[1]:.2f})")
            else:
                lines.append(f"Estimated success probability: {sp:.2f}")
        if laws:
            lines.append("Applicable laws:")
            for l in laws[:5]:
                sec = l.get('section', '')
                why = l.get('why', '')
                lines.append(f"- {sec}: {why}")
        docs = strategy.get('documents_checklist') or []
        if docs:
            lines.append("Documents to prepare:")
            for d in docs[:6]:
                doc = d.get('document','')
                purp = d.get('purpose','')
                src = d.get('required_from','')
                pr = d.get('priority','')
                suffix = f" (from: {src})" if src else ''
                prtxt = f" [{pr}]" if pr else ''
                lines.append(f"- {doc}: {purp}{suffix}{prtxt}")
        if args:
            lines.append("Key arguments:")
            for a in args[:5]:
                claim = a.get('claim', '')
                lines.append(f"- {claim}")
        if ctr:
            lines.append("Expected counter-arguments & responses:")
            for c in ctr[:3]:
                lines.append(f"- {c.get('claim','')}: {c.get('response','')}")
        wps = strategy.get('winning_points') or []
        if wps:
            lines.append("Points to emphasize:")
            for wp in wps[:6]:
                lines.append(f"- {wp}")
        if strengths:
            lines.append("Strengths: " + "; ".join(strengths[:5]))
        if weaknesses:
            lines.append("Weaknesses: " + "; ".join(weaknesses[:5]))
        return "\n".join(lines)

