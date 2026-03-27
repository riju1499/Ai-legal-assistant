"""
Response synthesizer that combines information from multiple sources
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


SYNTHESIS_PROMPT = """You are Wakalat Sewa, an intelligent legal AI assistant for Nepali law.

{conversation_context}

Current User Query: {query}

You have gathered information from multiple sources. Synthesize a comprehensive, accurate response.

{sources_context}

INSTRUCTIONS:
1. **Synthesize information** from all sources into a coherent answer
2. **Prioritize authoritative sources** (Constitution, laws) over cases over web
3. **Cite all sources** clearly:
   - Legal documents: (Source: [PDF name], Page [number])
   - Court cases: (Case: [case number], [court])
   - Web sources: (Source: [URL/title])
4. **Structure your response**:
   - Start with direct answer to the query
   - Provide legal basis from documents
   - Support with relevant case precedents if available
   - Add practical context from web if available
5. **Be specific and actionable** - if user asks "what should I do?", provide clear guidance
6. **Acknowledge limitations** - if sources don't fully answer, say so
7. **Use markdown** for clarity

Provide your comprehensive response:"""


class ResponseSynthesizer:
    """
    Synthesizes responses from multiple tool results
    """
    
    def __init__(self, llm_client):
        """
        Initialize synthesizer
        
        Args:
            llm_client: LLMClient instance
        """
        self.llm = llm_client
    
    def synthesize(
        self, 
        query: str, 
        tool_results: Dict[str, Any],
        routing_info: Dict[str, Any],
        conversation_history: list = None
    ) -> str:
        """
        Synthesize response from tool results
        
        Args:
            query: User query
            tool_results: Results from tool execution
            routing_info: Routing decision information
            conversation_history: Previous conversation for context
            
        Returns:
            Synthesized response text
        """
        logger.info(f"🔨 [SYNTHESIZER] Combining sources for: {query[:100]}")
        
        try:
            # Build conversation context
            conversation_context = self._build_conversation_context(conversation_history)
            
            # Build context from all sources
            sources_context = self._build_sources_context(tool_results)
            
            # Check if we have any results
            if not sources_context or "No information found" in sources_context:
                logger.warning("⚠️  [SYNTHESIZER] No information found in any source")
                return self._no_results_response(query)
            
            # Generate synthesized response
            prompt = SYNTHESIS_PROMPT.format(
                query=query,
                conversation_context=conversation_context,
                sources_context=sources_context
            )
            
            response = self.llm.generate(prompt, max_tokens=1024, temperature=0.7)
            
            # Add source summary at the end
            source_summary = self._create_source_summary(tool_results)
            if source_summary:
                response += f"\n\n---\n**Sources consulted:** {source_summary}"
            
            logger.info(f"✅ [SYNTHESIZER] Response generated ({len(response)} chars)")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ [SYNTHESIZER] Error: {e}", exc_info=True)
            return "I apologize, but I encountered an error synthesizing the response. Please try again."
    
    def _build_conversation_context(self, conversation_history: list) -> str:
        """Build conversation context from history"""
        if not conversation_history or len(conversation_history) == 0:
            return ""
        
        context = ["=== PREVIOUS CONVERSATION ==="]
        # Take last 3 exchanges (6 messages)
        recent = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:300]  # Truncate long responses
            context.append(f"{role}: {content}")
        
        context.append("=== END OF PREVIOUS CONVERSATION ===\n")
        return "\n".join(context)
    
    def _build_sources_context(self, tool_results: Dict[str, Any]) -> str:
        """Build formatted context from all tool results"""
        
        # Handle hybrid results
        if tool_results.get('tool') == 'hybrid':
            sources = tool_results.get('sources', {})
            context_parts = []
            
            # Knowledge base results
            if 'knowledge_base' in sources:
                kb_context = self._format_knowledge_base(sources['knowledge_base'])
                if kb_context:
                    context_parts.append("=== LEGAL DOCUMENTS ===\n" + kb_context)
            
            # Case results
            if 'cases' in sources:
                case_context = self._format_cases(sources['cases'])
                if case_context:
                    context_parts.append("=== COURT CASES ===\n" + case_context)
            
            # Web results
            if 'web' in sources:
                web_context = self._format_web(sources['web'])
                if web_context:
                    context_parts.append("=== WEB SOURCES ===\n" + web_context)
            
            return "\n\n".join(context_parts) if context_parts else "No information found from any source."
        
        # Handle single tool results
        tool_type = tool_results.get('tool')
        
        if tool_type == 'knowledge_base':
            context = self._format_knowledge_base(tool_results)
            return "=== LEGAL DOCUMENTS ===\n" + context if context else "No legal documents found."
        
        elif tool_type == 'case_search':
            context = self._format_cases(tool_results)
            return "=== COURT CASES ===\n" + context if context else "No court cases found."
        
        elif tool_type == 'web_search':
            context = self._format_web(tool_results)
            return "=== WEB SOURCES ===\n" + context if context else "No web results found."
        
        return "No information available."
    
    def _format_knowledge_base(self, kb_results: Dict[str, Any]) -> str:
        """Format knowledge base results"""
        results = kb_results.get('results', [])
        
        if not results:
            return ""
        
        formatted = []
        for i, chunk in enumerate(results[:5], 1):
            source = chunk.get('source', 'Unknown')
            page = chunk.get('page', 'N/A')
            text = chunk.get('text', '')
            score = chunk.get('score', 0)
            
            # Truncate if too long
            if len(text) > 800:
                text = text[:800] + "..."
            
            formatted.append(
                f"\nDocument {i}: {source}, Page {page} (Relevance: {score:.2f})\n"
                f"{text}\n"
                f"{'-' * 80}"
            )
        
        return '\n'.join(formatted)
    
    def _format_cases(self, case_results: Dict[str, Any]) -> str:
        """Format case search results"""
        results = case_results.get('results', [])
        
        if not results:
            return ""
        
        formatted = []
        for i, case in enumerate(results[:5], 1):
            case_num = case.get('case_number_english', 'N/A')
            case_type = case.get('case_type_english', 'N/A')
            court = case.get('court_english', 'N/A')
            verdict = case.get('verdict_english', 'N/A')
            summary = case.get('summary', 'No summary available')
            
            # Truncate summary
            if len(summary) > 400:
                summary = summary[:400] + "..."
            
            formatted.append(
                f"\nCase {i}: {case_num}\n"
                f"Type: {case_type}\n"
                f"Court: {court}\n"
                f"Verdict: {verdict}\n"
                f"Summary: {summary}\n"
                f"{'-' * 80}"
            )
        
        return '\n'.join(formatted)
    
    def _format_web(self, web_results: Dict[str, Any]) -> str:
        """Format web search results"""
        results = web_results.get('results', [])
        
        if not results:
            return ""
        
        # Placeholder for now
        return "Web search results (feature coming soon)"
    
    def _create_source_summary(self, tool_results: Dict[str, Any]) -> str:
        """Create a summary of sources used"""
        summaries = []
        
        if tool_results.get('tool') == 'hybrid':
            sources = tool_results.get('sources', {})
            
            for source_name, source_data in sources.items():
                count = source_data.get('count', 0)
                if count > 0:
                    if source_name == 'knowledge_base':
                        summaries.append(f"{count} legal documents")
                    elif source_name == 'cases':
                        summaries.append(f"{count} court cases")
                    elif source_name == 'web':
                        summaries.append(f"{count} web sources")
        else:
            count = tool_results.get('count', 0)
            tool_type = tool_results.get('tool')
            
            if count > 0:
                if tool_type == 'knowledge_base':
                    summaries.append(f"{count} legal documents")
                elif tool_type == 'case_search':
                    summaries.append(f"{count} court cases")
                elif tool_type == 'web_search':
                    summaries.append(f"{count} web sources")
        
        return ", ".join(summaries) if summaries else "No sources"
    
    def _no_results_response(self, query: str) -> str:
        """Generate response when no results found"""
        return f"""I apologize, but I couldn't find relevant information in the available sources to answer your query: "{query}"

This could be because:
- The specific information is not in the legal documents database
- The court cases database doesn't contain relevant precedents
- The query might need to be rephrased for better results

**Suggestions:**
1. Try rephrasing your query with different keywords
2. Break down complex questions into simpler parts
3. Consult with a qualified legal professional for specific advice

**Note:** I have access to:
- Constitutional provisions and legal acts
- 10,456 processed court cases
- Legal glossaries and terms

Please feel free to ask another question or rephrase your query."""

