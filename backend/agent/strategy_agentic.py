"""
Agentic Strategy Generator - Specialized agent for case fighting strategies
"""

import logging
from typing import Dict, Any, List, Optional

from .agent_state import AgentState
from .agents.case_analysis_agent import CaseAnalysisAgent
from .agents.precedent_retrieval_agent import PrecedentRetrievalAgent
from .agents.precedent_mining_agent import PrecedentMiningAgent
from .agents.argumentation_agent import ArgumentationAgent
from .agents.verifier_agent import VerifierAgent
from .agents.success_estimator_agent import SuccessEstimatorAgent
from .agents.case_type_analysis_agent import CaseTypeAnalysisAgent
from .agents.law_identification_agent import LawIdentificationAgent
from .agents.document_identification_agent import DocumentIdentificationAgent
from .agentic_agent import AgenticLegalAgent

logger = logging.getLogger(__name__)


STRATEGY_DECISION_PROMPT = """You are a legal strategy agent. Your goal is to generate a comprehensive case fighting strategy.

Case Facts: {case_facts}
Desired Outcome: {desired_outcome}
Case Type: {case_type}

Current State:
- Iteration: {iteration}/{max_iterations}
- Tools used: {tools_used}
- Information gathered: {info_summary}

Available Tools:
1. search_knowledge_base(query, limit) - Find relevant laws and legal provisions
2. search_cases(query, limit) - Find similar court cases and precedents
3. extract_precedents(cases) - Extract winning strategies from cases
4. generate_strategy_draft(case_facts, laws, cases) - Generate the final strategy

Think step-by-step:
1. What laws should I search for? (Constitution, Civil Code, Penal Code, etc.)
2. What similar cases should I find?
3. Should I extract precedents from the cases?
4. When do I have enough information to generate the strategy?

Respond with ONLY a valid JSON object (no prose). Use one of these tool_name values exactly: "search_knowledge_base" | "search_cases" | "extract_precedents" | "generate_strategy_draft".

Examples:
{{"action":"continue","tool_name":"search_knowledge_base","parameters":{{"query":"{case_facts}","limit":5}},"reasoning":"Find applicable provisions"}}
{{"action":"continue","tool_name":"search_cases","parameters":{{"query":"{case_facts}","limit":10}},"reasoning":"Retrieve similar precedents"}}
{{"action":"continue","tool_name":"extract_precedents","parameters":{{"cases":[]}},"reasoning":"Summarize winning arguments"}}
{{"action":"continue","tool_name":"generate_strategy_draft","parameters":{{"case_facts":"{case_facts}","laws":[],"cases":[]}},"reasoning":"Synthesize final plan"}}

Respond with ONLY a valid JSON object:
{{
  "action": "tool_name" or "DONE",
  "tool_name": "search_knowledge_base" | "search_cases" | "extract_precedents" | "generate_strategy_draft",
  "parameters": {{"query": "...", "limit": 5}},
  "reasoning": "Why I'm choosing this action"
}}

If you have laws and cases, set action to "generate_strategy_draft"."""


STRATEGY_REFLECTION_PROMPT = """Evaluate if you have enough information to generate a comprehensive legal strategy.

Case Facts: {case_facts}
Desired Outcome: {desired_outcome}

Information Gathered:
{info_summary}

Tools Used: {tools_used}
Iterations: {iteration}

Questions:
1. Do I have relevant laws/provisions?
2. Do I have similar cases/precedents?
3. Can I generate a complete strategy with arguments, counter-arguments, evidence plan?

Respond with ONLY a valid JSON object:
{{
  "complete": true or false,
  "gaps": ["missing laws", "need more cases"],
  "needs_more": true or false,
  "reasoning": "Explanation"
}}"""


class StrategyAgenticAgent:
    """
    Specialized agentic agent for case fighting strategy generation
    """
    
    def __init__(self, agentic_tools, llm_client, strategy_generator, max_iterations=5):
        """
        Initialize strategy agentic agent
        
        Args:
            agentic_tools: AgenticTools instance
            llm_client: LLMClient instance
            strategy_generator: StrategyGenerator instance
            max_iterations: Maximum iterations
        """
        self.tools = agentic_tools
        self.llm = llm_client
        self.strategy_generator = strategy_generator
        self.max_iterations = max_iterations
        # Specialized agents
        self.case_type_analysis_agent = CaseTypeAnalysisAgent(llm_client)
        self.case_analysis_agent = CaseAnalysisAgent(llm_client)
        self.law_identification_agent = LawIdentificationAgent(llm_client, agentic_tools)
        self.precedent_retrieval_agent = PrecedentRetrievalAgent(agentic_tools)
        self.precedent_mining_agent = PrecedentMiningAgent(strategy_generator)
        self.document_identification_agent = DocumentIdentificationAgent(llm_client)
        self.argumentation_agent = ArgumentationAgent(strategy_generator)
        self.success_estimator_agent = SuccessEstimatorAgent(llm_client)
        self.verifier_agent = VerifierAgent(llm_client)
    
    def generate_strategy(
        self,
        case_facts: str,
        desired_outcome: Optional[str] = None,
        case_type_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate strategy using agentic reasoning
        
        Args:
            case_facts: Case description
            desired_outcome: Desired outcome (optional)
            case_type_hint: Case type hint (optional)
            
        Returns:
            Complete strategy dictionary
        """
        logger.info(f"🤖 [STRATEGY_AGENTIC] Generating strategy for: {case_facts[:100]}")
        
        state = AgentState(
            query=f"Generate strategy for: {case_facts}",
            conversation_history=[]
        )
        
        # Deterministic multi-agent pipeline (ensures core steps are executed)
        logger.info("=" * 80)
        logger.info("🤖 MULTI-AGENT STRATEGY GENERATION PIPELINE")
        logger.info("=" * 80)
        
        # Step 0: Deep case type analysis (NEW)
        logger.info("🎯 [AGENT 0/7] CaseTypeAnalysisAgent: Analyzing case type and legal framework...")
        type_analysis = self.case_type_analysis_agent.run(case_facts)
        case_type = type_analysis.get("case_type") or case_type_hint
        is_prosecution = type_analysis.get("is_prosecution", False)
        is_defense = type_analysis.get("is_defense", False)
        legal_domain = type_analysis.get("legal_domain", "unknown")
        logger.info(f"   ✅ Case type analysis: {case_type} | Domain: {legal_domain} | Prosecution: {is_prosecution} | Defense: {is_defense}")
        state.update("case_type_analysis", type_analysis, "Analyzed case type and legal framework")
        
        # Step 1: Analyze case (enhanced)
        logger.info("📋 [AGENT 1/7] CaseAnalysisAgent: Analyzing case facts and extracting entities...")
        analysis = self.case_analysis_agent.run(case_facts)
        hint = analysis.get("case_type_hint") or case_type_hint or case_type
        logger.info(f"   ✅ Analysis complete: Case type = {hint}, Entities extracted: {len(analysis.get('entities', {}))}")
        state.update("case_analysis", analysis, "Analyzed facts and extracted entities")

        # Step 2: Identify laws (NEW - case-type specific)
        logger.info(f"⚖️  [AGENT 2/7] LawIdentificationAgent: Finding laws for {case_type}...")
        law_results = self.law_identification_agent.run(case_facts, case_type, legal_domain)
        identified_laws = law_results.get("laws", [])
        logger.info(f"   ✅ Law identification: {len(identified_laws)} laws found")
        state.update("law_identification", law_results, "Identified case-type-specific laws")
        
        # Step 3: Retrieve cases (enhanced with case-type context)
        logger.info("🔍 [AGENT 3/7] PrecedentRetrievalAgent: Searching similar cases...")
        retrieval = self.precedent_retrieval_agent.run(case_facts)
        cases = retrieval.get("cases", [])
        retrieval_laws = retrieval.get("laws", [])
        # Merge law results
        all_laws = identified_laws + retrieval_laws
        logger.info(f"   ✅ Retrieval complete: {len(retrieval_laws)} additional laws, {len(cases)} similar cases found")
        state.update("search_knowledge_base", {"results": all_laws, "count": len(all_laws)}, "Retrieved legal provisions")
        state.update("search_cases", {"results": cases, "count": len(cases)}, "Retrieved similar cases")

        # Step 4: Mine precedents
        logger.info(f"⚖️  [AGENT 4/7] PrecedentMiningAgent: Extracting winning strategies from {len(cases)} cases...")
        mined = self.precedent_mining_agent.run(cases)
        precedents_count = len(mined.get("precedents", []))
        logger.info(f"   ✅ Mining complete: {precedents_count} precedents extracted")
        state.update("extract_precedents", mined, "Extracted winning arguments/principles")
        
        # Step 5: Identify documents (NEW - case-type specific)
        logger.info("📄 [AGENT 5/7] DocumentIdentificationAgent: Identifying required documents...")
        doc_results = self.document_identification_agent.run(case_facts, case_type, legal_domain, type_analysis)
        required_documents = doc_results.get("documents", [])
        logger.info(f"   ✅ Document identification: {len(required_documents)} documents identified")
        state.update("document_identification", doc_results, "Identified case-type-specific documents")

        # Step 6: Generate strategy with case-type context
        logger.info("📝 [AGENT 6/7] ArgumentationAgent: Generating comprehensive strategy draft...")
        draft = self.argumentation_agent.run(
            case_facts, 
            all_laws, 
            cases, 
            hint,
            desired_outcome,
            required_documents,
            type_analysis
        )
        strategy_draft = draft.get("strategy", {})
        has_strategy = bool(strategy_draft.get("strategic_paragraph") or strategy_draft.get("raw"))
        logger.info(f"   ✅ Draft complete: Strategy generated ({'with' if has_strategy else 'without'} strategic paragraph)")
        state.update("generate_strategy_draft", {"strategy": strategy_draft, "count": 1}, "Generated strategy draft")

        # Step 7: Estimate success probability using a dedicated agent
        logger.info("📈 [AGENT 7/8] SuccessEstimatorAgent: Estimating success probability...")
        # Provide richer context to the estimator without changing signatures
        try:
            strategy_draft.setdefault("precedents", mined.get("precedents", []))
            strategy_draft.setdefault("documents", required_documents or [])
            if desired_outcome:
                strategy_draft.setdefault("desired_outcome", desired_outcome)
        except Exception:
            pass
        estimation = self.success_estimator_agent.run(
            case_facts,
            all_laws,
            cases,
            strategy_draft,
            case_type
        )
        sp = estimation.get("success_probability")
        if isinstance(sp, dict):
            strategy_draft["success_probability"] = sp
        state.update("success_estimation", {"result": estimation, "count": 1}, "Estimated success probability")

        # Step 8: Verify/refine with case-type awareness
        logger.info("✨ [AGENT 8/8] VerifierAgent: Verifying and refining strategy for accuracy and appeal...")
        final_strategy = self.verifier_agent.run(
            strategy_draft, 
            all_laws, 
            cases, 
            case_facts,
            case_type,
            required_documents,
            type_analysis
        )
        logger.info("   ✅ Verification complete: Strategy polished and ready")
        logger.info("=" * 80)
        logger.info("🎉 ALL 8 AGENTS COMPLETED SUCCESSFULLY")
        logger.info(f"   Agents used: CaseTypeAnalysis → CaseAnalysis → LawIdentification → PrecedentRetrieval → PrecedentMining → DocumentIdentification → Argumentation → SuccessEstimator → Verifier")
        logger.info("=" * 80)
        
        # Use the verified strategy (multi-agent pipeline is complete)
        strategy = final_strategy
        
        # Add agentic reasoning info
        strategy['agentic_reasoning'] = {
            'iterations': state.iteration_count,
            'tools_used': state.tools_used,
            'reasoning_steps': state.reasoning_history
        }
        
        return strategy
    
    def _think_and_decide(self, state: AgentState, case_facts: str, desired_outcome: Optional[str], case_type: Optional[str], iteration: int) -> Dict[str, Any]:
        """Decide next action for strategy generation"""
        info_summary = []
        for tool, results in state.gathered_info.items():
            if isinstance(results, dict):
                count = results.get('count', len(results.get('results', [])))
                info_summary.append(f"{tool}: {count} results")
        
        info_text = "\n".join(info_summary) if info_summary else "No information gathered yet"
        
        prompt = STRATEGY_DECISION_PROMPT.format(
            case_facts=case_facts,
            desired_outcome=desired_outcome or "Not specified",
            case_type=case_type or "Unknown",
            iteration=iteration + 1,
            max_iterations=self.max_iterations,
            tools_used=", ".join(state.tools_used) if state.tools_used else "None",
            info_summary=info_text
        )
        
        # Try up to 2 attempts with stricter then looser decoding
        for attempt in range(2):
            try:
                temperature = 0.1 if attempt == 0 else 0.4
                response = self.llm.generate(prompt, max_tokens=384, temperature=temperature)
                decision = self._parse_decision(response, state, case_facts)
                if decision:
                    return decision
            except Exception:
                pass
        # Fallback logic
        return self._fallback_next_action(state, case_facts)
    
    def _parse_decision(self, response: str, state: AgentState, case_facts: str) -> Dict[str, Any]:
        """Parse decision response"""
        import json
        import re
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                decision = json.loads(json_match.group(0))
                
                if decision.get("action") == "DONE":
                    return {"action": "DONE", "reasoning": decision.get("reasoning", "Sufficient information")}
                
                tool_name = decision.get("tool_name", "")
                if not tool_name:
                    # Fallback to deterministic next step
                    return self._fallback_next_action(state, case_facts)
                
                return {
                    "action": "continue",
                    "tool_name": tool_name,
                    "parameters": decision.get("parameters", {}),
                    "reasoning": decision.get("reasoning", "")
                }
        except Exception as e:
            logger.warning(f"Failed to parse decision: {e}")
        
        return None

    def _fallback_next_action(self, state: AgentState, case_facts: str) -> Dict[str, Any]:
        """Heuristic next action when parsing fails"""
        if not state.has_info_from("search_knowledge_base"):
            return {"action": "continue", "tool_name": "search_knowledge_base", "parameters": {"query": case_facts, "limit": 5}, "reasoning": "Find applicable provisions first"}
        if not state.has_info_from("search_cases"):
            return {"action": "continue", "tool_name": "search_cases", "parameters": {"query": case_facts, "limit": 10}, "reasoning": "Gather similar cases"}
        if not state.has_info_from("extract_precedents") and state.has_info_from("search_cases"):
            # Pass top cases to precedent extractor
            cases = state.gathered_info["search_cases"].get("results", [])
            return {"action": "continue", "tool_name": "extract_precedents", "parameters": {"cases": cases[:10]}, "reasoning": "Summarize winning arguments"}
        # Otherwise, generate draft
        return {"action": "continue", "tool_name": "generate_strategy_draft", "parameters": {"case_facts": case_facts, "laws": state.gathered_info.get("search_knowledge_base", {}).get("results", [])[:5], "cases": state.gathered_info.get("search_cases", {}).get("results", [])[:10]}, "reasoning": "Synthesize final plan"}
    
    def _should_stop(self, state: AgentState, case_facts: str, desired_outcome: Optional[str], iteration: int) -> bool:
        """Check if ready to generate strategy"""
        if iteration == 0 or iteration >= self.max_iterations - 1:
            return iteration >= self.max_iterations - 1
        
        info_summary = []
        has_laws = False
        has_cases = False
        
        for tool, results in state.gathered_info.items():
            if tool == "search_knowledge_base":
                kb_results = results.get('results', [])
                info_summary.append(f"Laws: {len(kb_results)} found")
                has_laws = len(kb_results) > 0
            elif tool == "search_cases":
                case_results = results.get('results', [])
                info_summary.append(f"Cases: {len(case_results)} found")
                has_cases = len(case_results) > 0
        
        info_text = "\n".join(info_summary) if info_summary else "No information"
        
        prompt = STRATEGY_REFLECTION_PROMPT.format(
            case_facts=case_facts,
            desired_outcome=desired_outcome or "Not specified",
            info_summary=info_text,
            tools_used=", ".join(state.tools_used),
            iteration=state.iteration_count
        )
        
        try:
            response = self.llm.generate(prompt, max_tokens=256, temperature=0.3)
            result = self._parse_reflection(response)
            return result.get("complete", False) or (has_laws and has_cases)
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            return iteration >= 3 or (has_laws and has_cases)
    
    def _parse_reflection(self, response: str) -> Dict[str, Any]:
        """Parse reflection response"""
        import json
        import re
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.warning(f"Failed to parse reflection: {e}")
        
        return {"complete": False, "needs_more": True}
    
    def _generate_final_strategy(
        self,
        state: AgentState,
        case_facts: str,
        desired_outcome: Optional[str],
        case_type_hint: Optional[str]
    ) -> Dict[str, Any]:
        """Generate final strategy from gathered information"""
        
        # Collect laws
        laws = []
        if state.has_info_from("search_knowledge_base"):
            kb_results = state.gathered_info["search_knowledge_base"].get('results', [])
            laws = kb_results[:5]
        
        # Collect cases
        cases = []
        if state.has_info_from("search_cases"):
            case_results = state.gathered_info["search_cases"].get('results', [])
            cases = case_results[:10]
        
        # Build retrieval context
        retrieval_lines = []
        if laws:
            retrieval_lines.append("=== LEGAL DOCUMENTS ===")
            for law in laws[:5]:
                source = law.get('source', 'Unknown')
                page = law.get('page', 'N/A')
                text = law.get('text', '')[:200]
                retrieval_lines.append(f"{source} (Page {page}): {text}")
        
        if cases:
            retrieval_lines.append("\n=== COURT CASES ===")
            for case in cases[:10]:
                num = case.get('case_number_english', 'N/A')
                ctype = case.get('case_type_english', 'N/A')
                summary = case.get('summary', '')[:200]
                retrieval_lines.append(f"Case {num} ({ctype}): {summary}")
        
        retrieval_context = "\n".join(retrieval_lines) if retrieval_lines else None
        
        # Generate strategy using strategy generator
        if self.strategy_generator:
            strategy = self.strategy_generator.generate(
                case_facts=case_facts,
                desired_outcome=desired_outcome,
                case_type_hint=case_type_hint,
                retrieval_context=retrieval_context
            )
            return strategy
        else:
            # Fallback basic strategy
            return {
                'case_type': case_type_hint or 'Unknown',
                'desired_outcome': desired_outcome or '',
                'strategic_paragraph': 'Strategy generation unavailable',
                'error': 'Strategy generator not available'
            }

    # verifier moved to VerifierAgent

