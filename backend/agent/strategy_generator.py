"""
Strategy generation module: builds AI-driven, precedent-backed case strategies for any case type.
Precedents are extracted dynamically from retrieved cases using LLM analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class StrategyGenerator:
    """
    Generates a structured strategy plan using:
    - User-provided case facts and desired outcome
    - Precedents extracted dynamically from retrieved cases using AI
    - Optional retrieval hits (laws/cases) provided by caller
    """

    def __init__(self, llm_client) -> None:
        self.llm = llm_client
    
    def _load_json(self, file_path: Path) -> Optional[Any]:
        """Helper method for loading JSON (kept for potential future use)"""
        try:
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return None
        return None

    def _select_case_type(self, case_type_hint: Optional[str], fallback_from_facts: str) -> Optional[str]:
        # Prefer explicit hint; else use LLM to identify case type
        if case_type_hint and case_type_hint.strip():
            return case_type_hint.strip()
        
        # Use LLM to identify case type from facts
        prompt = f"""Analyze this legal case description and identify the case type.

Case Facts: {fallback_from_facts}

Identify the most appropriate case type. Common types include:
- Property Dispute
- Inheritance
- Family/Property
- Constitutional
- Contract
- Criminal
- Administrative

Respond with ONLY the case type name, nothing else."""
        
        try:
            response = self.llm.generate(prompt, max_tokens=50, temperature=0.3)
            case_type = response.strip()
            return case_type if case_type else None
        except Exception:
            # Fallback to simple keyword matching
            facts_lower = fallback_from_facts.lower()
            keyword_map = {
                "property": "Property Dispute",
                "land": "Property Dispute",
                "inherit": "Inheritance",
                "partition": "Inheritance",
                "share": "Inheritance",
                "will": "Inheritance",
                "family": "Family/Property",
            }
            for kw, mapped in keyword_map.items():
                if kw in facts_lower:
                    return mapped
            return None

    def _extract_precedents_with_ai(self, retrieval_context: Optional[str], case_type: Optional[str] = None, case_facts: str = "") -> List[Dict[str, Any]]:
        """Extract precedents from retrieved cases using AI"""
        if not retrieval_context:
            return []
        
        # Add context about what type of precedents to look for
        focus_hint = ""
        if case_type:
            if "criminal" in case_type.lower():
                if "alibi" in case_facts.lower() or "was in" in case_facts.lower() or "was at" in case_facts.lower():
                    focus_hint = "Focus on cases where ALIBI DEFENSE was successful - cases where accused proved they were elsewhere.\n"
                elif "convict" in case_facts.lower() or "want to convict" in case_facts.lower() or "prosecution" in case_facts.lower() or "state" in case_facts.lower():
                    focus_hint = "Focus on CRIMINAL PROSECUTION cases - cases where the State successfully prosecuted murder/serious crimes.\n"
                elif "accused" in case_facts.lower() or "defending" in case_facts.lower():
                    focus_hint = "Focus on criminal DEFENSE cases with similar charges.\n"
            elif "property" in case_type.lower() or "inherit" in case_type.lower():
                focus_hint = "Focus on property/inheritance cases with similar legal issues.\n"
        
        prompt = f"""Analyze these court cases and extract key precedents (winning strategies, legal principles, patterns).

{focus_hint}For each relevant case, extract:
- Case number
- Winning argument/strategy (especially relevant to the case type)
- Legal principle applied
- Cause-effect relationship if applicable
- Why this precedent is relevant to the current case

Cases:
{retrieval_context[:2000]}

Return as JSON array with fields: case_id, holding, winning_argument, principle, why_relevant.
If no relevant precedents, return empty array."""
        
        try:
            response = self.llm.generate(prompt, max_tokens=800, temperature=0.2)
            # Try to parse JSON from response
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                precedents = json.loads(json_match.group(0))
                return precedents[:12]  # Limit to 12 precedents
        except Exception:
            pass
        
        return []

    def _prompt_strategy(self,
                          case_facts: str,
                          desired_outcome: Optional[str],
                          case_type: Optional[str],
                          precedents: List[Dict[str, Any]],
                          retrieval_context: Optional[str] = None,
                          case_type_info: Optional[Dict[str, Any]] = None) -> str:
        # Format AI-extracted precedents
        if precedents:
            precedents_text = "\n".join(
                [f"- Case {p.get('case_id', 'N/A')}: {p.get('holding', 'N/A')} | Strategy: {p.get('winning_argument', 'N/A')} | Principle: {p.get('principle', 'N/A')} | Why: {p.get('why_relevant', 'N/A')}"
                 for p in precedents[:12]]
            )
        else:
            precedents_text = "- (no precedents extracted yet)"

        context_block = f"\n\nRetrieved Context:\n{retrieval_context}" if retrieval_context else ""
        goal_line = f"Desired Outcome: {desired_outcome}" if desired_outcome else ""
        
        # Extract case type info
        is_prosecution = case_type_info.get("is_prosecution", False) if case_type_info else False
        is_defense = case_type_info.get("is_defense", False) if case_type_info else False
        legal_domain = case_type_info.get("legal_domain", "") if case_type_info else ""
        case_type_lower = (case_type or "").lower()
        
        # Build adaptive guidance based on case type
        if "criminal" in case_type_lower:
            if is_prosecution:
                documents_guidance = """6. Include AT LEAST 8 documents - CRITICAL: For CRIMINAL PROSECUTION cases:
   - CCTV footage/video evidence (prove accused's presence at crime scene)
   - Forensic reports (DNA, fingerprints, blood analysis) - prove connection to crime
   - Murder weapon/physical evidence (prove means of crime)
   - Police investigation reports (Thana Patra) - official investigation record
   - Medical examiner's report/Autopsy report (prove cause of death/injury)
   - Witness statements (Saksi Praman) - eyewitness accounts
   - Evidence collection reports (chain of custody)
   - Medical records (if injuries involved)
   ⚠️ DO NOT include property documents like Lalpurja or Jaminko Nam for criminal cases"""
                timeline_guidance = """8. Procedural_timeline - For CRIMINAL PROSECUTION cases:
   - Arrest and booking: Within 24 hours
   - Bail hearing: Within 48 hours
   - Investigation completion: Within 90 days
   - Charge sheet filing: Within deadline
   - Trial dates: Court scheduling
   ⚠️ DO NOT include civil steps like 'File partition petition'"""
                laws_guidance = """3. Include AT LEAST 5 applicable_laws - For CRIMINAL cases focus on:
   - National Penal Code Act 2017 (specific sections for the charge)
   - National Criminal Procedure Act 2017 (procedural sections)
   - Nepal Constitution (criminal rights provisions)
   Use SPECIFIC section numbers, not ranges"""
            else:  # defense
                documents_guidance = """6. Include AT LEAST 8 documents - CRITICAL: For CRIMINAL DEFENSE cases:
   - Alibi evidence (passport stamps, flight tickets, hotel receipts, credit card statements) - prove location elsewhere
   - CCTV footage proving location (visual proof of alibi)
   - Witness testimonies from alibi location (people who saw accused elsewhere)
   - Character certificates (establish good character)
   - Phone records/call logs (location data)
   - Immigration records (official entry/exit records)
   ⚠️ DO NOT include property documents"""
                timeline_guidance = """8. Procedural_timeline - For CRIMINAL DEFENSE cases:
   - File bail application: Immediately
   - Challenge evidence: Within deadlines
   - Present defense evidence: As scheduled
   - Cross-examination: During trial
   ⚠️ DO NOT include civil steps"""
                laws_guidance = """3. Include AT LEAST 5 applicable_laws - For CRIMINAL DEFENSE cases focus on:
   - National Criminal Procedure Act 2017 (defense rights, evidence rules)
   - National Penal Code Act 2017 (relevant charge sections)
   - Nepal Constitution (right to defense, presumption of innocence)
   Use SPECIFIC section numbers"""
        elif "property" in case_type_lower or "inherit" in case_type_lower:
            documents_guidance = """6. Include AT LEAST 8 documents - For PROPERTY cases:
   - Lalpurja (land ownership certificate) - prove ownership
   - Jaminko Nam (cadastral survey map) - prove boundaries
   - Nagarikta (citizenship certificate) - identity verification
   - Nata Pramanit (relationship certificate) - prove family relationships
   - Mritak Suchi/Mrityu Suchi (death certificate) - prove death/inheritance timeline
   - Rajinama/Parsar (affidavit) - sworn statements
   - Property tax receipts - ownership history
   - Witness testimonies (Saksi Praman) - support claims"""
            timeline_guidance = """8. Procedural_timeline - For PROPERTY cases:
   - File partition petition: Within 15 days from case initiation
   - Serve notice/summons: Within 7 days after filing
   - Submit supporting documents: Within 21 days from filing
   - First hearing: Within 30-45 days from filing
   - Evidence presentation: As scheduled (typically 60-90 days)
   - Final arguments and judgment: Within 120-180 days from filing"""
            laws_guidance = """3. Include AT LEAST 5 applicable_laws - For PROPERTY cases focus on:
   - Muluki Civil Code 2017 Sections 145-280 (joint family property)
   - Section 259 (majority decision for joint property management)
   - Section 124 (registration requirements)
   - Constitution Article 15 (property rights)
   Use SPECIFIC section numbers (Section 259, Section 145, etc.) not ranges"""
        else:
            documents_guidance = """6. Include AT LEAST 8 documents - Analyze case facts and identify relevant documents:
   - Common documents: Nagarikta (citizenship), Rajinama/Parsar (affidavit)
   - Case-specific documents based on case type
   - Each with SPECIFIC purpose for THIS case"""
            timeline_guidance = """8. Procedural_timeline - Provide CONCRETE steps with deadlines SPECIFIC to THIS case type:
   - Use realistic deadlines (e.g., 'File petition: Within 15 days', 'First hearing: Within 30 days')
   - Match timeline to actual case type procedures"""
            laws_guidance = """3. Include AT LEAST 5 applicable_laws with SPECIFIC section numbers:
   - Match laws to the actual case type
   - Include relevant acts: National Penal Code Act 2017, Muluki Civil Code 2017, National Criminal Procedure Act 2017, etc.
   - For each law, explain WHY it applies to THIS specific case"""

        prompt = (
            "You are an expert Nepali legal strategy assistant. Generate a COMPREHENSIVE, DETAILED, and CASE-SPECIFIC strategy.\n"
            "CRITICAL: Make all content SPECIFIC to the provided case facts. Do NOT use generic placeholders or repetitive statements.\n\n"
            "⚠️ CRITICAL REQUIREMENT: READ THE CASE FACTS CAREFULLY AND MATCH YOUR STRATEGY TO THE ACTUAL CASE TYPE!\n"
            "- If the case involves MURDER, KILLING, STABBING, CRIMINAL CHARGES → This is a CRIMINAL case\n"
            "- If the case involves LAND, PROPERTY, INHERITANCE → This is a PROPERTY/CIVIL case\n"
            "- DO NOT include property documents (Lalpurja, Jaminko Nam) for CRIMINAL cases\n"
            "- DO NOT include criminal evidence (CCTV, murder weapon) for PROPERTY cases\n"
            "- DO NOT mix case types - each case type has its own documents, laws, and procedures\n\n"
            
            "REQUIREMENTS:\n"
            "1. Understand the case structure:\n"
            "   - Identify WHO are the parties (claimant/plaintiff vs defendant/respondent)\n"
            "   - Identify relationships between parties\n"
            "   - Identify WHAT is being claimed or sought\n"
            "2. Analyze the legal framework:\n"
            "   - What legal domain does this fall under?\n"
            "   - Is this prosecution or defense? (if criminal)\n"
            "   - What type of civil claim? (if civil)\n"
            f"{laws_guidance}\n"
            "4. Include AT LEAST 5 precedents - use REAL case numbers from provided cases, include case_number_english if available\n"
            "   - Ensure precedents are actually relevant to THIS case type and legal issue\n"
            "   - Explain WHY each precedent is relevant\n"
            "5. Include AT LEAST 5 UNIQUE arguments - each should be different, not variations of the same point:\n"
            "   - For defense cases: focus on defense arguments (alibi, lack of evidence, procedural issues)\n"
            "   - For prosecution cases: focus on prosecution arguments (motive, means, opportunity, evidence)\n"
            "   - For civil cases: focus on legal rights, claims, evidence, defenses\n"
            "   - Include causal_path showing the logical chain: [premise1, premise2, conclusion]\n"
            f"{documents_guidance}\n"
            "   - For each document, specify: document name (in Nepali/English), purpose (why needed for THIS case), required_from (who provides it), priority (high/medium/low)\n"
            "7. Include AT LEAST 3 realistic counter-arguments opponents might raise, with specific legal responses\n"
            f"{timeline_guidance}\n"
            "9. Strategic_paragraph must be 4-5 sentences explaining the SPECIFIC legal approach for THIS case\n"
            "10. Success probability: 0.55-0.85 with confidence interval\n\n"
            f"Case Type: {case_type or 'Unknown'}\n"
            f"Legal Domain: {legal_domain}\n"
            f"Is Prosecution: {is_prosecution}\n"
            f"Is Defense: {is_defense}\n"
            f"{goal_line}\n\n"
            f"Case Facts:\n{case_facts.strip()}\n\n"
            f"Relevant Precedents:\n{precedents_text}{context_block}\n\n"
            "Generate a COMPLETE JSON object with ALL these fields populated with DETAILED, CASE-SPECIFIC information:\n"
            "{\n"
            "  \"case_type\": str, \n"
            "  \"desired_outcome\": str, \n"
            "  \"strategic_paragraph\": str,  \n"
            "  \"key_factors\": [{\"node\": str, \"value\": str, \"confidence\": float}],\n"
            "  \"applicable_laws\": [{\"section\": str, \"why\": str}],\n"
            "  \"precedents\": [{\"case_id\": str, \"holding\": str, \"why_relevant\": str}],\n"
            "  \"arguments\": [{\"claim\": str, \"support\": [str], \"causal_path\": [str]}],\n"
            "  \"counter_arguments\": [{\"opponent\": str, \"claim\": str, \"response\": str}],\n"
            "  \"evidence_plan\": [{\"item\": str, \"purpose\": str, \"node\": str, \"priority\": str}],\n"
            "  \"documents_checklist\": [{\"document\": str, \"purpose\": str, \"required_from\": str, \"priority\": \"high|medium|low\"}],\n"
            "  \"witness_plan\": [{\"type\": str, \"goal\": str}],\n"
            "  \"procedural_timeline\": [{\"step\": str, \"deadline\": str}],\n"
            "  \"what_if\": [{\"change\": str, \"impact\": {\"success_prob_delta\": float, \"explanation\": str}}],\n"
            "  \"necessary_vs_sufficient\": [{\"node\": str, \"role\": str, \"evidence_needed\": str}],\n"
            "  \"winning_points\": [str],\n"
            "  \"strengths\": [str], \n"
            "  \"weaknesses\": [str], \n"
            "  \"success_probability\": {\"point\": float, \"ci\": [float, float]}, \n"
            "  \"recommendation\": str,\n"
            "  \"citations\": [{\"type\": \"law\"|\"case\", \"source\": str, \"page\": int|null}]\n"
            "}\n"
        )
        return prompt

    def generate(self,
                 case_facts: str,
                 desired_outcome: Optional[str] = None,
                 case_type_hint: Optional[str] = None,
                 retrieval_context: Optional[str] = None,
                 case_type_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Choose case type using AI
        case_type = self._select_case_type(case_type_hint, case_facts)
        
        # Extract precedents dynamically from retrieval context using AI
        precedents = self._extract_precedents_with_ai(retrieval_context, case_type, case_facts)

        # Build prompt and query LLM
        prompt = self._prompt_strategy(
            case_facts=case_facts,
            desired_outcome=desired_outcome,
            case_type=case_type,
            precedents=precedents,
            retrieval_context=retrieval_context,
            case_type_info=case_type_info,
        )

        # response_text = self.llm.generate(prompt, max_tokens=2500, temperature=0.3)
        response_text = self.llm.generate(prompt, max_tokens=800, temperature=0.3)

        # Try to parse JSON response; if fail, wrap as text
        strategy: Dict[str, Any]
        try:
            # Extract JSON block if wrapped in markdown
            import re
            # Prefer fenced code block content
            fenced = re.search(r"```(?:json)?\n([\s\S]*?)\n```", response_text)
            if fenced:
                candidate = fenced.group(1)
            else:
                # Strip leading verbal wrappers and code fences
                cleaned = response_text.replace('```json', '```').strip()
                cleaned = re.sub(r"^Here is[\s\S]*?\n", "", cleaned)
                candidate = cleaned
            # Extract first JSON object from candidate
            match = re.search(r"\{[\s\S]*\}", candidate)
            if match:
                json_block = match.group(0)
            else:
                json_block = candidate
            # Heuristic fix: quote bare string values like: "key": Article 59
            def _quote_bare_values(txt: str) -> str:
                # Replace: : <letters/digits/spaces/dots> (until , or }) with quoted value if not already quoted/number/boolean/null
                def repl(m):
                    val = m.group(1).strip()
                    if val.startswith('"') or val.startswith('[') or val.startswith('{'):
                        return ": " + val
                    if re.fullmatch(r"-?\d+(?:\.\d+)?", val) or val in ["true","false","null"]:
                        return ": " + val
                    return ": \"" + val.replace('"','\\"') + "\""
                return re.sub(r":\s*([^,}\n]+)\s*(?=[,}])", repl, txt)
            json_block = _quote_bare_values(json_block)
            strategy = json.loads(json_block)
        except Exception:
            strategy = {
                "case_type": case_type or "Unknown",
                "desired_outcome": desired_outcome or "",
                "raw": response_text,
            }

        # Ensure minimal fields
        strategy.setdefault("case_type", case_type or "Unknown")
        strategy.setdefault("desired_outcome", desired_outcome or "")

        # Second-pass recovery: if only 'raw' exists, try to parse JSON from 'raw'
        if isinstance(strategy.get("raw"), str) and (len(strategy.keys()) <= 3):
            raw_text = strategy["raw"]
            import re
            fenced = re.search(r"```(?:json)?\n([\s\S]*?)\n```", raw_text)
            candidate = fenced.group(1) if fenced else raw_text
            match = re.search(r"\{[\s\S]*\}", candidate)
            if match:
                candidate_json = match.group(0)
                # Quote bare values again
                def _repl(m):
                    val = m.group(1).strip()
                    if val.startswith('"') or val.startswith('[') or val.startswith('{'):
                        return ": " + val
                    if re.fullmatch(r"-?\d+(?:\.\d+)?", val) or val in ["true","false","null"]:
                        return ": " + val
                    return ": \"" + val.replace('"','\\"') + "\""
                candidate_json = re.sub(r":\s*([^,}\n]+)\s*(?=[,}])", _repl, candidate_json)
                try:
                    parsed = json.loads(candidate_json)
                    parsed.setdefault("case_type", strategy.get("case_type"))
                    parsed.setdefault("desired_outcome", strategy.get("desired_outcome"))
                    parsed["_parsed_from_raw"] = True
                    strategy = parsed
                except Exception:
                    pass

        # Simple sanitization: avoid clearly irrelevant criminal procedure citations for civil/property cases
        if (case_type or "").lower() in ["property dispute", "inheritance", "family/property"] or ("land" in (case_facts or "").lower()):
            laws = strategy.get("applicable_laws") or []
            filtered = []
            for law in laws:
                sec = (law.get("section") or "").lower()
                if "criminal procedure" in sec or "penal code" in sec:
                    continue
                filtered.append(law)
            strategy["applicable_laws"] = filtered

        return strategy


