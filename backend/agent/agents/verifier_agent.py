"""
VerifierAgent: refines the strategy for accuracy, citations, and clarity
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class VerifierAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, strategy: Dict[str, Any], laws: List[Dict[str, Any]], cases: List[Dict[str, Any]], facts: str, case_type: Optional[str] = None, required_documents: Optional[List[Dict[str, Any]]] = None, case_type_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import json, re
        
        # First, enhance the strategy with missing fields from laws/cases
        strategy = self._enhance_strategy_with_context(strategy, laws, cases, facts, case_type, required_documents, case_type_info)
        
        # Build adaptive prompt based on case type
        case_type_lower = (case_type or "").lower()
        is_prosecution = case_type_info.get("is_prosecution", False) if case_type_info else False
        is_defense = case_type_info.get("is_defense", False) if case_type_info else False
        
        if "criminal" in case_type_lower:
            if is_prosecution:
                doc_guidance = """6. Ensure AT LEAST 8 documents - For CRIMINAL PROSECUTION:
   - CCTV footage/video evidence
   - Forensic reports (DNA, fingerprints, blood analysis)
   - Murder weapon/physical evidence
   - Police investigation reports (Thana Patra)
   - Medical examiner's report/Autopsy report
   - Witness statements (Saksi Praman)
   - Evidence collection reports
   ⚠️ DO NOT include property documents like Lalpurja or Jaminko Nam"""
            else:  # defense
                doc_guidance = """6. Ensure AT LEAST 8 documents - For CRIMINAL DEFENSE:
   - Alibi evidence (passport stamps, flight tickets, hotel receipts)
   - CCTV footage proving location
   - Witness testimonies from alibi location
   - Character certificates
   ⚠️ DO NOT include property documents"""
        elif "property" in case_type_lower or "inherit" in case_type_lower:
            doc_guidance = """6. Ensure AT LEAST 8 documents - For PROPERTY cases:
   - Lalpurja (land ownership certificate)
   - Jaminko Nam (cadastral map)
   - Nagarikta (citizenship)
   - Nata Pramanit (relationship certificate)
   - Mritak Suchi (death certificate if applicable)
   - Rajinama/Parsar (affidavit)
   - Each document needs SPECIFIC purpose related to THIS case"""
        else:
            doc_guidance = """6. Ensure AT LEAST 8 documents with SPECIFIC document names:
   - Match documents to the ACTUAL case type
   - Each document needs SPECIFIC purpose related to THIS case"""
        
        prompt = (
            "You are an expert legal strategy verifier for Nepali law. EXPAND and ENHANCE the strategy to be COMPREHENSIVE, ACCURATE, and ACTIONABLE.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Make content SPECIFIC to the case facts. Do NOT use generic placeholders - use actual parties and relationships from the facts.\n"
            "2. Match content to the ACTUAL case type - do NOT mix case types\n"
            f"3. Case Type: {case_type or 'Unknown'}\n"
            f"4. Is Prosecution: {is_prosecution}, Is Defense: {is_defense}\n"
            "3. Ensure AT LEAST 5 applicable_laws with SPECIFIC section numbers (e.g., 'Section 259', 'Section 145') and WHY each applies\n"
            "4. Ensure AT LEAST 5 precedents with REAL case IDs from provided cases (if case_id is missing or empty, use case_number_english from cases list)\n"
            "5. Ensure AT LEAST 5 detailed arguments - each should be UNIQUE and specific, not repetitive\n"
            f"{doc_guidance}\n"
            "7. Counter-arguments must be REALISTIC opponent claims (not generic), with specific legal responses\n"
            "8. Procedural_timeline must have REAL deadlines SPECIFIC to THIS case type\n"
            "9. Winning_points must be SPECIFIC legal arguments, not generic statements\n"
            "10. Strategic_paragraph must be 4-5 sentences explaining the SPECIFIC approach for THIS case\n"
            "11. Ensure witness_plan specifies WHO should testify and WHAT they should prove\n\n"
            f"Case Facts: {facts}\n\n"
            f"Available Laws (USE THESE):\n{json.dumps((laws or [])[:10], indent=2)}\n\n"
            f"Available Cases (USE THESE):\n{json.dumps((cases or [])[:10], indent=2)}\n\n"
            f"Current Strategy (EXPAND THIS):\n{json.dumps(strategy, indent=2)}\n\n"
            "Return a COMPLETE, EXPANDED JSON object with ALL fields filled with DETAILED, SPECIFIC information.\n"
            "DO NOT return minimal or empty fields. Make this strategy comprehensive and actionable."
        )
        try:
            original_sp = strategy.get('success_probability') if isinstance(strategy, dict) else None
            text = self.llm.generate(prompt, max_tokens=3000, temperature=0.3)
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                refined = json.loads(m.group(0))
                if original_sp and isinstance(refined, dict) and 'success_probability' not in refined:
                    refined['success_probability'] = original_sp
                return self._normalize_and_format(refined, laws, cases, facts, case_type, case_type_info)
        except Exception:
            pass
        return self._normalize_and_format(strategy, laws, cases, facts, case_type, case_type_info)

    def _enhance_strategy_with_context(self, strategy: Dict[str, Any], laws: List[Dict[str, Any]], cases: List[Dict[str, Any]], facts: str, case_type: Optional[str] = None, required_documents: Optional[List[Dict[str, Any]]] = None, case_type_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhance strategy with missing fields from laws and cases context"""
        s = strategy or {}
        
        # Ensure applicable_laws if missing but laws exist
        if not s.get('applicable_laws') and laws:
            s['applicable_laws'] = [
                {
                    'section': f"{l.get('source', 'Unknown')} Section {l.get('page', 'N/A')}",
                    'why': l.get('text', '')[:150] + '...' if l.get('text') else 'Relevant provision for the case'
                }
                for l in laws[:5]
            ]
        
        # Ensure precedents if missing but cases exist
        if not s.get('precedents') and cases:
            s['precedents'] = [
                {
                    'case_id': c.get('case_number_english', 'N/A'),
                    'holding': c.get('summary', '')[:200] + '...' if c.get('summary') else 'Similar case precedent',
                    'why_relevant': 'Similar facts and legal issues'
                }
                for c in cases[:5]
            ]
        
        # Ensure documents_checklist - use required_documents if provided, otherwise generate based on case type
        if not s.get('documents_checklist') or len(s.get('documents_checklist', [])) < 5:
            base_docs = s.get('documents_checklist', [])
            
            # Use provided documents if available
            if required_documents:
                for doc in required_documents:
                    if not any(doc.get('document', '').lower() in d.get('document', '').lower() for d in base_docs):
                        base_docs.append(doc)
            else:
                # Fallback: generate based on case type
                facts_lower = facts.lower()
                case_type_lower = (case_type or "").lower()
                
                if "criminal" in case_type_lower:
                    is_prosecution = case_type_info.get("is_prosecution", False) if case_type_info else False
                    if is_prosecution:
                        if not any('cctv' in d.get('document', '').lower() for d in base_docs):
                            base_docs.append({'document': 'CCTV footage/video evidence', 'purpose': 'Prove accused\'s presence at crime scene', 'priority': 'high'})
                    else:  # defense
                        if not any('passport' in d.get('document', '').lower() or 'alibi' in d.get('document', '').lower() for d in base_docs):
                            base_docs.append({'document': 'Passport with entry/exit stamps', 'purpose': 'Prove location elsewhere during crime', 'priority': 'high'})
                
                if 'land' in facts_lower or 'property' in facts_lower or 'inherit' in facts_lower:
                    if not any('lalpurja' in d.get('document', '').lower() for d in base_docs):
                        base_docs.append({'document': 'Lalpurja (Land Ownership Certificate)', 'purpose': 'Prove ownership of disputed land', 'priority': 'high'})
                    if not any('jaminko' in d.get('document', '').lower() for d in base_docs):
                        base_docs.append({'document': 'Jaminko Nam (Cadastral Map)', 'purpose': 'Identify exact location and boundaries', 'priority': 'high'})
                
                if not any('nagarikta' in d.get('document', '').lower() or 'citizenship' in d.get('document', '').lower() for d in base_docs):
                    base_docs.append({'document': 'Nagarikta (Citizenship Certificate)', 'purpose': 'Identity verification of all parties', 'priority': 'high'})
            
            s['documents_checklist'] = base_docs
        
        # Ensure arguments with at least one based on facts
        if not s.get('arguments') or len(s.get('arguments', [])) < 2:
            base_args = s.get('arguments', [])
            if 'land' in facts.lower() or 'property' in facts.lower():
                base_args.append({
                    'claim': 'We have legal right to equal share of ancestral property under Civil Code 2017',
                    'support': ['Right to inherit from ancestor', 'Equal distribution principles'],
                    'causal_path': ['Property Ownership', 'Inheritance Rights']
                })
            if 'co wife' in facts.lower() or 'co-wife' in facts.lower():
                base_args.append({
                    'claim': 'Children from co-wives have equal inheritance rights under Nepali law',
                    'support': ['Civil Code provisions on joint families', 'Precedent cases'],
                    'causal_path': ['Family Law', 'Inheritance']
                })
            s['arguments'] = base_args
        
        # Ensure winning_points
        if not s.get('winning_points'):
            s['winning_points'] = [
                'Strong legal foundation in Civil Code',
                'Clear evidence of ownership/relationship',
                'Precedent support from similar cases'
            ]
        
        # Ensure strengths and weaknesses
        if not s.get('strengths'):
            s['strengths'] = ['Legal basis exists', 'Documentation can be collected', 'Similar precedents available']
        if not s.get('weaknesses'):
            s['weaknesses'] = ['May require additional documentation', 'Opposing party may contest']
        
        return s
    
    def _normalize_and_format(self, strategy: Dict[str, Any], laws: List[Dict[str, Any]], cases: List[Dict[str, Any]], facts: str, case_type: Optional[str] = None, case_type_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ensure minimal fields exist and add a pretty_markdown presentation."""
        s = strategy or {}
        case_type_lower = (case_type or "").lower()
        is_prosecution = case_type_info.get("is_prosecution", False) if case_type_info else False
        is_defense = case_type_info.get("is_defense", False) if case_type_info else False
        
        # Ensure fields
        s.setdefault('case_type', case_type or 'Unknown')
        s.setdefault('desired_outcome', '')
        s.setdefault('arguments', [])
        s.setdefault('counter_arguments', [])
        s.setdefault('documents_checklist', [])
        s.setdefault('witness_plan', [])
        s.setdefault('procedural_timeline', [])
        s.setdefault('applicable_laws', [])
        s.setdefault('precedents', [])
        
        # Ensure procedural_timeline has concrete deadlines - CASE-TYPE SPECIFIC
        if not s.get('procedural_timeline') or len(s.get('procedural_timeline', [])) < 4:
            base_timeline = s.get('procedural_timeline', [])
            
            # REMOVE property defaults for criminal cases - this was the bug!
            if "criminal" in case_type_lower:
                if is_prosecution:
                    defaults = [
                        {'step': 'Arrest and booking', 'deadline': 'Within 24 hours'},
                        {'step': 'Bail hearing', 'deadline': 'Within 48 hours'},
                        {'step': 'Investigation completion', 'deadline': 'Within 90 days'},
                        {'step': 'Charge sheet filing', 'deadline': 'Within deadline'},
                        {'step': 'Trial dates', 'deadline': 'Court scheduling'},
                        {'step': 'Final arguments and judgment', 'deadline': 'As scheduled'}
                    ]
                else:  # defense
                    defaults = [
                        {'step': 'File bail application', 'deadline': 'Immediately'},
                        {'step': 'Challenge evidence', 'deadline': 'Within deadlines'},
                        {'step': 'Present defense evidence', 'deadline': 'As scheduled'},
                        {'step': 'Cross-examination', 'deadline': 'During trial'},
                        {'step': 'Final arguments', 'deadline': 'As scheduled'}
                    ]
            elif "property" in case_type_lower or "inherit" in case_type_lower:
                defaults = [
                    {'step': 'File partition petition at District Court', 'deadline': 'Within 15 days from case initiation'},
                    {'step': 'Serve notice/summons to all parties', 'deadline': 'Within 7 days after filing'},
                    {'step': 'Submit supporting documents (Lalpurja, certificates)', 'deadline': 'Within 21 days from filing'},
                    {'step': 'First hearing and case registration', 'deadline': 'Within 30-45 days from filing'},
                    {'step': 'Evidence presentation and witness testimonies', 'deadline': 'As scheduled by court (typically 60-90 days)'},
                    {'step': 'Final arguments and judgment', 'deadline': 'Within 120-180 days from filing'}
                ]
            else:
                # Generic defaults
                defaults = [
                    {'step': 'File petition/complaint', 'deadline': 'Within 15 days'},
                    {'step': 'Serve notice/summons', 'deadline': 'Within 7 days after filing'},
                    {'step': 'Submit supporting documents', 'deadline': 'Within 21 days from filing'},
                    {'step': 'First hearing', 'deadline': 'Within 30-45 days from filing'},
                    {'step': 'Evidence presentation', 'deadline': 'As scheduled'},
                    {'step': 'Final arguments and judgment', 'deadline': 'As scheduled'}
                ]
            
            for default in defaults:
                if not any(default['step'] in t.get('step', '') for t in base_timeline):
                    base_timeline.append(default)
            s['procedural_timeline'] = base_timeline[:6]
        # Success probability is expected to be provided by SuccessEstimatorAgent.
        # Do not compute heuristically here; if missing, leave it absent so callers can supply it.
        # Strategic paragraph fallback
        if not s.get('strategic_paragraph') and s.get('raw'):
            try:
                import re
                m = re.search(r'"strategic_paragraph"\s*:\s*"([^"]+)"', s['raw'])
                if m:
                    s['strategic_paragraph'] = m.group(1)
            except Exception:
                pass
        # Improve strategic_paragraph to be case-specific
        if not s.get('strategic_paragraph') or len(s.get('strategic_paragraph', '')) < 100:
            # Build case-specific strategic paragraph
            facts_lower = facts.lower()
            strategic_parts = []
            
            if "criminal" in case_type_lower:
                if is_defense:
                    strategic_parts.append(f"This defense strategy focuses on establishing our client's alibi for the alleged crime date.")
                    strategic_parts.append("We will utilize location evidence (passport stamps, flight tickets, hotel receipts) and witness testimonies to prove the client was elsewhere.")
                    strategic_parts.append("We will challenge the prosecution's evidence and argue that the burden of proof lies with the prosecution to establish guilt beyond a reasonable doubt.")
                elif is_prosecution:
                    strategic_parts.append(f"This prosecution strategy focuses on establishing the accused's guilt through evidence.")
                    strategic_parts.append("We will present CCTV footage, forensic reports, and witness statements to prove the accused's presence and involvement.")
            elif 'grandfather' in facts_lower and ('co wife' in facts_lower or 'co-wife' in facts_lower):
                strategic_parts.append("This strategy focuses on establishing equal inheritance rights for the grandson through the father, who was the son of the grandfather's co-wife.")
                strategic_parts.append("We will rely on Muluki Civil Code 2017 provisions on joint family property (Sections 145-280) and the principle of equal distribution among heirs.")
            elif 'land' in facts_lower or 'property' in facts_lower:
                strategic_parts.append("This strategy aims to secure property rights through comprehensive legal documentation and precedent-backed arguments.")
            
            s['strategic_paragraph'] = " ".join(strategic_parts) if strategic_parts else s.get('strategic_paragraph', '')
        # Build pretty_markdown presentation
        s['pretty_markdown'] = self._build_markdown(s, laws, cases, facts)
        return s

    def _build_markdown(self, s: Dict[str, Any], laws: List[Dict[str, Any]], cases: List[Dict[str, Any]], facts: str) -> str:
        lines = []
        lines.append(f"## Strategy – {s.get('case_type','Unknown')}")
        if s.get('desired_outcome'):
            lines.append(f"**Goal:** {s['desired_outcome']}\n")
        
        # Strategic paragraph
        if s.get('strategic_paragraph'):
            lines.append(f"**Strategy:** {s['strategic_paragraph']}\n")
        
        # Success probability
        sp = s.get('success_probability', {})
        if isinstance(sp, dict) and 'point' in sp:
            ci = sp.get('ci') or []
            if isinstance(ci, list) and len(ci) == 2:
                lines.append(f"**Success Probability:** {sp['point']:.2f} (CI: {ci[0]:.2f}–{ci[1]:.2f})\n")
            else:
                lines.append(f"**Success Probability:** {sp['point']:.2f}\n")
        
        # DOCUMENTS FIRST - Most important for user
        docs = s.get('documents_checklist') or []
        if docs:
            lines.append("### 📄 Documents Required")
            lines.append("**You need these documents to prove your case:**\n")
            for d in docs[:10]:
                doc_name = d.get('document','')
                purpose = d.get('purpose','')
                priority = d.get('priority','')
                priority_badge = f"[{priority.upper()}]" if priority else ""
                lines.append(f"- **{doc_name}** {priority_badge}")
                if purpose:
                    lines.append(f"  - Purpose: {purpose}")
            lines.append("")
        
        # LAWS - What laws apply
        laws_list = s.get('applicable_laws') or []
        if laws_list:
            lines.append("### ⚖️ Applicable Laws")
            lines.append("**These laws fall under your case:**\n")
            for l in laws_list[:8]:
                sec = l.get('section','')
                why = l.get('why','')
                if sec:
                    lines.append(f"- **{sec}**")
                    if why:
                        lines.append(f"  - {why}")
            lines.append("")
        
        # STRATEGY/ARGUMENTS - How to fight
        args = s.get('arguments') or []
        if args:
            lines.append("### 🎯 How to Fight This Case")
            lines.append("**Use these arguments and strategies:**\n")
            for a in args[:6]:
                claim = a.get('claim','')
                support = a.get('support', [])
                if claim:
                    lines.append(f"- **{claim}**")
                    if support and isinstance(support, list):
                        for sup in support[:2]:
                            if sup:
                                lines.append(f"  - {sup}")
            lines.append("")
        
        # Precedents
        precedents = s.get('precedents') or []
        if precedents:
            lines.append("### 📚 Relevant Precedents")
            for p in precedents[:5]:
                cid = p.get('case_id','') or p.get('case_number_english','') or 'N/A'
                if cid == '()' or cid.strip() == '':
                    cid = 'N/A'
                hold = p.get('holding','') or p.get('winning_argument','') or 'Relevant precedent'
                why = p.get('why_relevant','') or p.get('principle','') or ''
                lines.append(f"- **Case {cid}**: {hold}")
                if why and why != hold:
                    lines.append(f"  - {why}")
            lines.append("")
        
        # Timeline
        tl = s.get('procedural_timeline') or []
        if tl:
            lines.append("### ⏰ Procedural Timeline")
            for t in tl[:6]:
                step = t.get('step','')
                deadline = t.get('deadline','')
                if step:
                    lines.append(f"- **{step}**: {deadline}")
            lines.append("")
        
        # Counter-arguments
        ctr = s.get('counter_arguments') or []
        if ctr:
            lines.append("### 🛡️ Anticipated Counter-Arguments")
            for c in ctr[:4]:
                claim = c.get('claim','')
                response = c.get('response','')
                if claim:
                    lines.append(f"- **Opponent may claim**: {claim}")
                    if response:
                        lines.append(f"  - **Our response**: {response}")
            lines.append("")
        
        # Witness plan
        w = s.get('witness_plan') or []
        if w:
            lines.append("### 👥 Witness Plan")
            for wi in w[:5]:
                wtype = wi.get('type','')
                goal = wi.get('goal','')
                if wtype or goal:
                    lines.append(f"- {wtype}: {goal}")
            lines.append("")
        
        # Winning points
        wp = s.get('winning_points') or []
        if wp:
            lines.append("### ✅ Key Points to Emphasize")
            for item in wp[:6]:
                if isinstance(item, str):
                    lines.append(f"- {item}")
        
        return "\n".join(lines).strip()


