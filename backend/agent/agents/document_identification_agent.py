"""
DocumentIdentificationAgent: Identifies required documents based on case type and facts
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class DocumentIdentificationAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def run(self, facts: str, case_type: str, legal_domain: str, case_type_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Identify required documents based on case type and facts
        
        Args:
            facts: Case facts description
            case_type: Detected case type
            legal_domain: Legal domain
            case_type_info: Additional case type information
            
        Returns:
            Dictionary with identified documents
        """
        facts_lower = (facts or "").lower()
        case_type_lower = (case_type or "").lower()
        
        is_prosecution = case_type_info.get("is_prosecution", False) if case_type_info else False
        is_defense = case_type_info.get("is_defense", False) if case_type_info else False
        
        prompt = f"""Based on this case, identify ALL required documents.

Case Facts: {facts}
Case Type: {case_type}
Legal Domain: {legal_domain}
Is Prosecution: {is_prosecution}
Is Defense: {is_defense}

CRITICAL: Match documents to the ACTUAL case type!

For CRIMINAL PROSECUTION cases, consider:
- CCTV footage/video evidence (prove accused's presence at crime scene)
- Forensic reports (DNA, fingerprints, blood analysis) - prove connection to crime
- Murder weapon/physical evidence (prove means of crime)
- Police investigation reports (Thana Patra) - official investigation record
- Medical examiner's report/autopsy report (prove cause of death/injury)
- Witness statements (Saksi Praman) - eyewitness accounts
- Evidence collection reports (chain of custody)
- Medical records (if injuries involved)
- DO NOT include property documents like Lalpurja or Jaminko Nam

For CRIMINAL DEFENSE cases, consider:
- Alibi evidence (passport stamps, flight tickets, hotel receipts, credit card statements) - prove location elsewhere
- CCTV footage proving location (visual proof of alibi)
- Witness testimonies (people who saw accused elsewhere)
- Character certificates (establish good character)
- Phone records/call logs (location data)
- Immigration records (official entry/exit records)
- DO NOT include property documents

For PROPERTY cases, consider:
- Lalpurja (land ownership certificate) - prove ownership
- Jaminko Nam (cadastral survey map) - prove boundaries
- Nagarikta (citizenship certificate) - identity verification
- Nata Pramanit (relationship certificate) - prove family relationships
- Mritak Suchi/Mrityu Suchi (death certificate) - prove death/inheritance timeline
- Rajinama/Parsar (affidavit) - sworn statements

For CONTRACT cases, consider:
- Written contracts (original agreement)
- Communication records (emails, letters) - prove terms
- Payment receipts (prove payment)
- Delivery receipts (prove delivery)
- Witness statements (prove agreement)

For each document, specify:
- document name (in Nepali/English)
- purpose (why needed for THIS specific case)
- required_from (who provides it: client, court, police, etc.)
- priority (high/medium/low)

Return ONLY a JSON array:
[
    {{
        "document": "Document name",
        "purpose": "Why needed for THIS case",
        "required_from": "who provides it",
        "priority": "high|medium|low"
    }}
]
Include at least 8 documents relevant to THIS case type."""
        
        try:
            response = self.llm.generate(prompt, max_tokens=800, temperature=0.3)
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                documents = json.loads(json_match.group(0))
                if isinstance(documents, list) and len(documents) > 0:
                    return {
                        "documents": documents[:15],  # Limit to top 15
                        "case_type": case_type,
                        "legal_domain": legal_domain
                    }
        except Exception as e:
            logger.warning(f"Document identification failed: {e}")
        
        # Fallback: generate documents based on case type
        documents = self._generate_fallback_documents(facts_lower, case_type_lower, is_prosecution, is_defense)
        
        return {
            "documents": documents,
            "case_type": case_type,
            "legal_domain": legal_domain
        }
    
    def _generate_fallback_documents(self, facts_lower: str, case_type_lower: str, is_prosecution: bool, is_defense: bool) -> List[Dict[str, Any]]:
        """Generate fallback documents based on case type keywords"""
        documents = []
        
        if "criminal" in case_type_lower:
            if is_prosecution:
                documents.extend([
                    {"document": "CCTV footage/video evidence", "purpose": "Prove accused's presence at crime scene", "required_from": "Police", "priority": "high"},
                    {"document": "Forensic reports (DNA, fingerprints, blood analysis)", "purpose": "Prove connection to crime", "required_from": "Forensic lab", "priority": "high"},
                    {"document": "Murder weapon/physical evidence", "purpose": "Prove means of crime", "required_from": "Police", "priority": "high"},
                    {"document": "Police investigation reports (Thana Patra)", "purpose": "Official investigation record", "required_from": "Police", "priority": "high"},
                    {"document": "Medical examiner's report/Autopsy report", "purpose": "Prove cause of death", "required_from": "Hospital/Medical examiner", "priority": "high"},
                    {"document": "Witness statements (Saksi Praman)", "purpose": "Eyewitness accounts", "required_from": "Witnesses", "priority": "high"},
                    {"document": "Evidence collection reports", "purpose": "Chain of custody", "required_from": "Police", "priority": "medium"},
                    {"document": "Nagarikta (Citizenship Certificate)", "purpose": "Identity verification of parties", "required_from": "Client", "priority": "medium"}
                ])
            elif is_defense:
                documents.extend([
                    {"document": "Passport with entry/exit stamps", "purpose": "Prove location elsewhere during crime", "required_from": "Client", "priority": "high"},
                    {"document": "Flight tickets/itineraries", "purpose": "Proof of travel to different location", "required_from": "Client", "priority": "high"},
                    {"document": "Hotel receipts/booking confirmations", "purpose": "Proof of stay in different location", "required_from": "Client", "priority": "high"},
                    {"document": "Credit card statements", "purpose": "Transaction records showing location", "required_from": "Client", "priority": "high"},
                    {"document": "CCTV footage from alibi location", "purpose": "Visual proof of location elsewhere", "required_from": "Business establishment", "priority": "high"},
                    {"document": "Witness testimonies from alibi location", "purpose": "People who saw accused elsewhere", "required_from": "Witnesses", "priority": "high"},
                    {"document": "Phone records/call logs", "purpose": "Location data", "required_from": "Telecom provider", "priority": "medium"},
                    {"document": "Character certificates", "purpose": "Establish good character", "required_from": "References", "priority": "medium"}
                ])
        elif "property" in case_type_lower or "inherit" in case_type_lower:
            documents.extend([
                {"document": "Lalpurja (Land Ownership Certificate)", "purpose": "Prove ownership of disputed land", "required_from": "Land Revenue Office", "priority": "high"},
                {"document": "Jaminko Nam (Cadastral Survey Map)", "purpose": "Identify exact location and boundaries", "required_from": "Survey Department", "priority": "high"},
                {"document": "Nagarikta (Citizenship Certificate)", "purpose": "Identity verification of all parties", "required_from": "Client", "priority": "high"},
                {"document": "Nata Pramanit (Relationship Certificate)", "purpose": "Prove family relationships", "required_from": "Local administration", "priority": "high"},
                {"document": "Mritak Suchi/Mrityu Suchi (Death Certificate)", "purpose": "Prove death and inheritance timeline", "required_from": "Local administration", "priority": "high"},
                {"document": "Rajinama/Parsar (Affidavit)", "purpose": "Sworn statement regarding relationships", "required_from": "Client", "priority": "medium"},
                {"document": "Witness Testimonies (Saksi Praman)", "purpose": "Support claims of relationships/ownership", "required_from": "Witnesses", "priority": "medium"},
                {"document": "Property tax receipts", "purpose": "Proof of property ownership history", "required_from": "Tax office", "priority": "medium"}
            ])
        else:
            # Generic documents
            documents.extend([
                {"document": "Nagarikta (Citizenship Certificate)", "purpose": "Identity verification", "required_from": "Client", "priority": "high"},
                {"document": "Rajinama/Parsar (Affidavit)", "purpose": "Sworn statement of facts", "required_from": "Client", "priority": "medium"},
                {"document": "Witness statements", "purpose": "Support claims", "required_from": "Witnesses", "priority": "medium"}
            ])
        
        return documents[:10]  # Limit to 10

