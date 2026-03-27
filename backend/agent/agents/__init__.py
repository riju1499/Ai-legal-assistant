"""
Specialized agent modules for multi-agent strategy generation
"""

from .case_type_analysis_agent import CaseTypeAnalysisAgent
from .law_identification_agent import LawIdentificationAgent
from .document_identification_agent import DocumentIdentificationAgent

__all__ = [
    'CaseAnalysisAgent',
    'PrecedentRetrievalAgent',
    'PrecedentMiningAgent',
    'ArgumentationAgent',
    'VerifierAgent',
    'CaseTypeAnalysisAgent',
    'LawIdentificationAgent',
    'DocumentIdentificationAgent'
]


