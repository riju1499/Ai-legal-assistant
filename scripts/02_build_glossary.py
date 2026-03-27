#!/usr/bin/env python3
"""
Phase 2.1: Build Bilingual Legal Glossary
Creates Nepali-English mappings for legal terminology, case types, verdicts, and courts.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List

# Paths
ANALYSIS_DIR = Path(r"D:FinalAI/Wakalat Sewa/wakalt/tozip/analysis")
OUTPUT_DIR = Path(r"D:FinalAI/Wakalat Sewa/wakalt/tozip/glossary")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load metadata from Phase 1
metadata_df = pd.read_csv(ANALYSIS_DIR / "case_metadata.csv")


def create_court_glossary() -> Dict[str, str]:
    """
    Map Nepali court names to English equivalents.
    """
    court_glossary = {
        # Court levels
        'सर्वोच्च अदालत': 'Supreme Court',
        'पूर्ण इजलास': 'Full Bench',
        'डिभिजन बेञ्च': 'Division Bench',
        'डिभिजन बेंच': 'Division Bench',
        'जिल्ला अदालत': 'District Court',
        'उच्च अदालत': 'High Court',
        'अपील अदालत': 'Appellate Court',
        'क्षेत्रीय अदालत': 'Regional Court',
        'अञ्चल अदालत': 'Zonal Court',
        
        # Bench types
        'संयुक्त इजलास': 'Joint Bench',
        'एकल इजलास': 'Single Bench',
        'संवैधानिक इजलास': 'Constitutional Bench',
    }
    return court_glossary


def create_verdict_glossary() -> Dict[str, str]:
    """
    Map Nepali verdict terms to English equivalents.
    """
    verdict_glossary = {
        # Verdicts
        'सदर': 'Upheld',
        'बदर': 'Overturned',
        'खारेज': 'Dismissed',
        'ठहर्छ': 'Decided',
        'मनासिब': 'Appropriate',
        'मुनासिब': 'Reasonable',
        'बेमुनासिब': 'Unreasonable',
        
        # Actions
        'स्वीकृत': 'Approved',
        'अस्वीकृत': 'Rejected',
        'निर्णय': 'Judgment',
        'फैसला': 'Decision',
        'आदेश': 'Order',
        'हुकुम': 'Command',
        
        # Legal procedures
        'पुनरावेदन': 'Appeal',
        'पुनरावलोकन': 'Review',
        'निवेदन': 'Petition',
        'रिट': 'Writ',
    }
    return verdict_glossary


def create_case_type_glossary() -> Dict[str, str]:
    """
    Map Nepali case types to English equivalents.
    Based on analysis from Phase 1.
    """
    case_type_glossary = {
        # Criminal cases - Homicide/Life
        'कर्तव्य ज्यान': 'Culpable Homicide',
        'ज्यान': 'Homicide',
        'हत्या': 'Murder',
        'आत्महत्या': 'Suicide',
        'हत्याको प्रयास': 'Attempted Murder',
        
        # Criminal cases - Violence
        'जबरजस्ती करणी': 'Rape',
        'जबरजस्ती': 'Force/Coercion',
        'बलात्कार': 'Sexual Assault',
        'अपहरण': 'Kidnapping',
        'शरीर बन्धक': 'Hostage Taking',
        'कुटपिट': 'Assault',
        'अंगभंग': 'Grievous Hurt',
        
        # Criminal cases - Property
        'चोरी': 'Theft',
        'डकैती': 'Robbery',
        'लुटपाट': 'Looting',
        'आगजनी': 'Arson',
        'तोडफोड': 'Vandalism',
        
        # Criminal cases - Fraud
        'ठगी': 'Fraud',
        'जालसाजी': 'Forgery',
        'धोखाधडी': 'Cheating',
        'भ्रष्टाचार': 'Corruption',
        'घूसखोरी': 'Bribery',
        'आर्थिक अनियमितता': 'Financial Irregularity',
        
        # Criminal cases - Drugs
        'लागु औषध': 'Narcotics',
        'लागु पदार्थ': 'Drugs',
        'लागु औषध खरिद हेरफेर': 'Drug Trafficking',
        
        # Criminal cases - Other
        'हतियार': 'Weapons',
        'हात हतियार': 'Arms',
        'बम विस्फोट': 'Bomb Blast',
        'राजद्रोह': 'Sedition',
        'मानव बेचबिखन': 'Human Trafficking',
        'ओसारपसार': 'Trafficking',
        
        # Civil cases - Property
        'अंश': 'Inheritance/Share',
        'जग्गा': 'Land',
        'जग्गा खिचोला': 'Land Dispute',
        'घर जग्गा': 'House and Land',
        'सम्पत्ति': 'Property',
        'बाटो': 'Road/Path',
        'हक': 'Right',
        'हक कायम': 'Establishing Right',
        'चलन': 'Possession',
        'कब्जा': 'Occupancy',
        'भोगचलन': 'Enjoyment of Property',
        
        # Civil cases - Documents
        'लिखत': 'Document',
        'लिखत बदर': 'Document Cancellation',
        'बक्सपत्र': 'Deed of Gift',
        'बक्सपत्र बदर': 'Cancellation of Gift Deed',
        'मोहीयानी': 'Tenant Right',
        'रजिष्ट्रेसन': 'Registration',
        'नामसारी': 'Name Transfer',
        
        # Civil cases - Family
        'विवाह': 'Marriage',
        'विवाह विच्छेद': 'Divorce',
        'तलाक': 'Divorce',
        'भरण पोषण': 'Maintenance',
        'नाता': 'Relationship',
        'बहुविवाह': 'Polygamy',
        
        # Civil cases - Financial
        'लेनदेन': 'Transaction',
        'ऋण': 'Debt',
        'तमसुक': 'Promissory Note',
        'रकम': 'Amount',
        'ब्याज': 'Interest',
        'क्षतिपूर्ति': 'Compensation',
        'बीमा': 'Insurance',
        
        # Civil cases - Employment
        'जागिर': 'Employment',
        'तलब': 'Salary',
        'पेन्सन': 'Pension',
        'बर्खास्त': 'Dismissal',
        'सेवा': 'Service',
        
        # Civil cases - Business
        'व्यापार': 'Business',
        'कम्पनी': 'Company',
        'साझेदारी': 'Partnership',
        'ठेक्का': 'Contract',
        'बैंकिंग': 'Banking',
        'चेक अनादर': 'Check Dishonor',
        
        # Writ petitions
        'उत्प्रेषण': 'Certiorari',
        'परमादेश': 'Mandamus',
        'प्रतिषेध': 'Prohibition',
        'बन्दी प्रत्यक्षीकरण': 'Habeas Corpus',
        'अधिकार पृच्छा': 'Quo Warranto',
        'निषेधाज्ञा': 'Injunction',
        
        # Constitutional
        'संविधान': 'Constitution',
        'मौलिक हक': 'Fundamental Rights',
        'नागरिकता': 'Citizenship',
        'राजनीतिक': 'Political',
        
        # Administrative
        'सेवा सम्बन्धी': 'Service Related',
        'प्रशासनिक': 'Administrative',
        'कर': 'Tax',
        'आयकर': 'Income Tax',
        'राजस्व': 'Revenue',
        
        # Other
        'निर्वाचन': 'Election',
        'अदालत अवहेलना': 'Contempt of Court',
        'शपथपत्र': 'Affidavit',
        'जमानत': 'Bail',
    }
    return case_type_glossary


def create_legal_terms_glossary() -> Dict[str, str]:
    """
    Common legal terminology mappings.
    """
    legal_terms = {
        # Parties
        'वादी': 'Plaintiff',
        'प्रतिवादी': 'Defendant',
        'अभियुक्त': 'Accused',
        'पीडित': 'Victim',
        'साक्षी': 'Witness',
        'फरियादी': 'Complainant',
        
        # Legal professionals
        'न्यायाधीश': 'Judge',
        'न्यायमूर्ति': 'Justice',
        'अधिवक्ता': 'Advocate',
        'वकिल': 'Lawyer',
        'विद्वान': 'Learned',
        'प्रधान न्यायाधीश': 'Chief Justice',
        
        # Legal documents
        'ऐन': 'Act',
        'दफा': 'Section',
        'नियम': 'Rule',
        'विधेयक': 'Bill',
        'संहिता': 'Code',
        'अध्यादेश': 'Ordinance',
        
        # Court terms
        'इजलास': 'Bench',
        'सुनुवाई': 'Hearing',
        'मुद्दा': 'Case',
        'मुचुल्का': 'Statement',
        'प्रमाण': 'Evidence',
        'बयान': 'Testimony',
        'सबुद': 'Proof',
        
        # Legal actions
        'मुद्दा दर्ता': 'Case Registration',
        'जाहेरी': 'Complaint',
        'अनुसन्धान': 'Investigation',
        'पक्राउ': 'Arrest',
        'थुनामा': 'In Custody',
        'धरौटी': 'Bail Bond',
        'सजाय': 'Punishment',
        'जरिवाना': 'Fine',
        'कैद': 'Imprisonment',
        'मृत्युदण्ड': 'Death Penalty',
    }
    return legal_terms


def extract_unique_case_types(df: pd.DataFrame) -> List[str]:
    """
    Extract all unique case types from the dataset.
    """
    case_types = df['case_type'].dropna().unique()
    # Count frequency
    case_type_counts = df['case_type'].value_counts()
    
    print(f"\n📊 Found {len(case_types)} unique case types")
    print(f"\nTop 30 case types by frequency:\n")
    
    for case_type, count in case_type_counts.head(30).items():
        print(f"  {case_type}: {count}")
    
    return list(case_types)


def save_glossaries():
    """
    Save all glossaries to JSON files.
    """
    print("=" * 80)
    print("BUILDING BILINGUAL LEGAL GLOSSARY")
    print("=" * 80)
    print()
    
    # Create individual glossaries
    courts = create_court_glossary()
    verdicts = create_verdict_glossary()
    case_types = create_case_type_glossary()
    legal_terms = create_legal_terms_glossary()
    
    # Extract actual case types from data
    print("📋 Extracting case types from dataset...")
    unique_case_types = extract_unique_case_types(metadata_df)
    
    # Combined glossary
    combined = {
        'courts': courts,
        'verdicts': verdicts,
        'case_types': case_types,
        'legal_terms': legal_terms,
        'metadata': {
            'version': '1.0',
            'total_terms': len(courts) + len(verdicts) + len(case_types) + len(legal_terms),
            'courts_count': len(courts),
            'verdicts_count': len(verdicts),
            'case_types_count': len(case_types),
            'legal_terms_count': len(legal_terms),
        }
    }
    
    # Save combined glossary
    with open(OUTPUT_DIR / 'legal_glossary.json', 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved combined glossary: {OUTPUT_DIR / 'legal_glossary.json'}")
    
    # Save individual glossaries for easier access
    with open(OUTPUT_DIR / 'courts.json', 'w', encoding='utf-8') as f:
        json.dump(courts, f, ensure_ascii=False, indent=2)
    
    with open(OUTPUT_DIR / 'verdicts.json', 'w', encoding='utf-8') as f:
        json.dump(verdicts, f, ensure_ascii=False, indent=2)
    
    with open(OUTPUT_DIR / 'case_types.json', 'w', encoding='utf-8') as f:
        json.dump(case_types, f, ensure_ascii=False, indent=2)
    
    with open(OUTPUT_DIR / 'legal_terms.json', 'w', encoding='utf-8') as f:
        json.dump(legal_terms, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved individual glossaries to {OUTPUT_DIR}/")
    print()
    
    # Statistics
    print("=" * 80)
    print("📊 GLOSSARY STATISTICS")
    print("=" * 80)
    print(f"\n  Court terms:      {len(courts)}")
    print(f"  Verdict terms:    {len(verdicts)}")
    print(f"  Case types:       {len(case_types)}")
    print(f"  Legal terms:      {len(legal_terms)}")
    print(f"  {'─' * 40}")
    print(f"  TOTAL:            {combined['metadata']['total_terms']}")
    print()
    
    # Coverage analysis
    print("=" * 80)
    print("📈 COVERAGE ANALYSIS")
    print("=" * 80)
    print()
    
    # Check how many case types from data are in glossary
    covered = 0
    uncovered = []
    
    for case_type in unique_case_types[:50]:  # Check top 50
        if case_type and any(nepali in case_type for nepali in case_types.keys()):
            covered += 1
        elif case_type and case_type not in ['x', 'ːX', ':', '']:
            uncovered.append(case_type)
    
    print(f"  Case types in glossary: {len(case_types)}")
    print(f"  Case types in dataset: {len(unique_case_types)}")
    print(f"  Coverage (top 50): {covered}/50 ({covered/50*100:.1f}%)")
    print()
    
    if uncovered[:10]:
        print("  ⚠️  Uncovered case types (sample):")
        for ct in uncovered[:10]:
            print(f"     • {ct}")
        print()
        print("  💡 These should be added to the glossary manually")
        print()
    
    # Save uncovered terms for review
    if uncovered:
        with open(OUTPUT_DIR / 'uncovered_case_types.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(uncovered))
        print(f"✅ Saved uncovered terms to: {OUTPUT_DIR / 'uncovered_case_types.txt'}")
        print()
    
    print("=" * 80)
    print("✅ GLOSSARY BUILD COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review uncovered_case_types.txt")
    print("  2. Add missing terms to glossary")
    print("  3. Get legal expert validation")
    print("  4. Proceed to text preprocessing")
    print()


if __name__ == "__main__":
    save_glossaries()

