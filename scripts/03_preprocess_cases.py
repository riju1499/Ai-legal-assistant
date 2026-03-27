#!/usr/bin/env python3
"""
Phase 2.2: Text Preprocessing and Structured Data Extraction
Cleans Nepali text, extracts structured fields, normalizes content.
"""

import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# Paths
CASE_FILES_DIR = Path(r"D:FinalAI/Wakalat Sewa/CaseFiles")
GLOSSARY_DIR = Path(r"D:FinalAI/Wakalat Sewa/wakalt/tozip/glossary")
OUTPUT_DIR = Path("Wakalat Sewa/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load glossaries
with open(GLOSSARY_DIR / 'legal_glossary.json', 'r', encoding='utf-8') as f:
    GLOSSARY = json.load(f)


class NepaliTextCleaner:
    """
    Clean and normalize Nepali legal text.
    """
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Nepali Unicode characters."""
        # Common OCR errors and variations
        replacements = {
            'ऊ': 'ु',  # Normalize vowel signs if needed
            '\u200c': '',  # Remove zero-width non-joiner
            '\u200d': '',  # Remove zero-width joiner  
            '\xa0': ' ',  # Non-breaking space to regular space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Clean excessive whitespace."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove spaces before punctuation
        text = re.sub(r'\s+([।॥,;:!?])', r'\1', text)
        return text.strip()
    
    @staticmethod
    def remove_page_markers(text: str) -> str:
        """Remove page numbers and markers."""
        # Remove standalone numbers (page numbers)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove "Page X of Y" patterns
        text = re.sub(r'पृष्ठ\s*\d+', '', text)
        return text
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Main cleaning pipeline."""
        text = NepaliTextCleaner.normalize_unicode(text)
        text = NepaliTextCleaner.remove_page_markers(text)
        text = NepaliTextCleaner.clean_whitespace(text)
        return text


class CaseFieldExtractor:
    """
    Extract structured fields from case documents.
    """
    
    def __init__(self, glossary: Dict):
        self.glossary = glossary
    
    def extract_case_number(self, text: str) -> Optional[str]:
        """Extract case/decision number."""
        patterns = [
            r'निर्णय\s*नं[.\s]*([०-९0-9]+)',
            r'निर्णय\s*नम्बर\s*([०-९0-9]+)',
            r'निर्णय\s*संख्या\s*([०-९0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def extract_case_type(self, text: str) -> Dict[str, Optional[str]]:
        """Extract case type in both Nepali and English."""
        # Try to find मुद्दा (case type)
        match = re.search(r'मुद्दा[:\s]*([^\n।]+)', text)
        
        if match:
            nepali_type = match.group(1).strip()
            
            # Try to translate using glossary
            english_type = None
            for nep, eng in self.glossary['case_types'].items():
                if nep in nepali_type:
                    english_type = eng
                    break
            
            return {
                'nepali': nepali_type,
                'english': english_type
            }
        
        return {'nepali': None, 'english': None}
    
    def extract_court(self, text: str) -> Dict[str, Optional[str]]:
        """Extract court level."""
        nepali_court = None
        english_court = None
        
        # Try each court pattern
        for nep_court, eng_court in self.glossary['courts'].items():
            if nep_court in text:
                nepali_court = nep_court
                english_court = eng_court
                break
        
        return {
            'nepali': nepali_court,
            'english': english_court
        }
    
    def extract_judges(self, text: str) -> List[str]:
        """Extract judge names."""
        judges = re.findall(r'न्यायाधीश\s*श्री\s*([^\n]+)', text)
        judges += re.findall(r'न्यायमूर्ति\s*श्री\s*([^\n]+)', text)
        
        # Clean and deduplicate
        judges = [j.strip() for j in judges if j.strip()]
        return list(set(judges))
    
    def extract_lawyers(self, text: str) -> Dict[str, List[str]]:
        """Extract lawyer names for plaintiff and defendant."""
        plaintiff_lawyers = []
        defendant_lawyers = []
        
        # Plaintiff lawyers
        pl_patterns = [
            r'वादी.*?तर्फबाट.*?अधिवक्ता\s*श्री\s*([^\n]+)',
            r'निवेदक.*?तर्फबाट.*?अधिवक्ता\s*श्री\s*([^\n]+)',
        ]
        
        for pattern in pl_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            plaintiff_lawyers.extend(matches)
        
        # Defendant lawyers  
        def_patterns = [
            r'प्रतिवादी.*?तर्फबाट.*?अधिवक्ता\s*श्री\s*([^\n]+)',
            r'विपक्षी.*?तर्फबाट.*?अधिवक्ता\s*श्री\s*([^\n]+)',
        ]
        
        for pattern in def_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            defendant_lawyers.extend(matches)
        
        return {
            'plaintiff': [l.strip() for l in plaintiff_lawyers if l.strip()],
            'defendant': [l.strip() for l in defendant_lawyers if l.strip()]
        }
    
    def extract_parties(self, text: str) -> Dict[str, Optional[str]]:
        """Extract plaintiff and defendant names."""
        plaintiff = None
        defendant = None
        
        # Plaintiff patterns
        plaintiff_match = re.search(r'वादी[:\s]*([^\n।]+)', text)
        if plaintiff_match:
            plaintiff = plaintiff_match.group(1).strip()
        
        # Defendant patterns
        defendant_match = re.search(r'प्रतिवादी[:\s]*([^\n।]+)', text)
        if defendant_match:
            defendant = defendant_match.group(1).strip()
        
        return {
            'plaintiff': plaintiff,
            'defendant': defendant
        }
    
    def extract_decision_date(self, text: str) -> Optional[str]:
        """Extract decision date (Nepali calendar)."""
        patterns = [
            r'फैसला\s*मिति[:\s]*([०-९0-9।.]+)',
            r'इति\s*सम्वत्?\s*([०-९0-9]+)\s*साल\s*([^\s]+)\s*([०-९0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return None
    
    def extract_verdict_keywords(self, text: str) -> List[str]:
        """Extract verdict keywords."""
        keywords = []
        
        for nep_verdict, eng_verdict in self.glossary['verdicts'].items():
            if nep_verdict in text:
                keywords.append(eng_verdict)
        
        return list(set(keywords))
    
    def extract_legal_provisions(self, text: str) -> List[str]:
        """Extract cited legal provisions (sections/articles)."""
        # Find दफा (sections) and नं (numbers)
        sections = re.findall(r'(?:दफा|नं\.?)\s*([०-९0-9]+)', text)
        return list(set(sections))[:20]  # Limit to top 20
    
    def extract_all_fields(self, text: str, filename: str) -> Dict:
        """Extract all structured fields from a case."""
        cleaned_text = NepaliTextCleaner.clean_text(text)
        
        return {
            'filename': filename,
            'case_number': self.extract_case_number(cleaned_text),
            'case_type': self.extract_case_type(cleaned_text),
            'court': self.extract_court(cleaned_text),
            'judges': self.extract_judges(cleaned_text),
            'lawyers': self.extract_lawyers(cleaned_text),
            'parties': self.extract_parties(cleaned_text),
            'decision_date': self.extract_decision_date(cleaned_text),
            'verdict_keywords': self.extract_verdict_keywords(cleaned_text),
            'legal_provisions': self.extract_legal_provisions(cleaned_text),
            'text_length': len(cleaned_text),
            'cleaned_text': cleaned_text,  # Store cleaned version
        }


def process_all_cases(sample_size: Optional[int] = None):
    """
    Process all case files and extract structured data.
    
    Args:
        sample_size: If provided, only process this many cases (for testing)
    """
    print("=" * 80)
    print("PHASE 2: CASE PREPROCESSING & FIELD EXTRACTION")
    print("=" * 80)
    print()
    
    # Get all case files
    case_files = list(CASE_FILES_DIR.glob("*.txt"))
    
    if sample_size:
        print(f"⚠️  Processing sample of {sample_size} cases for testing")
        case_files = case_files[:sample_size]
    else:
        print(f"📁 Processing all {len(case_files)} case files")
    
    print()
    
    # Initialize extractor
    extractor = CaseFieldExtractor(GLOSSARY)
    
    # Process each case
    processed_cases = []
    errors = []
    
    for filepath in tqdm(case_files, desc="Processing cases"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract fields
            case_data = extractor.extract_all_fields(content, filepath.name)
            processed_cases.append(case_data)
            
        except Exception as e:
            errors.append({
                'filename': filepath.name,
                'error': str(e)
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_cases)
    
    # Flatten nested dictionaries for CSV export
    df_export = df.copy()
    
    # Expand case_type
    df_export['case_type_nepali'] = df['case_type'].apply(lambda x: x['nepali'] if x else None)
    df_export['case_type_english'] = df['case_type'].apply(lambda x: x['english'] if x else None)
    
    # Expand court
    df_export['court_nepali'] = df['court'].apply(lambda x: x['nepali'] if x else None)
    df_export['court_english'] = df['court'].apply(lambda x: x['english'] if x else None)
    
    # Expand parties
    df_export['plaintiff'] = df['parties'].apply(lambda x: x['plaintiff'] if x else None)
    df_export['defendant'] = df['parties'].apply(lambda x: x['defendant'] if x else None)
    
    # Convert lists to strings for CSV
    df_export['judges'] = df['judges'].apply(lambda x: '; '.join(x) if isinstance(x, list) else '')
    df_export['verdict_keywords'] = df['verdict_keywords'].apply(lambda x: '; '.join(x) if isinstance(x, list) else '')
    df_export['legal_provisions'] = df['legal_provisions'].apply(lambda x: '; '.join(x) if isinstance(x, list) else '')
    
    # Drop original nested columns
    df_export = df_export.drop(['case_type', 'court', 'parties', 'lawyers'], axis=1)
    
    # Save processed data
    df_export.to_csv(OUTPUT_DIR / 'processed_cases.csv', index=False)
    print(f"\n✅ Saved processed data: {OUTPUT_DIR / 'processed_cases.csv'}")
    
    # Save full structured data (with nested fields) as JSON
    df.to_json(OUTPUT_DIR / 'processed_cases.json', orient='records', force_ascii=False, indent=2)
    print(f"✅ Saved structured JSON: {OUTPUT_DIR / 'processed_cases.json'}")
    
    # Save errors if any
    if errors:
        with open(OUTPUT_DIR / 'processing_errors.json', 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"⚠️  {len(errors)} errors logged: {OUTPUT_DIR / 'processing_errors.json'}")
    
    # Generate statistics
    print()
    print("=" * 80)
    print("📊 PREPROCESSING STATISTICS")
    print("=" * 80)
    print()
    
    print(f"Total cases processed: {len(df)}")
    print(f"Errors encountered: {len(errors)}")
    print()
    
    print("Field extraction success rates:")
    print(f"  Case numbers: {(df['case_number'].notna().sum() / len(df) * 100):.1f}%")
    print(f"  Case types (Nepali): {(df_export['case_type_nepali'].notna().sum() / len(df) * 100):.1f}%")
    print(f"  Case types (English): {(df_export['case_type_english'].notna().sum() / len(df) * 100):.1f}%")
    print(f"  Courts: {(df_export['court_nepali'].notna().sum() / len(df) * 100):.1f}%")
    print(f"  Judges: {(df['judges'].apply(lambda x: len(x) > 0).sum() / len(df) * 100):.1f}%")
    print(f"  Parties: {(df_export['plaintiff'].notna().sum() / len(df) * 100):.1f}%")
    print(f"  Decision dates: {(df['decision_date'].notna().sum() / len(df) * 100):.1f}%")
    print()
    
    # Bilingual coverage
    bilingual_coverage = (df_export['case_type_english'].notna().sum() / 
                          df_export['case_type_nepali'].notna().sum() * 100)
    print(f"Bilingual translation coverage: {bilingual_coverage:.1f}%")
    print()
    
    print("=" * 80)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review processed_cases.csv")
    print("  2. Improve glossary coverage")
    print("  3. Test Nepali NLP tools")
    print("  4. Design database schema")
    print()
    
    return df


if __name__ == "__main__":
    # For testing, process a sample first
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        print("🧪 Running in SAMPLE mode (100 cases)")
        df = process_all_cases(sample_size=100)
    else:
        print("🚀 Processing ALL cases (this may take a few minutes)")
        df = process_all_cases()

