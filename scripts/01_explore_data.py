#!/usr/bin/env python3
"""
Phase 1: Data Exploration & Understanding
Analyzes Nepali court case files to extract metadata patterns and understand content structure.
"""

import os
import re
from pathlib import Path
from collections import Counter, defaultdict
import json
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

# Paths
CASE_FILES_DIR = Path("/CaseFiles")
OUTPUT_DIR = Path("Wakalat Sewa/analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_case_number_from_filename(filename: str) -> str:
    """Extract case number from filename."""
    # Filenames like: "निर्णय नं ६३८८ - वकसपत्र वदर.txt"
    match = re.search(r'निर्णय\s*नं?\s*([०-९0-9]+)', filename)
    if match:
        return match.group(1)
    return ""


def extract_case_type_from_filename(filename: str) -> str:
    """Extract case type from filename (after the dash)."""
    parts = filename.replace('.txt', '').split('-')
    if len(parts) > 1:
        return parts[1].strip()
    return ""


def parse_case_file(filepath: Path) -> Dict:
    """
    Parse a single case file and extract key metadata.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata using regex patterns
        metadata = {
            'filename': filepath.name,
            'file_size_kb': filepath.stat().st_size / 1024,
            'line_count': len(content.split('\n')),
            'char_count': len(content),
        }
        
        # Try to extract case number from content
        case_num_match = re.search(r'निर्णय\s*नं[.\s]*([०-९0-9]+)', content)
        if case_num_match:
            metadata['case_number'] = case_num_match.group(1)
        else:
            metadata['case_number'] = extract_case_number_from_filename(filepath.name)
        
        # Extract case type (मुद्दा)
        case_type_match = re.search(r'मुद्दा[:\s]*([^\n।]+)', content)
        if case_type_match:
            metadata['case_type'] = case_type_match.group(1).strip()
        else:
            metadata['case_type'] = extract_case_type_from_filename(filepath.name)
        
        # Extract court level
        court_patterns = [
            (r'पूर्ण\s*इजलास', 'Full Bench'),
            (r'डिभिजन\s*बेञ्च', 'Division Bench'),
            (r'सर्वोच्च\s*अदालत', 'Supreme Court'),
            (r'अपील\s*अदालत', 'Appellate Court'),
            (r'जिल्ला\s*अदालत', 'District Court'),
            (r'उच्च\s*अदालत', 'High Court'),
        ]
        
        metadata['court_level'] = ''
        for pattern, court_name in court_patterns:
            if re.search(pattern, content):
                metadata['court_level'] = court_name
                break
        
        # Extract judges (न्यायाधीश)
        judges = re.findall(r'न्यायाधीश\s*श्री\s*([^\n]+)', content)
        metadata['judges'] = judges if judges else []
        metadata['judge_count'] = len(judges)
        
        # Extract parties (वादी/प्रतिवादी)
        plaintiff_match = re.search(r'वादी[:\s]*([^\n।]+)', content)
        defendant_match = re.search(r'प्रतिवादी[:\s]*([^\n।]+)', content)
        
        metadata['plaintiff'] = plaintiff_match.group(1).strip() if plaintiff_match else ''
        metadata['defendant'] = defendant_match.group(1).strip() if defendant_match else ''
        
        # Extract decision date (फैसला मिति)
        date_match = re.search(r'फैसला\s*मिति[:\s]*([०-९0-9।.]+)', content)
        if date_match:
            metadata['decision_date'] = date_match.group(1).strip()
        else:
            # Alternative date format
            date_match = re.search(r'इति\s*सम्वत्?\s*([०-९0-9]+)\s*साल\s*([^\s]+)\s*([०-९0-9]+)', content)
            if date_match:
                metadata['decision_date'] = f"{date_match.group(1)}.{date_match.group(2)}.{date_match.group(3)}"
            else:
                metadata['decision_date'] = ''
        
        # Check for lawyers/advocates
        lawyers = re.findall(r'(?:अधिवक्ता|विद्वान|प्लीडर)\s*श्री\s*([^\n]+)', content)
        metadata['lawyers'] = lawyers if lawyers else []
        metadata['lawyer_count'] = len(lawyers)
        
        # Detect verdict keywords
        verdict_keywords = {
            'सदर': 'upheld',
            'बदर': 'overturned',
            'खारेज': 'dismissed',
            'ठहर्छ': 'decided',
            'मनासिब': 'appropriate',
        }
        
        metadata['verdict_keywords'] = []
        for nepali_word, english_meaning in verdict_keywords.items():
            if nepali_word in content:
                metadata['verdict_keywords'].append(english_meaning)
        
        # Detect sections/articles mentioned (ऐन)
        sections = re.findall(r'(?:दफा|नं\.?)\s*([०-९0-9]+)', content)
        metadata['mentioned_sections'] = list(set(sections))[:10]  # Top 10 unique
        metadata['section_count'] = len(set(sections))
        
        return metadata
        
    except Exception as e:
        return {
            'filename': filepath.name,
            'error': str(e),
        }


def analyze_case_files():
    """
    Analyze all case files and generate comprehensive statistics.
    """
    print("=" * 80)
    print("WAKALAT SEWA - Phase 1: Data Exploration & Understanding")
    print("=" * 80)
    print()
    
    # Get all case files
    case_files = list(CASE_FILES_DIR.glob("*.txt"))
    print(f"📁 Found {len(case_files)} case files")
    print()
    
    # Parse all files
    print("🔍 Parsing case files...")
    all_metadata = []
    
    for filepath in tqdm(case_files, desc="Processing files"):
        metadata = parse_case_file(filepath)
        all_metadata.append(metadata)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_metadata)
    
    # Save raw metadata
    df.to_csv(OUTPUT_DIR / "case_metadata.csv", index=False)
    print(f"✅ Saved metadata to: {OUTPUT_DIR / 'case_metadata.csv'}")
    print()
    
    # Generate statistics
    print("=" * 80)
    print("📊 DATA STATISTICS")
    print("=" * 80)
    print()
    
    # 1. File size statistics
    print("1. FILE SIZE STATISTICS:")
    print(f"   Total files: {len(df)}")
    print(f"   Average file size: {df['file_size_kb'].mean():.2f} KB")
    print(f"   Median file size: {df['file_size_kb'].median():.2f} KB")
    print(f"   Min file size: {df['file_size_kb'].min():.2f} KB")
    print(f"   Max file size: {df['file_size_kb'].max():.2f} KB")
    print()
    
    # 2. Content statistics
    print("2. CONTENT STATISTICS:")
    print(f"   Average lines per file: {df['line_count'].mean():.0f}")
    print(f"   Average characters per file: {df['char_count'].mean():.0f}")
    print()
    
    # 3. Court level distribution
    print("3. COURT LEVEL DISTRIBUTION:")
    court_dist = df['court_level'].value_counts()
    for court, count in court_dist.items():
        if court:
            print(f"   {court}: {count} ({count/len(df)*100:.1f}%)")
    print()
    
    # 4. Case type distribution (top 20)
    print("4. MOST COMMON CASE TYPES (Top 20):")
    case_type_dist = df['case_type'].value_counts().head(20)
    for case_type, count in case_type_dist.items():
        if case_type:
            print(f"   {case_type}: {count}")
    print()
    
    # 5. Judge statistics
    print("5. JUDGE STATISTICS:")
    print(f"   Average judges per case: {df['judge_count'].mean():.2f}")
    print(f"   Cases with judges identified: {(df['judge_count'] > 0).sum()}")
    
    # Get most frequent judges
    all_judges = []
    for judges_list in df['judges']:
        if isinstance(judges_list, list):
            all_judges.extend(judges_list)
    
    if all_judges:
        judge_counter = Counter(all_judges)
        print(f"   Top 10 Most Frequent Judges:")
        for judge, count in judge_counter.most_common(10):
            print(f"      {judge}: {count} cases")
    print()
    
    # 6. Lawyer statistics
    print("6. LAWYER STATISTICS:")
    print(f"   Average lawyers per case: {df['lawyer_count'].mean():.2f}")
    print(f"   Cases with lawyers identified: {(df['lawyer_count'] > 0).sum()}")
    
    # Get most frequent lawyers
    all_lawyers = []
    for lawyers_list in df['lawyers']:
        if isinstance(lawyers_list, list):
            all_lawyers.extend(lawyers_list)
    
    if all_lawyers:
        lawyer_counter = Counter(all_lawyers)
        print(f"   Top 10 Most Frequent Lawyers:")
        for lawyer, count in lawyer_counter.most_common(10):
            print(f"      {lawyer}: {count} cases")
    print()
    
    # 7. Verdict distribution
    print("7. VERDICT KEYWORD DISTRIBUTION:")
    verdict_counter = Counter()
    for keywords in df['verdict_keywords']:
        if isinstance(keywords, list):
            verdict_counter.update(keywords)
    
    for verdict, count in verdict_counter.most_common():
        print(f"   {verdict}: {count} cases")
    print()
    
    # 8. Year distribution (from decision dates)
    print("8. TEMPORAL DISTRIBUTION:")
    years = []
    for date in df['decision_date']:
        if isinstance(date, str) and date:
            # Extract year (Nepali calendar year)
            year_match = re.search(r'([०-९0-9]{4})', str(date))
            if year_match:
                years.append(year_match.group(1))
    
    if years:
        year_counter = Counter(years)
        print(f"   Cases by year (Top 15):")
        for year, count in year_counter.most_common(15):
            print(f"      {year}: {count} cases")
    print()
    
    # Save summary statistics
    summary = {
        'total_files': len(df),
        'avg_file_size_kb': float(df['file_size_kb'].mean()),
        'avg_lines': float(df['line_count'].mean()),
        'avg_chars': float(df['char_count'].mean()),
        'court_distribution': court_dist.to_dict(),
        'top_case_types': case_type_dist.head(10).to_dict(),
        'verdict_distribution': dict(verdict_counter),
    }
    
    with open(OUTPUT_DIR / "summary_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved summary statistics to: {OUTPUT_DIR / 'summary_statistics.json'}")
    print()
    
    # Sample cases for manual review
    print("=" * 80)
    print("📋 SAMPLE CASES FOR MANUAL REVIEW")
    print("=" * 80)
    print()
    
    # Pick a few representative samples
    samples = df.sample(min(5, len(df)))
    for idx, row in samples.iterrows():
        print(f"Sample {idx + 1}:")
        print(f"  Filename: {row['filename']}")
        print(f"  Case Number: {row.get('case_number', 'N/A')}")
        print(f"  Case Type: {row.get('case_type', 'N/A')}")
        print(f"  Court: {row.get('court_level', 'N/A')}")
        print(f"  Judge Count: {row.get('judge_count', 0)}")
        print(f"  File Size: {row.get('file_size_kb', 0):.2f} KB")
        print()
    
    print("=" * 80)
    print("✅ PHASE 1 COMPLETE!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Review the generated metadata CSV")
    print("  2. Identify patterns for structured data extraction")
    print("  3. Move to Phase 2: Data Preprocessing")
    print()


if __name__ == "__main__":
    analyze_case_files()

