#!/usr/bin/env python3
"""
Explore the processed case data
Quick analysis and examples of the bilingual dataset
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter

# Paths
PROCESSED_DIR = Path(r"D:FinalAI/Wakalat Sewa/processed_llm")
GLOSSARY_DIR = Path(r"D:FinalAI/Wakalat Sewa/wakalt/tozip/glossary")

# Load data
df = pd.read_csv(PROCESSED_DIR / 'processed_cases.csv')

print("=" * 80)
print("PROCESSED DATA EXPLORATION")
print("=" * 80)
print()

# Basic info
print(f"📊 Total Cases: {len(df)}")
print(f"📝 Total Columns: {len(df.columns)}")
print()

# Column names
print("📋 Available Fields:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")
print()

# Sample case
print("=" * 80)
print("🔍 SAMPLE CASE (Random Selection)")
print("=" * 80)
print()

sample = df.sample(1).iloc[0]
print(f"Filename: {sample['filename']}")
print(f"Case Number: {sample['case_number']}")
print(f"Case Type (Nepali): {sample['case_type_nepali']}")
print(f"Case Type (English): {sample['case_type_english']}")
print(f"Court (Nepali): {sample['court_nepali']}")
print(f"Court (English): {sample['court_english']}")
print(f"Judges: {sample['judges']}")
print(f"Plaintiff: {sample['plaintiff']}")
print(f"Defendant: {sample['defendant']}")
print(f"Verdict Keywords: {sample['verdict_keywords']}")
print(f"Legal Provisions: {sample['legal_provisions']}")
print(f"Text Length: {sample['text_length']} characters")
print()

# Top case types
print("=" * 80)
print("📈 TOP 15 CASE TYPES (Bilingual)")
print("=" * 80)
print()

case_type_counts = df['case_type_nepali'].value_counts().head(15)
print(f"{'Rank':<5} {'Nepali':<35} {'English':<30} {'Count':<8}")
print("-" * 80)

for rank, (nepali, count) in enumerate(case_type_counts.items(), 1):
    # Get corresponding English
    english = df[df['case_type_nepali'] == nepali]['case_type_english'].iloc[0]
    if pd.isna(english):
        english = "⚠️  Not Translated"
    print(f"{rank:<5} {str(nepali)[:34]:<35} {str(english)[:29]:<30} {count:<8}")

print()

# Court distribution
print("=" * 80)
print("⚖️  COURT DISTRIBUTION")
print("=" * 80)
print()

court_counts = df['court_nepali'].value_counts()
print(f"{'Court (Nepali)':<40} {'English':<30} {'Cases':<10}")
print("-" * 80)

for court_nepali, count in court_counts.items():
    if pd.notna(court_nepali):
        english = df[df['court_nepali'] == court_nepali]['court_english'].iloc[0]
        print(f"{str(court_nepali)[:39]:<40} {str(english)[:29]:<30} {count:<10}")

print()

# Verdict keywords analysis
print("=" * 80)
print("📋 VERDICT KEYWORDS (Most Common)")
print("=" * 80)
print()

# Extract all verdict keywords
all_verdicts = []
for verdicts in df['verdict_keywords'].dropna():
    if verdicts:
        all_verdicts.extend([v.strip() for v in str(verdicts).split(';')])

verdict_counter = Counter(all_verdicts)
print(f"{'Verdict Keyword':<30} {'Frequency':<10}")
print("-" * 40)
for verdict, count in verdict_counter.most_common(10):
    if verdict:
        print(f"{verdict:<30} {count:<10}")

print()

# Bilingual coverage analysis
print("=" * 80)
print("🌐 BILINGUAL COVERAGE ANALYSIS")
print("=" * 80)
print()

total_with_nepali = df['case_type_nepali'].notna().sum()
total_with_english = df['case_type_english'].notna().sum()
coverage = (total_with_english / total_with_nepali * 100) if total_with_nepali > 0 else 0

print(f"Cases with Nepali case type: {total_with_nepali:,} ({total_with_nepali/len(df)*100:.1f}%)")
print(f"Cases with English translation: {total_with_english:,} ({total_with_english/len(df)*100:.1f}%)")
print(f"Translation coverage: {coverage:.1f}%")
print()

# Find untranslated common types
untranslated = df[df['case_type_nepali'].notna() & df['case_type_english'].isna()]
if len(untranslated) > 0:
    print(f"⚠️  {len(untranslated)} cases need translation")
    print("\nMost common untranslated case types:")
    untrans_types = untranslated['case_type_nepali'].value_counts().head(10)
    for i, (case_type, count) in enumerate(untrans_types.items(), 1):
        print(f"  {i:2d}. {case_type} ({count} cases)")
    print()

# Judge statistics
print("=" * 80)
print("👨‍⚖️ JUDGE STATISTICS")
print("=" * 80)
print()

# Count unique judges
all_judges = []
for judges in df['judges'].dropna():
    if judges:
        all_judges.extend([j.strip() for j in str(judges).split(';')])

unique_judges = len(set(all_judges))
print(f"Total unique judges: {unique_judges:,}")
print(f"\nMost frequent judges:")

judge_counter = Counter(all_judges)
for i, (judge, count) in enumerate(judge_counter.most_common(10), 1):
    print(f"  {i:2d}. {judge} ({count} cases)")

print()

# Text length statistics
print("=" * 80)
print("📏 TEXT LENGTH STATISTICS")
print("=" * 80)
print()

print(f"Average length: {df['text_length'].mean():.0f} characters")
print(f"Median length: {df['text_length'].median():.0f} characters")
print(f"Shortest case: {df['text_length'].min()} characters")
print(f"Longest case: {df['text_length'].max():,} characters")
print()

# Sample search examples
print("=" * 80)
print("🔍 SAMPLE SEARCH QUERIES (Examples)")
print("=" * 80)
print()

print("Example 1: Find all murder cases")
print("-" * 40)
murder_cases = df[df['case_type_nepali'].str.contains('ज्यान', na=False)]
print(f"Found {len(murder_cases)} cases related to 'ज्यान' (homicide)")
print()

print("Example 2: Find Supreme Court cases")
print("-" * 40)
supreme_cases = df[df['court_nepali'].str.contains('सर्वोच्च', na=False)]
print(f"Found {len(supreme_cases)} Supreme Court cases")
print()

print("Example 3: Find property dispute cases")
print("-" * 40)
property_cases = df[df['case_type_nepali'].str.contains('जग्गा|अंश|सम्पत्ति', na=False)]
print(f"Found {len(property_cases)} property-related cases")
print()

print("=" * 80)
print("✅ DATA READY FOR:")
print("=" * 80)
print()
print("  1. ✓ Bilingual search (Nepali + English)")
print("  2. ✓ Court-level filtering")
print("  3. ✓ Case type classification")
print("  4. ✓ Judge/lawyer tracking")
print("  5. ✓ Statistical analysis")
print("  6. → Next: Semantic search with embeddings")
print("  7. → Next: AI-powered analysis")
print()

print("=" * 80)
print("💾 EXPORT SAMPLES FOR REVIEW")
print("=" * 80)
print()

# Export a sample for review
sample_export = df.sample(min(50, len(df)))[[
    'filename', 'case_number', 'case_type_nepali', 'case_type_english',
    'court_nepali', 'court_english', 'judges', 'plaintiff', 'defendant'
]]

sample_export.to_csv(PROCESSED_DIR / 'sample_50_cases.csv', index=False)
print(f"✅ Exported 50 random cases to: {PROCESSED_DIR / 'sample_50_cases.csv'}")
print()

print("📖 To explore further:")
print("  • Open processed/processed_cases.csv in Excel/LibreOffice")
print("  • Open processed/sample_50_cases.csv for quick review")
print("  • Load processed/processed_cases.json for full structured data")
print()

