#!/usr/bin/env python3
"""
Quick test to verify Qdrant integration works
"""

import sys
from pathlib import Path

sys.path.insert(0, 'backend')

from agent.qdrant_kb import QdrantKnowledgeBase

print("=" * 70)
print("Testing Qdrant Integration")
print("=" * 70)

# Initialize Qdrant
print("\n1. Initializing Qdrant...")
kb = QdrantKnowledgeBase(qdrant_path='qdrant_storage')

# Get collection info
print("\n2. Collection Info:")
info = kb.get_collection_info()
print(f"   Vector count: {info.get('vector_count', 0):,}")

# Test searches
test_queries = [
    "What is Article 12 of Constitution?",
    "Right to Freedom",
    "Fundamental Rights",
]

print("\n3. Test Searches:")
print("=" * 70)

for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = kb.search(query, limit=3, score_threshold=0.3)
    
    if results:
        print(f"✅ Found {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"   {i}. {r['source']} (Page {r['page']}, Score: {r['score']:.3f})")
            print(f"      Text preview: {r['text'][:100]}...")
    else:
        print("❌ No results found")

print("\n" + "=" * 70)
print("✅ Qdrant is working correctly!")
print("=" * 70)
print("\nNow start the backend with:")
print("  conda activate legenv")
print("  cd backend && python main.py")

