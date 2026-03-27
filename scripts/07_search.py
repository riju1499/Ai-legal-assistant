#!/usr/bin/env python3
"""
Phase 7: Semantic Search Interface
===================================
Query the FAISS index with natural language questions in Nepali or English.

Usage:
    python 07_search.py "your query here"
    python 07_search.py  (interactive mode)
"""

import os
import sys
import json
import numpy as np
import faiss
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# Configuration
# Paths are relative to project root
INDEX_DIR = Path("..") / "search_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
CORPUS_FILE = INDEX_DIR / "corpus.json"
METADATA_FILE = INDEX_DIR / "metadata.json"
CONFIG_FILE = INDEX_DIR / "config.json"

# Model (must match the one used for indexing)
MODEL_ID = "intfloat/multilingual-e5-base"

class SemanticSearch:
    """Semantic search engine for legal cases."""
    
    def __init__(self):
        """Initialize search engine by loading index, corpus, and model."""
        print("🔍 Initializing Semantic Search Engine...")
        
        # Load config
        with open(CONFIG_FILE, 'r') as f:
            self.config = json.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(str(INDEX_FILE))
        print(f"   ✓ Loaded index: {self.index.ntotal:,} documents")
        
        # Load corpus
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        
        # Load metadata
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"   ✓ Loaded corpus: {len(self.corpus):,} documents")
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   ✓ Device: {self.device}")
        print(f"   ✓ Loading model: {MODEL_ID}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()
        
        print("✅ Search engine ready!\n")
    
    def mean_pool(self, last_hidden_state, attention_mask):
        """Mean pooling."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts
    
    def encode_query(self, query):
        """
        Encode a search query into an embedding.
        E5 model requires 'query: ' prefix for search queries.
        """
        # Add query prefix
        query_with_prefix = f"query: {query}"
        
        # Tokenize
        inputs = self.tokenizer(
            [query_with_prefix],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embedding = self.mean_pool(
            outputs.last_hidden_state,
            inputs["attention_mask"]
        ).detach().cpu().numpy()
        
        # Normalize
        faiss.normalize_L2(embedding)
        
        return embedding.astype('float32')
    
    def search(self, query, k=5):
        """
        Search for top-k most similar documents.
        
        Args:
            query: Search query (Nepali or English)
            k: Number of results to return
        
        Returns:
            List of results with metadata and scores
        """
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            result = {
                'rank': i + 1,
                'score': float(score),
                'metadata': self.metadata[idx],
                'text_preview': self.corpus[idx][:300] + '...' if len(self.corpus[idx]) > 300 else self.corpus[idx]
            }
            results.append(result)
        
        return results
    
    def print_results(self, query, results):
        """Pretty print search results."""
        print("=" * 80)
        print(f"🔍 Query: {query}")
        print("=" * 80)
        print()
        
        if not results:
            print("❌ No results found.")
            return
        
        for result in results:
            print(f"📄 Rank #{result['rank']} | Score: {result['score']:.4f}")
            print("-" * 80)
            
            meta = result['metadata']
            
            # Case information
            if meta.get('case_number_english'):
                print(f"   📋 Case Number: {meta['case_number_english']}")
            if meta.get('case_number_nepali'):
                print(f"   📋 केस नम्बर: {meta['case_number_nepali']}")
            
            if meta.get('case_type_english'):
                print(f"   ⚖️  Case Type: {meta['case_type_english']}")
            if meta.get('case_type_nepali'):
                print(f"   ⚖️  मुद्दा प्रकार: {meta['case_type_nepali']}")
            
            if meta.get('court_english'):
                print(f"   🏛️  Court: {meta['court_english']}")
            if meta.get('court_nepali'):
                print(f"   🏛️  अदालत: {meta['court_nepali']}")
            
            print(f"   📁 File: {meta.get('filename', 'Unknown')}")
            
            # Summary preview
            if meta.get('summary'):
                print()
                print(f"   📝 Summary:")
                summary_lines = meta['summary'].split('\n')
                for line in summary_lines[:3]:  # Show first 3 lines
                    if line.strip():
                        print(f"      {line.strip()}")
                if len(summary_lines) > 3:
                    print(f"      ...")
            
            print()
        
        print("=" * 80)

def interactive_mode(search_engine):
    """Interactive search mode."""
    print()
    print("=" * 80)
    print("🔍 INTERACTIVE SEMANTIC SEARCH MODE")
    print("=" * 80)
    print()
    print("Enter your search query in Nepali or English.")
    print("Type 'quit' or 'exit' to stop.")
    print()
    
    while True:
        try:
            query = input("🔎 Search: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Perform search
            results = search_engine.search(query, k=5)
            print()
            search_engine.print_results(query, results)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

def main():
    # Check if index exists
    if not INDEX_FILE.exists():
        print("❌ Error: Search index not found!")
        print(f"   Expected: {INDEX_FILE}")
        print()
        print("Please run 06_build_semantic_index.py first to build the index.")
        sys.exit(1)
    
    # Initialize search engine
    search_engine = SemanticSearch()
    
    # Check if query provided as argument
    if len(sys.argv) > 1:
        # Single query mode
        query = ' '.join(sys.argv[1:])
        results = search_engine.search(query, k=5)
        search_engine.print_results(query, results)
    else:
        # Interactive mode
        interactive_mode(search_engine)

if __name__ == "__main__":
    main()

