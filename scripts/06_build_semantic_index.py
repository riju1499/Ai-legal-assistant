#!/usr/bin/env python3
"""
Phase 6: Build Semantic Search Index
=====================================
Creates FAISS vector index from LLM-extracted case data for semantic search.

Uses multilingual embeddings to support both Nepali and English queries.
"""

import os
import json
import numpy as np
import faiss
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# Configuration
# Paths are relative to project root
PROCESSED_LLM_DIR = Path("..") / "processed_llm"
SUMMARIES_FILE = PROCESSED_LLM_DIR / "case_summaries.json"
FULL_DATA_FILE = PROCESSED_LLM_DIR / "llm_extracted_cases.json"
OUTPUT_DIR = Path("..") / "search_index"

# Model: Multilingual E5 - supports 100+ languages including Nepali
MODEL_ID = "intfloat/multilingual-e5-base"  # 768-dimensional embeddings

def load_case_data():
    """Load and prepare case data for indexing."""
    print("📂 Loading case data...")
    
    # Load summaries
    with open(SUMMARIES_FILE, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    
    print(f"✅ Loaded {len(summaries):,} case summaries")
    return summaries

def prepare_corpus(cases):
    """
    Prepare corpus for embedding.
    Each case becomes a searchable document with:
    - English summary (main search content)
    - Metadata (case number, type, court, parties)
    """
    corpus = []
    metadata = []
    
    print("📝 Preparing corpus...")
    
    for case in tqdm(cases):
        # Get case information
        filename = case.get('filename', 'Unknown')
        case_num_en = case.get('case_number_english', '')
        case_type_en = case.get('case_type_english', '')
        court_en = case.get('court_english', '')
        summary = case.get('summary', '')
        
        # Skip if no summary
        if not summary or summary.strip() == '':
            continue
        
        # Build searchable text (summary + metadata)
        searchable_text = f"{summary}"
        
        # Add metadata context for better search
        if case_num_en:
            searchable_text = f"Case Number: {case_num_en}\n{searchable_text}"
        if case_type_en:
            searchable_text = f"Case Type: {case_type_en}\n{searchable_text}"
        if court_en:
            searchable_text = f"Court: {court_en}\n{searchable_text}"
        
        corpus.append(searchable_text)
        
        # Store metadata for retrieval
        metadata.append({
            'filename': filename,
            'case_number_nepali': case.get('case_number_nepali', ''),
            'case_number_english': case_num_en,
            'case_type_nepali': case.get('case_type_nepali', ''),
            'case_type_english': case_type_en,
            'court_nepali': case.get('court_nepali', ''),
            'court_english': court_en,
            'summary': summary[:500] + '...' if len(summary) > 500 else summary
        })
    
    print(f"✅ Prepared {len(corpus):,} documents")
    return corpus, metadata

def mean_pool(last_hidden_state, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts

def encode_corpus(corpus, batch_size=16):
    """
    Encode corpus into embeddings using multilingual E5 model.
    
    E5 model requires 'passage: ' prefix for documents.
    """
    print(f"🤖 Loading model: {MODEL_ID}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    
    embeddings = []
    
    print(f"⚡ Encoding {len(corpus):,} documents...")
    
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch = corpus[i:i+batch_size]
        
        # E5 model requires 'passage: ' prefix for documents
        batch_with_prefix = [f"passage: {text}" for text in batch]
        
        # Tokenize
        inputs = tokenizer(
            batch_with_prefix,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling
        batch_embeddings = mean_pool(
            outputs.last_hidden_state,
            inputs["attention_mask"]
        ).detach().cpu().numpy()
        
        # Normalize for cosine similarity (inner product search)
        faiss.normalize_L2(batch_embeddings)
        
        embeddings.append(batch_embeddings.astype('float32'))
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings, axis=0)
    
    print(f"✅ Generated embeddings: {all_embeddings.shape}")
    return all_embeddings

def build_faiss_index(embeddings):
    """
    Build FAISS index for fast similarity search.
    Uses IndexFlatIP (inner product) for normalized vectors = cosine similarity.
    """
    print("🔨 Building FAISS index...")
    
    d = embeddings.shape[1]  # Dimension (768 for E5-base)
    
    # IndexFlatIP for inner product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    print(f"✅ Index built: {index.ntotal:,} vectors, {d} dimensions")
    return index

def save_artifacts(index, corpus, metadata):
    """Save index, corpus, and metadata."""
    print(f"💾 Saving artifacts to {OUTPUT_DIR}/")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save FAISS index
    index_path = OUTPUT_DIR / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"   ✓ Saved: {index_path}")
    
    # Save corpus
    corpus_path = OUTPUT_DIR / "corpus.json"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"   ✓ Saved: {corpus_path}")
    
    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"   ✓ Saved: {metadata_path}")
    
    # Save config
    config = {
        'model_id': MODEL_ID,
        'num_documents': len(corpus),
        'embedding_dim': index.d,
        'index_type': 'IndexFlatIP',
        'normalized': True
    }
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"   ✓ Saved: {config_path}")

def main():
    print("=" * 60)
    print("🔍 SEMANTIC SEARCH INDEX BUILDER")
    print("=" * 60)
    print()
    
    # Check if input file exists
    if not SUMMARIES_FILE.exists():
        print(f"❌ Error: {SUMMARIES_FILE} not found!")
        print(f"   Run 05_llm_extraction.py first.")
        return
    
    # Load data
    cases = load_case_data()
    
    # Prepare corpus
    corpus, metadata = prepare_corpus(cases)
    
    if len(corpus) == 0:
        print("❌ No valid documents to index!")
        return
    
    # Generate embeddings
    embeddings = encode_corpus(corpus)
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Save everything
    save_artifacts(index, corpus, metadata)
    
    print()
    print("=" * 60)
    print("✅ SEMANTIC INDEX COMPLETE!")
    print("=" * 60)
    print()
    print(f"📊 Statistics:")
    print(f"   • Total documents indexed: {len(corpus):,}")
    print(f"   • Embedding dimensions: {embeddings.shape[1]}")
    print(f"   • Model: {MODEL_ID}")
    print(f"   • Index size: {index.ntotal:,} vectors")
    print()
    print("📁 Output files:")
    print(f"   • {OUTPUT_DIR}/index.faiss")
    print(f"   • {OUTPUT_DIR}/corpus.json")
    print(f"   • {OUTPUT_DIR}/metadata.json")
    print(f"   • {OUTPUT_DIR}/config.json")
    print()
    print("🚀 Next step: Run 07_search.py to test semantic search!")
    print()

if __name__ == "__main__":
    main()

