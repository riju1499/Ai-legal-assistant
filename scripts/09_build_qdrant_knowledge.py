#!/usr/bin/env python3
"""
Build Qdrant vector database from legal PDF knowledge base
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from agent.qdrant_kb import QdrantKnowledgeBase
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Build Qdrant knowledge base from PDFs"""
    
    print("=" * 70)
    print("Wakalat Sewa - Qdrant Knowledge Base Builder")
    print("=" * 70)
    print()
    
    # Paths
    pdf_dir = Path("../global_knowledge_base")
    qdrant_path = Path("../qdrant_storage")
    
    print(f"📂 PDF Directory: {pdf_dir.absolute()}")
    print(f"💾 Qdrant Storage: {qdrant_path.absolute()}")
    print()
    
    # Check if PDFs exist
    if not pdf_dir.exists():
        print(f"❌ Error: PDF directory not found at {pdf_dir}")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    print()
    
    # Ask user for confirmation
    choice = input("Do you want to proceed with building the knowledge base? (y/n) [y]: ").strip().lower()
    if choice and choice != 'y':
        print("Cancelled by user")
        return
    
    print("\n" + "=" * 70)
    print("Building Qdrant Knowledge Base")
    print("=" * 70)
    print()
    
    try:
        # Initialize Qdrant KB
        kb = QdrantKnowledgeBase(
            collection_name="legal_knowledge",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            qdrant_path=str(qdrant_path)
        )
        
        # Check if collection has data
        info = kb.get_collection_info()
        if info and info.get('vector_count', 0) > 0:
            print(f"⚠️  Collection already contains {info['vector_count']} vectors")
            rebuild = input("Rebuild from scratch? (y/n) [n]: ").strip().lower()
            if rebuild == 'y':
                kb.clear_collection()
                print("✓ Collection cleared\n")
        
        # Ingest PDFs
        print("📄 Processing PDFs...")
        print()
        
        results = kb.ingest_directory(pdf_dir, chunk_size=1000)
        
        # Print results
        print("\n" + "=" * 70)
        print("Ingestion Complete")
        print("=" * 70)
        print()
        
        total_chunks = 0
        for filename, count in results.items():
            print(f"  {filename}: {count:,} chunks")
            total_chunks += count
        
        print()
        print(f"✅ Total: {total_chunks:,} knowledge chunks indexed")
        
        # Get final collection info
        final_info = kb.get_collection_info()
        if final_info:
            print(f"📊 Collection size: {final_info.get('vector_count', 0):,} vectors")
        
        print()
        print("=" * 70)
        print("Testing Search")
        print("=" * 70)
        print()
        
        # Test search
        test_queries = [
            "What is Article 12 of Constitution?",
            "Electronic transactions",
            "Criminal procedure"
        ]
        
        for query in test_queries:
            print(f"Query: '{query}'")
            results = kb.search(query, limit=2, score_threshold=0.3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Source: {result['source']} (Page {result['page']}, Score: {result['score']:.3f})")
                    print(f"     Text: {result['text'][:150]}...")
            else:
                print("  No results found")
            print()
        
        print("=" * 70)
        print("✅ Qdrant Knowledge Base Ready!")
        print("=" * 70)
        print()
        print("The chatbot now has access to:")
        for pdf in pdf_files:
            print(f"  ✓ {pdf.name}")
        print()
        print("You can now start the backend and use the AI chatbot!")
        
    except Exception as e:
        logger.error(f"Failed to build knowledge base: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

