#!/usr/bin/env python3
"""
Simple Metrics Test
Focused on core functionality without complex dependencies
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

def test_router_only():
    """Test only the router functionality"""
    print("🧠 Testing Router (LLM-based routing)...")
    
    test_cases = [
        ("What is Article 12?", "knowledge_base"),
        ("Find property cases", "case_search"),
        ("What should I do about citizenship?", "hybrid"),
        ("Recent legal news", "web_search"),
        ("Explain the Civil Code", "knowledge_base")
    ]
    
    try:
        from agent.llm_client import LLMClient
        from agent.intelligent_router import IntelligentRouter
        
        llm = LLMClient()
        router = IntelligentRouter(llm)
        
        correct = 0
        total = len(test_cases)
        response_times = []
        
        for query, expected in test_cases:
            start_time = time.time()
            try:
                result = router.route_query(query)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                predicted = result["primary_tool"]
                confidence = result.get("confidence", "unknown")
                
                if predicted == expected:
                    correct += 1
                    print(f"   ✅ {query} → {predicted} ({confidence})")
                else:
                    print(f"   ❌ {query} → {predicted} (expected: {expected})")
                    
            except Exception as e:
                print(f"   ⚠️ {query} → Error: {e}")
                response_times.append(0)
        
        accuracy = correct / total
        avg_time = sum(response_times) / len(response_times) if response_times else 0
        
        print(f"   📊 Router Accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"   ⏱️ Average Response Time: {avg_time:.2f}s")
        
        return {
            "accuracy": accuracy,
            "avg_response_time": avg_time,
            "total_tests": total,
            "correct_predictions": correct
        }
        
    except Exception as e:
        print(f"   ❌ Router test failed: {e}")
        return {"accuracy": 0, "avg_response_time": 0, "error": str(e)}

def test_system_components():
    """Test system components availability"""
    print("🔧 Testing System Components...")
    
    components = {
        "llm_client": False,
        "search_index": False,
        "qdrant_storage": False,
        "legal_documents": False,
        "case_files": False
    }
    
    # Test LLM Client
    try:
        from agent.llm_client import LLMClient
        llm = LLMClient()
        components["llm_client"] = llm.is_available()
        print(f"   {'✅' if components['llm_client'] else '❌'} LLM Client: {llm.get_status()}")
    except Exception as e:
        print(f"   ❌ LLM Client: {e}")
    
    # Test Search Index
    try:
        index_dir = Path(r"D:/FinalAIproj/Wakalat Sewa/wakalt/tozip/search_index")
        if (index_dir / "index.faiss").exists() and (index_dir / "metadata.json").exists():
            with open(index_dir / "metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            components["search_index"] = len(metadata) > 0
            print(f"   {'✅' if components['search_index'] else '❌'} Search Index: {len(metadata):,} cases")
        else:
            print("   ❌ Search Index: Files not found")
    except Exception as e:
        print(f"   ❌ Search Index: {e}")
    
    # Test Qdrant Storage
    try:
        from agent.qdrant_kb import QdrantKnowledgeBase
        qdrant_kb = QdrantKnowledgeBase()
        info = qdrant_kb.get_collection_info()
        vector_count = info.get("vector_count", 0)
        components["qdrant_storage"] = vector_count > 0
        print(f"   {'✅' if components['qdrant_storage'] else '❌'} Qdrant Storage: {vector_count:,} vectors")
    except Exception as e:
        print(f"   ❌ Qdrant Storage: {e}")
    
    # Test Legal Documents
    try:
        kb_path = Path(r"D:/FinalAIproj/Wakalat Sewa/Wakalt/tozip/global_knowledge_base")
        if kb_path.exists():
            pdf_files = list(kb_path.glob("*.pdf"))
            components["legal_documents"] = len(pdf_files) > 0
            print(f"   {'✅' if components['legal_documents'] else '❌'} Legal Documents: {len(pdf_files)} PDFs")
        else:
            print("   ❌ Legal Documents: Directory not found")
    except Exception as e:
        print(f"   ❌ Legal Documents: {e}")
    
    # Test Case Files
    try:
        case_path = Path(r"D:/FinalAIproj/Wakalat Sewa/CaseFiles/CaseFiles")
        if case_path.exists():
            case_files = list(case_path.glob("*.txt"))
            components["case_files"] = len(case_files) > 0
            print(f"   {'✅' if components['case_files'] else '❌'} Case Files: {len(case_files):,} files")
        else:
            print("   ❌ Case Files: Directory not found")
    except Exception as e:
        print(f"   ❌ Case Files: {e}")
    
    # Calculate component health
    healthy_components = sum(components.values())
    total_components = len(components)
    health_score = healthy_components / total_components
    
    print(f"   📊 Component Health: {health_score:.1%} ({healthy_components}/{total_components})")
    
    return {
        "components": components,
        "health_score": health_score,
        "healthy_count": healthy_components,
        "total_count": total_components
    }

def test_basic_search():
    """Test basic search functionality"""
    print("🔍 Testing Basic Search...")
    
    try:
        # Test FAISS search directly
        import faiss
        import torch
        from transformers import AutoTokenizer, AutoModel
        import json
        
        # Load search index
        index_dir = Path(r"D:/FinalAIproj/Wakalat Sewa/wakalt/tozip/search_index")
        index_file = index_dir / "index.faiss"
        metadata_file = index_dir / "metadata.json"
        config_file = index_dir / "config.json"
        
        if not all([index_file.exists(), metadata_file.exists(), config_file.exists()]):
            print("   ❌ Search index files not found")
            return {"status": "not_found"}
        
        # Load components
        search_index = faiss.read_index(str(index_file))
        # Ensure index is on CPU
        if hasattr(search_index, 'is_trained') and search_index.is_trained:
            # Move to CPU if it's on GPU
            if hasattr(search_index, 'gpu_index'):
                search_index = faiss.index_gpu_to_cpu(search_index)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Initialize model (force CPU to avoid device mismatch)
        device = torch.device("cpu")  # Force CPU to avoid CUDA issues
        model_id = config.get('model_id', 'intfloat/multilingual-e5-base')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        
        # Test search
        test_query = "property dispute"
        print(f"   🔍 Testing query: '{test_query}'")
        
        # Encode query
        inputs = tokenizer(test_query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Search
        start_time = time.time()
        scores, indices = search_index.search(query_embedding, k=5)
        search_time = time.time() - start_time
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(metadata):
                case = metadata[idx]
                results.append({
                    "rank": i + 1,
                    "score": float(score),
                    "case_number": case.get('case_number_english', 'Unknown'),
                    "title": case.get('title_english', 'Unknown')[:100]
                })
        
        print(f"   ✅ Search completed in {search_time:.3f}s")
        print(f"   📊 Found {len(results)} results")
        for result in results[:3]:
            print(f"      {result['rank']}. {result['case_number']} (score: {result['score']:.3f})")
        
        return {
            "status": "success",
            "search_time": search_time,
            "results_count": len(results),
            "top_score": results[0]["score"] if results else 0
        }
        
    except Exception as e:
        print(f"   ❌ Search test failed: {e}")
        return {"status": "error", "error": str(e)}

def test_qdrant_search():
    """Test Qdrant search functionality"""
    print("🗄️ Testing Qdrant Search...")
    
    try:
        from agent.qdrant_kb import QdrantKnowledgeBase
        qdrant_kb = QdrantKnowledgeBase()
        
        # Test search
        test_query = "constitutional rights"
        print(f"   🔍 Testing query: '{test_query}'")
        
        start_time = time.time()
        results = qdrant_kb.search(test_query, limit=3)
        search_time = time.time() - start_time
        
        print(f"   ✅ Qdrant search completed in {search_time:.3f}s")
        print(f"   📊 Found {len(results)} results")
        
        for i, result in enumerate(results[:3], 1):
            source = result.get('source', 'Unknown')
            page = result.get('page', 'N/A')
            score = result.get('score', 0)
            print(f"      {i}. {source} (Page {page}, score: {score:.3f})")
        
        return {
            "status": "success",
            "search_time": search_time,
            "results_count": len(results),
            "top_score": results[0]["score"] if results else 0
        }
        
    except Exception as e:
        print(f"   ❌ Qdrant search failed: {e}")
        return {"status": "error", "error": str(e)}

def main():
    """Run simple metrics test"""
    print("🚀 Wakalat Sewa V2 - Simple Metrics Test")
    print("=" * 60)
    
    results = {}
    
    # Test router
    results["router"] = test_router_only()
    print()
    
    # Test system components
    results["components"] = test_system_components()
    print()
    
    # Test basic search
    results["search"] = test_basic_search()
    print()
    
    # Test Qdrant search
    results["qdrant"] = test_qdrant_search()
    print()
    
    # Calculate overall score
    router_score = results["router"].get("accuracy", 0)
    component_score = results["components"].get("health_score", 0)
    search_score = 1.0 if results["search"].get("status") == "success" else 0.0
    qdrant_score = 1.0 if results["qdrant"].get("status") == "success" else 0.0
    
    overall_score = (router_score + component_score + search_score + qdrant_score) / 4
    
    # Summary
    print("=" * 60)
    print("📊 SIMPLE METRICS SUMMARY")
    print("=" * 60)
    
    print(f"🧠 Router Accuracy: {router_score:.1%}")
    print(f"🔧 Component Health: {component_score:.1%}")
    print(f"🔍 Search Functionality: {'✅ Working' if search_score > 0 else '❌ Failed'}")
    print(f"🗄️ Qdrant Search: {'✅ Working' if qdrant_score > 0 else '❌ Failed'}")
    print(f"🎯 Overall Score: {overall_score:.1%}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if router_score < 0.9:
        print("   🔧 Router accuracy needs improvement")
    
    if component_score < 0.8:
        print("   🔧 Some system components need attention")
    
    if search_score == 0:
        print("   🔧 Search functionality needs to be fixed")
    
    if qdrant_score == 0:
        print("   🔧 Qdrant search needs to be fixed")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "overall_score": overall_score,
        "results": results
    }
    
    output_file = Path(__file__).parent / "simple_metrics_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("✅ Simple metrics test completed!")

if __name__ == "__main__":
    main()
