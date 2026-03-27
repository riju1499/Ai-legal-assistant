#!/usr/bin/env python3
"""
System Status Check
Quick health check without requiring full backend initialization
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

def check_file_system():
    """Check if required files and directories exist"""
    print("📁 Checking File System...")
    
    required_paths = {
        "Search Index": Path("../search_index"),
        "Qdrant Storage": Path("../qdrant_storage"),
        "Glossary": Path("../glossary"),
        "Case Files": Path("../CaseFiles"),
        "Global Knowledge Base": Path("../global_knowledge_base")
    }
    
    status = {}
    for name, path in required_paths.items():
        exists = path.exists()
        status[name] = exists
        print(f"   {'✅' if exists else '❌'} {name}: {path}")
    
    return status

def check_search_index():
    """Check search index status"""
    print("🔍 Checking Search Index...")
    
    try:
        index_dir = Path("../search_index")
        required_files = ["index.faiss", "metadata.json", "config.json"]
        
        all_exist = all((index_dir / f).exists() for f in required_files)
        
        if all_exist:
            # Load metadata to get case count
            with open(index_dir / "metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            case_count = len(metadata)
            print(f"   ✅ Search index ready: {case_count:,} cases")
            return {"status": "ready", "case_count": case_count}
        else:
            missing = [f for f in required_files if not (index_dir / f).exists()]
            print(f"   ❌ Missing files: {missing}")
            return {"status": "incomplete", "missing": missing}
            
    except Exception as e:
        print(f"   ❌ Error checking search index: {e}")
        return {"status": "error", "error": str(e)}

def check_qdrant_storage():
    """Check Qdrant storage status"""
    print("🗄️ Checking Qdrant Storage...")
    
    try:
        qdrant_path = Path("../qdrant_storage")
        
        if not qdrant_path.exists():
            print("   ❌ Qdrant storage directory not found")
            return {"status": "not_found"}
        
        # Check if it's locked by another process
        try:
            from agent.qdrant_kb import QdrantKnowledgeBase
            qdrant_kb = QdrantKnowledgeBase()
            info = qdrant_kb.get_collection_info()
            vector_count = info.get("vector_count", 0)
            
            if vector_count > 0:
                print(f"   ✅ Qdrant ready: {vector_count:,} vectors")
                return {"status": "ready", "vector_count": vector_count}
            else:
                print("   ⚠️ Qdrant empty: No vectors found")
                return {"status": "empty"}
                
        except Exception as e:
            if "already accessed by another instance" in str(e):
                print("   ⚠️ Qdrant locked by another process")
                return {"status": "locked"}
            else:
                print(f"   ❌ Qdrant error: {e}")
                return {"status": "error", "error": str(e)}
                
    except Exception as e:
        print(f"   ❌ Error checking Qdrant: {e}")
        return {"status": "error", "error": str(e)}

def check_llm_availability():
    """Check LLM availability"""
    print("🤖 Checking LLM Availability...")
    
    try:
        from agent.llm_client import LLMClient
        llm = LLMClient()
        
        status = llm.get_status()
        print(f"   📊 LLM Status: {status}")
        
        if llm.is_available():
            print("   ✅ LLM available")
            return {"status": "available", "details": status}
        else:
            print("   ❌ LLM not available")
            return {"status": "unavailable", "details": status}
            
    except Exception as e:
        print(f"   ❌ Error checking LLM: {e}")
        return {"status": "error", "error": str(e)}

def check_legal_documents():
    """Check legal documents in knowledge base"""
    print("📚 Checking Legal Documents...")
    
    try:
        kb_path = Path("../global_knowledge_base")
        
        if not kb_path.exists():
            print("   ❌ Knowledge base directory not found")
            return {"status": "not_found"}
        
        # Check for PDF files
        pdf_files = list(kb_path.glob("*.pdf"))
        
        if pdf_files:
            print(f"   ✅ Found {len(pdf_files)} PDF files:")
            for pdf in pdf_files:
                size_mb = pdf.stat().st_size / (1024 * 1024)
                print(f"      - {pdf.name} ({size_mb:.1f} MB)")
            return {"status": "ready", "file_count": len(pdf_files), "files": [f.name for f in pdf_files]}
        else:
            print("   ❌ No PDF files found in knowledge base")
            return {"status": "empty"}
            
    except Exception as e:
        print(f"   ❌ Error checking documents: {e}")
        return {"status": "error", "error": str(e)}

def check_case_files():
    """Check case files"""
    print("⚖️ Checking Case Files...")
    
    try:
        case_path = Path("../CaseFiles")
        
        if not case_path.exists():
            print("   ❌ Case files directory not found")
            return {"status": "not_found"}
        
        # Count case files
        case_files = list(case_path.glob("*.txt"))
        
        if case_files:
            total_size = sum(f.stat().st_size for f in case_files) / (1024 * 1024 * 1024)  # GB
            print(f"   ✅ Found {len(case_files):,} case files ({total_size:.1f} GB)")
            return {"status": "ready", "file_count": len(case_files), "total_size_gb": total_size}
        else:
            print("   ❌ No case files found")
            return {"status": "empty"}
            
    except Exception as e:
        print(f"   ❌ Error checking case files: {e}")
        return {"status": "error", "error": str(e)}

def calculate_system_health(results):
    """Calculate overall system health score"""
    print("🏥 Calculating System Health...")
    
    # Weight different components
    weights = {
        "file_system": 0.1,
        "search_index": 0.25,
        "qdrant_storage": 0.25,
        "llm_availability": 0.2,
        "legal_documents": 0.1,
        "case_files": 0.1
    }
    
    scores = {}
    total_score = 0
    
    # File system score
    fs_status = results.get("file_system", {})
    fs_score = sum(fs_status.values()) / len(fs_status) if fs_status else 0
    scores["file_system"] = fs_score
    total_score += fs_score * weights["file_system"]
    
    # Search index score
    si_status = results.get("search_index", {})
    if si_status.get("status") == "ready":
        si_score = 1.0
    elif si_status.get("status") == "incomplete":
        si_score = 0.5
    else:
        si_score = 0.0
    scores["search_index"] = si_score
    total_score += si_score * weights["search_index"]
    
    # Qdrant score
    qdrant_status = results.get("qdrant_storage", {})
    if qdrant_status.get("status") == "ready":
        qdrant_score = 1.0
    elif qdrant_status.get("status") == "locked":
        qdrant_score = 0.7  # Partial credit if locked
    else:
        qdrant_score = 0.0
    scores["qdrant_storage"] = qdrant_score
    total_score += qdrant_score * weights["qdrant_storage"]
    
    # LLM score
    llm_status = results.get("llm_availability", {})
    if llm_status.get("status") == "available":
        llm_score = 1.0
    else:
        llm_score = 0.0
    scores["llm_availability"] = llm_score
    total_score += llm_score * weights["llm_availability"]
    
    # Documents score
    doc_status = results.get("legal_documents", {})
    if doc_status.get("status") == "ready":
        doc_score = 1.0
    else:
        doc_score = 0.0
    scores["legal_documents"] = doc_score
    total_score += doc_score * weights["legal_documents"]
    
    # Case files score
    case_status = results.get("case_files", {})
    if case_status.get("status") == "ready":
        case_score = 1.0
    else:
        case_score = 0.0
    scores["case_files"] = case_score
    total_score += case_score * weights["case_files"]
    
    print(f"   📊 Component Scores:")
    for component, score in scores.items():
        print(f"      {component}: {score:.1%}")
    
    print(f"   🎯 Overall Health Score: {total_score:.1%}")
    
    return {
        "overall_score": total_score,
        "component_scores": scores,
        "health_level": "excellent" if total_score >= 0.9 else "good" if total_score >= 0.7 else "fair" if total_score >= 0.5 else "poor"
    }

def main():
    """Run system status check"""
    print("🚀 Wakalat Sewa V2 - System Status Check")
    print("=" * 60)
    
    results = {}
    
    # Run all checks
    results["file_system"] = check_file_system()
    print()
    
    results["search_index"] = check_search_index()
    print()
    
    results["qdrant_storage"] = check_qdrant_storage()
    print()
    
    results["llm_availability"] = check_llm_availability()
    print()
    
    results["legal_documents"] = check_legal_documents()
    print()
    
    results["case_files"] = check_case_files()
    print()
    
    # Calculate overall health
    health = calculate_system_health(results)
    print()
    
    # Summary
    print("=" * 60)
    print("📊 SYSTEM STATUS SUMMARY")
    print("=" * 60)
    
    print(f"🏥 Overall Health: {health['overall_score']:.1%} ({health['health_level']})")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if results["search_index"].get("status") != "ready":
        print("   🔧 Build search index: python scripts/06_build_semantic_index.py")
    
    if results["qdrant_storage"].get("status") == "empty":
        print("   🔧 Build Qdrant knowledge base: python scripts/09_build_qdrant_knowledge.py")
    elif results["qdrant_storage"].get("status") == "locked":
        print("   🔧 Stop other Qdrant processes or restart system")
    
    if results["llm_availability"].get("status") != "available":
        print("   🔧 Start Ollama: ollama serve")
        print("   🔧 Pull model: ollama pull llama3.2")
    
    if results["legal_documents"].get("status") != "ready":
        print("   🔧 Add legal PDFs to global_knowledge_base/ directory")
    
    if results["case_files"].get("status") != "ready":
        print("   🔧 Add case files to CaseFiles/ directory")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "system_health": health,
        "component_results": results
    }
    
    output_file = Path(__file__).parent / "system_status_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("✅ System status check completed!")

if __name__ == "__main__":
    main()
