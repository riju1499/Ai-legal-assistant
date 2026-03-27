from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import faiss
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import logging

# Import AI agent modules
from agent.llm_client import LLMClient
from agent.tools import AgentTools
from agent.rag_pipeline import RAGPipeline
from agent.agent_graph import LegalAgentGraph
from agent.qdrant_kb import QdrantKnowledgeBase
from agent.intelligent_tools import IntelligentTools
from agent.intelligent_router import IntelligentRouter
from agent.response_synthesizer import ResponseSynthesizer
from agent.strategy_generator import StrategyGenerator
from agent.agentic_tools import AgenticTools
from agent.strategy_agentic import StrategyAgenticAgent

import torch
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))
    print("Memory allocated:", torch.cuda.memory_allocated(0))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wakalat Sewa API V2",
    description="Semantic search and AI chatbot API for Nepali legal cases",
    version="2.0.0"
)

# CORS middleware - allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to search index
INDEX_DIR = Path("../search_index")
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"
CONFIG_FILE = INDEX_DIR / "config.json"

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

class CaseResult(BaseModel):
    rank: int
    score: float
    index: int  # Index in metadata for detail view
    case_number_english: Optional[str] = ""
    case_number_nepali: Optional[str] = ""
    case_type_english: Optional[str] = ""
    case_type_nepali: Optional[str] = ""
    court_english: Optional[str] = ""
    court_nepali: Optional[str] = ""
    summary: Optional[str] = ""
    filename: Optional[str] = ""

class SearchResponse(BaseModel):
    query: str
    results: List[CaseResult]
    total_found: int

# V2: Chat request/response models
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    conversation_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    query_type: str
    similar_cases: List[Dict[str, Any]]
    causal_explanation: str
    disclaimer: str

class CaseAnalysisRequest(BaseModel):
    situation: str
    include_loopholes: Optional[bool] = True

class StrategyRequest(BaseModel):
    case_facts: str
    desired_outcome: Optional[str] = None
    case_type: Optional[str] = None
    jurisdiction: Optional[str] = "Nepal"
    include_retrieval: Optional[bool] = True

class StrategyResponse(BaseModel):
    strategy: Dict[str, Any]

# Global model variables
search_engine = None
llm_client = None
agent_tools = None
rag_pipeline = None
legal_agent = None
qdrant_kb = None
strategy_generator = None
agentic_tools = None
strategy_agentic_agent = None

class SemanticSearchEngine:
    """Semantic search engine for legal cases"""
    
    def __init__(self):
        print("🔍 Initializing Semantic Search Engine...")
        
        # Load config
        with open(CONFIG_FILE, 'r') as f:
            self.config = json.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(str(INDEX_FILE))
        print(f"   ✓ Loaded index: {self.index.ntotal:,} documents")
        
        # Load metadata
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"   ✓ Loaded metadata: {len(self.metadata):,} cases")
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = self.config.get('model_id', 'intfloat/multilingual-e5-base')
        print(f"   ✓ Loading model: {model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        print(f"✅ Search engine ready on {self.device}!\n")
    
    def mean_pool(self, last_hidden_state, attention_mask):
        """Mean pooling for embeddings"""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts
    
    def encode_query(self, query: str):
        """Encode search query into embedding"""
        query_with_prefix = f"query: {query}"
        
        inputs = self.tokenizer(
            [query_with_prefix],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embedding = self.mean_pool(
            outputs.last_hidden_state,
            inputs["attention_mask"]
        ).detach().cpu().numpy()
        
        faiss.normalize_L2(embedding)
        return embedding.astype('float32')
    
    def search(self, query: str, k: int = 5):
        """Search for top-k most similar cases"""
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Helper function to safely convert to string
        def safe_str(value):
            if value is None:
                return ""
            if isinstance(value, list):
                # Join list items or take first item
                return ", ".join(str(v) for v in value if v) if value else ""
            return str(value)
        
        # Prepare results
        results = []
        rank = 1
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.metadata):
                meta = self.metadata[idx]
                summary = safe_str(meta.get('summary', ''))
                
                # Skip cases with failed summaries or empty summaries
                if not summary or summary == "Summary generation failed":
                    continue
                
                results.append({
                    'rank': rank,
                    'score': float(score),
                    'index': int(idx),  # Add index for detail view
                    'case_number_english': safe_str(meta.get('case_number_english', '')),
                    'case_number_nepali': safe_str(meta.get('case_number_nepali', '')),
                    'case_type_english': safe_str(meta.get('case_type_english', '')),
                    'case_type_nepali': safe_str(meta.get('case_type_nepali', '')),
                    'court_english': safe_str(meta.get('court_english', '')),
                    'court_nepali': safe_str(meta.get('court_nepali', '')),
                    'summary': summary,
                    'filename': safe_str(meta.get('filename', ''))
                })
                rank += 1
        
        return results

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize search engine and AI agent on startup"""
    global search_engine, llm_client, agent_tools, rag_pipeline, legal_agent, qdrant_kb
    global intelligent_tools, router, synthesizer, strategy_generator
    global agentic_tools, strategy_agentic_agent
    
    try:
        # Initialize search engine (V1)
        search_engine = SemanticSearchEngine()
        
        # Initialize AI agent components (V2)
        logger.info("🤖 Initializing Intelligent Legal Agent...")
        
        # LLM Client
        llm_client = LLMClient()
        llm_status = llm_client.get_status()
        logger.info(f"   LLM Status: {llm_status}")
        
        if not llm_client.is_available():
            logger.warning("   ⚠️  No LLM available. Chat functionality will be limited.")
        
        # Initialize Qdrant Knowledge Base
        try:
            qdrant_path = Path("../qdrant_storage")
            logger.info(f"   Checking for Qdrant at: {qdrant_path.absolute()}")
            
            if qdrant_path.exists():
                logger.info("   📚 Loading Qdrant knowledge base...")
                qdrant_kb = QdrantKnowledgeBase(
                    collection_name="legal_knowledge",
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                    qdrant_path=str(qdrant_path)
                )
                kb_info = qdrant_kb.get_collection_info()
                vector_count = kb_info.get('vector_count', 0)
                
                if vector_count > 0:
                    logger.info(f"   ✅ Qdrant KB loaded: {vector_count:,} knowledge chunks")
                else:
                    logger.error(f"   ❌ Qdrant collection is EMPTY! Run scripts/09_build_qdrant_knowledge.py")
                    qdrant_kb = None
            else:
                logger.error(f"   ❌ Qdrant storage not found at: {qdrant_path.absolute()}")
                logger.error("   📖 Run: cd scripts && python 09_build_qdrant_knowledge.py")
                qdrant_kb = None
        except Exception as e:
            logger.error(f"   ❌ Could not load Qdrant KB: {e}", exc_info=True)
            qdrant_kb = None
        
        # Agent Tools (legacy)
        glossary_dir = Path("../glossary")
        agent_tools = AgentTools(
            search_engine=search_engine,
            metadata=search_engine.metadata,
            glossary_dir=glossary_dir,
            qdrant_kb=qdrant_kb
        )
        logger.info("   ✓ Agent tools initialized")
        
        # RAG Pipeline (legacy, for backward compatibility)
        rag_pipeline = RAGPipeline(llm_client, agent_tools)
        logger.info("   ✓ RAG pipeline initialized")
        
        # NEW: Strategy Generator initialization (AI-driven, no static knowledge files)
        try:
            strategy_generator = StrategyGenerator(llm_client=llm_client)
            logger.info("   ✓ Strategy generator initialized")
        except Exception as e:
            logger.error(f"   ❌ Strategy generator init failed: {e}")
            strategy_generator = None

        # NEW: Intelligent Tools
        intelligent_tools = IntelligentTools(
            agent_tools=agent_tools,
            qdrant_kb=qdrant_kb,
            web_search_enabled=False,  # Enable when web search is implemented
            strategy_generator=strategy_generator
        )
        logger.info("   ✓ Intelligent tools initialized")
        
        # NEW: Intelligent Router
        router = IntelligentRouter(llm_client)
        logger.info("   ✓ Intelligent router initialized")
        
        # NEW: Response Synthesizer
        synthesizer = ResponseSynthesizer(llm_client)
        logger.info("   ✓ Response synthesizer initialized")
        
        # NEW: Intelligent Agent Graph
        legal_agent = LegalAgentGraph(
            rag_pipeline=rag_pipeline,
            intelligent_tools=intelligent_tools,
            router=router,
            synthesizer=synthesizer
        )
        logger.info("   ✓ Intelligent agent graph initialized")
        
        # NEW: Agentic Strategy System (for case fighting)
        try:
            agentic_tools = AgenticTools(
                intelligent_tools=intelligent_tools,
                agent_tools=agent_tools,
                strategy_generator=strategy_generator
            )
            strategy_agentic_agent = StrategyAgenticAgent(
                agentic_tools=agentic_tools,
                llm_client=llm_client,
                strategy_generator=strategy_generator,
                max_iterations=5
            )
            logger.info("   ✓ Agentic strategy agent initialized")
        except Exception as e:
            logger.error(f"   ❌ Agentic strategy agent init failed: {e}")
            agentic_tools = None
            strategy_agentic_agent = None

        logger.info("✅ Intelligent Legal Agent ready!\n")
        
    except Exception as e:
        logger.error(f"❌ Error during initialization: {e}")
        logger.info("   The API will start but some features may not work.")
        search_engine = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Wakalat Sewa API",
        "status": "online",
        "search_available": search_engine is not None,
        "total_cases": len(search_engine.metadata) if search_engine else 0
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    return {
        "status": "healthy",
        "total_cases": len(search_engine.metadata),
        "index_size": search_engine.index.ntotal,
        "device": str(search_engine.device)
    }

@app.post("/search", response_model=SearchResponse)
async def search_cases(request: SearchRequest):
    """
    Search legal cases using semantic similarity
    
    - **query**: Search query in English or Nepali
    - **limit**: Number of results to return (default: 5, max: 20)
    """
    if search_engine is None:
        raise HTTPException(
            status_code=503, 
            detail="Search engine not initialized. Please contact administrator."
        )
    
    if not request.query or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Limit results to max 20
    limit = min(request.limit, 20)
    
    try:
        # Fetch extra results to account for filtering out failed summaries
        # Request 2x the limit to ensure we get enough valid results
        fetch_limit = min(limit * 2, 50)
        results = search_engine.search(request.query, k=fetch_limit)
        
        # Trim to requested limit
        results = results[:limit]
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )

@app.get("/case/{case_index}")
async def get_case_detail(case_index: int):
    """
    Get full details of a specific case by index
    
    - **case_index**: Index of the case in the metadata array
    """
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    if case_index < 0 or case_index >= len(search_engine.metadata):
        raise HTTPException(status_code=404, detail="Case not found")
    
    def safe_str(value):
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if v) if value else ""
        return str(value)
    
    meta = search_engine.metadata[case_index]
    
    return {
        "index": case_index,
        "case_number_english": safe_str(meta.get('case_number_english', '')),
        "case_number_nepali": safe_str(meta.get('case_number_nepali', '')),
        "case_type_english": safe_str(meta.get('case_type_english', '')),
        "case_type_nepali": safe_str(meta.get('case_type_nepali', '')),
        "court_english": safe_str(meta.get('court_english', '')),
        "court_nepali": safe_str(meta.get('court_nepali', '')),
        "summary": safe_str(meta.get('summary', '')),
        "filename": safe_str(meta.get('filename', ''))
    }

@app.get("/download/{filename}")
async def download_case_file(filename: str):
    """
    Download the original case file
    
    - **filename**: Name of the case file to download
    """
    from urllib.parse import quote
    
    # Path to CaseFiles directory
    case_files_dir = Path("../CaseFiles")
    
    # Search for the file recursively
    file_path = None
    for root, dirs, files in os.walk(case_files_dir):
        if filename in files:
            file_path = Path(root) / filename
            break
    
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Case file not found")
    
    # Encode filename for Content-Disposition header (RFC 5987)
    # This handles Unicode characters properly
    encoded_filename = quote(filename)
    
    # Return the file for download
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='text/plain; charset=utf-8',
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )

@app.get("/stats")
async def get_statistics():
    """Get database statistics"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    def safe_str(value):
        """Convert value to string, handling lists and None"""
        if value is None:
            return "Unknown"
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if v) if value else "Unknown"
        return str(value) if value else "Unknown"
    
    # Count case types
    case_types = {}
    courts = {}
    
    for case in search_engine.metadata:
        case_type = safe_str(case.get('case_type_english', 'Unknown'))
        court = safe_str(case.get('court_english', 'Unknown'))
        
        case_types[case_type] = case_types.get(case_type, 0) + 1
        courts[court] = courts.get(court, 0) + 1
    
    return {
        "total_cases": len(search_engine.metadata),
        "case_types": dict(sorted(case_types.items(), key=lambda x: x[1], reverse=True)[:10]),
        "courts": dict(sorted(courts.items(), key=lambda x: x[1], reverse=True)),
        "model": search_engine.config.get('model_id'),
        "embedding_dim": search_engine.config.get('embedding_dim')
    }

# ============================================
# V2: AI CHATBOT ENDPOINTS
# ============================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    AI legal chatbot endpoint with conversation memory
    
    - **message**: User's legal question or query
    - **conversation_id**: Optional conversation ID for tracking
    - **conversation_history**: Previous messages for context
    
    Returns AI-generated legal information with causal analysis and case references
    """
    if legal_agent is None or not llm_client.is_available():
        raise HTTPException(
            status_code=503,
            detail="AI chatbot is not available. Please ensure GOOGLE_API_KEY is configured or Ollama is running."
        )
    
    if not request.message or request.message.strip() == "":
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Build conversation context from history
        conversation_context = []
        if request.conversation_history:
            # Take last 5 exchanges (10 messages) to keep context manageable
            recent_history = request.conversation_history[-10:] if len(request.conversation_history) > 10 else request.conversation_history
            conversation_context = [
                {"role": msg.role, "content": msg.content[:500]}  # Truncate long messages
                for msg in recent_history
            ]
        
        # Run agent workflow with conversation context
        response = legal_agent.run(
            query=request.message,
            conversation_history=conversation_context
        )
        
        return ChatResponse(
            response=response.get('response', ''),
            query_type=response.get('query_type', ''),
            similar_cases=response.get('similar_cases', []),
            causal_explanation=response.get('causal_explanation', ''),
            disclaimer=response.get('disclaimer', '')
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing error: {str(e)}"
        )

@app.post("/chat/analyze-case")
async def analyze_case_for_loopholes(request: CaseAnalysisRequest):
    """
    Detailed legal case analysis with loophole detection
    
    - **situation**: Description of the user's legal situation
    - **include_loopholes**: Whether to include loophole analysis (default: true)
    
    Returns comprehensive analysis with potential legal arguments and case references
    """
    if rag_pipeline is None or not llm_client.is_available():
        raise HTTPException(
            status_code=503,
            detail="AI analysis is not available. Please ensure GOOGLE_API_KEY is configured or Ollama is running."
        )
    
    if not request.situation or request.situation.strip() == "":
        raise HTTPException(status_code=400, detail="Situation description cannot be empty")
    
    try:
        # Process as loophole analysis query
        query_enhanced = f"Analyze this legal situation and identify potential legal arguments: {request.situation}"
        
        response = rag_pipeline.process_query(
            query_enhanced,
            include_similar_cases=True
        )
        
        return {
            "analysis": response.get('response', ''),
            "similar_cases": response.get('similar_cases', []),
            "causal_patterns": response.get('causal_explanation', ''),
            "disclaimer": response.get('disclaimer', ''),
            "query_type": "LOOPHOLE_ANALYSIS"
        }
    
    except Exception as e:
        logger.error(f"Case analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )

@app.get("/chat/status")
async def get_chat_status():
    """Get AI chatbot system status"""
    if llm_client is None:
        return {
            "available": False,
            "message": "AI agent not initialized"
        }
    
    status = llm_client.get_status()
    return {
        "available": status.get('any_available', False),
        "gemini_available": status.get('gemini', {}).get('available', False),
        "ollama_available": status.get('ollama', {}).get('available', False),
        "agent_ready": legal_agent is not None,
        "agentic_strategy_ready": strategy_agentic_agent is not None
    }


# ============================================
# Strategy Generation Endpoint (All Case Types)
# ============================================

@app.post("/strategy", response_model=StrategyResponse)
async def generate_strategy(request: StrategyRequest):
    """
    Generate an AI-driven, precedent-backed legal strategy using agentic reasoning.
    - Requires: case_facts (free text). Optional: desired_outcome, case_type, include_retrieval.
    - Uses agentic AI with iterative reasoning to gather information and generate comprehensive strategy.
    """
    if strategy_agentic_agent is None or not llm_client or not llm_client.is_available():
        raise HTTPException(status_code=503, detail="Agentic strategy generator not available. Please ensure GOOGLE_API_KEY is configured or Ollama is running.")

    if not request.case_facts or request.case_facts.strip() == "":
        raise HTTPException(status_code=400, detail="case_facts cannot be empty")

    try:
        # Use agentic strategy agent for iterative reasoning
        strategy = strategy_agentic_agent.generate_strategy(
            case_facts=request.case_facts,
            desired_outcome=request.desired_outcome,
            case_type_hint=request.case_type
        )

        return {"strategy": strategy}
    
    except Exception as e:
        logger.error(f"Strategy generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Strategy generation error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

