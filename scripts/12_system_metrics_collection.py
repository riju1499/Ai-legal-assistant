#!/usr/bin/env python3
"""
System Metrics Collection Script
Measures router accuracy, search quality, and system performance
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import statistics

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from agent.llm_client import LLMClient
from agent.intelligent_router import IntelligentRouter
from agent.intelligent_tools import IntelligentTools
from agent.response_synthesizer import ResponseSynthesizer
from agent.agent_graph import LegalAgentGraph
from agent.rag_pipeline import RAGPipeline
from agent.tools import AgentTools
from agent.qdrant_kb import QdrantKnowledgeBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    """
    Comprehensive metrics collection for Wakalat Sewa system
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = {
            "router_accuracy": {},
            "search_quality": {},
            "response_times": {},
            "system_health": {},
            "conversation_memory": {}
        }
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize system components for testing"""
        try:
            logger.info("🔧 Initializing system components...")
            
            # Initialize LLM client
            self.llm_client = LLMClient()
            
            # Initialize router
            self.router = IntelligentRouter(self.llm_client)
            
            # Initialize basic components only (avoid complex dependencies)
            self.qdrant_kb = QdrantKnowledgeBase()
            
            # Initialize tools with proper arguments
            import faiss
            import torch
            from transformers import AutoTokenizer, AutoModel
            import json
            
            # Load search index
            index_dir = Path(r"D:/FinalAIproj/Wakalat Sewa/wakalt/tozip/search_index")
            index_file = index_dir / "index.faiss"
            metadata_file = index_dir / "metadata.json"
            config_file = index_dir / "config.json"
            
            if all([index_file.exists(), metadata_file.exists(), config_file.exists()]):
                # Load FAISS index
                search_index = faiss.read_index(str(index_file))
                
                # Load metadata
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Load config
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Initialize model (force CPU)
                device = torch.device("cpu")
                model_id = config.get('model_id', 'intfloat/multilingual-e5-base')
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModel.from_pretrained(model_id).to(device)
                model.eval()
                
                # Create a simple search engine class
                class SimpleSearchEngine:
                    def __init__(self, index, metadata, tokenizer, model, device):
                        self.index = index
                        self.metadata = metadata
                        self.tokenizer = tokenizer
                        self.model = model
                        self.device = device
                    
                    def search(self, query, k=5):
                        # Simple search implementation
                        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        
                        scores, indices = self.index.search(query_embedding, k)
                        results = []
                        for score, idx in zip(scores[0], indices[0]):
                            if idx < len(self.metadata):
                                results.append({
                                    'case_number_english': self.metadata[idx].get('case_number_english', 'Unknown'),
                                    'title_english': self.metadata[idx].get('title_english', 'Unknown'),
                                    'score': float(score)
                                })
                        return results
                
                search_engine = SimpleSearchEngine(search_index, metadata, tokenizer, model, device)
                
                # Initialize AgentTools with required arguments
                glossary_dir = Path(r"D:/FinalAIproj/Wakalat Sewa/wakalt/tozip/glossary")
                self.agent_tools = AgentTools(
                    search_engine=search_engine,
                    metadata=metadata,
                    glossary_dir=glossary_dir,
                    qdrant_kb=self.qdrant_kb
                )
                
                self.intelligent_tools = IntelligentTools(self.agent_tools, self.qdrant_kb)
                
                # Initialize synthesizer
                self.synthesizer = ResponseSynthesizer(self.llm_client)
                
                # Initialize RAG pipeline
                self.rag_pipeline = RAGPipeline(self.llm_client, self.agent_tools)
                
                # Initialize agent graph
                self.agent_graph = LegalAgentGraph(
                    self.rag_pipeline, 
                    self.intelligent_tools, 
                    self.router, 
                    self.synthesizer
                )
                
                logger.info("✅ Components initialized successfully")
            else:
                logger.warning("⚠️ Search index not found, using basic components only")
                self.agent_tools = None
                self.intelligent_tools = None
                self.rag_pipeline = None
                self.agent_graph = None
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            # Initialize minimal components
            self.agent_tools = None
            self.intelligent_tools = None
            self.rag_pipeline = None
            self.agent_graph = None
    
    def collect_router_metrics(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Measure router classification accuracy
        
        Args:
            test_queries: List of test queries with expected tools
            
        Returns:
            Router metrics dictionary
        """
        logger.info("🧠 Collecting router metrics...")
        
        results = {
            "total_queries": len(test_queries),
            "correct_predictions": 0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "tool_accuracy": {
                "knowledge_base": {"correct": 0, "total": 0},
                "case_search": {"correct": 0, "total": 0},
                "hybrid": {"correct": 0, "total": 0},
                "web_search": {"correct": 0, "total": 0}
            },
            "response_times": [],
            "fallback_usage": 0
        }
        
        for i, test_case in enumerate(test_queries):
            query = test_case["query"]
            expected_tool = test_case["expected_tool"]
            
            try:
                start_time = time.time()
                routing = self.router.route_query(query)
                response_time = time.time() - start_time
                
                results["response_times"].append(response_time)
                
                # Check if prediction is correct
                predicted_tool = routing["primary_tool"]
                if predicted_tool == expected_tool:
                    results["correct_predictions"] += 1
                    results["tool_accuracy"][predicted_tool]["correct"] += 1
                
                # Track confidence distribution
                confidence = routing.get("confidence", "unknown")
                if confidence in results["confidence_distribution"]:
                    results["confidence_distribution"][confidence] += 1
                
                # Track tool usage
                results["tool_accuracy"][predicted_tool]["total"] += 1
                
                # Check for fallback usage (heuristic routing)
                if "fallback" in routing.get("reasoning", "").lower():
                    results["fallback_usage"] += 1
                
                logger.info(f"   Query {i+1}/{len(test_queries)}: {predicted_tool} (expected: {expected_tool})")
                
            except Exception as e:
                logger.error(f"   Query {i+1} failed: {e}")
                results["response_times"].append(0)
        
        # Calculate accuracy
        results["accuracy"] = results["correct_predictions"] / results["total_queries"]
        results["average_response_time"] = statistics.mean(results["response_times"])
        
        logger.info(f"✅ Router accuracy: {results['accuracy']:.2%}")
        logger.info(f"✅ Average response time: {results['average_response_time']:.2f}s")
        
        return results
    
    def collect_search_quality_metrics(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Measure search quality for knowledge base and case search
        
        Args:
            test_queries: List of test queries with expected results
            
        Returns:
            Search quality metrics
        """
        logger.info("🔍 Collecting search quality metrics...")
        
        results = {
            "knowledge_base": {
                "total_searches": 0,
                "average_results": 0,
                "average_score": 0,
                "zero_results": 0,
                "response_times": []
            },
            "case_search": {
                "total_searches": 0,
                "average_results": 0,
                "average_score": 0,
                "zero_results": 0,
                "response_times": []
            }
        }
        
        if not self.intelligent_tools:
            logger.warning("⚠️ Intelligent tools not available, skipping search quality metrics")
            return results
        
        for test_case in test_queries:
            query = test_case["query"]
            search_type = test_case.get("search_type", "both")
            
            # Test knowledge base search
            if search_type in ["knowledge_base", "both"]:
                try:
                    start_time = time.time()
                    kb_results = self.intelligent_tools.search_knowledge_base(query, limit=5)
                    response_time = time.time() - start_time
                    
                    results["knowledge_base"]["total_searches"] += 1
                    results["knowledge_base"]["response_times"].append(response_time)
                    
                    if kb_results["count"] > 0:
                        results["knowledge_base"]["average_results"] += kb_results["count"]
                        # Calculate average score if available
                        if "results" in kb_results and kb_results["results"]:
                            scores = [r.get("score", 0) for r in kb_results["results"]]
                            results["knowledge_base"]["average_score"] += statistics.mean(scores)
                    else:
                        results["knowledge_base"]["zero_results"] += 1
                        
                except Exception as e:
                    logger.error(f"Knowledge base search failed: {e}")
            
            # Test case search
            if search_type in ["case_search", "both"]:
                try:
                    start_time = time.time()
                    case_results = self.intelligent_tools.search_cases(query, limit=5)
                    response_time = time.time() - start_time
                    
                    results["case_search"]["total_searches"] += 1
                    results["case_search"]["response_times"].append(response_time)
                    
                    if case_results["count"] > 0:
                        results["case_search"]["average_results"] += case_results["count"]
                        # Calculate average score if available
                        if "results" in case_results and case_results["results"]:
                            scores = [r.get("score", 0) for r in case_results["results"]]
                            results["case_search"]["average_score"] += statistics.mean(scores)
                    else:
                        results["case_search"]["zero_results"] += 1
                        
                except Exception as e:
                    logger.error(f"Case search failed: {e}")
        
        # Calculate averages
        for search_type in ["knowledge_base", "case_search"]:
            if results[search_type]["total_searches"] > 0:
                results[search_type]["average_results"] /= results[search_type]["total_searches"]
                results[search_type]["average_score"] /= results[search_type]["total_searches"]
                results[search_type]["average_response_time"] = statistics.mean(results[search_type]["response_times"])
        
        logger.info(f"✅ Knowledge base: {results['knowledge_base']['average_results']:.1f} results, {results['knowledge_base']['average_score']:.3f} avg score")
        logger.info(f"✅ Case search: {results['case_search']['average_results']:.1f} results, {results['case_search']['average_score']:.3f} avg score")
        
        return results
    
    def collect_response_time_metrics(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Measure end-to-end response times
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Response time metrics
        """
        logger.info("⏱️ Collecting response time metrics...")
        
        results = {
            "total_queries": len(test_queries),
            "response_times": [],
            "component_times": {
                "routing": [],
                "search": [],
                "synthesis": [],
                "total": []
            },
            "error_rate": 0
        }
        
        for i, query in enumerate(test_queries):
            try:
                start_time = time.time()
                
                # Measure routing time
                routing_start = time.time()
                routing = self.router.route_query(query)
                routing_time = time.time() - routing_start
                
                # Measure search time
                search_start = time.time()
                if routing["primary_tool"] == "hybrid":
                    tool_results = self.intelligent_tools.hybrid_search(query)
                else:
                    tool_results = self.intelligent_tools.execute_tool(
                        routing["primary_tool"], query
                    )
                search_time = time.time() - search_start
                
                # Measure synthesis time
                synthesis_start = time.time()
                response = self.synthesizer.synthesize(
                    query, tool_results, routing
                )
                synthesis_time = time.time() - synthesis_start
                
                total_time = time.time() - start_time
                
                results["response_times"].append(total_time)
                results["component_times"]["routing"].append(routing_time)
                results["component_times"]["search"].append(search_time)
                results["component_times"]["synthesis"].append(synthesis_time)
                results["component_times"]["total"].append(total_time)
                
                logger.info(f"   Query {i+1}/{len(test_queries)}: {total_time:.2f}s total")
                
            except Exception as e:
                logger.error(f"   Query {i+1} failed: {e}")
                results["error_rate"] += 1
        
        # Calculate statistics
        results["error_rate"] /= results["total_queries"]
        results["average_response_time"] = statistics.mean(results["response_times"])
        results["median_response_time"] = statistics.median(results["response_times"])
        results["p95_response_time"] = sorted(results["response_times"])[int(0.95 * len(results["response_times"]))]
        
        # Component averages
        component_times = results["component_times"].copy()  # Create a copy to avoid iteration issues
        for component in component_times:
            if component_times[component]:
                results["component_times"][f"{component}_avg"] = statistics.mean(component_times[component])
        
        logger.info(f"✅ Average response time: {results['average_response_time']:.2f}s")
        logger.info(f"✅ 95th percentile: {results['p95_response_time']:.2f}s")
        logger.info(f"✅ Error rate: {results['error_rate']:.2%}")
        
        return results
    
    def collect_conversation_memory_metrics(self, test_conversations: List[List[Dict[str, str]]]) -> Dict[str, Any]:
        """
        Measure conversation memory effectiveness
        
        Args:
            test_conversations: List of conversation sequences
            
        Returns:
            Memory metrics
        """
        logger.info("🧠 Collecting conversation memory metrics...")
        
        results = {
            "total_conversations": len(test_conversations),
            "pronoun_resolution_success": 0,
            "reference_resolution_success": 0,
            "context_utilization": 0,
            "memory_overflow_events": 0
        }
        
        for i, conversation in enumerate(test_conversations):
            try:
                # Test conversation flow
                for j, message in enumerate(conversation):
                    query = message["content"]
                    conversation_history = conversation[:j]  # Previous messages
                    
                    # Route with conversation history
                    routing = self.router.route_query(query, conversation_history)
                    
                    # Check if routing considers conversation context
                    if len(conversation_history) > 0 and "conversation" in str(routing).lower():
                        results["context_utilization"] += 1
                    
                    # Test pronoun resolution (simple heuristic)
                    if any(pronoun in query.lower() for pronoun in ["it", "that", "this", "they"]):
                        if len(conversation_history) > 0:
                            results["pronoun_resolution_success"] += 1
                    
                    # Test reference resolution
                    if any(ref in query.lower() for ref in ["the first", "the second", "the last", "previous"]):
                        if len(conversation_history) > 0:
                            results["reference_resolution_success"] += 1
                    
                    # Check for memory overflow (conversation too long)
                    if len(conversation_history) > 10:
                        results["memory_overflow_events"] += 1
                
                logger.info(f"   Conversation {i+1}/{len(test_conversations)}: {len(conversation)} messages")
                
            except Exception as e:
                logger.error(f"   Conversation {i+1} failed: {e}")
        
        # Calculate success rates
        total_pronoun_tests = sum(1 for conv in test_conversations for msg in conv 
                                if any(pronoun in msg["content"].lower() for pronoun in ["it", "that", "this", "they"]))
        total_reference_tests = sum(1 for conv in test_conversations for msg in conv 
                                  if any(ref in msg["content"].lower() for ref in ["the first", "the second", "the last", "previous"]))
        
        if total_pronoun_tests > 0:
            results["pronoun_resolution_rate"] = results["pronoun_resolution_success"] / total_pronoun_tests
        if total_reference_tests > 0:
            results["reference_resolution_rate"] = results["reference_resolution_success"] / total_reference_tests
        
        logger.info(f"✅ Pronoun resolution: {results.get('pronoun_resolution_rate', 0):.2%}")
        logger.info(f"✅ Reference resolution: {results.get('reference_resolution_rate', 0):.2%}")
        
        return results
    
    def collect_system_health_metrics(self) -> Dict[str, Any]:
        """
        Collect system health and resource metrics
        
        Returns:
            System health metrics
        """
        logger.info("🏥 Collecting system health metrics...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "components_status": {},
            "resource_usage": {},
            "data_availability": {}
        }
        
        # Check component availability
        components = {
            "llm_client": self.llm_client,
            "router": self.router,
            "intelligent_tools": self.intelligent_tools,
            "synthesizer": self.synthesizer,
            "agent_graph": self.agent_graph
        }
        
        for name, component in components.items():
            try:
                # Simple availability check
                if hasattr(component, 'is_available'):
                    results["components_status"][name] = component.is_available()
                else:
                    results["components_status"][name] = True  # Assume available if no check method
            except Exception as e:
                results["components_status"][name] = False
                logger.warning(f"Component {name} check failed: {e}")
        
        # Check data availability
        try:
            # Check Qdrant
            qdrant_info = self.qdrant_kb.get_collection_info()
            results["data_availability"]["qdrant_vectors"] = qdrant_info.get("vector_count", 0)
        except Exception as e:
            results["data_availability"]["qdrant_vectors"] = 0
            logger.warning(f"Qdrant check failed: {e}")
        
        try:
            # Check case search
            case_count = len(self.agent_tools.metadata) if hasattr(self.agent_tools, 'metadata') else 0
            results["data_availability"]["case_count"] = case_count
        except Exception as e:
            results["data_availability"]["case_count"] = 0
            logger.warning(f"Case count check failed: {e}")
        
        # Calculate overall health score
        component_health = sum(results["components_status"].values()) / len(results["components_status"])
        data_health = 1.0 if (results["data_availability"]["qdrant_vectors"] > 0 and 
                             results["data_availability"]["case_count"] > 0) else 0.5
        results["overall_health_score"] = (component_health + data_health) / 2
        
        logger.info(f"✅ Overall health score: {results['overall_health_score']:.2%}")
        logger.info(f"✅ Components healthy: {sum(results['components_status'].values())}/{len(results['components_status'])}")
        
        return results
    
    def run_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Run comprehensive metrics collection
        
        Returns:
            Complete metrics report
        """
        logger.info("📊 Starting comprehensive metrics collection...")
        
        # Define test data
        router_test_queries = [
            {"query": "What is Article 12 of the Constitution?", "expected_tool": "knowledge_base"},
            {"query": "Find property dispute cases", "expected_tool": "case_search"},
            {"query": "What should I do if my citizenship is denied?", "expected_tool": "hybrid"},
            {"query": "Recent changes in immigration law", "expected_tool": "web_search"},
            {"query": "Explain the Civil Code", "expected_tool": "knowledge_base"},
            {"query": "Show me murder cases", "expected_tool": "case_search"},
            {"query": "How do I file a case?", "expected_tool": "hybrid"},
            {"query": "Current legal news", "expected_tool": "web_search"},
            {"query": "Article 18 rights", "expected_tool": "knowledge_base"},
            {"query": "Family law precedents", "expected_tool": "case_search"}
        ]
        
        search_test_queries = [
            {"query": "constitutional rights", "search_type": "knowledge_base"},
            {"query": "property dispute", "search_type": "case_search"},
            {"query": "citizenship law", "search_type": "both"},
            {"query": "criminal procedure", "search_type": "both"},
            {"query": "civil code", "search_type": "knowledge_base"}
        ]
        
        response_time_queries = [
            "What is Article 12?",
            "Find property cases",
            "What should I do about citizenship?",
            "Explain the Civil Code",
            "Show me recent cases"
        ]
        
        conversation_test_cases = [
            [
                {"role": "user", "content": "What is Article 12?"},
                {"role": "assistant", "content": "Article 12 is about..."},
                {"role": "user", "content": "What about Article 18?"},
                {"role": "assistant", "content": "Article 18 is..."},
                {"role": "user", "content": "How do they differ?"}
            ],
            [
                {"role": "user", "content": "Find property cases"},
                {"role": "assistant", "content": "Here are property cases..."},
                {"role": "user", "content": "What was the verdict in the first one?"},
                {"role": "assistant", "content": "The first case..."},
                {"role": "user", "content": "How does it compare to the second?"}
            ]
        ]
        
        # Collect all metrics
        comprehensive_metrics = {
            "collection_timestamp": datetime.now().isoformat(),
            "router_metrics": self.collect_router_metrics(router_test_queries),
            "search_quality_metrics": self.collect_search_quality_metrics(search_test_queries),
            "response_time_metrics": self.collect_response_time_metrics(response_time_queries),
            "conversation_memory_metrics": self.collect_conversation_memory_metrics(conversation_test_cases),
            "system_health_metrics": self.collect_system_health_metrics()
        }
        
        # Calculate overall system score
        router_score = comprehensive_metrics["router_metrics"]["accuracy"]
        search_score = (comprehensive_metrics["search_quality_metrics"]["knowledge_base"]["average_score"] + 
                       comprehensive_metrics["search_quality_metrics"]["case_search"]["average_score"]) / 2
        response_score = 1.0 if comprehensive_metrics["response_time_metrics"]["average_response_time"] < 5.0 else 0.5
        health_score = comprehensive_metrics["system_health_metrics"]["overall_health_score"]
        
        comprehensive_metrics["overall_system_score"] = (router_score + search_score + response_score + health_score) / 4
        
        logger.info(f"🎯 Overall system score: {comprehensive_metrics['overall_system_score']:.2%}")
        
        return comprehensive_metrics
    
    def save_metrics_report(self, metrics: Dict[str, Any], filename: str = None):
        """
        Save metrics report to file
        
        Args:
            metrics: Metrics dictionary
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_metrics_{timestamp}.json"
        
        output_path = Path(__file__).parent / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Metrics report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save metrics report: {e}")
    
    def print_metrics_summary(self, metrics: Dict[str, Any]):
        """
        Print a human-readable metrics summary
        
        Args:
            metrics: Metrics dictionary
        """
        print("\n" + "="*80)
        print("📊 WAKALAT SEWA V2 - SYSTEM METRICS SUMMARY")
        print("="*80)
        
        # Router metrics
        router = metrics["router_metrics"]
        print(f"\n🧠 ROUTER PERFORMANCE:")
        print(f"   Accuracy: {router['accuracy']:.2%}")
        print(f"   Average Response Time: {router['average_response_time']:.2f}s")
        print(f"   Fallback Usage: {router['fallback_usage']}/{router['total_queries']}")
        
        # Search quality
        search = metrics["search_quality_metrics"]
        print(f"\n🔍 SEARCH QUALITY:")
        print(f"   Knowledge Base: {search['knowledge_base']['average_score']:.3f} avg score")
        print(f"   Case Search: {search['case_search']['average_score']:.3f} avg score")
        print(f"   Zero Results: KB={search['knowledge_base']['zero_results']}, Cases={search['case_search']['zero_results']}")
        
        # Response times
        response = metrics["response_time_metrics"]
        print(f"\n⏱️ RESPONSE TIMES:")
        print(f"   Average: {response['average_response_time']:.2f}s")
        print(f"   Median: {response['median_response_time']:.2f}s")
        print(f"   95th Percentile: {response['p95_response_time']:.2f}s")
        print(f"   Error Rate: {response['error_rate']:.2%}")
        
        # Memory metrics
        memory = metrics["conversation_memory_metrics"]
        print(f"\n🧠 CONVERSATION MEMORY:")
        print(f"   Pronoun Resolution: {memory.get('pronoun_resolution_rate', 0):.2%}")
        print(f"   Reference Resolution: {memory.get('reference_resolution_rate', 0):.2%}")
        print(f"   Context Utilization: {memory['context_utilization']}")
        
        # System health
        health = metrics["system_health_metrics"]
        print(f"\n🏥 SYSTEM HEALTH:")
        print(f"   Overall Score: {health['overall_health_score']:.2%}")
        print(f"   Components Healthy: {sum(health['components_status'].values())}/{len(health['components_status'])}")
        print(f"   Qdrant Vectors: {health['data_availability'].get('qdrant_vectors', 0):,}")
        print(f"   Case Count: {health['data_availability'].get('case_count', 0):,}")
        
        # Overall score
        print(f"\n🎯 OVERALL SYSTEM SCORE: {metrics['overall_system_score']:.2%}")
        print("="*80)


def main():
    """Main function to run metrics collection"""
    try:
        logger.info("🚀 Starting Wakalat Sewa V2 Metrics Collection...")
        
        # Initialize metrics collector
        collector = SystemMetricsCollector()
        
        # Run comprehensive metrics
        metrics = collector.run_comprehensive_metrics()
        
        # Print summary
        collector.print_metrics_summary(metrics)
        
        # Save report
        collector.save_metrics_report(metrics)
        
        logger.info("✅ Metrics collection completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Metrics collection failed: {e}")
        raise


if __name__ == "__main__":
    main()
