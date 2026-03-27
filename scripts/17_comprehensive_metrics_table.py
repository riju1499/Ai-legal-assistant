#!/usr/bin/env python3
"""
Comprehensive Metrics Table Generator
Creates detailed metrics tables with all collected data
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveMetricsTable:
    """Generate comprehensive metrics tables"""
    
    def __init__(self, json_file_path):
        self.json_file_path = Path(json_file_path)
        self.data = None
        self.output_dir = self.json_file_path.parent / "metrics_tables"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_json_data(self):
        """Load JSON metrics data"""
        try:
            with open(self.json_file_path, 'r') as f:
                self.data = json.load(f)
            logger.info(f"✅ Loaded JSON data from: {self.json_file_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load JSON data: {e}")
            return False
    
    def create_router_metrics_table(self):
        """Create comprehensive router metrics table"""
        logger.info("🧠 Creating router metrics table...")
        
        if 'router_metrics' not in self.data:
            logger.warning("No router metrics data found")
            return None
        
        router_data = self.data['router_metrics']
        detailed_results = router_data.get('detailed_results', [])
        
        if not detailed_results:
            logger.warning("No detailed router results found")
            return None
        
        # Create comprehensive router table
        router_table = []
        
        # Summary metrics
        router_table.append({
            'Category': 'Router Summary',
            'Metric': 'Total Queries',
            'Value': router_data.get('total_queries', 0),
            'Unit': 'queries',
            'Description': 'Total number of queries processed by router'
        })
        
        router_table.append({
            'Category': 'Router Summary',
            'Metric': 'Accuracy',
            'Value': f"{router_data.get('accuracy', 0):.2f}",
            'Unit': '%',
            'Description': 'Percentage of correct routing decisions'
        })
        
        router_table.append({
            'Category': 'Router Summary',
            'Metric': 'Average Response Time',
            'Value': f"{router_data.get('avg_response_time', 0):.3f}",
            'Unit': 'seconds',
            'Description': 'Average time taken for routing decisions'
        })
        
        router_table.append({
            'Category': 'Router Summary',
            'Metric': '95th Percentile Response Time',
            'Value': f"{router_data.get('p95_response_time', 0):.3f}",
            'Unit': 'seconds',
            'Description': '95th percentile of response times'
        })
        
        router_table.append({
            'Category': 'Router Summary',
            'Metric': 'Correct Predictions',
            'Value': router_data.get('correct_predictions', 0),
            'Unit': 'count',
            'Description': 'Number of correct routing predictions'
        })
        
        router_table.append({
            'Category': 'Router Summary',
            'Metric': 'Incorrect Predictions',
            'Value': router_data.get('incorrect_predictions', 0),
            'Unit': 'count',
            'Description': 'Number of incorrect routing predictions'
        })
        
        # Tool distribution
        tool_distribution = router_data.get('tool_distribution', {})
        for tool, count in tool_distribution.items():
            router_table.append({
                'Category': 'Tool Distribution',
                'Metric': f'{tool.title()} Usage',
                'Value': count,
                'Unit': 'queries',
                'Description': f'Number of queries routed to {tool}'
            })
        
        # Confidence distribution
        confidence_scores = router_data.get('confidence_scores', [])
        if confidence_scores:
            confidence_counts = {}
            for score in confidence_scores:
                confidence_counts[score] = confidence_counts.get(score, 0) + 1
            
            for confidence, count in confidence_counts.items():
                router_table.append({
                    'Category': 'Confidence Distribution',
                    'Metric': f'{confidence.title()} Confidence',
                    'Value': count,
                    'Unit': 'queries',
                    'Description': f'Number of queries with {confidence} confidence'
                })
        
        # Export router table
        router_df = pd.DataFrame(router_table)
        router_file = self.output_dir / "router_comprehensive_table.csv"
        router_df.to_csv(router_file, index=False)
        logger.info(f"✅ Router metrics table saved to: {router_file}")
        
        return str(router_file)
    
    def create_search_metrics_table(self):
        """Create comprehensive search metrics table"""
        logger.info("🔍 Creating search metrics table...")
        
        if 'search_metrics' not in self.data:
            logger.warning("No search metrics data found")
            return None
        
        search_data = self.data['search_metrics']
        search_table = []
        
        # Knowledge base metrics
        kb_data = search_data.get('knowledge_base', {})
        search_table.append({
            'Category': 'Knowledge Base',
            'Metric': 'Total Queries',
            'Value': kb_data.get('total_queries', 0),
            'Unit': 'queries',
            'Description': 'Total number of knowledge base searches'
        })
        
        search_table.append({
            'Category': 'Knowledge Base',
            'Metric': 'Average Score',
            'Value': f"{kb_data.get('avg_score', 0):.3f}",
            'Unit': 'score',
            'Description': 'Average relevance score for knowledge base results'
        })
        
        search_table.append({
            'Category': 'Knowledge Base',
            'Metric': 'Zero Results',
            'Value': kb_data.get('zero_results', 0),
            'Unit': 'queries',
            'Description': 'Number of queries with no results'
        })
        
        # Case search metrics
        case_data = search_data.get('case_search', {})
        search_table.append({
            'Category': 'Case Search',
            'Metric': 'Total Queries',
            'Value': case_data.get('total_queries', 0),
            'Unit': 'queries',
            'Description': 'Total number of case searches'
        })
        
        search_table.append({
            'Category': 'Case Search',
            'Metric': 'Average Score',
            'Value': f"{case_data.get('avg_score', 0):.3f}",
            'Unit': 'score',
            'Description': 'Average relevance score for case search results'
        })
        
        search_table.append({
            'Category': 'Case Search',
            'Metric': 'Zero Results',
            'Value': case_data.get('zero_results', 0),
            'Unit': 'queries',
            'Description': 'Number of queries with no results'
        })
        
        # Detailed results analysis
        detailed_results = search_data.get('detailed_results', [])
        if detailed_results:
            # Calculate statistics
            kb_scores = [r['knowledge_base']['avg_score'] for r in detailed_results if r['knowledge_base']['avg_score'] > 0]
            case_scores = [r['case_search']['avg_score'] for r in detailed_results if r['case_search']['avg_score'] > 0]
            
            if kb_scores:
                search_table.append({
                    'Category': 'Knowledge Base Analysis',
                    'Metric': 'Score Range',
                    'Value': f"{min(kb_scores):.3f} - {max(kb_scores):.3f}",
                    'Unit': 'score',
                    'Description': 'Range of knowledge base scores'
                })
            
            if case_scores:
                search_table.append({
                    'Category': 'Case Search Analysis',
                    'Metric': 'Score Range',
                    'Value': f"{min(case_scores):.3f} - {max(case_scores):.3f}",
                    'Unit': 'score',
                    'Description': 'Range of case search scores'
                })
        
        # Export search table
        search_df = pd.DataFrame(search_table)
        search_file = self.output_dir / "search_comprehensive_table.csv"
        search_df.to_csv(search_file, index=False)
        logger.info(f"✅ Search metrics table saved to: {search_file}")
        
        return str(search_file)
    
    def create_response_metrics_table(self):
        """Create comprehensive response metrics table"""
        logger.info("⏱️ Creating response metrics table...")
        
        if 'response_metrics' not in self.data:
            logger.warning("No response metrics data found")
            return None
        
        response_data = self.data['response_metrics']
        response_table = []
        
        # Basic metrics
        response_table.append({
            'Category': 'Response Summary',
            'Metric': 'Total Queries',
            'Value': response_data.get('total_queries', 0),
            'Unit': 'queries',
            'Description': 'Total number of response queries processed'
        })
        
        response_table.append({
            'Category': 'Response Summary',
            'Metric': 'Success Rate',
            'Value': f"{response_data.get('success_rate', 0):.2f}",
            'Unit': '%',
            'Description': 'Percentage of successful responses'
        })
        
        response_table.append({
            'Category': 'Response Summary',
            'Metric': 'Error Count',
            'Value': response_data.get('error_count', 0),
            'Unit': 'errors',
            'Description': 'Number of failed responses'
        })
        
        # Response time metrics
        response_table.append({
            'Category': 'Response Times',
            'Metric': 'Average Response Time',
            'Value': f"{response_data.get('avg_response_time', 0):.3f}",
            'Unit': 'seconds',
            'Description': 'Average time to generate responses'
        })
        
        response_table.append({
            'Category': 'Response Times',
            'Metric': '95th Percentile Response Time',
            'Value': f"{response_data.get('p95_response_time', 0):.3f}",
            'Unit': 'seconds',
            'Description': '95th percentile of response times'
        })
        
        response_table.append({
            'Category': 'Response Times',
            'Metric': 'Maximum Response Time',
            'Value': f"{response_data.get('max_response_time', 0):.3f}",
            'Unit': 'seconds',
            'Description': 'Longest response time recorded'
        })
        
        response_table.append({
            'Category': 'Response Times',
            'Metric': 'Minimum Response Time',
            'Value': f"{response_data.get('min_response_time', 0):.3f}",
            'Unit': 'seconds',
            'Description': 'Shortest response time recorded'
        })
        
        # Detailed analysis
        detailed_results = response_data.get('detailed_results', [])
        if detailed_results:
            # Tool usage analysis
            tool_usage = {}
            for result in detailed_results:
                tool = result.get('tools_used', 'unknown')
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
            
            for tool, count in tool_usage.items():
                response_table.append({
                    'Category': 'Tool Usage',
                    'Metric': f'{tool.title()} Usage',
                    'Value': count,
                    'Unit': 'queries',
                    'Description': f'Number of queries using {tool}'
                })
            
            # Response length analysis
            response_lengths = [r.get('response_length', 0) for r in detailed_results]
            if response_lengths:
                response_table.append({
                    'Category': 'Response Analysis',
                    'Metric': 'Average Response Length',
                    'Value': f"{sum(response_lengths) / len(response_lengths):.0f}",
                    'Unit': 'characters',
                    'Description': 'Average length of generated responses'
                })
                
                response_table.append({
                    'Category': 'Response Analysis',
                    'Metric': 'Response Length Range',
                    'Value': f"{min(response_lengths)} - {max(response_lengths)}",
                    'Unit': 'characters',
                    'Description': 'Range of response lengths'
                })
        
        # Export response table
        response_df = pd.DataFrame(response_table)
        response_file = self.output_dir / "response_comprehensive_table.csv"
        response_df.to_csv(response_file, index=False)
        logger.info(f"✅ Response metrics table saved to: {response_file}")
        
        return str(response_file)
    
    def create_conversation_metrics_table(self):
        """Create comprehensive conversation metrics table"""
        logger.info("🧠 Creating conversation metrics table...")
        
        if 'conversation_metrics' not in self.data:
            logger.warning("No conversation metrics data found")
            return None
        
        conversation_data = self.data['conversation_metrics']
        conversation_table = []
        
        # Basic metrics
        conversation_table.append({
            'Category': 'Conversation Summary',
            'Metric': 'Total Conversations',
            'Value': conversation_data.get('total_conversations', 0),
            'Unit': 'conversations',
            'Description': 'Total number of conversation tests'
        })
        
        conversation_table.append({
            'Category': 'Conversation Summary',
            'Metric': 'Pronoun Resolution',
            'Value': f"{conversation_data.get('pronoun_resolution', 0):.2f}",
            'Unit': '%',
            'Description': 'Percentage of successful pronoun resolution'
        })
        
        conversation_table.append({
            'Category': 'Conversation Summary',
            'Metric': 'Reference Resolution',
            'Value': f"{conversation_data.get('reference_resolution', 0):.2f}",
            'Unit': '%',
            'Description': 'Percentage of successful reference resolution'
        })
        
        conversation_table.append({
            'Category': 'Conversation Summary',
            'Metric': 'Context Utilization',
            'Value': f"{conversation_data.get('context_utilization', 0):.2f}",
            'Unit': '%',
            'Description': 'Percentage of context utilization in responses'
        })
        
        # Detailed conversation analysis
        detailed_results = conversation_data.get('detailed_results', [])
        if detailed_results:
            # Calculate conversation statistics
            total_messages = sum(r.get('messages_processed', 0) for r in detailed_results)
            total_pronoun_resolution = sum(r.get('pronoun_resolution_count', 0) for r in detailed_results)
            total_reference_resolution = sum(r.get('reference_resolution_count', 0) for r in detailed_results)
            total_context_utilization = sum(r.get('context_utilization_count', 0) for r in detailed_results)
            
            conversation_table.append({
                'Category': 'Conversation Analysis',
                'Metric': 'Total Messages Processed',
                'Value': total_messages,
                'Unit': 'messages',
                'Description': 'Total number of messages processed across all conversations'
            })
            
            if total_messages > 0:
                conversation_table.append({
                    'Category': 'Conversation Analysis',
                    'Metric': 'Average Pronoun Resolution Rate',
                    'Value': f"{(total_pronoun_resolution / total_messages) * 100:.2f}",
                    'Unit': '%',
                    'Description': 'Average pronoun resolution rate across all conversations'
                })
                
                conversation_table.append({
                    'Category': 'Conversation Analysis',
                    'Metric': 'Average Reference Resolution Rate',
                    'Value': f"{(total_reference_resolution / total_messages) * 100:.2f}",
                    'Unit': '%',
                    'Description': 'Average reference resolution rate across all conversations'
                })
                
                conversation_table.append({
                    'Category': 'Conversation Analysis',
                    'Metric': 'Average Context Utilization Rate',
                    'Value': f"{(total_context_utilization / total_messages) * 100:.2f}",
                    'Unit': '%',
                    'Description': 'Average context utilization rate across all conversations'
                })
        
        # Export conversation table
        conversation_df = pd.DataFrame(conversation_table)
        conversation_file = self.output_dir / "conversation_comprehensive_table.csv"
        conversation_df.to_csv(conversation_file, index=False)
        logger.info(f"✅ Conversation metrics table saved to: {conversation_file}")
        
        return str(conversation_file)
    
    def create_system_health_table(self):
        """Create comprehensive system health table"""
        logger.info("🏥 Creating system health table...")
        
        if 'system_health' not in self.data:
            logger.warning("No system health data found")
            return None
        
        health_data = self.data['system_health']
        health_table = []
        
        # Overall health metrics
        health_table.append({
            'Category': 'System Health Summary',
            'Metric': 'Overall Health Score',
            'Value': f"{health_data.get('overall_score', 0):.2f}",
            'Unit': '%',
            'Description': 'Overall system health score'
        })
        
        health_table.append({
            'Category': 'System Health Summary',
            'Metric': 'Healthy Components',
            'Value': health_data.get('healthy_components', 0),
            'Unit': 'components',
            'Description': 'Number of healthy components'
        })
        
        health_table.append({
            'Category': 'System Health Summary',
            'Metric': 'Total Components',
            'Value': health_data.get('total_components', 0),
            'Unit': 'components',
            'Description': 'Total number of system components'
        })
        
        # Component details
        components = health_data.get('components', {})
        for component_name, component_info in components.items():
            health_table.append({
                'Category': 'Component Status',
                'Metric': f'{component_name.title()} Status',
                'Value': component_info.get('status', 'unknown'),
                'Unit': 'status',
                'Description': f'Health status of {component_name} component'
            })
            
            if 'response_time' in component_info:
                health_table.append({
                    'Category': 'Component Performance',
                    'Metric': f'{component_name.title()} Response Time',
                    'Value': f"{component_info.get('response_time', 0):.3f}",
                    'Unit': 'seconds',
                    'Description': f'Response time of {component_name} component'
                })
            
            if 'error' in component_info and component_info['error']:
                health_table.append({
                    'Category': 'Component Errors',
                    'Metric': f'{component_name.title()} Error',
                    'Value': component_info.get('error', ''),
                    'Unit': 'error',
                    'Description': f'Error message for {component_name} component'
                })
        
        # Export health table
        health_df = pd.DataFrame(health_table)
        health_file = self.output_dir / "system_health_comprehensive_table.csv"
        health_df.to_csv(health_file, index=False)
        logger.info(f"✅ System health table saved to: {health_file}")
        
        return str(health_file)
    
    def create_performance_metrics_table(self):
        """Create comprehensive performance metrics table"""
        logger.info("📊 Creating performance metrics table...")
        
        if 'performance_metrics' not in self.data:
            logger.warning("No performance metrics data found")
            return None
        
        performance_data = self.data['performance_metrics']
        performance_table = []
        
        # System resources
        system_resources = performance_data.get('system_resources', {})
        for resource, value in system_resources.items():
            performance_table.append({
                'Category': 'System Resources',
                'Metric': resource.replace('_', ' ').title(),
                'Value': str(value),
                'Unit': 'resource',
                'Description': f'System resource: {resource}'
            })
        
        # Throughput metrics
        throughput = performance_data.get('throughput', {})
        for metric, value in throughput.items():
            performance_table.append({
                'Category': 'Throughput',
                'Metric': metric.replace('_', ' ').title(),
                'Value': f"{value:.3f}" if isinstance(value, (int, float)) else str(value),
                'Unit': 'performance',
                'Description': f'Throughput metric: {metric}'
            })
        
        # Scalability metrics
        scalability = performance_data.get('scalability', {})
        for metric, value in scalability.items():
            performance_table.append({
                'Category': 'Scalability',
                'Metric': metric.replace('_', ' ').title(),
                'Value': f"{value:.3f}" if isinstance(value, (int, float)) else str(value),
                'Unit': 'scalability',
                'Description': f'Scalability metric: {metric}'
            })
        
        # Reliability metrics
        reliability = performance_data.get('reliability', {})
        for metric, value in reliability.items():
            performance_table.append({
                'Category': 'Reliability',
                'Metric': metric.replace('_', ' ').title(),
                'Value': f"{value:.3f}" if isinstance(value, (int, float)) else str(value),
                'Unit': 'reliability',
                'Description': f'Reliability metric: {metric}'
            })
        
        # Export performance table
        performance_df = pd.DataFrame(performance_table)
        performance_file = self.output_dir / "performance_comprehensive_table.csv"
        performance_df.to_csv(performance_file, index=False)
        logger.info(f"✅ Performance metrics table saved to: {performance_file}")
        
        return str(performance_file)
    
    def create_master_metrics_table(self):
        """Create master metrics table combining all metrics"""
        logger.info("📋 Creating master metrics table...")
        
        master_table = []
        
        # Metadata
        metadata = self.data.get('metadata', {})
        master_table.append({
            'Category': 'System Information',
            'Metric': 'Test Timestamp',
            'Value': metadata.get('timestamp', 'N/A'),
            'Unit': 'timestamp',
            'Description': 'When the metrics were collected'
        })
        
        master_table.append({
            'Category': 'System Information',
            'Metric': 'Test Type',
            'Value': metadata.get('test_type', 'N/A'),
            'Unit': 'type',
            'Description': 'Type of metrics test performed'
        })
        
        master_table.append({
            'Category': 'System Information',
            'Metric': 'System Version',
            'Value': metadata.get('version', 'N/A'),
            'Unit': 'version',
            'Description': 'Version of the system being tested'
        })
        
        # Router metrics summary
        router_data = self.data.get('router_metrics', {})
        master_table.append({
            'Category': 'Router Performance',
            'Metric': 'Router Accuracy',
            'Value': f"{router_data.get('accuracy', 0):.2f}",
            'Unit': '%',
            'Description': 'Percentage of correct routing decisions'
        })
        
        master_table.append({
            'Category': 'Router Performance',
            'Metric': 'Router Response Time',
            'Value': f"{router_data.get('avg_response_time', 0):.3f}",
            'Unit': 'seconds',
            'Description': 'Average router response time'
        })
        
        # Search metrics summary
        search_data = self.data.get('search_metrics', {})
        kb_data = search_data.get('knowledge_base', {})
        case_data = search_data.get('case_search', {})
        
        master_table.append({
            'Category': 'Search Quality',
            'Metric': 'Knowledge Base Score',
            'Value': f"{kb_data.get('avg_score', 0):.3f}",
            'Unit': 'score',
            'Description': 'Average knowledge base search score'
        })
        
        master_table.append({
            'Category': 'Search Quality',
            'Metric': 'Case Search Score',
            'Value': f"{case_data.get('avg_score', 0):.3f}",
            'Unit': 'score',
            'Description': 'Average case search score'
        })
        
        # Response metrics summary
        response_data = self.data.get('response_metrics', {})
        master_table.append({
            'Category': 'Response Performance',
            'Metric': 'Success Rate',
            'Value': f"{response_data.get('success_rate', 0):.2f}",
            'Unit': '%',
            'Description': 'Percentage of successful responses'
        })
        
        master_table.append({
            'Category': 'Response Performance',
            'Metric': 'Average Response Time',
            'Value': f"{response_data.get('avg_response_time', 0):.3f}",
            'Unit': 'seconds',
            'Description': 'Average response generation time'
        })
        
        # System health summary
        health_data = self.data.get('system_health', {})
        master_table.append({
            'Category': 'System Health',
            'Metric': 'Overall Health Score',
            'Value': f"{health_data.get('overall_score', 0):.2f}",
            'Unit': '%',
            'Description': 'Overall system health score'
        })
        
        master_table.append({
            'Category': 'System Health',
            'Metric': 'Healthy Components',
            'Value': f"{health_data.get('healthy_components', 0)}/{health_data.get('total_components', 0)}",
            'Unit': 'components',
            'Description': 'Number of healthy components vs total'
        })
        
        # Performance summary
        performance_data = self.data.get('performance_metrics', {})
        throughput = performance_data.get('throughput', {})
        reliability = performance_data.get('reliability', {})
        
        master_table.append({
            'Category': 'Performance',
            'Metric': 'Queries Per Second',
            'Value': f"{throughput.get('queries_per_second', 0):.2f}",
            'Unit': 'qps',
            'Description': 'System throughput in queries per second'
        })
        
        master_table.append({
            'Category': 'Performance',
            'Metric': 'Error Rate',
            'Value': f"{reliability.get('error_rate', 0):.2f}",
            'Unit': '%',
            'Description': 'System error rate percentage'
        })
        
        # Export master table
        master_df = pd.DataFrame(master_table)
        master_file = self.output_dir / "master_metrics_table.csv"
        master_df.to_csv(master_file, index=False)
        logger.info(f"✅ Master metrics table saved to: {master_file}")
        
        return str(master_file)
    
    def generate_all_tables(self):
        """Generate all comprehensive metrics tables"""
        logger.info("📊 Generating all comprehensive metrics tables...")
        
        if not self.load_json_data():
            return False
        
        results = {}
        
        try:
            # Generate all tables
            results['router'] = self.create_router_metrics_table()
            results['search'] = self.create_search_metrics_table()
            results['response'] = self.create_response_metrics_table()
            results['conversation'] = self.create_conversation_metrics_table()
            results['system_health'] = self.create_system_health_table()
            results['performance'] = self.create_performance_metrics_table()
            results['master'] = self.create_master_metrics_table()
            
            # Create table index
            self._create_table_index(results)
            
            logger.info(f"✅ All comprehensive metrics tables generated successfully!")
            logger.info(f"📁 Tables saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Table generation failed: {e}")
            return False
    
    def _create_table_index(self, results):
        """Create HTML index for all tables"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wakalat Sewa V2 - Comprehensive Metrics Tables</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .table {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa; }}
                .table h3 {{ color: #2c3e50; margin-top: 0; }}
                .table p {{ color: #7f8c8d; margin: 10px 0; }}
                .table a {{ color: #3498db; text-decoration: none; font-weight: bold; }}
                .table a:hover {{ color: #2980b9; }}
                .timestamp {{ text-align: center; color: #7f8c8d; font-style: italic; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Wakalat Sewa V2 - Comprehensive Metrics Tables</h1>
                
                <div class="table">
                    <h3>🧠 Router Metrics Table</h3>
                    <p>Comprehensive router performance metrics including accuracy, response times, and tool distribution</p>
                    <a href="router_comprehensive_table.csv" target="_blank">Download Router Metrics CSV</a>
                </div>
                
                <div class="table">
                    <h3>🔍 Search Metrics Table</h3>
                    <p>Detailed search quality metrics for knowledge base and case search functionality</p>
                    <a href="search_comprehensive_table.csv" target="_blank">Download Search Metrics CSV</a>
                </div>
                
                <div class="table">
                    <h3>⏱️ Response Metrics Table</h3>
                    <p>Comprehensive response time and performance metrics</p>
                    <a href="response_comprehensive_table.csv" target="_blank">Download Response Metrics CSV</a>
                </div>
                
                <div class="table">
                    <h3>🧠 Conversation Metrics Table</h3>
                    <p>Conversation memory and context utilization metrics</p>
                    <a href="conversation_comprehensive_table.csv" target="_blank">Download Conversation Metrics CSV</a>
                </div>
                
                <div class="table">
                    <h3>🏥 System Health Table</h3>
                    <p>System health status and component performance metrics</p>
                    <a href="system_health_comprehensive_table.csv" target="_blank">Download System Health CSV</a>
                </div>
                
                <div class="table">
                    <h3>📊 Performance Metrics Table</h3>
                    <p>System performance, throughput, and reliability metrics</p>
                    <a href="performance_comprehensive_table.csv" target="_blank">Download Performance Metrics CSV</a>
                </div>
                
                <div class="table">
                    <h3>📋 Master Metrics Table</h3>
                    <p>Master table combining all key metrics for comprehensive overview</p>
                    <a href="master_metrics_table.csv" target="_blank">Download Master Metrics CSV</a>
                </div>
                
                <div class="timestamp">
                    Generated on: {timestamp}
                </div>
            </div>
        </body>
        </html>
        """
        
        index_file = self.output_dir / "index.html"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Table index created: {index_file}")


def main():
    """Main function to generate comprehensive metrics tables"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python 17_comprehensive_metrics_table.py <json_file_path>")
        print("Example: python 17_comprehensive_metrics_table.py large_scale_metrics_20250123_120000.json")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    if not Path(json_file_path).exists():
        print(f"❌ JSON file not found: {json_file_path}")
        sys.exit(1)
    
    table_generator = ComprehensiveMetricsTable(json_file_path)
    success = table_generator.generate_all_tables()
    
    if success:
        print("🎉 All comprehensive metrics tables generated successfully!")
        print(f"📁 Check the metrics_tables directory for all CSV files.")
        print(f"🌐 Open the index.html file to view all tables.")
    else:
        print("❌ Table generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

