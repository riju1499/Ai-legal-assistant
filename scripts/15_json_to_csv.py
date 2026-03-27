#!/usr/bin/env python3
"""
JSON to CSV Converter
Converts detailed JSON metrics to structured CSV files
"""

import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsJSONToCSV:
    """Convert JSON metrics to structured CSV files"""
    
    def __init__(self, json_file_path):
        self.json_file_path = Path(json_file_path)
        self.data = None
        self.output_dir = self.json_file_path.parent / "csv_exports"
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
    
    def export_router_metrics(self):
        """Export router metrics to CSV"""
        if not self.data or 'router_metrics' not in self.data:
            logger.warning("No router metrics data found")
            return None
        
        router_data = self.data['router_metrics']
        detailed_results = router_data.get('detailed_results', [])
        
        if not detailed_results:
            logger.warning("No detailed router results found")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(detailed_results)
        
        # Add summary metrics
        summary_data = {
            'metric': ['accuracy', 'avg_response_time', 'p95_response_time', 'total_queries', 'correct_predictions', 'incorrect_predictions'],
            'value': [
                router_data.get('accuracy', 0),
                router_data.get('avg_response_time', 0),
                router_data.get('p95_response_time', 0),
                router_data.get('total_queries', 0),
                router_data.get('correct_predictions', 0),
                router_data.get('incorrect_predictions', 0)
            ]
        }
        
        # Export detailed results
        detailed_file = self.output_dir / "router_detailed_metrics.csv"
        df.to_csv(detailed_file, index=False)
        logger.info(f"✅ Router detailed metrics exported to: {detailed_file}")
        
        # Export summary
        summary_file = self.output_dir / "router_summary_metrics.csv"
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✅ Router summary metrics exported to: {summary_file}")
        
        return {
            'detailed_file': str(detailed_file),
            'summary_file': str(summary_file)
        }
    
    def export_search_metrics(self):
        """Export search metrics to CSV"""
        if not self.data or 'search_metrics' not in self.data:
            logger.warning("No search metrics data found")
            return None
        
        search_data = self.data['search_metrics']
        detailed_results = search_data.get('detailed_results', [])
        
        if not detailed_results:
            logger.warning("No detailed search results found")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(detailed_results)
        
        # Flatten nested data
        flattened_data = []
        for result in detailed_results:
            flattened_result = {
                'query': result['query'],
                'kb_results_count': result['knowledge_base']['results_count'],
                'kb_avg_score': result['knowledge_base']['avg_score'],
                'kb_response_time': result['knowledge_base']['response_time'],
                'case_results_count': result['case_search']['results_count'],
                'case_avg_score': result['case_search']['avg_score'],
                'case_response_time': result['case_search']['response_time']
            }
            flattened_data.append(flattened_result)
        
        # Export detailed results
        detailed_file = self.output_dir / "search_detailed_metrics.csv"
        detailed_df = pd.DataFrame(flattened_data)
        detailed_df.to_csv(detailed_file, index=False)
        logger.info(f"✅ Search detailed metrics exported to: {detailed_file}")
        
        # Export summary
        summary_data = {
            'source': ['knowledge_base', 'case_search'],
            'total_queries': [
                search_data.get('knowledge_base', {}).get('total_queries', 0),
                search_data.get('case_search', {}).get('total_queries', 0)
            ],
            'avg_score': [
                search_data.get('knowledge_base', {}).get('avg_score', 0),
                search_data.get('case_search', {}).get('avg_score', 0)
            ],
            'zero_results': [
                search_data.get('knowledge_base', {}).get('zero_results', 0),
                search_data.get('case_search', {}).get('zero_results', 0)
            ]
        }
        
        summary_file = self.output_dir / "search_summary_metrics.csv"
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✅ Search summary metrics exported to: {summary_file}")
        
        return {
            'detailed_file': str(detailed_file),
            'summary_file': str(summary_file)
        }
    
    def export_response_metrics(self):
        """Export response metrics to CSV"""
        if not self.data or 'response_metrics' not in self.data:
            logger.warning("No response metrics data found")
            return None
        
        response_data = self.data['response_metrics']
        detailed_results = response_data.get('detailed_results', [])
        
        if not detailed_results:
            logger.warning("No detailed response results found")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(detailed_results)
        
        # Export detailed results
        detailed_file = self.output_dir / "response_detailed_metrics.csv"
        df.to_csv(detailed_file, index=False)
        logger.info(f"✅ Response detailed metrics exported to: {detailed_file}")
        
        # Export summary
        summary_data = {
            'metric': [
                'total_queries', 'success_rate', 'error_count', 'avg_response_time',
                'p95_response_time', 'max_response_time', 'min_response_time'
            ],
            'value': [
                response_data.get('total_queries', 0),
                response_data.get('success_rate', 0),
                response_data.get('error_count', 0),
                response_data.get('avg_response_time', 0),
                response_data.get('p95_response_time', 0),
                response_data.get('max_response_time', 0),
                response_data.get('min_response_time', 0)
            ]
        }
        
        summary_file = self.output_dir / "response_summary_metrics.csv"
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✅ Response summary metrics exported to: {summary_file}")
        
        return {
            'detailed_file': str(detailed_file),
            'summary_file': str(summary_file)
        }
    
    def export_conversation_metrics(self):
        """Export conversation metrics to CSV"""
        if not self.data or 'conversation_metrics' not in self.data:
            logger.warning("No conversation metrics data found")
            return None
        
        conversation_data = self.data['conversation_metrics']
        detailed_results = conversation_data.get('detailed_results', [])
        
        if not detailed_results:
            logger.warning("No detailed conversation results found")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(detailed_results)
        
        # Export detailed results
        detailed_file = self.output_dir / "conversation_detailed_metrics.csv"
        df.to_csv(detailed_file, index=False)
        logger.info(f"✅ Conversation detailed metrics exported to: {detailed_file}")
        
        # Export summary
        summary_data = {
            'metric': [
                'total_conversations', 'pronoun_resolution', 'reference_resolution', 'context_utilization'
            ],
            'value': [
                conversation_data.get('total_conversations', 0),
                conversation_data.get('pronoun_resolution', 0),
                conversation_data.get('reference_resolution', 0),
                conversation_data.get('context_utilization', 0)
            ]
        }
        
        summary_file = self.output_dir / "conversation_summary_metrics.csv"
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✅ Conversation summary metrics exported to: {summary_file}")
        
        return {
            'detailed_file': str(detailed_file),
            'summary_file': str(summary_file)
        }
    
    def export_system_health_metrics(self):
        """Export system health metrics to CSV"""
        if not self.data or 'system_health' not in self.data:
            logger.warning("No system health data found")
            return None
        
        health_data = self.data['system_health']
        components = health_data.get('components', {})
        
        # Create component status DataFrame
        component_data = []
        for component_name, component_info in components.items():
            component_data.append({
                'component': component_name,
                'status': component_info.get('status', 'unknown'),
                'response_time': component_info.get('response_time', 0),
                'error': component_info.get('error', ''),
                'details': str(component_info)
            })
        
        # Export component status
        components_file = self.output_dir / "system_health_components.csv"
        components_df = pd.DataFrame(component_data)
        components_df.to_csv(components_file, index=False)
        logger.info(f"✅ System health components exported to: {components_file}")
        
        # Export summary
        summary_data = {
            'metric': ['overall_score', 'healthy_components', 'total_components'],
            'value': [
                health_data.get('overall_score', 0),
                health_data.get('healthy_components', 0),
                health_data.get('total_components', 0)
            ]
        }
        
        summary_file = self.output_dir / "system_health_summary.csv"
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✅ System health summary exported to: {summary_file}")
        
        return {
            'components_file': str(components_file),
            'summary_file': str(summary_file)
        }
    
    def export_performance_metrics(self):
        """Export performance metrics to CSV"""
        if not self.data or 'performance_metrics' not in self.data:
            logger.warning("No performance metrics data found")
            return None
        
        performance_data = self.data['performance_metrics']
        
        # Flatten performance data
        flattened_data = []
        
        # System resources
        for resource, value in performance_data.get('system_resources', {}).items():
            flattened_data.append({
                'category': 'system_resources',
                'metric': resource,
                'value': value
            })
        
        # Throughput
        for metric, value in performance_data.get('throughput', {}).items():
            flattened_data.append({
                'category': 'throughput',
                'metric': metric,
                'value': value
            })
        
        # Scalability
        for metric, value in performance_data.get('scalability', {}).items():
            flattened_data.append({
                'category': 'scalability',
                'metric': metric,
                'value': value
            })
        
        # Reliability
        for metric, value in performance_data.get('reliability', {}).items():
            flattened_data.append({
                'category': 'reliability',
                'metric': metric,
                'value': value
            })
        
        # Export performance metrics
        performance_file = self.output_dir / "performance_metrics.csv"
        performance_df = pd.DataFrame(flattened_data)
        performance_df.to_csv(performance_file, index=False)
        logger.info(f"✅ Performance metrics exported to: {performance_file}")
        
        return str(performance_file)
    
    def export_comprehensive_summary(self):
        """Export comprehensive summary to CSV"""
        if not self.data or 'comprehensive_summary' not in self.data:
            logger.warning("No comprehensive summary data found")
            return None
        
        summary_data = self.data['comprehensive_summary']
        
        # Test summary
        test_summary = summary_data.get('test_summary', {})
        test_summary_data = []
        for metric, value in test_summary.items():
            test_summary_data.append({
                'category': 'test_summary',
                'metric': metric,
                'value': value
            })
        
        # Performance summary
        performance_summary = summary_data.get('performance_summary', {})
        performance_summary_data = []
        for metric, value in performance_summary.items():
            performance_summary_data.append({
                'category': 'performance_summary',
                'metric': metric,
                'value': value
            })
        
        # Recommendations
        recommendations = summary_data.get('recommendations', [])
        recommendations_data = []
        for i, recommendation in enumerate(recommendations):
            recommendations_data.append({
                'category': 'recommendations',
                'metric': f'recommendation_{i+1}',
                'value': recommendation
            })
        
        # Combine all summary data
        all_summary_data = test_summary_data + performance_summary_data + recommendations_data
        
        # Export comprehensive summary
        summary_file = self.output_dir / "comprehensive_summary.csv"
        summary_df = pd.DataFrame(all_summary_data)
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✅ Comprehensive summary exported to: {summary_file}")
        
        return str(summary_file)
    
    def create_metrics_overview_table(self):
        """Create a comprehensive metrics overview table"""
        if not self.data:
            logger.warning("No data loaded")
            return None
        
        overview_data = []
        
        # Add metadata
        metadata = self.data.get('metadata', {})
        overview_data.append({
            'Category': 'Metadata',
            'Metric': 'Timestamp',
            'Value': metadata.get('timestamp', 'N/A')
        })
        overview_data.append({
            'Category': 'Metadata',
            'Metric': 'Test Type',
            'Value': metadata.get('test_type', 'N/A')
        })
        overview_data.append({
            'Category': 'Metadata',
            'Metric': 'Version',
            'Value': metadata.get('version', 'N/A')
        })
        
        # Router metrics
        router = self.data.get('router_metrics', {})
        overview_data.append({
            'Category': 'Router',
            'Metric': 'Accuracy (%)',
            'Value': router.get('accuracy', 0)
        })
        overview_data.append({
            'Category': 'Router',
            'Metric': 'Avg Response Time (s)',
            'Value': router.get('avg_response_time', 0)
        })
        overview_data.append({
            'Category': 'Router',
            'Metric': 'Total Queries',
            'Value': router.get('total_queries', 0)
        })
        
        # Search metrics
        search = self.data.get('search_metrics', {})
        overview_data.append({
            'Category': 'Search',
            'Metric': 'KB Avg Score',
            'Value': search.get('knowledge_base', {}).get('avg_score', 0)
        })
        overview_data.append({
            'Category': 'Search',
            'Metric': 'Case Avg Score',
            'Value': search.get('case_search', {}).get('avg_score', 0)
        })
        
        # Response metrics
        response = self.data.get('response_metrics', {})
        overview_data.append({
            'Category': 'Response',
            'Metric': 'Success Rate (%)',
            'Value': response.get('success_rate', 0)
        })
        overview_data.append({
            'Category': 'Response',
            'Metric': 'Avg Response Time (s)',
            'Value': response.get('avg_response_time', 0)
        })
        
        # System health
        health = self.data.get('system_health', {})
        overview_data.append({
            'Category': 'System Health',
            'Metric': 'Overall Score (%)',
            'Value': health.get('overall_score', 0)
        })
        overview_data.append({
            'Category': 'System Health',
            'Metric': 'Healthy Components',
            'Value': f"{health.get('healthy_components', 0)}/{health.get('total_components', 0)}"
        })
        
        # Export overview table
        overview_file = self.output_dir / "metrics_overview_table.csv"
        overview_df = pd.DataFrame(overview_data)
        overview_df.to_csv(overview_file, index=False)
        logger.info(f"✅ Metrics overview table exported to: {overview_file}")
        
        return str(overview_file)
    
    def export_all_metrics(self):
        """Export all metrics to CSV files"""
        logger.info("🔄 Converting JSON metrics to CSV files...")
        
        if not self.load_json_data():
            return False
        
        results = {}
        
        try:
            # Export all metric types
            results['router'] = self.export_router_metrics()
            results['search'] = self.export_search_metrics()
            results['response'] = self.export_response_metrics()
            results['conversation'] = self.export_conversation_metrics()
            results['system_health'] = self.export_system_health_metrics()
            results['performance'] = self.export_performance_metrics()
            results['comprehensive_summary'] = self.export_comprehensive_summary()
            results['overview_table'] = self.create_metrics_overview_table()
            
            # Create export summary
            export_summary = {
                'export_timestamp': datetime.now().isoformat(),
                'source_json': str(self.json_file_path),
                'output_directory': str(self.output_dir),
                'exported_files': results
            }
            
            # Save export summary
            summary_file = self.output_dir / "export_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(export_summary, f, indent=2)
            
            logger.info(f"✅ All metrics exported successfully to: {self.output_dir}")
            logger.info(f"📁 Export summary saved to: {summary_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False


def main():
    """Main function to convert JSON to CSV"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python 15_json_to_csv.py <json_file_path>")
        print("Example: python 15_json_to_csv.py large_scale_metrics_20250123_120000.json")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    if not Path(json_file_path).exists():
        print(f"❌ JSON file not found: {json_file_path}")
        sys.exit(1)
    
    converter = MetricsJSONToCSV(json_file_path)
    success = converter.export_all_metrics()
    
    if success:
        print("🎉 JSON to CSV conversion completed successfully!")
        print(f"📁 Check the csv_exports directory for all CSV files.")
    else:
        print("❌ JSON to CSV conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

