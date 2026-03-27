#!/usr/bin/env python3
"""
Metrics Visualization Generator
Creates comprehensive visualizations from CSV metrics data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsVisualizer:
    """Generate comprehensive visualizations from metrics CSV data"""
    
    def __init__(self, csv_dir_path):
        self.csv_dir = Path(csv_dir_path)
        self.output_dir = self.csv_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_csv_data(self):
        """Load all CSV data files"""
        logger.info("📊 Loading CSV data files...")
        
        self.data = {}
        
        # Load overview table
        overview_file = self.csv_dir / "metrics_overview_table.csv"
        if overview_file.exists():
            self.data['overview'] = pd.read_csv(overview_file)
            logger.info(f"✅ Loaded overview data: {len(self.data['overview'])} rows")
        
        # Load router metrics
        router_detailed = self.csv_dir / "router_detailed_metrics.csv"
        router_summary = self.csv_dir / "router_summary_metrics.csv"
        if router_detailed.exists():
            self.data['router_detailed'] = pd.read_csv(router_detailed)
            logger.info(f"✅ Loaded router detailed data: {len(self.data['router_detailed'])} rows")
        if router_summary.exists():
            self.data['router_summary'] = pd.read_csv(router_summary)
            logger.info(f"✅ Loaded router summary data: {len(self.data['router_summary'])} rows")
        
        # Load search metrics
        search_detailed = self.csv_dir / "search_detailed_metrics.csv"
        search_summary = self.csv_dir / "search_summary_metrics.csv"
        if search_detailed.exists():
            self.data['search_detailed'] = pd.read_csv(search_detailed)
            logger.info(f"✅ Loaded search detailed data: {len(self.data['search_detailed'])} rows")
        if search_summary.exists():
            self.data['search_summary'] = pd.read_csv(search_summary)
            logger.info(f"✅ Loaded search summary data: {len(self.data['search_summary'])} rows")
        
        # Load response metrics
        response_detailed = self.csv_dir / "response_detailed_metrics.csv"
        response_summary = self.csv_dir / "response_summary_metrics.csv"
        if response_detailed.exists():
            self.data['response_detailed'] = pd.read_csv(response_detailed)
            logger.info(f"✅ Loaded response detailed data: {len(self.data['response_detailed'])} rows")
        if response_summary.exists():
            self.data['response_summary'] = pd.read_csv(response_summary)
            logger.info(f"✅ Loaded response summary data: {len(self.data['response_summary'])} rows")
        
        # Load conversation metrics
        conversation_detailed = self.csv_dir / "conversation_detailed_metrics.csv"
        conversation_summary = self.csv_dir / "conversation_summary_metrics.csv"
        if conversation_detailed.exists():
            self.data['conversation_detailed'] = pd.read_csv(conversation_detailed)
            logger.info(f"✅ Loaded conversation detailed data: {len(self.data['conversation_detailed'])} rows")
        if conversation_summary.exists():
            self.data['conversation_summary'] = pd.read_csv(conversation_summary)
            logger.info(f"✅ Loaded conversation summary data: {len(self.data['conversation_summary'])} rows")
        
        # Load system health metrics
        health_components = self.csv_dir / "system_health_components.csv"
        health_summary = self.csv_dir / "system_health_summary.csv"
        if health_components.exists():
            self.data['health_components'] = pd.read_csv(health_components)
            logger.info(f"✅ Loaded health components data: {len(self.data['health_components'])} rows")
        if health_summary.exists():
            self.data['health_summary'] = pd.read_csv(health_summary)
            logger.info(f"✅ Loaded health summary data: {len(self.data['health_summary'])} rows")
        
        # Load performance metrics
        performance_file = self.csv_dir / "performance_metrics.csv"
        if performance_file.exists():
            self.data['performance'] = pd.read_csv(performance_file)
            logger.info(f"✅ Loaded performance data: {len(self.data['performance'])} rows")
        
        logger.info(f"📊 Loaded {len(self.data)} datasets")
        return len(self.data) > 0
    
    def create_overview_dashboard(self):
        """Create comprehensive overview dashboard"""
        logger.info("📊 Creating overview dashboard...")
        
        if 'overview' not in self.data:
            logger.warning("No overview data available")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'System Health', 'Router Performance', 'Search Quality'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Performance metrics
        performance_data = self.data['overview'][self.data['overview']['Category'].isin(['Router', 'Response'])]
        if not performance_data.empty:
            fig.add_trace(
                go.Bar(
                    x=performance_data['Metric'],
                    y=performance_data['Value'],
                    name='Performance',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
        
        # System health pie chart
        health_data = self.data['overview'][self.data['overview']['Category'] == 'System Health']
        if not health_data.empty:
            fig.add_trace(
                go.Pie(
                    labels=health_data['Metric'],
                    values=health_data['Value'],
                    name='System Health'
                ),
                row=1, col=2
            )
        
        # Router performance
        router_data = self.data['overview'][self.data['overview']['Category'] == 'Router']
        if not router_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=router_data['Metric'],
                    y=router_data['Value'],
                    mode='markers+lines',
                    name='Router',
                    marker=dict(size=10, color='red')
                ),
                row=2, col=1
            )
        
        # Search quality
        search_data = self.data['overview'][self.data['overview']['Category'] == 'Search']
        if not search_data.empty:
            fig.add_trace(
                go.Bar(
                    x=search_data['Metric'],
                    y=search_data['Value'],
                    name='Search Quality',
                    marker_color='green'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="AI legal Assistant - Comprehensive Metrics Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save dashboard
        dashboard_file = self.output_dir / "overview_dashboard.html"
        fig.write_html(str(dashboard_file))
        logger.info(f"✅ Overview dashboard saved to: {dashboard_file}")
        
        return str(dashboard_file)
    
    def create_router_analysis(self):
        """Create router performance analysis"""
        logger.info("🧠 Creating router analysis...")
        
        if 'router_detailed' not in self.data:
            logger.warning("No router detailed data available")
            return None
        
        router_data = self.data['router_detailed']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time Distribution', 'Tool Distribution', 'Accuracy by Tool', 'Confidence Distribution'),
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Response time distribution
        fig.add_trace(
            go.Histogram(
                x=router_data['response_time'],
                name='Response Time',
                nbinsx=20,
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Tool distribution
        tool_counts = router_data['predicted_tool'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=tool_counts.index,
                values=tool_counts.values,
                name='Tool Distribution'
            ),
            row=1, col=2
        )
        
        # Accuracy by tool
        accuracy_by_tool = router_data.groupby('predicted_tool')['is_correct'].mean() * 100
        fig.add_trace(
            go.Bar(
                x=accuracy_by_tool.index,
                y=accuracy_by_tool.values,
                name='Accuracy by Tool',
                marker_color='green'
            ),
            row=2, col=1
        )
        
        # Confidence distribution
        confidence_counts = router_data['confidence'].value_counts()
        fig.add_trace(
            go.Bar(
                x=confidence_counts.index,
                y=confidence_counts.values,
                name='Confidence Distribution',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Router Performance Analysis",
            showlegend=True,
            height=800
        )
        
        # Save analysis
        analysis_file = self.output_dir / "router_analysis.html"
        fig.write_html(str(analysis_file))
        logger.info(f"✅ Router analysis saved to: {analysis_file}")
        
        return str(analysis_file)
    
    def create_search_quality_analysis(self):
        """Create search quality analysis"""
        logger.info("🔍 Creating search quality analysis...")
        
        if 'search_detailed' not in self.data:
            logger.warning("No search detailed data available")
            return None
        
        search_data = self.data['search_detailed']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Knowledge Base vs Case Search Scores', 'Response Time Comparison', 'Results Count Distribution', 'Score Distribution'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # KB vs Case Search scores
        fig.add_trace(
            go.Scatter(
                x=search_data['kb_avg_score'],
                y=search_data['case_avg_score'],
                mode='markers',
                name='KB vs Case Scores',
                marker=dict(size=8, color='blue', opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Response time comparison
        fig.add_trace(
            go.Bar(
                x=['Knowledge Base', 'Case Search'],
                y=[search_data['kb_response_time'].mean(), search_data['case_response_time'].mean()],
                name='Avg Response Time',
                marker_color=['lightblue', 'lightgreen']
            ),
            row=1, col=2
        )
        
        # Results count distribution
        fig.add_trace(
            go.Histogram(
                x=search_data['kb_results_count'],
                name='KB Results Count',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Score distribution
        fig.add_trace(
            go.Histogram(
                x=search_data['kb_avg_score'],
                name='KB Score Distribution',
                marker_color='lightgreen',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Search Quality Analysis",
            showlegend=True,
            height=800
        )
        
        # Save analysis
        analysis_file = self.output_dir / "search_quality_analysis.html"
        fig.write_html(str(analysis_file))
        logger.info(f"✅ Search quality analysis saved to: {analysis_file}")
        
        return str(analysis_file)
    
    def create_response_time_analysis(self):
        """Create response time analysis"""
        logger.info("⏱️ Creating response time analysis...")
        
        if 'response_detailed' not in self.data:
            logger.warning("No response detailed data available")
            return None
        
        response_data = self.data['response_detailed']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time Distribution', 'Response Time by Tool', 'Success Rate Analysis', 'Response Length vs Time'),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Response time distribution
        fig.add_trace(
            go.Histogram(
                x=response_data['response_time'],
                name='Response Time Distribution',
                nbinsx=20,
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Response time by tool
        tools = response_data['tools_used'].unique()
        for tool in tools:
            tool_data = response_data[response_data['tools_used'] == tool]['response_time']
            fig.add_trace(
                go.Box(
                    y=tool_data,
                    name=tool,
                    boxpoints='outliers'
                ),
                row=1, col=2
            )
        
        # Success rate analysis
        success_by_tool = response_data.groupby('tools_used')['success'].mean() * 100
        fig.add_trace(
            go.Bar(
                x=success_by_tool.index,
                y=success_by_tool.values,
                name='Success Rate by Tool',
                marker_color='green'
            ),
            row=2, col=1
        )
        
        # Response length vs time
        fig.add_trace(
            go.Scatter(
                x=response_data['response_length'],
                y=response_data['response_time'],
                mode='markers',
                name='Length vs Time',
                marker=dict(size=6, color='red', opacity=0.6)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Response Time Analysis",
            showlegend=True,
            height=800
        )
        
        # Save analysis
        analysis_file = self.output_dir / "response_time_analysis.html"
        fig.write_html(str(analysis_file))
        logger.info(f"✅ Response time analysis saved to: {analysis_file}")
        
        return str(analysis_file)
    
    def create_system_health_dashboard(self):
        """Create system health dashboard"""
        logger.info("🏥 Creating system health dashboard...")
        
        if 'health_components' not in self.data:
            logger.warning("No health components data available")
            return None
        
        health_data = self.data['health_components']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Component Status', 'Response Times', 'Error Analysis', 'Health Overview'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Component status pie chart
        status_counts = health_data['status'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                name='Component Status'
            ),
            row=1, col=1
        )
        
        # Response times
        healthy_components = health_data[health_data['status'] == 'healthy']
        if not healthy_components.empty:
            fig.add_trace(
                go.Bar(
                    x=healthy_components['component'],
                    y=healthy_components['response_time'],
                    name='Response Times',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
        
        # Error analysis
        error_components = health_data[health_data['status'] == 'unhealthy']
        if not error_components.empty:
            fig.add_trace(
                go.Bar(
                    x=error_components['component'],
                    y=[1] * len(error_components),
                    name='Components with Errors',
                    marker_color='red'
                ),
                row=2, col=1
            )
        
        # Health overview indicator
        if 'health_summary' in self.data:
            health_summary = self.data['health_summary']
            overall_score = health_summary[health_summary['metric'] == 'overall_score']['value'].iloc[0]
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    title={'text': "Overall Health Score"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "green"}]}
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="System Health Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save dashboard
        dashboard_file = self.output_dir / "system_health_dashboard.html"
        fig.write_html(str(dashboard_file))
        logger.info(f"✅ System health dashboard saved to: {dashboard_file}")
        
        return str(dashboard_file)
    
    def create_performance_trends(self):
        """Create performance trends analysis"""
        logger.info("📈 Creating performance trends...")
        
        if 'performance' not in self.data:
            logger.warning("No performance data available")
            return None
        
        performance_data = self.data['performance']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Categories', 'Throughput Metrics', 'Reliability Metrics', 'System Resources'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Performance categories
        category_counts = performance_data['category'].value_counts()
        fig.add_trace(
            go.Bar(
                x=category_counts.index,
                y=category_counts.values,
                name='Performance Categories',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Throughput metrics
        throughput_data = performance_data[performance_data['category'] == 'throughput']
        if not throughput_data.empty:
            fig.add_trace(
                go.Bar(
                    x=throughput_data['metric'],
                    y=throughput_data['value'],
                    name='Throughput',
                    marker_color='green'
                ),
                row=1, col=2
            )
        
        # Reliability metrics
        reliability_data = performance_data[performance_data['category'] == 'reliability']
        if not reliability_data.empty:
            fig.add_trace(
                go.Bar(
                    x=reliability_data['metric'],
                    y=reliability_data['value'],
                    name='Reliability',
                    marker_color='orange'
                ),
                row=2, col=1
            )
        
        # System resources
        resources_data = performance_data[performance_data['category'] == 'system_resources']
        if not resources_data.empty:
            fig.add_trace(
                go.Bar(
                    x=resources_data['metric'],
                    y=resources_data['value'],
                    name='System Resources',
                    marker_color='purple'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Performance Trends Analysis",
            showlegend=True,
            height=800
        )
        
        # Save trends
        trends_file = self.output_dir / "performance_trends.html"
        fig.write_html(str(trends_file))
        logger.info(f"✅ Performance trends saved to: {trends_file}")
        
        return str(trends_file)
    
    def create_static_plots(self):
        """Create static matplotlib plots"""
        logger.info("📊 Creating static plots...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Wakalat Sewa V2 - Metrics Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Router accuracy
        if 'router_summary' in self.data:
            router_summary = self.data['router_summary']
            accuracy_data = router_summary[router_summary['metric'] == 'accuracy']
            if not accuracy_data.empty:
                axes[0, 0].bar(['Router Accuracy'], [accuracy_data['value'].iloc[0]])
                axes[0, 0].set_title('Router Accuracy (%)')
                axes[0, 0].set_ylim(0, 100)
        
        # Plot 2: Response times
        if 'response_summary' in self.data:
            response_summary = self.data['response_summary']
            avg_time = response_summary[response_summary['metric'] == 'avg_response_time']
            if not avg_time.empty:
                axes[0, 1].bar(['Avg Response Time'], [avg_time['value'].iloc[0]])
                axes[0, 1].set_title('Average Response Time (s)')
        
        # Plot 3: Search quality
        if 'search_summary' in self.data:
            search_summary = self.data['search_summary']
            kb_score = search_summary[search_summary['source'] == 'knowledge_base']['avg_score'].iloc[0]
            case_score = search_summary[search_summary['source'] == 'case_search']['avg_score'].iloc[0]
            axes[0, 2].bar(['Knowledge Base', 'Case Search'], [kb_score, case_score])
            axes[0, 2].set_title('Search Quality Scores')
        
        # Plot 4: System health
        if 'health_summary' in self.data:
            health_summary = self.data['health_summary']
            overall_score = health_summary[health_summary['metric'] == 'overall_score']['value'].iloc[0]
            axes[1, 0].bar(['System Health'], [overall_score])
            axes[1, 0].set_title('System Health Score (%)')
            axes[1, 0].set_ylim(0, 100)
        
        # Plot 5: Success rate
        if 'response_summary' in self.data:
            response_summary = self.data['response_summary']
            success_rate = response_summary[response_summary['metric'] == 'success_rate']
            if not success_rate.empty:
                axes[1, 1].bar(['Success Rate'], [success_rate['value'].iloc[0]])
                axes[1, 1].set_title('Success Rate (%)')
                axes[1, 1].set_ylim(0, 100)
        
        # Plot 6: Component health
        if 'health_components' in self.data:
            health_components = self.data['health_components']
            status_counts = health_components['status'].value_counts()
            axes[1, 2].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
            axes[1, 2].set_title('Component Health Status')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save static plots
        static_file = self.output_dir / "metrics_overview_static.png"
        plt.savefig(static_file, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Static plots saved to: {static_file}")
        
        return str(static_file)
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        logger.info("🎨 Generating all visualizations...")
        
        if not self.load_csv_data():
            logger.error("❌ Failed to load CSV data")
            return False
        
        results = {}
        
        try:
            # Generate all visualizations
            results['overview_dashboard'] = self.create_overview_dashboard()
            results['router_analysis'] = self.create_router_analysis()
            results['search_quality_analysis'] = self.create_search_quality_analysis()
            results['response_time_analysis'] = self.create_response_time_analysis()
            results['system_health_dashboard'] = self.create_system_health_dashboard()
            results['performance_trends'] = self.create_performance_trends()
            results['static_plots'] = self.create_static_plots()
            
            # Create visualization index
            index_content = self._create_visualization_index(results)
            index_file = self.output_dir / "index.html"
            # with open(index_file, 'w') as f:
            #     f.write(index_content)
            
            with open(index_file, 'w', encoding='utf-8') as f:   # ★ fix: specify utf-8 encoding
                f.write(index_content)

            logger.info(f"✅ All visualizations generated successfully!")
            logger.info(f"📁 Visualizations saved to: {self.output_dir}")
            logger.info(f"🌐 Open {index_file} to view all visualizations")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Visualization generation failed: {e}")
            return False
    
    def _create_visualization_index(self, results):
        """Create HTML index for all visualizations"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI legal Assistant - Metrics Visualizations</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .visualization {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa; }}
                .visualization h3 {{ color: #2c3e50; margin-top: 0; }}
                .visualization p {{ color: #7f8c8d; margin: 10px 0; }}
                .visualization a {{ color: #3498db; text-decoration: none; font-weight: bold; }}
                .visualization a:hover {{ color: #2980b9; }}
                .timestamp {{ text-align: center; color: #7f8c8d; font-style: italic; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏛️ AI legal Assistant - Metrics Visualizations</h1>
                
                <div class="visualization">
                    <h3>📊 Overview Dashboard</h3>
                    <p>Comprehensive metrics dashboard showing all key performance indicators</p>
                    <a href="overview_dashboard.html" target="_blank">View Interactive Dashboard</a>
                </div>
                
                <div class="visualization">
                    <h3>🧠 Router Analysis</h3>
                    <p>Detailed analysis of router performance, accuracy, and tool distribution</p>
                    <a href="router_analysis.html" target="_blank">View Router Analysis</a>
                </div>
                
                <div class="visualization">
                    <h3>🔍 Search Quality Analysis</h3>
                    <p>Analysis of search quality metrics for knowledge base and case search</p>
                    <a href="search_quality_analysis.html" target="_blank">View Search Analysis</a>
                </div>
                
                <div class="visualization">
                    <h3>⏱️ Response Time Analysis</h3>
                    <p>Detailed analysis of response times, success rates, and performance trends</p>
                    <a href="response_time_analysis.html" target="_blank">View Response Analysis</a>
                </div>
                
                <div class="visualization">
                    <h3>🏥 System Health Dashboard</h3>
                    <p>Comprehensive system health monitoring and component status</p>
                    <a href="system_health_dashboard.html" target="_blank">View Health Dashboard</a>
                </div>
                
                <div class="visualization">
                    <h3>📈 Performance Trends</h3>
                    <p>Analysis of performance trends, throughput, and reliability metrics</p>
                    <a href="performance_trends.html" target="_blank">View Performance Trends</a>
                </div>
                
                <div class="visualization">
                    <h3>📊 Static Overview</h3>
                    <p>Static overview plots for quick reference and reporting</p>
                    <a href="metrics_overview_static.png" target="_blank">View Static Plots</a>
                </div>
                
                <div class="timestamp">
                    Generated on: {timestamp}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content


def main():
    """Main function to generate visualizations"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python 16_metrics_visualization.py <csv_directory_path>")
        print("Example: python 16_metrics_visualization.py csv_exports")
        sys.exit(1)
    
    csv_dir_path = sys.argv[1]
    
    if not Path(csv_dir_path).exists():
        print(f"❌ CSV directory not found: {csv_dir_path}")
        sys.exit(1)
    
    visualizer = MetricsVisualizer(csv_dir_path)
    success = visualizer.generate_all_visualizations()
    
    if success:
        print("🎉 All visualizations generated successfully!")
        print(f"📁 Check the visualizations directory for all HTML files and images.")
        print(f"🌐 Open the index.html file in your browser to view all visualizations.")
    else:
        print("❌ Visualization generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

