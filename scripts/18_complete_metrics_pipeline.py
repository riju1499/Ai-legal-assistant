#!/usr/bin/env python3
"""
Complete Metrics Pipeline
Runs the entire metrics collection, analysis, and visualization pipeline
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteMetricsPipeline:
    """Complete metrics pipeline orchestrator"""
    
    def __init__(self):
        self.scripts_dir = Path(r"D:/FinalAIproj/Wakalat Sewa/wakalt/tozip/scripts")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def run_large_scale_metrics(self):
        """Run large-scale metrics collection"""
        logger.info("🚀 Step 1: Running large-scale metrics collection...")
        
        try:
            cmd = [
                sys.executable, 
                str(self.scripts_dir / "14_large_scale_metrics.py")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.scripts_dir)
            
            if result.returncode == 0:
                logger.info("✅ Large-scale metrics collection completed successfully")
                
                # Find the generated JSON file
                json_files = list(self.scripts_dir.glob("large_scale_metrics_*.json"))
                if json_files:
                    latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
                    self.results['json_file'] = str(latest_json)
                    logger.info(f"📁 JSON results saved to: {latest_json}")
                    return True
                else:
                    logger.error("❌ No JSON file generated")
                    return False
            else:
                logger.error(f"❌ Large-scale metrics collection failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running large-scale metrics: {e}")
            return False
    
    def run_json_to_csv(self):
        """Convert JSON to CSV"""
        logger.info("📊 Step 2: Converting JSON to CSV...")
        
        if 'json_file' not in self.results:
            logger.error("❌ No JSON file available for conversion")
            return False
        
        try:
            cmd = [
                sys.executable,
                str(self.scripts_dir / "15_json_to_csv.py"),
                self.results['json_file']
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.scripts_dir)
            
            if result.returncode == 0:
                logger.info("✅ JSON to CSV conversion completed successfully")
                
                # Find the CSV directory
                csv_dir = Path(self.results['json_file']).parent / "csv_exports"
                if csv_dir.exists():
                    self.results['csv_dir'] = str(csv_dir)
                    logger.info(f"📁 CSV files saved to: {csv_dir}")
                    return True
                else:
                    logger.error("❌ CSV directory not found")
                    return False
            else:
                logger.error(f"❌ JSON to CSV conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running JSON to CSV conversion: {e}")
            return False
    
    def run_visualization(self):
        """Generate visualizations"""
        logger.info("🎨 Step 3: Generating visualizations...")
        
        if 'csv_dir' not in self.results:
            logger.error("❌ No CSV directory available for visualization")
            return False
        
        try:
            cmd = [
                sys.executable,
                str(self.scripts_dir / "16_metrics_visualization.py"),
                self.results['csv_dir']
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.scripts_dir)
            
            if result.returncode == 0:
                logger.info("✅ Visualization generation completed successfully")
                
                # Find the visualizations directory
                viz_dir = Path(self.results['csv_dir']) / "visualizations"
                if viz_dir.exists():
                    self.results['viz_dir'] = str(viz_dir)
                    logger.info(f"📁 Visualizations saved to: {viz_dir}")
                    return True
                else:
                    logger.error("❌ Visualizations directory not found")
                    return False
            else:
                logger.error(f"❌ Visualization generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running visualization generation: {e}")
            return False
    
    def run_comprehensive_tables(self):
        """Generate comprehensive metrics tables"""
        logger.info("📋 Step 4: Generating comprehensive metrics tables...")
        
        if 'json_file' not in self.results:
            logger.error("❌ No JSON file available for table generation")
            return False
        
        try:
            cmd = [
                sys.executable,
                str(self.scripts_dir / "17_comprehensive_metrics_table.py"),
                self.results['json_file']
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.scripts_dir)
            
            if result.returncode == 0:
                logger.info("✅ Comprehensive tables generation completed successfully")
                
                # Find the metrics tables directory
                tables_dir = Path(self.results['json_file']).parent / "metrics_tables"
                if tables_dir.exists():
                    self.results['tables_dir'] = str(tables_dir)
                    logger.info(f"📁 Metrics tables saved to: {tables_dir}")
                    return True
                else:
                    logger.error("❌ Metrics tables directory not found")
                    return False
            else:
                logger.error(f"❌ Comprehensive tables generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running comprehensive tables generation: {e}")
            return False
    
    def create_pipeline_summary(self):
        """Create comprehensive pipeline summary"""
        logger.info("📋 Creating pipeline summary...")
        
        summary_content = f"""
# Wakalat Sewa V2 - Complete Metrics Pipeline Summary

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Pipeline Results

### 1. Large-Scale Metrics Collection
- **Status:** {'✅ Completed' if 'json_file' in self.results else '❌ Failed'}
- **JSON File:** {self.results.get('json_file', 'N/A')}
- **Description:** Comprehensive metrics collection with 50+ router tests, 30+ search tests, 20+ response tests, and 10+ conversation tests

### 2. JSON to CSV Conversion
- **Status:** {'✅ Completed' if 'csv_dir' in self.results else '❌ Failed'}
- **CSV Directory:** {self.results.get('csv_dir', 'N/A')}
- **Description:** Converted detailed JSON metrics to structured CSV files for analysis

### 3. Visualization Generation
- **Status:** {'✅ Completed' if 'viz_dir' in self.results else '❌ Failed'}
- **Visualizations Directory:** {self.results.get('viz_dir', 'N/A')}
- **Description:** Generated interactive HTML dashboards and static plots

### 4. Comprehensive Tables
- **Status:** {'✅ Completed' if 'tables_dir' in self.results else '❌ Failed'}
- **Tables Directory:** {self.results.get('tables_dir', 'N/A')}
- **Description:** Created detailed metrics tables for all system components

## Generated Files

### JSON Metrics
- Large-scale metrics JSON file with comprehensive test results
- Contains router, search, response, conversation, and system health metrics

### CSV Files
- Router detailed and summary metrics
- Search quality metrics (knowledge base and case search)
- Response time and performance metrics
- Conversation memory metrics
- System health component status
- Performance and reliability metrics
- Comprehensive overview table

### Visualizations
- Interactive HTML dashboards
- Router performance analysis
- Search quality analysis
- Response time analysis
- System health dashboard
- Performance trends
- Static overview plots

### Comprehensive Tables
- Detailed metrics tables for each component
- Master metrics table with all key indicators
- Component-specific analysis tables

## Usage Instructions

1. **View Visualizations:** Open `{self.results.get('viz_dir', 'N/A')}/index.html` in your browser
2. **Access Tables:** Open `{self.results.get('tables_dir', 'N/A')}/index.html` in your browser
3. **Download CSV Files:** All CSV files are available in the respective directories
4. **Analyze JSON Data:** Use the original JSON file for custom analysis

## Next Steps

1. Review the generated visualizations to identify performance trends
2. Analyze the comprehensive tables for detailed metrics
3. Use the CSV files for further analysis in Excel, Python, or other tools
4. Monitor system performance over time using these baseline metrics

## Troubleshooting

If any step failed:
1. Check the error messages in the logs
2. Ensure all dependencies are installed
3. Verify file paths and permissions
4. Re-run individual steps as needed

---
*Generated by Wakalat Sewa V2 Complete Metrics Pipeline*
"""
        
        summary_file = self.scripts_dir / f"pipeline_summary_{self.timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        logger.info(f"✅ Pipeline summary saved to: {summary_file}")
        return str(summary_file)
    
    def run_complete_pipeline(self):
        """Run the complete metrics pipeline"""
        logger.info("🚀 Starting Complete Metrics Pipeline for Wakalat Sewa V2")
        logger.info("=" * 80)
        
        pipeline_steps = [
            ("Large-Scale Metrics Collection", self.run_large_scale_metrics),
            ("JSON to CSV Conversion", self.run_json_to_csv),
            ("Visualization Generation", self.run_visualization),
            ("Comprehensive Tables Generation", self.run_comprehensive_tables)
        ]
        
        success_count = 0
        
        for step_name, step_function in pipeline_steps:
            logger.info(f"\n🔄 Running: {step_name}")
            try:
                if step_function():
                    success_count += 1
                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    logger.error(f"❌ {step_name} failed")
            except Exception as e:
                logger.error(f"❌ {step_name} failed with error: {e}")
        
        # Create pipeline summary
        summary_file = self.create_pipeline_summary()
        
        # Print final results
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPLETE METRICS PIPELINE RESULTS")
        logger.info("=" * 80)
        logger.info(f"✅ Successful Steps: {success_count}/{len(pipeline_steps)}")
        logger.info(f"📁 Pipeline Summary: {summary_file}")
        
        if 'json_file' in self.results:
            logger.info(f"📄 JSON Metrics: {self.results['json_file']}")
        if 'csv_dir' in self.results:
            logger.info(f"📊 CSV Files: {self.results['csv_dir']}")
        if 'viz_dir' in self.results:
            logger.info(f"🎨 Visualizations: {self.results['viz_dir']}")
        if 'tables_dir' in self.results:
            logger.info(f"📋 Tables: {self.results['tables_dir']}")
        
        if success_count == len(pipeline_steps):
            logger.info("🎉 Complete metrics pipeline completed successfully!")
            return True
        else:
            logger.warning(f"⚠️ Pipeline completed with {len(pipeline_steps) - success_count} failures")
            return False


def main():
    """Main function to run the complete metrics pipeline"""
    pipeline = CompleteMetricsPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Complete metrics pipeline finished successfully!")
        print("📁 Check the generated files and directories for all results.")
        print("🌐 Open the HTML index files in your browser to view visualizations and tables.")
    else:
        print("\n⚠️ Pipeline completed with some failures.")
        print("📋 Check the pipeline summary for details on what succeeded and what failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

