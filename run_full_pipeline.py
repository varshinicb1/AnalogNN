"""
Full End-to-End Pipeline Test
============================

Runs the complete OpenAnalogNN pipeline with all new integrations.
"""

import torch
import os
from experiments.runner import ExperimentRunner
from utils.mlflow_tracker import MLflowTracker

print("=" * 60)
print("Running Full End-to-End Pipeline")
print("=" * 60)

# Initialize MLflow tracker
tracker = MLflowTracker(experiment_name="OpenAnalogNN_Full_Pipeline")
tracker.start_run(run_name="full_pipeline_test")

try:
    # Load config
    tracker.log_config({'pipeline': 'full_test', 'mode': 'integration'})
    
    # Initialize runner
    runner = ExperimentRunner(config_path="./configs/config.yaml")
    
    # Run full pipeline
    print("\nStarting full pipeline execution...")
    results = runner.run_full_pipeline()
    
    # Log results to MLflow
    tracker.log_metrics({
        'pipeline_success': 1.0,
        'benchmark_rmse': results.get('benchmark', {}).get('rmse_post_calibration', 0.0)
    })
    
    # Log generated figures
    if os.path.exists("./figures"):
        for fig_file in os.listdir("./figures"):
            if fig_file.endswith('.html'):
                tracker.log_artifact(f"./figures/{fig_file}", "plotly_figures")
    
    print("\n" + "=" * 60)
    print("Full Pipeline Completed Successfully!")
    print("=" * 60)
    print(f"\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    tracker.end_run(status="FINISHED")
    
except Exception as e:
    print(f"\nPipeline failed: {e}")
    import traceback
    traceback.print_exc()
    tracker.end_run(status="FAILED")
    raise
