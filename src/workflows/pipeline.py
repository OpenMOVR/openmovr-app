"""
Workflow Pipeline for MOVR Clinical Analytics

Provides orchestration for common data science workflows including:
- Data loading and validation
- Data cleaning and preprocessing  
- Enrollment validation
- Analytics execution
- Report generation

Supports both interactive and batch execution modes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import traceback

from ..config import get_config
from ..data_processing.loader import get_loader
from ..cleaning.cleaner import get_cleaner
from ..validation.enrollment import get_enrollment_validator


logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Individual step in a workflow pipeline."""
    name: str
    function: Callable
    depends_on: List[str] = None
    params: Dict[str, Any] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Any = None
    error: str = None
    start_time: datetime = None
    end_time: datetime = None
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []
        if self.params is None:
            self.params = {}


class WorkflowPipeline:
    """
    Orchestrates multi-step data science workflows with dependency management.
    
    Features:
    - Step dependency resolution
    - Error handling and recovery
    - Progress tracking and logging
    - Result caching
    - Parallel execution support
    """
    
    def __init__(self, name: str, data_path: Optional[Path] = None):
        self.name = name
        self.config = get_config()
        self.data_path = data_path or self.config.paths.data_dir
        
        # Initialize components
        self.loader = get_loader(self.data_path)
        self.cleaner = get_cleaner(self.data_path)
        self.enrollment_validator = get_enrollment_validator(self.data_path)
        
        # Pipeline state
        self.steps = {}
        self.execution_order = []
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.status = WorkflowStatus.PENDING
    
    def add_step(self, 
                name: str,
                function: Callable,
                depends_on: List[str] = None,
                params: Dict[str, Any] = None) -> 'WorkflowPipeline':
        """
        Add a step to the pipeline.
        
        Args:
            name: Unique step name
            function: Function to execute
            depends_on: List of step names this step depends on
            params: Parameters to pass to the function
            
        Returns:
            Self for method chaining
        """
        if name in self.steps:
            raise ValueError(f"Step '{name}' already exists in pipeline")
        
        step = WorkflowStep(
            name=name,
            function=function,
            depends_on=depends_on or [],
            params=params or {}
        )
        
        self.steps[name] = step
        self._resolve_execution_order()
        
        return self
    
    def _resolve_execution_order(self):
        """Resolve step execution order based on dependencies."""
        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(step_name):
            if step_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{step_name}'")
            if step_name in visited:
                return
            
            temp_visited.add(step_name)
            
            step = self.steps[step_name]
            for dependency in step.depends_on:
                if dependency not in self.steps:
                    raise ValueError(f"Unknown dependency '{dependency}' for step '{step_name}'")
                visit(dependency)
            
            temp_visited.remove(step_name)
            visited.add(step_name)
            order.append(step_name)
        
        for step_name in self.steps:
            visit(step_name)
        
        self.execution_order = order
    
    def execute(self, 
               steps_to_run: Optional[List[str]] = None,
               fail_fast: bool = True,
               save_results: bool = True) -> Dict[str, Any]:
        """
        Execute the workflow pipeline.
        
        Args:
            steps_to_run: Specific steps to execute (None = all steps)
            fail_fast: Whether to stop on first error
            save_results: Whether to save results to output directory
            
        Returns:
            Execution results and metadata
        """
        logger.info(f"Starting workflow pipeline: {self.name}")
        
        self.start_time = datetime.now()
        self.status = WorkflowStatus.RUNNING
        
        steps_to_execute = steps_to_run or self.execution_order
        executed_steps = []
        
        try:
            for step_name in self.execution_order:
                if step_name not in steps_to_execute:
                    self.steps[step_name].status = WorkflowStatus.SKIPPED
                    logger.info(f"Skipping step: {step_name}")
                    continue
                
                # Check dependencies
                if not self._dependencies_satisfied(step_name):
                    if fail_fast:
                        raise RuntimeError(f"Dependencies not satisfied for step '{step_name}'")
                    else:
                        logger.error(f"Skipping step '{step_name}' due to unsatisfied dependencies")
                        self.steps[step_name].status = WorkflowStatus.SKIPPED
                        continue
                
                # Execute step
                success = self._execute_step(step_name)
                executed_steps.append(step_name)
                
                if not success and fail_fast:
                    break
            
            # Determine overall status
            failed_steps = [name for name, step in self.steps.items() 
                          if step.status == WorkflowStatus.FAILED]
            
            if failed_steps:
                self.status = WorkflowStatus.FAILED
                logger.error(f"Pipeline failed. Failed steps: {failed_steps}")
            else:
                self.status = WorkflowStatus.COMPLETED
                logger.info(f"Pipeline completed successfully")
        
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            logger.error(f"Pipeline execution failed: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            self.end_time = datetime.now()
        
        # Generate execution report
        execution_results = self._generate_execution_report()
        
        # Save results if requested
        if save_results:
            self._save_pipeline_results(execution_results)
        
        return execution_results
    
    def _dependencies_satisfied(self, step_name: str) -> bool:
        """Check if all dependencies for a step are satisfied."""
        step = self.steps[step_name]
        
        for dependency in step.depends_on:
            dep_step = self.steps[dependency]
            if dep_step.status != WorkflowStatus.COMPLETED:
                return False
        
        return True
    
    def _execute_step(self, step_name: str) -> bool:
        """Execute a single pipeline step."""
        step = self.steps[step_name]
        
        logger.info(f"Executing step: {step_name}")
        step.start_time = datetime.now()
        step.status = WorkflowStatus.RUNNING
        
        try:
            # Prepare function parameters
            params = step.params.copy()
            
            # Add results from dependencies as parameters
            for dependency in step.depends_on:
                dep_result = self.steps[dependency].result
                params[f"{dependency}_result"] = dep_result
            
            # Execute the function
            step.result = step.function(**params)
            self.results[step_name] = step.result
            
            step.status = WorkflowStatus.COMPLETED
            step.end_time = datetime.now()
            
            execution_time = (step.end_time - step.start_time).total_seconds()
            logger.info(f"Step '{step_name}' completed in {execution_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            step.status = WorkflowStatus.FAILED
            step.error = str(e)
            step.end_time = datetime.now()
            
            logger.error(f"Step '{step_name}' failed: {e}")
            logger.error(traceback.format_exc())
            
            return False
    
    def _generate_execution_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution report."""
        total_time = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        step_reports = []
        for step_name in self.execution_order:
            step = self.steps[step_name]
            step_time = 0
            if step.start_time and step.end_time:
                step_time = (step.end_time - step.start_time).total_seconds()
            
            step_reports.append({
                'step_name': step_name,
                'status': step.status.value,
                'execution_time_seconds': step_time,
                'error': step.error,
                'has_result': step.result is not None
            })
        
        return {
            'pipeline_name': self.name,
            'overall_status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_execution_time_seconds': total_time,
            'total_steps': len(self.steps),
            'completed_steps': len([s for s in self.steps.values() if s.status == WorkflowStatus.COMPLETED]),
            'failed_steps': len([s for s in self.steps.values() if s.status == WorkflowStatus.FAILED]),
            'skipped_steps': len([s for s in self.steps.values() if s.status == WorkflowStatus.SKIPPED]),
            'step_details': step_reports
        }
    
    def _save_pipeline_results(self, execution_results: Dict[str, Any]):
        """Save pipeline results to output directory."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save execution report
            report_file = self.config.paths.output_dir / f"pipeline_{self.name}_{timestamp}_report.json"
            with open(report_file, 'w') as f:
                json.dump(execution_results, f, indent=2, default=str)
            
            # Save step results
            results_file = self.config.paths.output_dir / f"pipeline_{self.name}_{timestamp}_results.json"
            serializable_results = {}
            
            for step_name, result in self.results.items():
                if isinstance(result, pd.DataFrame):
                    # Save DataFrames as CSV
                    csv_file = self.config.paths.output_dir / f"{step_name}_{timestamp}.csv"
                    result.to_csv(csv_file, index=False)
                    serializable_results[step_name] = f"DataFrame saved to: {csv_file}"
                elif isinstance(result, dict):
                    serializable_results[step_name] = result
                else:
                    serializable_results[step_name] = str(result)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Pipeline results saved to: {self.config.paths.output_dir}")
            
        except Exception as e:
            logger.error(f"Could not save pipeline results: {e}")
    
    def get_step_result(self, step_name: str) -> Any:
        """Get result from a specific step."""
        if step_name not in self.steps:
            raise ValueError(f"Step '{step_name}' not found")
        
        return self.steps[step_name].result
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline configuration and status."""
        return {
            'name': self.name,
            'total_steps': len(self.steps),
            'execution_order': self.execution_order,
            'status': self.status.value,
            'step_status': {name: step.status.value for name, step in self.steps.items()}
        }


# Pre-built workflow templates
def create_data_loading_workflow(name: str = "data_loading") -> WorkflowPipeline:
    """Create a standard data loading workflow."""
    pipeline = WorkflowPipeline(name)
    
    # Step functions
    def load_data_dictionary(**kwargs):
        """Load and convert data dictionary."""
        loader = get_loader()
        data_dict = loader.data_dict
        data_dict.convert_to_json()
        return data_dict.save_json_config()
    
    def load_core_datasets(**kwargs):
        """Load core MOVR datasets."""
        loader = get_loader()
        return {
            'demographics': loader.load_demographics(),
            'diagnosis': loader.load_diagnosis(), 
            'encounters': loader.load_encounters()
        }
    
    def validate_data_quality(**kwargs):
        """Validate data quality."""
        cleaner = get_cleaner()
        datasets = kwargs.get('load_core_datasets_result', {})
        
        quality_report = {}
        for name, df in datasets.items():
            cleaned_df = cleaner.clean_dataset(df, name)
            quality_report[name] = {
                'original_rows': len(df),
                'cleaned_rows': len(cleaned_df),
                'quality_score': len(cleaned_df) / len(df) if len(df) > 0 else 0
            }
        
        return quality_report
    
    # Add steps
    pipeline.add_step("load_data_dictionary", load_data_dictionary)
    pipeline.add_step("load_core_datasets", load_core_datasets, depends_on=["load_data_dictionary"])
    pipeline.add_step("validate_data_quality", validate_data_quality, depends_on=["load_core_datasets"])
    
    return pipeline


def create_enrollment_validation_workflow(name: str = "enrollment_validation") -> WorkflowPipeline:
    """Create enrollment validation workflow."""
    pipeline = WorkflowPipeline(name)
    
    def validate_enrollment(**kwargs):
        """Validate participant enrollment."""
        validator = get_enrollment_validator()
        return validator.validate_participant_enrollment()
    
    def generate_enrollment_report(**kwargs):
        """Generate enrollment report."""
        validator = get_enrollment_validator()
        validate_enrollment_result = kwargs.get('validate_enrollment_result', {})
        validator.validation_results = validate_enrollment_result
        
        # Generate multiple report formats
        return {
            'summary': validator.get_enrollment_report('summary'),
            'detailed_csv': validator.get_enrollment_report('csv', save_to_file=True),
            'excel_report': validator.get_enrollment_report('excel', save_to_file=True)
        }
    
    pipeline.add_step("validate_enrollment", validate_enrollment)
    pipeline.add_step("generate_enrollment_report", generate_enrollment_report, 
                     depends_on=["validate_enrollment"])
    
    return pipeline


def create_full_analytics_workflow(name: str = "full_analytics") -> WorkflowPipeline:
    """Create comprehensive analytics workflow."""
    pipeline = WorkflowPipeline(name)
    
    # Combine data loading and enrollment workflows
    data_workflow = create_data_loading_workflow("data_loading_sub")
    enrollment_workflow = create_enrollment_validation_workflow("enrollment_validation_sub")
    
    # Copy steps from sub-workflows
    for step_name, step in data_workflow.steps.items():
        pipeline.add_step(f"data_{step_name}", step.function, 
                         [f"data_{dep}" for dep in step.depends_on], step.params)
    
    for step_name, step in enrollment_workflow.steps.items():
        depends_on = [f"enrollment_{dep}" for dep in step.depends_on]
        if not depends_on:  # First enrollment step depends on data loading
            depends_on = ["data_validate_data_quality"]
        
        pipeline.add_step(f"enrollment_{step_name}", step.function, depends_on, step.params)
    
    return pipeline


def get_pipeline(workflow_type: str, name: str = None) -> WorkflowPipeline:
    """Get a pre-configured workflow pipeline."""
    if name is None:
        name = workflow_type
    
    if workflow_type == "data_loading":
        return create_data_loading_workflow(name)
    elif workflow_type == "enrollment_validation":
        return create_enrollment_validation_workflow(name)
    elif workflow_type == "full_analytics":
        return create_full_analytics_workflow(name)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")