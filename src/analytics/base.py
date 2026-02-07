"""
Base Analytics Framework for MOVR Clinical Analytics

Provides foundational classes and utilities for creating modular, reusable
analytics components that can be used across different analysis types:
- Industry reports
- Publication studies  
- Annual reports
- Bespoke client analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

from ..config import get_config
from ..data_processing.loader import get_loader
from ..cleaning.cleaner import get_cleaner


logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Standard container for analysis results."""
    name: str
    description: str
    data: Union[pd.DataFrame, Dict[str, Any], Any]
    metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'data_type': type(self.data).__name__,
            'data_shape': getattr(self.data, 'shape', None) if hasattr(self.data, 'shape') else None
        }


class BaseAnalyzer(ABC):
    """
    Abstract base class for all MOVR analytics components.
    
    Provides standard interface and common functionality for:
    - Data loading and preprocessing
    - Configuration management
    - Result standardization
    - Output generation
    """
    
    def __init__(self, 
                 name: str,
                 data_path: Optional[Path] = None,
                 config_overrides: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = get_config()
        self.data_path = data_path or self.config.paths.data_dir
        
        # Initialize data components
        self.loader = get_loader(self.data_path)
        self.cleaner = get_cleaner(self.data_path)
        
        # Analysis state
        self.results = {}
        self.metadata = {
            'analyzer_name': name,
            'created_at': datetime.now(),
            'config_overrides': config_overrides or {}
        }
        
        # Apply configuration overrides
        if config_overrides:
            self._apply_config_overrides(config_overrides)
        
        # Initialize analyzer-specific components
        self._initialize()
    
    def _apply_config_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides."""
        for key, value in overrides.items():
            if hasattr(self.config.analytics, key):
                setattr(self.config.analytics, key, value)
    
    def _initialize(self):
        """Initialize analyzer-specific components. Override in subclasses."""
        pass
    
    @abstractmethod
    def run_analysis(self, **kwargs) -> Dict[str, AnalysisResult]:
        """
        Run the analysis. Must be implemented by subclasses.
        
        Returns:
            Dictionary of AnalysisResult objects
        """
        pass
    
    def load_required_data(self, 
                          datasets: Optional[List[str]] = None,
                          clean_data: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load required datasets for analysis.
        
        Args:
            datasets: List of dataset names to load (None = load default set)
            clean_data: Whether to apply data cleaning
            
        Returns:
            Dictionary of loaded DataFrames
        """
        if datasets is None:
            datasets = self.get_default_datasets()
        
        loaded_data = {}
        
        for dataset_name in datasets:
            try:
                logger.info(f"Loading dataset: {dataset_name}")
                df = self.loader.load_file(dataset_name)
                
                if clean_data:
                    df = self.cleaner.clean_dataset(df, dataset_name)
                
                loaded_data[dataset_name] = df
                
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                if self.config.analytics.strict_validation:
                    raise
        
        self.metadata['loaded_datasets'] = list(loaded_data.keys())
        self.metadata['data_load_timestamp'] = datetime.now()
        
        return loaded_data
    
    def get_default_datasets(self) -> List[str]:
        """Get default datasets required for this analyzer. Override in subclasses."""
        return ["Demographics_MainData", "Encounter_MainData", "Diagnosis_MainData"]
    
    def add_result(self, 
                   name: str, 
                   data: Any, 
                   description: str = "",
                   metadata: Optional[Dict[str, Any]] = None):
        """Add a result to the analyzer."""
        result = AnalysisResult(
            name=name,
            description=description,
            data=data,
            metadata=metadata or {}
        )
        
        self.results[name] = result
        logger.info(f"Added analysis result: {name}")
    
    def get_result(self, name: str) -> Optional[AnalysisResult]:
        """Get a specific analysis result."""
        return self.results.get(name)
    
    def get_all_results(self) -> Dict[str, AnalysisResult]:
        """Get all analysis results."""
        return self.results
    
    def save_results(self, 
                    output_dir: Optional[Path] = None,
                    formats: List[str] = None) -> Dict[str, Path]:
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results (None = use config default)
            formats: List of formats to save ('csv', 'excel', 'json', 'parquet')
            
        Returns:
            Dictionary mapping format to file paths
        """
        output_dir = output_dir or self.config.paths.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        
        formats = formats or self.config.analytics.output_formats
        saved_files = {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for result_name, result in self.results.items():
            file_base = f"{self.name}_{result_name}_{timestamp}"
            
            for format_type in formats:
                try:
                    if isinstance(result.data, pd.DataFrame):
                        file_path = self._save_dataframe(result.data, output_dir, file_base, format_type)
                    else:
                        file_path = self._save_object(result.data, output_dir, file_base, format_type)
                    
                    if format_type not in saved_files:
                        saved_files[format_type] = []
                    saved_files[format_type].append(file_path)
                    
                except Exception as e:
                    logger.error(f"Failed to save {result_name} as {format_type}: {e}")
        
        # Save metadata
        metadata_file = output_dir / f"{self.name}_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.get_metadata_summary(), f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_dir}")
        return saved_files
    
    def _save_dataframe(self, df: pd.DataFrame, output_dir: Path, file_base: str, format_type: str) -> Path:
        """Save DataFrame in specified format."""
        if format_type == 'csv':
            file_path = output_dir / f"{file_base}.csv"
            df.to_csv(file_path, index=False)
        elif format_type == 'excel':
            file_path = output_dir / f"{file_base}.xlsx"
            df.to_excel(file_path, index=False)
        elif format_type == 'parquet':
            file_path = output_dir / f"{file_base}.parquet"
            df.to_parquet(file_path, index=False)
        elif format_type == 'json':
            file_path = output_dir / f"{file_base}.json"
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format for DataFrame: {format_type}")
        
        return file_path
    
    def _save_object(self, obj: Any, output_dir: Path, file_base: str, format_type: str) -> Path:
        """Save arbitrary object in specified format."""
        if format_type == 'json':
            file_path = output_dir / f"{file_base}.json"
            with open(file_path, 'w') as f:
                json.dump(obj, f, indent=2, default=str)
        else:
            # Fallback to JSON for non-DataFrame objects
            file_path = output_dir / f"{file_base}_object.json"
            with open(file_path, 'w') as f:
                json.dump({'data': str(obj), 'type': type(obj).__name__}, f, indent=2, default=str)
        
        return file_path
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get comprehensive metadata summary."""
        summary = self.metadata.copy()
        summary.update({
            'total_results': len(self.results),
            'result_names': list(self.results.keys()),
            'result_summary': {name: result.to_dict() for name, result in self.results.items()}
        })
        return summary
    
    def create_summary_report(self) -> Dict[str, Any]:
        """Create a summary report of the analysis. Override in subclasses."""
        return {
            'analyzer_name': self.name,
            'execution_timestamp': datetime.now(),
            'total_results': len(self.results),
            'results_overview': {name: result.description for name, result in self.results.items()},
            'key_findings': self._extract_key_findings()
        }
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from results. Override in subclasses."""
        return [f"Generated {len(self.results)} analysis results"]
    
    def export_notebook_script(self, output_path: Optional[Path] = None) -> Path:
        """
        Export analysis as a Python script suitable for notebook execution.
        
        Args:
            output_path: Path for the output script
            
        Returns:
            Path to the generated script
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.paths.notebook_dir / f"{self.name}_analysis_{timestamp}.py"
        
        script_content = self._generate_notebook_script()
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Notebook script exported to: {output_path}")
        return output_path
    
    def _generate_notebook_script(self) -> str:
        """Generate Python script content for notebook execution."""
        script = f'''#!/usr/bin/env python3
"""
MOVR Clinical Analytics - {self.name}
Generated on: {datetime.now().isoformat()}

This script can be executed as a standalone analysis or imported into a notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# MOVR Analytics imports
from src.analytics.base import BaseAnalyzer
from src.data_processing.loader import get_loader
from src.cleaning.cleaner import get_cleaner
from src.config import get_config


class {self.__class__.__name__}Script:
    """Generated script version of {self.name} analyzer."""
    
    def __init__(self):
        self.config = get_config()
        self.loader = get_loader()
        self.cleaner = get_cleaner()
    
    def run_analysis(self):
        """Run the complete analysis."""
        print(f"Starting {self.name} analysis...")
        
        # Load data
        print("Loading data...")
        data = self._load_data()
        
        # Run analysis
        print("Running analysis...")
        results = self._analyze_data(data)
        
        # Generate outputs
        print("Generating outputs...")
        self._generate_outputs(results)
        
        print("Analysis complete!")
        return results
    
    def _load_data(self):
        """Load required datasets."""
        # Implementation would be generated based on analyzer specifics
        pass
    
    def _analyze_data(self, data):
        """Run the analysis logic."""
        # Implementation would be generated based on analyzer specifics
        pass
    
    def _generate_outputs(self, results):
        """Generate output visualizations and reports."""
        # Implementation would be generated based on analyzer specifics
        pass


if __name__ == "__main__":
    analyzer = {self.__class__.__name__}Script()
    results = analyzer.run_analysis()
'''
        
        return script
    
    def __repr__(self) -> str:
        """String representation of the analyzer."""
        return f"{self.__class__.__name__}(name='{self.name}', results={len(self.results)})"


class DescriptiveAnalyzer(BaseAnalyzer):
    """Analyzer for descriptive statistics and data summaries."""
    
    def run_analysis(self, **kwargs) -> Dict[str, AnalysisResult]:
        """Run descriptive analysis."""
        data = self.load_required_data()
        
        for dataset_name, df in data.items():
            # Basic descriptive statistics
            desc_stats = df.describe(include='all')
            self.add_result(
                f"{dataset_name}_descriptive_stats",
                desc_stats,
                f"Descriptive statistics for {dataset_name}"
            )
            
            # Missing data analysis
            missing_analysis = df.isnull().sum()
            missing_pct = (missing_analysis / len(df) * 100).round(2)
            missing_summary = pd.DataFrame({
                'Missing_Count': missing_analysis,
                'Missing_Percentage': missing_pct
            })
            
            self.add_result(
                f"{dataset_name}_missing_data",
                missing_summary,
                f"Missing data analysis for {dataset_name}"
            )
        
        return self.results
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from descriptive analysis."""
        findings = []
        
        for result_name, result in self.results.items():
            if 'missing_data' in result_name:
                if isinstance(result.data, pd.DataFrame):
                    high_missing = result.data[result.data['Missing_Percentage'] > 10]
                    if not high_missing.empty:
                        findings.append(f"High missing data (>10%) found in {len(high_missing)} fields in {result_name}")
        
        return findings


def get_analyzer(analyzer_type: str, name: str, **kwargs) -> BaseAnalyzer:
    """Factory function to get analyzer instances."""
    if analyzer_type == "descriptive":
        return DescriptiveAnalyzer(name, **kwargs)
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")