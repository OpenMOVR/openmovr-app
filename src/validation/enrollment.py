"""
Enrollment Validation for MOVR Clinical Analytics

Validates participant enrollment based on MOVR study requirements:
- At least one Demographics_MainData record
- At least one Diagnosis_MainData record  
- At least one Encounter_MainData record

Provides detailed enrollment status reporting and data quality metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import logging
from datetime import datetime
from pathlib import Path

from ..config import get_config
from ..data_processing.data_dictionary import DataDictionary
from ..data_processing.loader import get_loader


logger = logging.getLogger(__name__)


class EnrollmentValidator:
    """
    Validates participant enrollment status based on MOVR requirements.
    
    Key validation rules:
    1. Participant must have Demographics_MainData record
    2. Participant must have Diagnosis_MainData record
    3. Participant must have Encounter_MainData record
    4. All records must have valid key identifiers
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        self.config = get_config()
        self.data_path = data_path or self.config.paths.data_dir
        self.data_dict = DataDictionary(str(self.data_path))
        self.loader = get_loader(self.data_path)
        
        # Validation results storage
        self.validation_results = {}
        self.enrollment_summary = {}
        
    @property
    def key_fields(self) -> Dict[str, str]:
        """Get key field mappings."""
        return self.loader.key_fields
    
    def validate_participant_enrollment(self, 
                                       participant_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate enrollment status for specified participants or all participants.
        
        Args:
            participant_ids: Specific participant IDs to validate (None = all)
            
        Returns:
            Dictionary with enrollment validation results
        """
        logger.info("Starting participant enrollment validation")
        
        # Load required datasets
        try:
            demographics = self.loader.load_demographics()
            diagnosis = self.loader.load_diagnosis() 
            encounters = self.loader.load_encounters()
        except FileNotFoundError as e:
            logger.error(f"Required data file not found: {e}")
            return {"error": str(e)}
        
        participant_id_field = self.key_fields["participant_id"]
        
        # Get all unique participants across datasets
        all_participants = set()
        demo_participants = set(demographics[participant_id_field].dropna().unique())
        diag_participants = set(diagnosis[participant_id_field].dropna().unique())
        encounter_participants = set(encounters[participant_id_field].dropna().unique())
        
        all_participants.update(demo_participants)
        all_participants.update(diag_participants)
        all_participants.update(encounter_participants)
        
        # Filter to specific participants if requested
        if participant_ids:
            all_participants = all_participants.intersection(set(participant_ids))
        
        logger.info(f"Validating enrollment for {len(all_participants)} participants")
        
        # Validate each participant
        enrollment_results = []
        
        for participant_id in all_participants:
            result = self._validate_single_participant(
                participant_id,
                demographics,
                diagnosis, 
                encounters
            )
            enrollment_results.append(result)
        
        # Generate summary statistics
        enrollment_df = pd.DataFrame(enrollment_results)
        summary = self._generate_enrollment_summary(enrollment_df)
        
        validation_results = {
            "timestamp": datetime.now(),
            "total_participants": len(all_participants),
            "enrollment_details": enrollment_results,
            "enrollment_summary": summary,
            "data_quality_issues": self._identify_quality_issues(enrollment_df)
        }
        
        self.validation_results = validation_results
        return validation_results
    
    def _validate_single_participant(self, 
                                   participant_id: str,
                                   demographics: pd.DataFrame,
                                   diagnosis: pd.DataFrame,
                                   encounters: pd.DataFrame) -> Dict[str, Any]:
        """Validate enrollment for a single participant."""
        participant_id_field = self.key_fields["participant_id"]
        
        # Check presence in each required dataset
        has_demographics = participant_id in demographics[participant_id_field].values
        has_diagnosis = participant_id in diagnosis[participant_id_field].values
        has_encounters = participant_id in encounters[participant_id_field].values
        
        # Get record counts
        demo_count = len(demographics[demographics[participant_id_field] == participant_id])
        diag_count = len(diagnosis[diagnosis[participant_id_field] == participant_id])
        encounter_count = len(encounters[encounters[participant_id_field] == participant_id])
        
        # Determine enrollment status
        is_enrolled = has_demographics and has_diagnosis and has_encounters
        
        # Identify missing components
        missing_components = []
        if not has_demographics:
            missing_components.append("Demographics_MainData")
        if not has_diagnosis:
            missing_components.append("Diagnosis_MainData")
        if not has_encounters:
            missing_components.append("Encounter_MainData")
        
        # Check for legacy USNDR data
        is_usndr = False
        if has_demographics and self.key_fields["legacy_flag"] in demographics.columns:
            usndr_records = demographics[
                (demographics[participant_id_field] == participant_id) &
                (demographics[self.key_fields["legacy_flag"]] == True)
            ]
            is_usndr = len(usndr_records) > 0
        
        return {
            "participant_id": participant_id,
            "is_enrolled": is_enrolled,
            "has_demographics": has_demographics,
            "has_diagnosis": has_diagnosis, 
            "has_encounters": has_encounters,
            "demographics_records": demo_count,
            "diagnosis_records": diag_count,
            "encounter_records": encounter_count,
            "missing_components": missing_components,
            "is_usndr": is_usndr,
            "validation_timestamp": datetime.now()
        }
    
    def _generate_enrollment_summary(self, enrollment_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for enrollment validation."""
        total = len(enrollment_df)
        
        summary = {
            "total_participants": total,
            "enrolled_participants": enrollment_df['is_enrolled'].sum(),
            "enrollment_rate": enrollment_df['is_enrolled'].mean() if total > 0 else 0,
            
            # Component availability
            "has_demographics": enrollment_df['has_demographics'].sum(),
            "has_diagnosis": enrollment_df['has_diagnosis'].sum(), 
            "has_encounters": enrollment_df['has_encounters'].sum(),
            
            # Legacy data
            "usndr_participants": enrollment_df['is_usndr'].sum(),
            
            # Record count statistics
            "avg_demographics_records": enrollment_df['demographics_records'].mean(),
            "avg_diagnosis_records": enrollment_df['diagnosis_records'].mean(),
            "avg_encounter_records": enrollment_df['encounter_records'].mean(),
            
            # Missing component analysis
            "missing_only_demographics": len(enrollment_df[
                (~enrollment_df['has_demographics']) & 
                enrollment_df['has_diagnosis'] & 
                enrollment_df['has_encounters']
            ]),
            "missing_only_diagnosis": len(enrollment_df[
                enrollment_df['has_demographics'] & 
                (~enrollment_df['has_diagnosis']) & 
                enrollment_df['has_encounters']
            ]),
            "missing_only_encounters": len(enrollment_df[
                enrollment_df['has_demographics'] & 
                enrollment_df['has_diagnosis'] & 
                (~enrollment_df['has_encounters'])
            ]),
        }
        
        return summary
    
    def _identify_quality_issues(self, enrollment_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify data quality issues affecting enrollment."""
        issues = {
            "participants_with_zero_records": [],
            "participants_with_excessive_records": [],
            "usndr_vs_current_conflicts": []
        }
        
        # Find participants with zero records in any category
        zero_demographics = enrollment_df[enrollment_df['demographics_records'] == 0]['participant_id'].tolist()
        zero_diagnosis = enrollment_df[enrollment_df['diagnosis_records'] == 0]['participant_id'].tolist()
        zero_encounters = enrollment_df[enrollment_df['encounter_records'] == 0]['participant_id'].tolist()
        
        issues["participants_with_zero_records"] = {
            "demographics": zero_demographics,
            "diagnosis": zero_diagnosis,
            "encounters": zero_encounters
        }
        
        # Find participants with excessive records (potential duplicates)
        excessive_threshold = 10  # Configurable threshold
        
        excessive_demographics = enrollment_df[
            enrollment_df['demographics_records'] > excessive_threshold
        ]['participant_id'].tolist()
        
        excessive_encounters = enrollment_df[
            enrollment_df['encounter_records'] > excessive_threshold
        ]['participant_id'].tolist()
        
        issues["participants_with_excessive_records"] = {
            "demographics": excessive_demographics,
            "encounters": excessive_encounters,
            "threshold": excessive_threshold
        }
        
        return issues
    
    def get_enrollment_report(self, 
                             output_format: str = 'summary',
                             save_to_file: bool = False) -> Union[Dict, pd.DataFrame, str]:
        """
        Get enrollment validation report in various formats.
        
        Args:
            output_format: 'summary', 'detailed', 'csv', 'excel'
            save_to_file: Whether to save report to output directory
            
        Returns:
            Report in requested format
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_participant_enrollment first.")
        
        if output_format == 'summary':
            report = self.validation_results['enrollment_summary']
        
        elif output_format == 'detailed':
            report = self.validation_results
        
        elif output_format in ['csv', 'excel']:
            enrollment_df = pd.DataFrame(self.validation_results['enrollment_details'])
            
            if save_to_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if output_format == 'csv':
                    filename = f"enrollment_report_{timestamp}.csv"
                    output_path = self.config.paths.output_dir / filename
                    enrollment_df.to_csv(output_path, index=False)
                
                elif output_format == 'excel':
                    filename = f"enrollment_report_{timestamp}.xlsx"
                    output_path = self.config.paths.output_dir / filename
                    
                    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                        enrollment_df.to_excel(writer, sheet_name='Enrollment_Details', index=False)
                        
                        summary_df = pd.DataFrame([self.validation_results['enrollment_summary']])
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        issues_df = pd.DataFrame(self.validation_results['data_quality_issues'])
                        issues_df.to_excel(writer, sheet_name='Quality_Issues', index=False)
                
                logger.info(f"Enrollment report saved to: {output_path}")
                return str(output_path)
            
            else:
                report = enrollment_df
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return report
    
    def validate_enrollment_requirements(self) -> Dict[str, bool]:
        """
        Validate that enrollment requirements from data dictionary are met.
        
        Returns:
            Dictionary of requirement validation results
        """
        try:
            requirements = self.data_dict.get_enrollment_requirements()
            required_forms = requirements.get("required_forms", [])
            minimum_records = requirements.get("minimum_records", 1)
            
            validation_results = {
                "requirements_defined": bool(required_forms),
                "required_forms_available": True,
                "minimum_records_met": True,
                "missing_forms": []
            }
            
            # Check if required forms exist as data files
            available_files = self.loader.list_available_files()
            
            for form in required_forms:
                if form not in available_files:
                    validation_results["required_forms_available"] = False
                    validation_results["missing_forms"].append(form)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Could not validate enrollment requirements: {e}")
            return {"error": str(e)}
    
    def get_non_enrolled_participants(self) -> List[Dict[str, Any]]:
        """Get list of non-enrolled participants with reasons."""
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_participant_enrollment first.")
        
        non_enrolled = []
        for participant in self.validation_results['enrollment_details']:
            if not participant['is_enrolled']:
                non_enrolled.append({
                    'participant_id': participant['participant_id'],
                    'missing_components': participant['missing_components'],
                    'has_demographics': participant['has_demographics'],
                    'has_diagnosis': participant['has_diagnosis'],
                    'has_encounters': participant['has_encounters']
                })
        
        return non_enrolled
    
    def suggest_enrollment_fixes(self) -> List[Dict[str, Any]]:
        """Suggest actions to improve enrollment rates."""
        if not self.validation_results:
            return []
        
        suggestions = []
        quality_issues = self.validation_results.get('data_quality_issues', {})
        
        # Suggest fixes for zero records
        zero_records = quality_issues.get('participants_with_zero_records', {})
        for data_type, participant_list in zero_records.items():
            if participant_list:
                suggestions.append({
                    'issue': f'Missing {data_type} records',
                    'affected_participants': len(participant_list),
                    'suggestion': f'Verify data collection and loading process for {data_type}',
                    'priority': 'high'
                })
        
        # Suggest fixes for excessive records
        excessive_records = quality_issues.get('participants_with_excessive_records', {})
        for data_type, participant_list in excessive_records.items():
            if data_type != 'threshold' and participant_list:
                suggestions.append({
                    'issue': f'Excessive {data_type} records',
                    'affected_participants': len(participant_list),
                    'suggestion': f'Check for duplicate records in {data_type} data',
                    'priority': 'medium'
                })
        
        return suggestions


def get_enrollment_validator(data_path: Optional[Path] = None) -> EnrollmentValidator:
    """Get a configured enrollment validator instance."""
    return EnrollmentValidator(data_path)