"""
AWS S3 Data Loader for MOVR Clinical Analytics

Extends MOVRDataLoader to support loading parquet files from S3.
Provides secure, scalable data access with caching.
"""

import pandas as pd
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from typing import Optional, Union
import streamlit as st
import logging
from io import BytesIO

from .loader import MOVRDataLoader

logger = logging.getLogger(__name__)


class S3DataLoader(MOVRDataLoader):
    """
    S3-backed data loader for MOVR clinical data.
    
    Supports both local and S3 data sources with automatic fallback.
    Credentials can be provided via:
    - AWS credentials file (~/.aws/credentials)
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - Streamlit secrets (recommended for deployment)
    - IAM roles (for EC2/ECS deployments)
    
    Example Streamlit secrets (.streamlit/secrets.toml):
        [aws]
        access_key_id = "AKIA..."
        secret_access_key = "..."
        region = "us-east-1"
        
        [s3]
        bucket_name = "movr-clinical-data"
        data_prefix = "parquet/"
    """
    
    def __init__(
        self, 
        bucket_name: Optional[str] = None,
        data_prefix: str = "",
        local_cache: bool = True,
        local_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize S3 data loader.
        
        Args:
            bucket_name: S3 bucket name (defaults to secrets or env)
            data_prefix: Prefix/folder path in S3 bucket
            local_cache: Whether to cache downloaded files locally
            local_path: Local directory for cache (defaults to data/)
        """
        super().__init__(data_path=local_path)
        
        # S3 configuration
        self.bucket_name = bucket_name or self._get_bucket_name()
        self.data_prefix = data_prefix.rstrip('/') + '/' if data_prefix else ''
        self.local_cache = local_cache
        
        # Initialize S3 client
        self.s3_client = self._init_s3_client()
        
        logger.info(f"S3DataLoader initialized: bucket={self.bucket_name}, prefix={self.data_prefix}")
    
    def _get_bucket_name(self) -> str:
        """Get S3 bucket name from secrets or environment."""
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and 's3' in st.secrets:
            return st.secrets['s3']['bucket_name']
        
        # Try environment variable
        import os
        bucket = os.getenv('S3_BUCKET_NAME')
        if bucket:
            return bucket
        
        raise ValueError(
            "S3 bucket name not configured. Set in:\n"
            "- Streamlit secrets: [s3] bucket_name = '...'\n"
            "- Environment: S3_BUCKET_NAME=..."
        )
    
    def _init_s3_client(self):
        """Initialize boto3 S3 client with credentials from secrets or environment."""
        try:
            # Try Streamlit secrets first
            if hasattr(st, 'secrets') and 'aws' in st.secrets:
                aws_config = st.secrets['aws']
                session = boto3.Session(
                    aws_access_key_id=aws_config.get('access_key_id'),
                    aws_secret_access_key=aws_config.get('secret_access_key'),
                    region_name=aws_config.get('region', 'us-east-1')
                )
                return session.client('s3')
            
            # Fall back to default credential chain (env vars, ~/.aws/credentials, IAM roles)
            return boto3.client('s3')
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise ValueError(
                "AWS credentials not configured. Options:\n"
                "1. Add to Streamlit secrets (.streamlit/secrets.toml)\n"
                "2. Set environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
                "3. Configure ~/.aws/credentials\n"
                "4. Use IAM roles (for EC2/ECS)"
            )
    
    def _s3_key(self, table_name: str) -> str:
        """Construct full S3 key for a table."""
        normalized = self._normalize_table_name(table_name)
        if not normalized.endswith('.parquet'):
            normalized += '.parquet'
        return f"{self.data_prefix}{normalized}"
    
    def _file_exists_in_s3(self, table_name: str) -> bool:
        """Check if a file exists in S3."""
        try:
            s3_key = self._s3_key(table_name)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    @st.cache_data(ttl=3600)
    def load_from_s3(_self, table_name: str) -> pd.DataFrame:
        """
        Load parquet file from S3 with caching.
        
        Args:
            table_name: Name of the table to load
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist in S3
            ClientError: If S3 access fails
        """
        s3_key = _self._s3_key(table_name)
        logger.info(f"Loading {table_name} from S3: s3://{_self.bucket_name}/{s3_key}")
        
        try:
            # Download file to memory
            response = _self.s3_client.get_object(Bucket=_self.bucket_name, Key=s3_key)
            parquet_data = response['Body'].read()
            
            # Load into pandas
            df = pd.read_parquet(BytesIO(parquet_data))
            
            # Optional: cache locally for faster subsequent access
            if _self.local_cache:
                local_file = _self.data_path / f"{_self._normalize_table_name(table_name)}.parquet"
                local_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(local_file)
                logger.info(f"Cached to local: {local_file}")
            
            logger.info(f"Loaded {len(df)} rows from S3")
            return df
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(
                    f"File not found in S3: s3://{_self.bucket_name}/{s3_key}"
                )
            elif error_code == 'AccessDenied':
                raise PermissionError(
                    f"Access denied to S3 object: s3://{_self.bucket_name}/{s3_key}\n"
                    "Check your AWS credentials and bucket permissions."
                )
            else:
                logger.error(f"S3 error loading {table_name}: {e}")
                raise
    
    def load_table(
        self, 
        table_name: str, 
        use_s3: bool = True,
        force_s3: bool = False
    ) -> pd.DataFrame:
        """
        Load table with automatic fallback: S3 → Local → Error.
        
        Args:
            table_name: Name of table to load
            use_s3: Whether to attempt S3 first (default: True)
            force_s3: Only use S3, don't fall back to local (default: False)
            
        Returns:
            DataFrame with loaded data
        """
        normalized = self._normalize_table_name(table_name)
        
        # Try S3 first (if enabled)
        if use_s3:
            try:
                if self._file_exists_in_s3(table_name):
                    return self.load_from_s3(table_name)
                else:
                    logger.warning(f"{table_name} not found in S3: s3://{self.bucket_name}/{self._s3_key(table_name)}")
            except Exception as e:
                logger.warning(f"Failed to load {table_name} from S3: {e}")
                if force_s3:
                    raise
        
        # Fall back to local file (via parent class)
        if not force_s3:
            logger.info(f"Attempting local load for {table_name}")
            return super().load_table(table_name)
        
        # No options left
        raise FileNotFoundError(
            f"Table {table_name} not found in S3 or locally"
        )
    
    def list_available_files(self, source: str = 'both') -> dict:
        """
        List available parquet files from S3 and/or local.
        
        Args:
            source: 'both', 's3', or 'local'
            
        Returns:
            Dict with 's3' and 'local' lists of available files
        """
        result = {'s3': [], 'local': []}
        
        # List S3 files
        if source in ('both', 's3'):
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=self.data_prefix
                )
                if 'Contents' in response:
                    result['s3'] = [
                        obj['Key'].replace(self.data_prefix, '').replace('.parquet', '')
                        for obj in response['Contents']
                        if obj['Key'].endswith('.parquet')
                    ]
            except ClientError as e:
                logger.error(f"Failed to list S3 files: {e}")
        
        # List local files
        if source in ('both', 'local'):
            result['local'] = super().list_available_files()
        
        return result
    
    def sync_from_s3(self, tables: Optional[list] = None) -> dict:
        """
        Download tables from S3 to local cache.
        
        Args:
            tables: List of table names to sync (None = all)
            
        Returns:
            Dict with sync results: {'downloaded': [...], 'failed': [...]}
        """
        if tables is None:
            available = self.list_available_files(source='s3')
            tables = available['s3']
        
        results = {'downloaded': [], 'failed': []}
        
        for table in tables:
            try:
                df = self.load_from_s3(table)
                local_file = self.data_path / f"{self._normalize_table_name(table)}.parquet"
                df.to_parquet(local_file)
                results['downloaded'].append(table)
                logger.info(f"Synced {table} from S3")
            except Exception as e:
                results['failed'].append({'table': table, 'error': str(e)})
                logger.error(f"Failed to sync {table}: {e}")
        
        return results


# Convenience function to use in existing code
@st.cache_resource
def get_s3_loader(
    bucket_name: Optional[str] = None,
    data_prefix: str = "",
    local_cache: bool = True
) -> S3DataLoader:
    """
    Get cached S3DataLoader instance.
    
    This ensures only one S3 client is created per session.
    """
    return S3DataLoader(
        bucket_name=bucket_name,
        data_prefix=data_prefix,
        local_cache=local_cache
    )
