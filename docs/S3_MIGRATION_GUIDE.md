# Migration Example: Using S3DataLoader in Your API

This document shows how to migrate from local file storage to AWS S3 storage in your API files.

## Before (Local Files Only)

```python
# api/cohorts.py

from src.data_processing.loader import MOVRDataLoader
import pandas as pd
import streamlit as st

class CohortAPI:
    def __init__(self):
        self.loader = MOVRDataLoader()
    
    def get_demographics(self):
        return self.loader.load_table('demographics_maindata')
    
    def get_diagnosis(self):
        return self.loader.load_table('diagnosis_maindata')
```

## After (S3 with Fallback)

```python
# api/cohorts.py

import os
from src.data_processing.s3_loader import get_s3_loader
from src.data_processing.loader import MOVRDataLoader
import pandas as pd
import streamlit as st

class CohortAPI:
    def __init__(self):
        # Use S3 if configured, otherwise fall back to local
        if self._should_use_s3():
            self.loader = get_s3_loader()
        else:
            self.loader = MOVRDataLoader()
    
    def _should_use_s3(self) -> bool:
        """Check if S3 is configured and should be used."""
        # Check environment variable
        if os.getenv('USE_S3', '').lower() == 'true':
            return True
        
        # Check Streamlit secrets
        if hasattr(st, 'secrets') and 's3' in st.secrets:
            return True
        
        return False
    
    def get_demographics(self):
        # Works with both S3DataLoader and MOVRDataLoader!
        return self.loader.load_table('demographics_maindata')
    
    def get_diagnosis(self):
        return self.loader.load_table('diagnosis_maindata')
```

## Ultra-Simple (Always Try S3)

The S3DataLoader automatically falls back to local files if S3 isn't configured:

```python
# api/cohorts.py

from src.data_processing.s3_loader import get_s3_loader
import pandas as pd
import streamlit as st

class CohortAPI:
    def __init__(self):
        # This works whether S3 is configured or not!
        # - If S3 is configured: uses S3
        # - If S3 fails: falls back to local files
        # - If neither exists: raises FileNotFoundError
        self.loader = get_s3_loader()
    
    def get_demographics(self):
        return self.loader.load_table('demographics_maindata')
    
    def get_diagnosis(self):
        return self.loader.load_table('diagnosis_maindata')
```

## Advanced: Conditional S3 Usage

```python
# api/cohorts.py

from src.data_processing.s3_loader import get_s3_loader
import pandas as pd
import streamlit as st

class CohortAPI:
    def __init__(self, use_s3: bool = True, force_s3: bool = False):
        """
        Initialize CohortAPI with flexible storage options.
        
        Args:
            use_s3: Try S3 first if available (default: True)
            force_s3: Only use S3, no local fallback (default: False)
        """
        self.loader = get_s3_loader()
        self.use_s3 = use_s3
        self.force_s3 = force_s3
    
    def get_demographics(self):
        return self.loader.load_table(
            'demographics_maindata',
            use_s3=self.use_s3,
            force_s3=self.force_s3
        )
    
    def get_diagnosis(self):
        return self.loader.load_table(
            'diagnosis_maindata',
            use_s3=self.use_s3,
            force_s3=self.force_s3
        )
```

## Testing Locally

### Without S3 (Local Files)
```bash
# Just run the app - it will use local files
streamlit run app.py
```

### With S3 (Test Configuration)
```bash
# Create .streamlit/secrets.toml
cat > .streamlit/secrets.toml << EOF
[aws]
access_key_id = "AKIA..."
secret_access_key = "..."
region = "us-east-1"

[s3]
bucket_name = "movr-clinical-data"
data_prefix = "parquet/"
EOF

# Run app - it will use S3
streamlit run app.py
```

## Debugging

```python
# Add logging to see which storage is used
import logging
logging.basicConfig(level=logging.INFO)

from src.data_processing.s3_loader import get_s3_loader

loader = get_s3_loader()

# Check available files
files = loader.list_available_files(source='both')
print("S3 files:", files['s3'])
print("Local files:", files['local'])

# Load with verbose logging
df = loader.load_table('demographics_maindata')
# Logs will show: "Loading from S3: s3://bucket/path" or "Falling back to local"
```

## Caching Strategy

The S3DataLoader uses Streamlit's `@st.cache_data` decorator, so repeated loads are cached:

```python
from src.data_processing.s3_loader import get_s3_loader

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_demographics():
    loader = get_s3_loader()
    return loader.load_table('demographics_maindata')

# First call: downloads from S3 (slow)
df1 = load_demographics()

# Second call: returns cached data (instant)
df2 = load_demographics()
```

## Performance Comparison

| Storage | First Load | Cached Load | Concurrent Users |
|---------|-----------|-------------|------------------|
| **Local Files** | ~1-2s | Instant | Limited by disk I/O |
| **S3 (no cache)** | ~3-5s | Instant | Scales linearly |
| **S3 (with cache)** | ~3-5s | Instant | Unlimited* |

*With proper Streamlit caching

## Migration Checklist

- [ ] Add boto3 to requirements.txt ✅ (already done)
- [ ] Copy secrets.toml.example to .streamlit/secrets.toml
- [ ] Fill in AWS credentials and S3 bucket name
- [ ] Update API files to use get_s3_loader()
- [ ] Test locally: `streamlit run app.py`
- [ ] Verify S3 files are loaded (check logs)
- [ ] Push to GitHub
- [ ] Add secrets to Streamlit Cloud settings
- [ ] Verify deployed app uses S3
- [ ] Monitor CloudWatch for errors

## Rollback Plan

If S3 integration causes issues:

```bash
# 1. Remove S3 dependency from API files
git revert <commit-hash>

# 2. Or just change loader initialization
# OLD:
from src.data_processing.s3_loader import get_s3_loader
loader = get_s3_loader()

# NEW:
from src.data_processing.loader import MOVRDataLoader
loader = MOVRDataLoader()

# 3. Redeploy
git push
```

The beauty of the API facade pattern is that pages don't know or care where data comes from!

## Common Issues

### 1. "NoCredentialsError"
**Cause**: AWS credentials not configured
**Fix**: Add to `.streamlit/secrets.toml` or environment variables

### 2. "AccessDenied"
**Cause**: IAM policy doesn't grant S3 access
**Fix**: Update IAM policy to include `s3:GetObject` and `s3:ListBucket`

### 3. "NoSuchBucket"
**Cause**: Bucket name is wrong or doesn't exist
**Fix**: Verify bucket name in AWS console and secrets.toml

### 4. Slow loading
**Cause**: Downloading from S3 on every request
**Fix**: Enable caching with `@st.cache_data` decorator

### 5. "File not found" but file exists in S3
**Cause**: File path/prefix mismatch
**Fix**: Check `data_prefix` in secrets matches S3 folder structure

## Need Help?

- **S3 Integration Guide**: [docs/AWS_S3_INTEGRATION.md](AWS_S3_INTEGRATION.md)
- **boto3 Documentation**: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
- **Streamlit Secrets**: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
