# AWS S3 Integration Guide

## Overview

This guide explains how to integrate AWS S3 for secure, scalable data storage with your OpenMOVR App deployment.

## Why Use S3?

### Benefits ✅
- **Scalability**: Store unlimited data (no 1GB limit)
- **Security**: Industry-standard encryption, access controls, audit logs
- **Compliance**: HIPAA-eligible with proper configuration
- **Performance**: CDN integration, parallel downloads
- **Cost**: Pay only for what you use (~$0.023/GB/month)
- **Backup**: Built-in versioning and disaster recovery

### Use Cases
1. **Protected Data Deployment**: Store PHI-containing parquet files securely
2. **Multi-tenant Access**: Different users access different data based on permissions
3. **Large Datasets**: Beyond Streamlit Cloud's 1GB storage limit
4. **Regulatory Compliance**: HIPAA, GDPR, SOC 2 requirements

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           Streamlit App (on Cloud)                  │
│  - Lightweight (no data files committed)            │
│  - Credentials via Streamlit Secrets                │
│  - Caching for performance                          │
└─────────────────┬───────────────────────────────────┘
                  │
                  │ boto3 (AWS SDK)
                  │
┌─────────────────▼───────────────────────────────────┐
│              AWS S3 Bucket                          │
│  movr-clinical-data/                                │
│  ├── parquet/                                       │
│  │   ├── demographics_maindata.parquet              │
│  │   ├── diagnosis_maindata.parquet                 │
│  │   ├── encounter_maindata.parquet                 │
│  │   └── ...                                        │
│  └── snapshots/                                     │
│      ├── dmd_snapshot.json                          │
│      └── lgmd_snapshot.json                         │
│                                                     │
│  Security:                                          │
│  - Encryption: SSE-S3 or SSE-KMS                    │
│  - Access: IAM policies + bucket policies           │
│  - Audit: CloudTrail logging                        │
└─────────────────────────────────────────────────────┘
```

## Setup Steps

### 1. Create S3 Bucket

```bash
# Using AWS CLI
aws s3 mb s3://movr-clinical-data --region us-east-1

# Enable versioning (recommended for audit trails)
aws s3api put-bucket-versioning \
    --bucket movr-clinical-data \
    --versioning-configuration Status=Enabled

# Enable encryption (required for HIPAA compliance)
aws s3api put-bucket-encryption \
    --bucket movr-clinical-data \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }'

# Block public access (CRITICAL for PHI data)
aws s3api put-public-access-block \
    --bucket movr-clinical-data \
    --public-access-block-configuration \
        BlockPublicAcls=true,\
        IgnorePublicAcls=true,\
        BlockPublicPolicy=true,\
        RestrictPublicBuckets=true
```

### 2. Create IAM User for Streamlit App

Create a dedicated IAM user with minimal permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::movr-clinical-data",
        "arn:aws:s3:::movr-clinical-data/*"
      ]
    }
  ]
}
```

**Generate access keys** and save them securely (you'll need them for configuration).

### 3. Upload Data to S3

```bash
# Upload parquet files
aws s3 cp data/ s3://movr-clinical-data/parquet/ --recursive --exclude "*" --include "*.parquet"

# Upload snapshots (optional - can keep in git for public data)
aws s3 cp stats/ s3://movr-clinical-data/snapshots/ --recursive --include "*.json"

# Verify upload
aws s3 ls s3://movr-clinical-data/parquet/
```

### 4. Configure Streamlit Secrets

**For Streamlit Cloud:**

Go to your app settings and add to Secrets:

```toml
# .streamlit/secrets.toml

[aws]
access_key_id = "AKIA..."
secret_access_key = "your-secret-key"
region = "us-east-1"

[s3]
bucket_name = "movr-clinical-data"
data_prefix = "parquet/"

# Your existing access key for DUA pages
OPENMOVR_SITE_KEY = "your-dua-access-key"
```

**For Local Development:**

Create `.streamlit/secrets.toml` (git-ignored) or use environment variables:

```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET_NAME="movr-clinical-data"
```

### 5. Update Code to Use S3

**Option A: Replace existing loader (for full S3 migration)**

In your API files (e.g., `api/cohorts.py`, `api/dmd.py`):

```python
# OLD:
from src.data_processing.loader import MOVRDataLoader
loader = MOVRDataLoader()

# NEW:
from src.data_processing.s3_loader import get_s3_loader
loader = get_s3_loader()
```

**Option B: Hybrid mode (try S3, fallback to local)**

The S3DataLoader already supports this by default:

```python
from src.data_processing.s3_loader import get_s3_loader

# Automatically tries S3 first, falls back to local
loader = get_s3_loader()
df = loader.load_table('demographics_maindata')  # S3 → local → error
```

**Option C: Conditional (based on environment)**

```python
import os
from src.data_processing.loader import MOVRDataLoader
from src.data_processing.s3_loader import get_s3_loader

if os.getenv('USE_S3', 'false').lower() == 'true':
    loader = get_s3_loader()
else:
    loader = MOVRDataLoader()
```

### 6. Update requirements.txt

```bash
echo "boto3>=1.34.0" >> requirements.txt
```

### 7. Deploy and Test

```bash
# Commit changes
git add src/data_processing/s3_loader.py requirements.txt
git commit -m "Add AWS S3 integration"
git push

# Streamlit Cloud will auto-redeploy
# Check logs for S3 connection success
```

## Security Best Practices

### HIPAA Compliance Checklist

- ✅ **Encryption in transit**: Use HTTPS (enforced by boto3)
- ✅ **Encryption at rest**: Enable SSE-S3 or SSE-KMS
- ✅ **Access logging**: Enable CloudTrail for audit trails
- ✅ **Least privilege**: IAM policies with minimal permissions
- ✅ **No public access**: Block all public bucket access
- ✅ **Versioning**: Enable for data integrity and recovery
- ✅ **Lifecycle policies**: Auto-archive or delete old data
- ✅ **MFA delete**: Require MFA to delete objects
- ✅ **VPC endpoints**: Use VPC endpoints for private network access (advanced)

### IAM Policy Examples

**Read-only access (recommended for production app):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::movr-clinical-data",
        "arn:aws:s3:::movr-clinical-data/parquet/*"
      ]
    }
  ]
}
```

**Admin access (for data uploads/updates):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:*"
      ],
      "Resource": [
        "arn:aws:s3:::movr-clinical-data",
        "arn:aws:s3:::movr-clinical-data/*"
      ]
    }
  ]
}
```

### Credential Rotation

Rotate access keys every 90 days:

```bash
# 1. Create new access key
aws iam create-access-key --user-name streamlit-movr-app

# 2. Update Streamlit secrets with new key

# 3. Test app to ensure it works

# 4. Delete old key
aws iam delete-access-key --user-name streamlit-movr-app --access-key-id AKIA...
```

## Cost Estimation

### AWS S3 Pricing (us-east-1, as of 2026)

| Resource | Cost | Estimate |
|----------|------|----------|
| Storage (Standard) | $0.023/GB/month | 10 GB = $0.23/mo |
| GET requests | $0.0004/1000 | 100K/mo = $0.04/mo |
| Data transfer out | $0.09/GB | 5 GB/mo = $0.45/mo |
| **Total** | | **~$1-5/month** |

### Optimization Tips
1. **Enable caching**: The S3DataLoader caches in Streamlit's `@st.cache_data`
2. **Local cache**: Enable `local_cache=True` to save files to local disk on first load
3. **Compression**: Use Snappy or Gzip compression in parquet files
4. **Lifecycle policies**: Archive old snapshots to S3 Glacier

## Monitoring and Troubleshooting

### CloudWatch Metrics

Monitor these key metrics:
- `BucketSizeBytes`: Total data stored
- `NumberOfObjects`: File count
- `AllRequests`: API call volume
- `4xxErrors`, `5xxErrors`: Access issues

### Common Issues

**1. NoCredentialsError**
```
Problem: AWS credentials not found
Solution: 
  - Check Streamlit secrets are configured correctly
  - Verify IAM user has access keys generated
  - Test locally with AWS CLI: aws s3 ls s3://movr-clinical-data
```

**2. AccessDenied**
```
Problem: IAM policy doesn't grant required permissions
Solution:
  - Verify bucket policy doesn't block access
  - Check IAM policy includes s3:GetObject and s3:ListBucket
  - Ensure public access block isn't preventing access
```

**3. SlowLoading**
```
Problem: S3 downloads are slow
Solution:
  - Enable local_cache=True in S3DataLoader
  - Use smaller parquet files (partition large tables)
  - Enable compression
  - Consider AWS CloudFront CDN for caching
```

**4. CORS Issues (if accessing from browser)**
```
Problem: Browser blocks direct S3 access
Solution: Configure CORS policy on bucket (not needed for server-side access)
```

### Debug Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('src.data_processing.s3_loader')
logger.setLevel(logging.DEBUG)
```

## Migration Checklist

- [ ] Create S3 bucket with encryption and versioning
- [ ] Create IAM user with read-only policy
- [ ] Upload parquet files to S3
- [ ] Configure Streamlit secrets with AWS credentials
- [ ] Add boto3 to requirements.txt
- [ ] Update code to use S3DataLoader
- [ ] Test locally with `.streamlit/secrets.toml`
- [ ] Deploy to Streamlit Cloud
- [ ] Verify app loads data from S3 successfully
- [ ] Set up CloudWatch alarms for errors
- [ ] Document bucket name and IAM user for team
- [ ] Schedule quarterly credential rotation

## Alternative: Streamlit Cloud Enterprise

If S3 integration is too complex, consider **Streamlit Cloud Enterprise**:
- Private data storage
- SSO authentication
- Dedicated resources (no 1GB limit)
- Support and SLAs
- Cost: ~$500-1000/month

Contact: https://streamlit.io/enterprise

## Support

Questions? Contact:
- **AWS Support**: https://console.aws.amazon.com/support/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Project Lead**: andre.paredes@ymail.com
