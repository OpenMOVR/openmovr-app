# Documentation Index

This directory contains comprehensive documentation for the OpenMOVR App architecture, deployment, and AWS S3 integration.

## 📚 Documentation Overview

### 🏗️ [ARCHITECTURE_SCALABILITY.md](ARCHITECTURE_SCALABILITY.md)
**Read this first!** Comprehensive analysis of:
- Current architecture assessment
- Scalability on Streamlit free vs paid tiers
- Deployment recommendations
- Cost comparison
- Implementation roadmap

**Key Questions Answered**:
- ✅ Is my app built correctly for Streamlit?
- ✅ Can it scale on the free version?
- ✅ Do I need AWS S3 integration?
- ✅ What are my deployment options?

---

### ☁️ [AWS_S3_INTEGRATION.md](AWS_S3_INTEGRATION.md)
Complete guide to adding AWS S3 for scalable, secure data storage:
- Step-by-step setup instructions
- S3 bucket creation and configuration
- IAM policy examples
- Security best practices
- HIPAA compliance checklist
- Cost estimation (~$5/month)
- Troubleshooting guide

**Use this when**:
- Your data files exceed 1GB
- You need HIPAA-compliant storage
- You want to serve protected/PHI data
- You're deploying DUA-protected pages

---

### � [FEEDBACK_SETUP.md](FEEDBACK_SETUP.md)
Complete guide for setting up clinician-friendly feedback collection:
- Why Google Forms is best for clinicians
- Step-by-step form creation (30 minutes)
- Response management workflow
- Email templates for responding
- Privacy and HIPAA compliance
- Cost estimation (FREE)

**Use this when**:
- Setting up user feedback system
- Need alternative to GitHub Issues
- Want to collect structured feedback
- Managing feedback responses

---

### 📧 [CONTACT_UPDATE_SUMMARY.md](CONTACT_UPDATE_SUMMARY.md)
Summary of contact information updates made to the app:
- Centralized contact configuration
- Files modified with new team email
- Feedback button implementation
- Instructions for pushing changes
- Testing checklist

**Use this when**:
- Reviewing recent contact info changes
- Understanding feedback button setup
- Deploying contact updates

---

### �🔄 [S3_MIGRATION_GUIDE.md](S3_MIGRATION_GUIDE.md)
Practical examples showing how to update your code to use S3:
- Before/after code comparisons
- Migration strategies (simple to advanced)
- Caching optimization
- Testing locally
- Rollback plan
- Common issues and fixes

**Use this when**:
- You're ready to implement S3 in your API files
- You want to test S3 integration locally
- You need code examples

---

## 🚀 Quick Start

### Option 1: Deploy Snapshot Mode (FREE)
**Best for**: Public-facing dashboard with aggregated data

```bash
# Already ready to deploy!
streamlit run app.py
# Or push to Streamlit Cloud
```

**No setup needed** - uses pre-computed JSON statistics from `stats/` directory

---

### Option 2: Add S3 Integration ($5/month)
**Best for**: Protected data access, datasets > 1GB, HIPAA compliance

```bash
# 1. Read the guides
open docs/AWS_S3_INTEGRATION.md
open docs/S3_MIGRATION_GUIDE.md

# 2. Set up S3 (15 minutes)
# - Create S3 bucket
# - Upload parquet files
# - Configure Streamlit secrets

# 3. Update code (5 minutes)
# Replace: MOVRDataLoader()
# With:    get_s3_loader()

# 4. Deploy
git push
```

---

## 📊 Decision Matrix

| Your Situation | Recommended Solution | Read This |
|----------------|---------------------|-----------|
| Just exploring the app | Run locally | Main README.md |
| Want to deploy public dashboard | Snapshot mode (free) | DEPLOYMENT.md |
| Need to understand architecture | Review design | [ARCHITECTURE_SCALABILITY.md](ARCHITECTURE_SCALABILITY.md) |
| **Setting up user feedback** | **Google Forms** | **[FEEDBACK_SETUP.md](FEEDBACK_SETUP.md)** |
| **Updating contact info** | **Review changes** | **[CONTACT_UPDATE_SUMMARY.md](CONTACT_UPDATE_SUMMARY.md)** |
| Data files > 1GB | Add S3 | [AWS_S3_INTEGRATION.md](AWS_S3_INTEGRATION.md) |
| Need PHI/protected data | Add S3 | [AWS_S3_INTEGRATION.md](AWS_S3_INTEGRATION.md) |
| Ready to code integration | Update API files | [S3_MIGRATION_GUIDE.md](S3_MIGRATION_GUIDE.md) |
| Troubleshooting S3 issues | Debug | [AWS_S3_INTEGRATION.md](AWS_S3_INTEGRATION.md#monitoring-and-troubleshooting) |

---

## 🎯 Common Workflows

### Workflow 1: First-Time Deployment
1. Read [ARCHITECTURE_SCALABILITY.md](ARCHITECTURE_SCALABILITY.md) (10 min)
2. Deploy snapshot mode to Streamlit Cloud (5 min)
3. Test public pages
4. Decide if you need S3 for protected pages

### Workflow 2: Adding S3 to Existing App
1. Read [AWS_S3_INTEGRATION.md](AWS_S3_INTEGRATION.md) (20 min)
2. Create S3 bucket and upload data (15 min)
3. Read [S3_MIGRATION_GUIDE.md](S3_MIGRATION_GUIDE.md) (10 min)
4. Update 1-2 API files (10 min)
5. Test locally, then deploy (15 min)

**Total time**: ~1-2 hours for full S3 integration

### Workflow 3: Troubleshooting Deployment Issues
1. Check [ARCHITECTURE_SCALABILITY.md](ARCHITECTURE_SCALABILITY.md#deployment-recommendations)
2. Review deployment logs in Streamlit Cloud
3. Search [AWS_S3_INTEGRATION.md](AWS_S3_INTEGRATION.md#monitoring-and-troubleshooting) for error messages
4. Verify secrets configuration

---

## 💡 Key Insights

### Your App is Well-Built ✅
- Follows Streamlit best practices
- Proper separation of concerns
- Clean API facade pattern
- Smart dual-mode design (snapshot vs live)
- Ready to scale on free tier (snapshot mode)

### Scalability Path
```
Phase 1: Snapshot Mode (FREE)
  → Deploy public dashboard
  → Scales to 1000s of users
  → No infrastructure needed
  
Phase 2: Add S3 (~$5/month)
  → Unlock protected pages
  → Unlimited data storage
  → HIPAA-compliant
  → Still use Streamlit free tier
  
Phase 3: Optimize (if needed)
  → Streamlit paid tier ($200/mo) OR
  → Self-host ($50/mo) OR
  → Database backend ($15/mo)
```

### Cost-Effective Scaling
- **Current**: Free (snapshot mode)
- **With S3**: $5/month (unlimited data)
- **Paid Streamlit**: $200/month (more resources)
- **Self-host**: $50/month (full control)

S3 integration gives you 99% of scaling benefits at <3% of the cost!

---

## 📁 Related Files

### Configuration
- `.streamlit/secrets.toml.example` - Template for credentials
- `config/settings.py` - App configuration
- `requirements.txt` - Python dependencies (includes boto3)

### Core Code
- `src/data_processing/s3_loader.py` - S3 data loader implementation
- `src/data_processing/loader.py` - Base data loader
- `api/*` - API layer files (update these for S3)

### Documentation
- `../ARCHITECTURE.md` - Original architecture doc
- `../DEPLOYMENT.md` - Deployment instructions
- `../README.md` - Main project README

---

## 🆘 Getting Help

### Documentation Not Clear?
- Open an issue on GitHub
- Email: andre.paredes@ymail.com

### AWS/S3 Questions?
- AWS Documentation: https://docs.aws.amazon.com/s3/
- boto3 Reference: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html

### Streamlit Questions?
- Streamlit Docs: https://docs.streamlit.io/
- Community Forum: https://discuss.streamlit.io/

### Security/Compliance Questions?
- AWS HIPAA Compliance: https://aws.amazon.com/compliance/hipaa-compliance/
- HIPAA Security Rule: https://www.hhs.gov/hipaa/for-professionals/security/

---

## 🔄 Keep Updated

Check for updates to:
- ✅ boto3 library (security patches)
- ✅ Streamlit version (new features)
- ✅ AWS IAM policies (security best practices)
- ✅ This documentation (improvements based on feedback)

---

## 📝 Contributing

Found an issue or improvement? Feel free to:
1. Update the relevant doc file
2. Submit a pull request
3. Open an issue with suggestions

These docs are living documents - contributions welcome!

---

**Last Updated**: February 2026
**Maintainer**: Andre D Paredes (andre.paredes@ymail.com)
