# Contact Information Update - Summary

## Changes Made

### 1. **Centralized Contact Configuration** ✅
Created [`config/contact.py`](../config/contact.py) as single source of truth for all contact information:

- **Primary Contact**: `mdamovr@mdausa.org` (MDA MOVR Data Hub Team)
- **Developer Contact**: `andre.paredes@ymail.com` (for technical/GitHub issues)
- **Feedback Form URL**: Placeholder for Google Forms (see setup guide)
- **GitHub Links**: For developer contributions

### 2. **Updated Contact Information Across App** ✅

**Files Updated**:
- ✅ `components/sidebar.py` - Sidebar footer now shows team email
- ✅ `components/sidebar.py` - Page footer updated
- ✅ `pages/4_About.py` - Added comprehensive "Get Help" section
- ✅ `data/README.md` - Data access contact updated
- ✅ `DEPLOYMENT.md` - Contact section updated
- ✅ `ARCHITECTURE.md` - Contact section updated  
- ✅ `CLAUDE.md` - Contact section updated
- ✅ `utils/access.py` - Error messages now reference team email
- ✅ Documentation files in `docs/` folder

### 3. **Added Feedback Button** ✅

**New Feature in Sidebar**:
- 📝 **"Report Issue or Feedback"** button appears in every page's sidebar
- Links to Google Form (when configured) or email (fallback)
- Clinician-friendly - no GitHub account needed
- Mobile-responsive

**Configuration**:
```python
# In config/contact.py
FEEDBACK_FORM_ENABLED = False  # Change to True after creating form
FEEDBACK_FORM_URL = "https://forms.gle/YOUR_FORM_ID_HERE"
```

### 4. **Enhanced About Page** ✅

Added **"Get Help or Provide Feedback"** section with:
- **For Users** (Clinicians/Researchers):
  - Feedback form link
  - Email contact
  - Study participation info
  - PHI privacy warning
  
- **For Developers**:
  - GitHub Issues link
  - Repository link
  - Documentation reference
  - Contribution guidelines

## What You Need to Do Next

### Immediate (Required) ✅

**The app is ready to push with corrected contact info!**

All references now properly use:
- `mdamovr@mdausa.org` for user support
- `andre.paredes@ymail.com` for developer/technical issues (shown only when configured)

```bash
# Review changes
git status

# Commit the contact info fix
git add config/contact.py components/sidebar.py pages/4_About.py 
git add data/README.md DEPLOYMENT.md ARCHITECTURE.md CLAUDE.md utils/access.py
git commit -m "Fix contact information - use MDA MOVR team email"

# Push to deploy
git push
```

### Soon (Recommended) 📝

**Set up Google Forms feedback** (30 minutes):

1. **Create the form**:
   - Follow: [`docs/FEEDBACK_SETUP.md`](FEEDBACK_SETUP.md)
   - Estimated time: 30 minutes
   - Cost: FREE

2. **Update config**:
   ```python
   # In config/contact.py
   FEEDBACK_FORM_URL = "https://forms.gle/YOUR_ACTUAL_URL"
   FEEDBACK_FORM_ENABLED = True
   ```

3. **Redeploy**:
   ```bash
   git add config/contact.py
   git commit -m "Enable feedback form"
   git push
   ```

## Recommended Feedback Strategy

### **Multi-Channel Approach** 🎯

| User Type | Primary Method | Secondary | Use For |
|-----------|---------------|-----------|---------|
| **Clinicians** | Google Form button | Email | Bug reports, feature requests |
| **Researchers** | Google Form or Email | Email | Data questions, feedback |
| **Data Managers** | Email directly | Google Form | Access, data issues |
| **Developers** | GitHub Issues | Email | Code, technical issues |

### **Why This Works**

✅ **Google Forms for clinicians because**:
- No account required
- Familiar interface
- Works on mobile
- Can be anonymous
- Collects structured data
- Automatic email notifications

❌ **NOT GitHub Issues for clinicians because**:
- Requires GitHub account
- Technical interface
- Intimidating for non-developers
- Public by default
- Steep learning curve

✅ **Keep GitHub for developers**:
- Code-related discussions
- Pull requests
- Technical documentation
- Open source community

## Configuration Options

### Show/Hide Developer Contact

```python
# In config/contact.py
SHOW_DEVELOPER_CONTACT = False  # Default: team email only
SHOW_DEVELOPER_CONTACT = True   # Show both team and developer email
```

**Recommendation**: Keep `False` to avoid confusion. Users don't need to know about developer vs team distinction.

### Enable/Disable Feedback Button

```python
# In config/contact.py
SHOW_FEEDBACK_BUTTON = True   # Show "Report Issue" button (recommended)
SHOW_FEEDBACK_BUTTON = False  # Hide button (email only)
```

**Recommendation**: Keep `True` once you set up Google Form.

## What It Looks Like

### Sidebar Footer (Every Page)
```
───────────────────────────────
  [📝 Report Issue or Feedback] <- Blue button
  
  Contact: mdamovr@mdausa.org
  
  [MOVR Logo]
```

### About Page → "Get Help" Section
```
┌───────────────────────┬───────────────────────┐
│ For Users             │ For Developers        │
│ • Report issues       │ • GitHub Issues       │
│ • Data access         │ • Pull requests       │
│ • Study participation │ • Documentation       │
└───────────────────────┴───────────────────────┘

💡 Tip: Use sidebar button for quick feedback!
```

### Page Footer (Bottom of Every Page)
```
────────────────────────────────────────
Data Source: MDA MOVR Data Hub Study
Independently built via OpenMOVR Initiative
Gen1 | v0.2.0 (Prototype)
mdamovr@mdausa.org
```

## Response Management

### Email Workflow

**When `mdamovr@mdausa.org` receives feedback**:

1. ✅ Reply within 24-48 hours with acknowledgment
2. ✅ Categorize: Bug / Feature / Question / Other
3. ✅ Track in spreadsheet or issue tracker
4. ✅ Update user when resolved (if email provided)

### Google Forms Workflow

**When form submitted**:

1. Auto-notification email sent to `mdamovr@mdausa.org`
2. Response recorded in Google Sheets
3. Add tracking columns: Status, Priority, Assigned To, Date Resolved
4. Weekly review meeting to triage new submissions
5. Follow up with user if email provided

See [`docs/FEEDBACK_SETUP.md`](FEEDBACK_SETUP.md) for detailed workflow.

## Privacy & Compliance

### PHI Warning ⚠️

Added to About page and feedback form:
```
⚠️ Privacy Note: Do not include patient names, medical record 
numbers, or other PHI in feedback submissions.
```

### Anonymous Feedback

- Email field in form is **optional**
- No login required
- Users can submit anonymously if preferred

### HIPAA Considerations

✅ Feedback form is HIPAA-safe when:
- Users instructed NOT to include PHI
- Form doesn't collect identifying info by default
- Hosted by Google (BAA available if needed)

## Files Created

1. **[`config/contact.py`](../config/contact.py)** - Centralized contact config
2. **[`docs/FEEDBACK_SETUP.md`](FEEDBACK_SETUP.md)** - Complete setup guide
3. **[`docs/CONTACT_UPDATE_SUMMARY.md`](CONTACT_UPDATE_SUMMARY.md)** - This file

## Files Modified

1. **[`components/sidebar.py`](../components/sidebar.py)** - Sidebar footer + feedback button
2. **[`pages/4_About.py`](../pages/4_About.py)** - Added "Get Help" section
3. **[`data/README.md`](../data/README.md)** - Data access contact
4. **[`utils/access.py`](../utils/access.py)** - Error message contact
5. **[`DEPLOYMENT.md`](../DEPLOYMENT.md)** - Contact section
6. **[`ARCHITECTURE.md`](../ARCHITECTURE.md)** - Contact section
7. **[`CLAUDE.md`](../CLAUDE.md)** - Contact section

## Testing Checklist

Before deploying:

- [ ] Run app locally: `streamlit run app.py`
- [ ] Check sidebar footer on every page - shows `mdamovr@mdausa.org`
- [ ] Check About page - "Get Help" section displays correctly
- [ ] Click "Send Feedback" button (should open email if form not set up)
- [ ] Check page footers - show team email
- [ ] Verify no `andre.paredes@ymail.com` in user-facing UI (only in docs)

After setting up Google Form:

- [ ] Update `FEEDBACK_FORM_URL` in `config/contact.py`
- [ ] Set `FEEDBACK_FORM_ENABLED = True`
- [ ] Redeploy
- [ ] Test feedback button - should open Google Form
- [ ] Submit test feedback
- [ ] Verify email notification received
- [ ] Check Google Sheets has response

## Questions?

- **Setup help**: See [`docs/FEEDBACK_SETUP.md`](FEEDBACK_SETUP.md)
- **Contact config**: See [`config/contact.py`](../config/contact.py)
- **Developer questions**: andre.paredes@ymail.com
