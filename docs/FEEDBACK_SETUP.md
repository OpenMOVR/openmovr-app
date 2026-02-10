# Setting Up Feedback Collection

## Overview

Clinicians and researchers need an easy way to report issues, request features, and provide feedback. GitHub Issues are too technical for most users. This guide explains how to set up a clinician-friendly feedback system.

## Recommended Solution: Google Forms

**Why Google Forms?**
- ✅ Free and familiar to clinicians
- ✅ No login required (can be anonymous or ask for email)
- ✅ Responses go to Google Sheets for tracking
- ✅ Email notifications when submissions received
- ✅ Can include file uploads (screenshots)
- ✅ Works on mobile devices
- ✅ Professional appearance

## Step-by-Step Setup

### 1. Create Google Form

1. Go to [forms.google.com](https://forms.google.com)
2. Click **"+ Blank"** to create new form
3. Title: **"OpenMOVR App Feedback"**

### 2. Add Questions

**Recommended Form Structure**:

```
Title: OpenMOVR App Feedback

Description: Help us improve the OpenMOVR App by reporting issues, 
requesting features, or sharing your experience.

[Section 1: Contact Information]
1. Your Name (Optional)
   - Short answer
   
2. Your Email (Optional - for follow-up)
   - Short answer
   - Validation: Email
   
3. Your Role
   - Multiple choice
   - Options:
     * Clinician/Physician
     * Clinical Research Coordinator
     * Data Manager
     * Researcher
     * Patient/Caregiver
     * Other

[Section 2: Feedback Details]
4. Feedback Type (Required)
   - Multiple choice (required)
   - Options:
     * Bug/Error Report
     * Feature Request
     * Data Question
     * General Feedback
     * Other
   
5. Which page were you using? (Optional)
   - Multiple choice
   - Options:
     * Dashboard/Home
     * Disease Explorer
     * Facility View
     * Data Dictionary
     * Site Analytics
     * Download Center
     * DMD Clinical Analytics
     * LGMD Clinical Analytics
     * ALS Clinical Analytics
     * SMA Clinical Analytics
     * Other/Unsure

6. Describe your feedback (Required)
   - Paragraph
   - Required
   - Hint: "Please be as detailed as possible. For bugs, describe 
           what you expected vs. what happened."

7. Upload Screenshot (Optional)
   - File upload
   - Allow: Images only
   - Max files: 3

[Section 3: Priority (Optional)]
8. How urgent is this issue?
   - Multiple choice
   - Options:
     * Critical (prevents me from using the app)
     * High (major inconvenience)
     * Medium (nice to fix but not blocking)
     * Low (enhancement/suggestion)
```

### 3. Configure Form Settings

**Settings → Responses**:
- ✅ Collect email addresses: **OFF** (optional feedback should be easy)
- ✅ Limit to 1 response: **OFF** (allow multiple feedback)
- ✅ Edit after submit: **OFF**
- ✅ See summary charts: **ON**

**Settings → General**:
- Confirmation message: 
  ```
  Thank you for your feedback! We review all submissions and will 
  follow up if you provided an email address. Your input helps us 
  improve OpenMOVR for the entire community.
  ```

### 4. Set Up Notifications

1. Click **Responses** tab
2. Click three dots (**⋮**) → **Get email notifications for new responses**
3. This will send you an email each time someone submits feedback

### 5. Get Form URL

1. Click **Send** button (top right)
2. Click **link icon** (🔗)
3. Check **"Shorten URL"**
4. Copy the shortened URL (e.g., `https://forms.gle/abc123`)

### 6. Update OpenMOVR App Config

Edit `config/contact.py`:

```python
# Replace YOUR_FORM_ID_HERE with your actual form URL
FEEDBACK_FORM_URL = "https://forms.gle/abc123"  # Your URL from step 5
FEEDBACK_FORM_ENABLED = True  # Enable the feedback button
```

### 7. Test It

1. Restart your Streamlit app
2. Check sidebar - you should see **"📝 Report Issue or Feedback"** button
3. Click it to test the form
4. Verify you receive email notification

## Managing Responses

### View Responses in Google Sheets

1. In Google Forms, click **Responses** tab
2. Click **Google Sheets icon** (📊) → **Create Spreadsheet**
3. Name: "OpenMOVR Feedback Tracker"
4. This creates a live-updating spreadsheet with all responses

### Organize Responses

**Add columns to your sheet**:
- **Status**: Open, In Progress, Resolved, Won't Fix
- **Priority**: Critical, High, Medium, Low
- **Assigned To**: Team member
- **Notes**: Internal tracking
- **Date Resolved**: When fixed

### Set Up Automatic Labeling (Advanced)

Use Google Apps Script to auto-label urgent issues:

```javascript
function onFormSubmit(e) {
  var urgency = e.values[8]; // Column with urgency
  var email = "mdamovr@mdausa.org";
  
  if (urgency === "Critical (prevents me from using the app)") {
    MailApp.sendEmail({
      to: email,
      subject: "🚨 CRITICAL OpenMOVR Feedback",
      body: "A critical issue was reported. Check responses immediately."
    });
  }
}
```

## Alternative Options

### Option 2: Microsoft Forms
If your organization uses Microsoft 365:
- Similar to Google Forms
- Integrates with OneDrive/SharePoint
- https://forms.office.com

### Option 3: Typeform
More polished but costs money after 10 responses/month:
- Better UX design
- Logic jumps
- $25/month for Pro
- https://typeform.com

### Option 4: Streamlit Feedback Component
Built into Streamlit but limited:

```python
import streamlit as st

# Add to sidebar or bottom of page
feedback = st.feedback("thumbs")  # or "faces"
if feedback:
    st.write(f"Feedback recorded: {feedback}")
```

**Limitations**:
- Just thumbs up/down or faces (no text)
- No way to collect detailed feedback
- Stored in session only

### Option 5: Custom Streamlit Form
Build your own in-app form:

```python
with st.form("feedback_form"):
    st.markdown("### Send Feedback")
    feedback_type = st.selectbox("Type", ["Bug Report", "Feature Request", "Question"])
    message = st.text_area("Message", height=150)
    email = st.text_input("Your Email (optional)")
    
    if st.form_submit_button("Submit"):
        # Send email using SMTP or save to database
        send_feedback_email(feedback_type, message, email)
        st.success("Thank you for your feedback!")
```

**Requires**:
- Email server configuration (SMTP)
- Or database to store responses
- More maintenance

## Recommended Multi-Channel Strategy

### For Different User Types

| User Type | Primary Channel | Secondary |
|-----------|----------------|-----------|
| **Clinicians** | Google Form button in app | Email (mdamovr@mdausa.org) |
| **Researchers** | Google Form or Email | Email |
| **Developers** | GitHub Issues | Email |
| **Data Managers** | Email directly | Google Form |

### Communication Matrix

**In the App (About page)**:
```markdown
## Get Help or Provide Feedback

### For Users (Clinicians, Researchers)
- **Report Issues or Suggest Features**: Click "📝 Report Issue" in the sidebar
- **Questions about data access**: mdamovr@mdausa.org
- **General support**: mdamovr@mdausa.org

### For Developers
- **Technical issues or contributions**: 
  [GitHub Issues](https://github.com/OpenMOVR/openmovr-app/issues)
- **Pull requests welcome**: 
  [GitHub Repo](https://github.com/OpenMOVR/openmovr-app)
```

## Privacy Considerations

### HIPAA Compliance Note
- ⚠️ **Do NOT ask users to include PHI in feedback**
- Add disclaimer to form:
  ```
  IMPORTANT: Do not include any patient names, medical record 
  numbers, or other protected health information (PHI) in your 
  feedback. Describe issues in general terms only.
  ```

### Anonymous Feedback
- Make email field optional
- Don't require Google login
- Consider: "Allow anonymous submissions for sensitive topics"

## Monitoring and Response

### Best Practices

1. **Response Time Goals**:
   - Critical issues: 24 hours
   - High priority: 3 business days
   - Medium/Low: 1-2 weeks
   
2. **Acknowledgment**:
   If user provides email, send acknowledgment:
   ```
   Subject: Thank you for your OpenMOVR feedback
   
   Dear [Name],
   
   Thank you for submitting feedback about the OpenMOVR App. 
   We've received your [bug report/feature request/question] 
   and will review it shortly.
   
   Your feedback helps us improve the platform for the entire 
   research community.
   
   Best regards,
   MDA MOVR Data Hub Team
   ```

3. **Track Metrics**:
   - Number of submissions per month
   - Response time average
   - Resolution rate
   - User satisfaction (follow-up survey)

4. **Regular Reviews**:
   - Weekly review of new feedback
   - Monthly summary to stakeholders
   - Quarterly analysis of trends

## Sample Email Templates

### For Bug Reports
```
Subject: Re: OpenMOVR Bug Report - [Issue]

Hi [Name],

Thank you for reporting this issue with [specific page/feature]. 

We've confirmed the bug and have added it to our development 
roadmap. We expect to have a fix deployed by [timeframe].

We'll send you an update once it's resolved.

Best regards,
MDA MOVR Team
```

### For Feature Requests  
```
Subject: Re: OpenMOVR Feature Request - [Feature]

Hi [Name],

Thank you for suggesting [feature]. This is valuable feedback.

We've added your request to our roadmap for consideration. 
Feature prioritization is based on community need and 
development resources.

You can track our roadmap progress on the About page of the app.

Best regards,
MDA MOVR Team
```

## Implementation Checklist

- [ ] Create Google Form with recommended questions
- [ ] Enable email notifications for new responses
- [ ] Create linked Google Sheet for response tracking
- [ ] Set up response categorization (Status, Priority columns)
- [ ] Update `config/contact.py` with form URL
- [ ] Test feedback button in dev environment
- [ ] Add disclaimer about PHI to form
- [ ] Create email templates for responses
- [ ] Set up weekly feedback review meeting
- [ ] Announce feedback system to users
- [ ] Monitor first month's submissions and adjust

## Cost

**Google Forms**: FREE ✅
- Unlimited forms
- Unlimited responses
- Unlimited storage in Google Drive

**Time Investment**:
- Setup: 30 minutes
- Weekly review: 30-60 minutes
- Response time: 15 minutes per feedback item

## Questions?

Contact: andre.paredes@ymail.com
