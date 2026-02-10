# Google Form Copy-Paste Template for OpenMOVR App Feedback

## Instructions
1. Go to https://forms.google.com
2. Click "+ Blank" to create new form
3. Copy-paste the sections below into your form

---

## FORM HEADER

**Title:**
```
OpenMOVR App Feedback
```

**Description:**
```
Help us improve the OpenMOVR App by reporting issues, requesting features, or sharing your experience.

⚠️ PRIVACY NOTE: Do not include patient names, medical record numbers, or other protected health information (PHI) in your feedback. Describe issues in general terms only.
```

---

## QUESTIONS (Copy each section)

### Question 1
**Question:** Your Name (Optional)
**Type:** Short answer
**Required:** No

---

### Question 2
**Question:** Your Email (Optional - for follow-up)
**Type:** Short answer
**Required:** No
**Validation:** → Click three dots → Response validation → Email

---

### Question 3
**Question:** Your Role
**Type:** Multiple choice
**Required:** No

**Options:**
```
Clinician/Physician
Clinical Research Coordinator
Data Manager
Researcher
Patient/Caregiver
Pharmaceutical/Industry
Other
```

---

### Question 4
**Question:** Feedback Type
**Type:** Multiple choice
**Required:** Yes ✓

**Options:**
```
🐛 Bug/Error Report
✨ Feature Request
❓ Data Question
💭 General Feedback
🔒 Access Issue
📊 Data Dictionary Issue
📈 Clinical Analytics Issue
🗺️ Site/Facility Issue
Other
```

---

### Question 5
**Question:** Which page were you using?
**Type:** Dropdown
**Required:** No

**Options:**
```
Dashboard/Home
Disease Explorer
Facility View
Data Dictionary
Site Analytics
Download Center
DMD Clinical Analytics
LGMD Clinical Analytics
ALS Clinical Analytics
SMA Clinical Analytics
About Page
Sign the DUA
Not sure/Multiple pages
```

---

### Question 6
**Question:** Describe your feedback
**Type:** Paragraph
**Required:** Yes ✓

**Description text (below question):**
```
Please be as detailed as possible. 

For bugs: What did you expect to happen vs. what actually happened?
For features: What would you like to be able to do?
For data questions: Which specific field or table?
```

---

### Question 7
**Question:** Steps to Reproduce (For bugs only)
**Type:** Paragraph
**Required:** No

**Description text:**
```
If reporting a bug, please describe the steps:
1. I went to...
2. I clicked on...
3. Then I saw...
```

---

### Question 8
**Question:** Upload Screenshot (Optional)
**Type:** File upload
**Required:** No

**Settings:**
- Allow specific file types: Images only
- Maximum number of files: 3
- Maximum file size: 10 MB

---

### Question 9
**Question:** How urgent is this issue?
**Type:** Multiple choice
**Required:** No

**Options:**
```
🚨 Critical - Prevents me from using the app
🔴 High - Major inconvenience or blocking my work
🟡 Medium - Noticeable but I can work around it
🟢 Low - Minor issue or nice-to-have enhancement
```

---

### Question 10
**Question:** Any other comments or suggestions?
**Type:** Paragraph
**Required:** No

---

## FORM SETTINGS

### After Creating Form, Configure Settings:

**Click Settings (gear icon) → Presentation:**
- Confirmation message:
```
Thank you for your feedback!

We review all submissions and will follow up within 3-5 business days if you provided an email address. Your input helps us improve OpenMOVR for the entire community.

For urgent issues, please email mdamovr@mdausa.org directly.
```

**Click Settings → Responses:**
- [ ] Collect email addresses: OFF (email is optional in form)
- [ ] Limit to 1 response: OFF
- [ ] Allow response editing: OFF
- [✓] Show link to submit another response: ON

---

## AFTER CREATING FORM

### Step 1: Get Your Form URL
1. Click "Send" button (top right)
2. Click link icon (🔗)
3. Check "Shorten URL"
4. Copy the URL (looks like: `https://forms.gle/abc123XYZ`)

### Step 2: Set Up Email Notifications
1. Click "Responses" tab
2. Click three dots (⋮) → "Get email notifications for new responses"
3. Enter your email: `mdamovr@mdausa.org`

### Step 3: Create Response Spreadsheet
1. In "Responses" tab
2. Click Google Sheets icon (📊)
3. Select "Create a new spreadsheet"
4. Name it: "OpenMOVR Feedback Tracker"

### Step 4: Update Your App Config
Edit `config/contact.py`:
```python
FEEDBACK_FORM_URL = "https://forms.gle/abc123XYZ"  # Paste your actual URL
FEEDBACK_FORM_ENABLED = True
```

### Step 5: Commit and Deploy
```bash
git add config/contact.py
git commit -m "Enable feedback form"
git push
```

---

## RESPONSE TRACKING SPREADSHEET

### Add These Columns to Your Spreadsheet:
After the form creates the spreadsheet, add these manual tracking columns:

| Column | Purpose |
|--------|---------|
| Status | Open / In Progress / Resolved / Won't Fix |
| Priority | Critical / High / Medium / Low |
| Assigned To | Team member name |
| Date Resolved | When issue was fixed |
| Resolution Notes | What was done |
| User Notified | Yes/No - did you follow up? |

---

## SAMPLE RESPONSES FOR TESTING

Test your form with these examples:

**Test 1 - Bug Report:**
- Feedback Type: Bug/Error Report
- Page: Data Dictionary
- Description: "When I filter by DMD, the table shows 0 results but I know there are DMD fields"
- Urgency: High

**Test 2 - Feature Request:**
- Feedback Type: Feature Request
- Page: Disease Explorer
- Description: "Would love to see age distribution charts for each disease"
- Urgency: Low

**Test 3 - Data Question:**
- Feedback Type: Data Question
- Page: Download Center
- Description: "How often is the data updated? Can I get historical snapshots?"
- Urgency: Medium

---

## EMAIL RESPONSE TEMPLATES

### For Bug Reports:
```
Subject: Re: OpenMOVR Bug Report - [Brief Issue Description]

Hi [Name],

Thank you for reporting this issue. We've confirmed the bug with [specific feature/page] and have added it to our development queue.

Current status: [In Progress / Scheduled for Fix / Under Investigation]
Expected resolution: [Timeframe or "We'll update you when fixed"]

We'll send you another email when this is resolved.

Best regards,
MDA MOVR Data Hub Team
mdamovr@mdausa.org
```

### For Feature Requests:
```
Subject: Re: OpenMOVR Feature Request - [Feature Name]

Hi [Name],

Thank you for suggesting [feature]. This is valuable feedback from the community.

We've added your request to our roadmap for consideration. Feature prioritization is based on:
- Community demand (how many users request it)
- Development complexity
- Alignment with MOVR mission

You can track our development roadmap on the About page of the app.

Best regards,
MDA MOVR Data Hub Team
mdamovr@mdausa.org
```

### For Data Questions:
```
Subject: Re: OpenMOVR Data Question

Hi [Name],

Thank you for your question about [topic].

[Answer their specific question]

If you need more detailed information or have follow-up questions, please feel free to reply to this email or submit another form.

Best regards,
MDA MOVR Data Hub Team
mdamovr@mdausa.org
```

### Generic Acknowledgment (Auto-send within 24 hours):
```
Subject: OpenMOVR Feedback Received - [Ticket #123]

Hi [Name],

Thank you for submitting feedback about the OpenMOVR App. We've received your [bug report / feature request / question] and will review it shortly.

Your feedback is tracked as: Ticket #123

Response timeline:
• Critical issues: 24 hours
• High priority: 3 business days  
• Medium/Low: 1-2 weeks

We appreciate your help in improving OpenMOVR for the entire community.

Best regards,
MDA MOVR Data Hub Team
mdamovr@mdausa.org
```

---

## QUICK CHECKLIST

- [ ] Create form at forms.google.com
- [ ] Copy-paste title and description
- [ ] Add all 10 questions
- [ ] Configure form settings (confirmation message)
- [ ] Enable email notifications
- [ ] Create linked Google Sheet
- [ ] Add tracking columns to sheet
- [ ] Get shortened form URL
- [ ] Update config/contact.py with URL
- [ ] Set FEEDBACK_FORM_ENABLED = True
- [ ] Test form by submitting fake feedback
- [ ] Verify email notification works
- [ ] Commit and push to deploy
- [ ] Verify button works in deployed app

---

## ESTIMATED TIME: 30 MINUTES

- Form creation: 15 minutes
- Settings configuration: 5 minutes
- Testing: 5 minutes
- Deploy: 5 minutes

---

## NEED HELP?

Questions about setup: andre.paredes@ymail.com
Full guide: docs/FEEDBACK_SETUP.md
