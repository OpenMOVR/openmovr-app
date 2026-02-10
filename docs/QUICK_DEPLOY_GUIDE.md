# Quick Reference: Deploying Contact Info Updates

## ✅ Ready to Push Right Now

All contact information has been fixed and is ready to deploy!

### What Was Changed

**Primary Contact** → `mdamovr@mdausa.org` (MDA MOVR Data Hub Team)
- Sidebar footer (all pages)
- Page footer (all pages)  
- About page
- Error messages
- Documentation files

**Developer Contact** → `andre.paredes@ymail.com` (hidden by default, shows in docs only)

## Push Commands

```bash
# 1. Review what changed
git status
git diff

# 2. Stage contact info updates
git add config/contact.py
git add components/sidebar.py
git add pages/4_About.py
git add utils/access.py
git add data/README.md
git add DEPLOYMENT.md ARCHITECTURE.md CLAUDE.md

# 3. Stage documentation (optional but recommended)
git add docs/

# 4. Commit with clear message
git commit -m "Fix contact information across app

- Centralize contact config in config/contact.py
- Update all references to use mdamovr@mdausa.org
- Add feedback button infrastructure (disabled until form created)
- Add comprehensive 'Get Help' section to About page
- Update all documentation files"

# 5. Push to deploy
git push origin main
```

## What Happens After Deploy

### Users Will See
- ✅ `mdamovr@mdausa.org` in sidebar footer
- ✅ `mdamovr@mdausa.org` in page footer
- ✅ "Send Feedback" button in sidebar (opens email until form is set up)
- ✅ "Get Help or Provide Feedback" section in About page
- ✅ Professional team email instead of personal email

### After Streamlit Cloud Auto-Deploys
1. Check any page → sidebar footer shows team email
2. Go to About page → "Get Help" section looks good
3. Click "Send Feedback" button → opens email client
4. Everything works ✅

## Next Step: Set Up Google Form (Optional)

**When**: Soon (not urgent, email works for now)
**Time**: 30 minutes
**Cost**: FREE

### Quick Setup

1. Go to [forms.google.com](https://forms.google.com)
2. Follow: [`docs/FEEDBACK_SETUP.md`](../docs/FEEDBACK_SETUP.md)
3. Get form URL (e.g., `https://forms.gle/abc123`)
4. Update `config/contact.py`:
   ```python
   FEEDBACK_FORM_URL = "https://forms.gle/abc123"  # Your actual URL
   FEEDBACK_FORM_ENABLED = True
   ```
5. Commit and push:
   ```bash
   git add config/contact.py
   git commit -m "Enable feedback form"
   git push
   ```

Done! Button now opens Google Form instead of email.

## Verification Checklist

After deploying, verify:

- [ ] Open app in browser
- [ ] Check sidebar → shows `mdamovr@mdausa.org`
- [ ] Click "Send Feedback" button → opens email
- [ ] Go to About page → "Get Help" section displays
- [ ] Check page footer → shows team email
- [ ] No `aparedes@mdausa.org` anywhere user-facing
- [ ] No `andre.paredes@ymail.com` in UI (OK in docs)

## Questions About This Update?

**What changed?**: See [`docs/CONTACT_UPDATE_SUMMARY.md`](../docs/CONTACT_UPDATE_SUMMARY.md)
**How to set up form?**: See [`docs/FEEDBACK_SETUP.md`](../docs/FEEDBACK_SETUP.md)
**Need help?**: Email andre.paredes@ymail.com

## TL;DR

```bash
# Add, commit, push - done!
git add config/ components/ pages/ utils/ data/ docs/ *.md
git commit -m "Fix contact information and add feedback system"
git push

# Later: Create Google Form and update config/contact.py
```

That's it! 🚀
