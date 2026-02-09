# Deployment Guide

## Option 1: Streamlit Cloud (Recommended)

### Step 1: Push to GitHub

```bash
cd /home/aparedes/MDA/openmovr-app

# Commit
git add .
git commit -m "Initial OpenMOVR App"

# Create GitHub repo "openmovr-app" under OpenMOVR organization, then:
git remote add origin https://github.com/OpenMOVR/openmovr-app.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - Repository: `OpenMOVR/openmovr-app`
   - Branch: `main`
   - Main file: `app.py`
5. Click "Deploy"

Your app will be live at: `https://openmovr-app.streamlit.app`

### Step 3: Custom Domain (Optional)

In Streamlit Cloud settings:
1. Go to app settings → "Custom domain"
2. Add `app.openmovr.io`
3. Configure DNS:
   ```
   CNAME app.openmovr.io → openmovr-app.streamlit.app
   ```

---

## Option 2: GitHub Pages Redirect

Since GitHub Pages only serves static files, create a redirect page.

### Step 1: Deploy to Streamlit Cloud first (see above)

### Step 2: Create redirect repo

Create a new repo called `app` under OpenMOVR organization (will be `app.openmovr.github.io`).

Add `index.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Redirecting to OpenMOVR App...</title>
    <meta http-equiv="refresh" content="0; URL=https://openmovr-app.streamlit.app">
    <link rel="canonical" href="https://openmovr-app.streamlit.app">
</head>
<body>
    <p>Redirecting to <a href="https://openmovr-app.streamlit.app">OpenMOVR App</a>...</p>
</body>
</html>
```

### Step 3: Enable GitHub Pages

1. Go to repo Settings → Pages
2. Source: Deploy from branch `main`
3. The redirect will be live at `https://openmovr.github.io/app/`

---

## URL Options Summary

| URL | How |
|-----|-----|
| `openmovr-app.streamlit.app` | Default Streamlit Cloud URL |
| `app.openmovr.io` | Custom domain → Streamlit Cloud |
| `openmovr.github.io/app` | GitHub Pages redirect |

---

## Access Key Configuration

DUA-gated pages (Site Analytics, Download Center, DMD/LGMD Clinical Summaries) require an access key.

### Streamlit Cloud
Add to Streamlit secrets (Settings → Secrets):
```toml
OPENMOVR_SITE_KEY = "your-access-key"
```

### Local Development
```bash
export OPENMOVR_SITE_KEY="your-access-key"
```

---

## Updating the App

1. Make changes locally
2. Regenerate snapshots if data changed:
   ```bash
   python scripts/generate_stats_snapshot.py
   python scripts/generate_dmd_snapshot.py
   python scripts/generate_lgmd_snapshot.py
   python scripts/generate_curated_dictionary.py
   ```
3. Commit and push:
   ```bash
   cd /home/aparedes/MDA/openmovr-app
   git add .
   git commit -m "Update description"
   git push
   ```
4. Streamlit Cloud auto-deploys on push

---

## Troubleshooting

### "Module not found" errors
- Ensure `src/` directory is included
- Check `__init__.py` files exist

### App shows "Snapshot not found"
- Verify `stats/database_snapshot.json` exists
- Regenerate snapshot if needed

### Slow loading
- Snapshots should load instantly
- Live mode with parquet files takes longer

---

## Contact

Andre D Paredes
aparedes@mdausa.org
