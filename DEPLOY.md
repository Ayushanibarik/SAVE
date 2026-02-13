# Streamlit Cloud Deployment Guide

## Prerequisites
1. GitHub account
2. PythonAnywhere account (for backend, see `backend-python/DEPLOY_PYTHONANYWHERE.md`)

## Step 1: Initialize Git Repository

```bash
cd "a:\multiagent ai  damage management system"
git init
git add .
git commit -m "Initial S.A.V.E. deployment"
```

## Step 2: Create GitHub Repository
1. Go to https://github.com/new
2. Name: `save-disaster-response` (or any name)
3. **DO NOT** initialize with README (we already have code)
4. Click **Create repository**

## Step 3: Push to GitHub
```bash
git remote add origin https://github.com/<your-username>/save-disaster-response.git
git branch -M main
git push -u origin main
```

## Step 4: Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Click **New app**
3. Select your repository: `<your-username>/save-disaster-response`
4. **Main file path:** `dashboard/app.py`
5. **Python version:** 3.11
6. Click **Deploy**

## Step 5: Configure Secrets
1. In Streamlit Cloud app settings
2. Go to **⚙️ Settings** → **Secrets**
3. Paste (replace `<yourusername>` with your PythonAnywhere username):

```toml
BACKEND_URL = "https://<yourusername>.pythonanywhere.com"
TWILIO_ACCOUNT_SID = "AC3d8ad8a35d6bfa2f9560d194b115121c"
TWILIO_AUTH_TOKEN = "60e4ccc50ba0c94f596311fbfae1dc06"
TWILIO_PHONE_NUMBER = "+12523903034"
```

4. Click **Save**
5. App will auto-redeploy with secrets

## Done! 🎉

Your S.A.V.E. dashboard will be live at:
```
https://share.streamlit.io/<your-username>/save-disaster-response/main
```

Or custom URL:
```
https://<app-name>.streamlit.app
```

---

## Troubleshooting

### Dashboard shows "Backend: Offline"
- Check backend is running: visit `https://<yourusername>.pythonanywhere.com/health`
- Verify `BACKEND_URL` secret is correct in Streamlit Cloud

### AI calls not working
- Check Twilio credentials in both backend WSGI and Streamlit secrets
- Verify phone number is verified in Twilio (free trial limitation)

### Import errors
- Check `dashboard/requirements.txt` includes all dependencies
- Streamlit auto-installs from this file
