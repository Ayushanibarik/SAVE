# PythonAnywhere Deployment Guide for S.A.V.E. Backend

## Step 1: Create PythonAnywhere Account
1. Go to https://www.pythonanywhere.com/registration/register/beginner/
2. Create a **free Beginner account** (no credit card needed)
3. Confirm your email

## Step 2: Upload Backend Files
1. In PythonAnywhere dashboard, go to **Files**
2. Navigate to `/home/<yourusername>/`
3. Create folder: `save-backend`
4. Upload these files to `save-backend/`:
   - `optimize.py`
   - `ai_caller.py`
   - `config.py`
   - `data_loader.py`
   - `data_manager.py`
   - `data_sources.py`
   - `rl_optimizer.py`
   - All other `.py` files from `backend-python/`
   - `requirements.txt`

## Step 3: Install Dependencies
1. Go to **Consoles** → **Bash**
2. Run:
   ```bash
   cd save-backend
   pip3.10 install --user -r requirements.txt
   ```

## Step 4: Configure Web App
1. Go to **Web** tab
2. Click **Add a new web app**
3. Choose **Manual configuration** → **Python 3.10**
4. In **Code** section:
   - **Source code:** `/home/<yourusername>/save-backend`
   - **Working directory:** `/home/<yourusername>/save-backend`
   - **WSGI configuration file:** Click to edit

5. Replace entire WSGI file content with:
   ```python
   import sys
   import os

   # Add your project directory to the sys.path
   project_home = '/home/<yourusername>/save-backend'
   if project_home not in sys.path:
       sys.path.insert(0, project_home)

   # Set environment variables
   os.environ['TWILIO_ACCOUNT_SID'] = 'AC3d8ad8a35d6bfa2f9560d194b115121c'
   os.environ['TWILIO_AUTH_TOKEN'] = '60e4ccc50ba0c94f596311fbfae1dc06'
   os.environ['TWILIO_PHONE_NUMBER'] = '+12523903034'

   # Import Flask app
   from optimize import app as application
   ```

6. Click **Save**

## Step 5: Enable Flask App
1. Still in **Web** tab, scroll to **Virtualenv** section (optional, skip if dependencies installed with --user)
2. Click **Reload** button (big green button at top)

## Step 6: Get Your Backend URL
Your backend will be available at:
```
https://<yourusername>.pythonanywhere.com
```

## Step 7: Update Streamlit Secrets
In Streamlit Cloud, set this in **Secrets**:
```toml
BACKEND_URL = "https://<yourusername>.pythonanywhere.com"
TWILIO_ACCOUNT_SID = "AC3d8ad8a35d6bfa2f9560d194b115121c"
TWILIO_AUTH_TOKEN = "60e4ccc50ba0c94f596311fbfae1dc06"
TWILIO_PHONE_NUMBER = "+12523903034"
```

---

## Free Tier Limitations
- **Always-on:** No cold starts! ✅
- **Traffic:** 100k hits/day (plenty for demos)
- **Duration:** Free forever
- **Downsides:** CPU limited to 100 seconds/day (fine for API calls)

---

## Testing Backend
After reload, test:
```bash
curl https://<yourusername>.pythonanywhere.com/health
```

Should return:
```json
{
  "status": "healthy",
  "version": "3.0-AI",
  ...
}
```
