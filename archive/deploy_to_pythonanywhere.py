"""
S.A.V.E. Backend — Automated PythonAnywhere Deployment Script
Uploads all backend files, creates/configures webapp, installs deps, and verifies.
"""

import requests
import os
import time
import glob

# ============================================
# CONFIGURATION
# ============================================
USERNAME = "ayushanimesh"
API_TOKEN = "23d702e079ad2623ddf91ec16cd57dcec39f1b5c"
DOMAIN = f"{USERNAME}.pythonanywhere.com"
API_BASE = f"https://www.pythonanywhere.com/api/v0/user/{USERNAME}"
HEADERS = {"Authorization": f"Token {API_TOKEN}"}

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend-python")
REMOTE_DIR = f"/home/{USERNAME}/save-backend"

# Trimmed requirements (no torch/scikit-learn — they're unused and too large)
REQUIREMENTS_CONTENT = """flask>=3.0.0
flask-cors>=4.0.0
requests>=2.31.0
numpy>=1.24.0
geopy>=2.4.0
pymongo>=4.0.0
"""

WSGI_CONTENT = f"""import sys
import os

# Add project directory to sys.path
project_home = '{REMOTE_DIR}'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set environment variables
os.environ['TWILIO_ACCOUNT_SID'] = 'AC3d8ad8a35d6bfa2f9560d194b115121c'
os.environ['TWILIO_AUTH_TOKEN'] = '60e4ccc50ba0c94f596311fbfae1dc06'
os.environ['TWILIO_PHONE_NUMBER'] = '+12523903034'

# Import Flask app
from optimize import app as application
"""


def log(msg, level="INFO"):
    icons = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERR": "❌", "UP": "📤"}
    print(f"  {icons.get(level, '•')} [{level}] {msg}")


def api_get(path, **kwargs):
    return requests.get(f"{API_BASE}{path}", headers=HEADERS, **kwargs)


def api_post(path, **kwargs):
    return requests.post(f"{API_BASE}{path}", headers=HEADERS, **kwargs)


def api_delete(path, **kwargs):
    return requests.delete(f"{API_BASE}{path}", headers=HEADERS, **kwargs)


def api_patch(path, **kwargs):
    return requests.patch(f"{API_BASE}{path}", headers=HEADERS, **kwargs)


# ============================================
# STEP 1: Upload all backend .py files
# ============================================
def upload_files():
    print("\n" + "=" * 60)
    print("  STEP 1: Uploading backend files")
    print("=" * 60)

    py_files = glob.glob(os.path.join(BACKEND_DIR, "*.py"))
    if not py_files:
        log(f"No .py files found in {BACKEND_DIR}", "ERR")
        return False

    log(f"Found {len(py_files)} Python files to upload")

    for filepath in py_files:
        filename = os.path.basename(filepath)
        remote_path = f"{REMOTE_DIR}/{filename}"
        with open(filepath, "rb") as f:
            resp = api_post(
                f"/files/path{remote_path}",
                files={"content": f}
            )
        if resp.status_code in (200, 201):
            log(f"{filename} → uploaded", "UP")
        else:
            log(f"{filename} → FAILED ({resp.status_code}: {resp.text[:100]})", "ERR")
            return False

    # Upload trimmed requirements.txt
    resp = api_post(
        f"/files/path{REMOTE_DIR}/requirements.txt",
        files={"content": ("requirements.txt", REQUIREMENTS_CONTENT.encode())}
    )
    if resp.status_code in (200, 201):
        log("requirements.txt (trimmed) → uploaded", "UP")
    else:
        log(f"requirements.txt → FAILED ({resp.status_code})", "ERR")
        return False

    log("All files uploaded successfully!", "OK")
    return True


# ============================================
# STEP 2: Create or update web app
# ============================================
def setup_webapp():
    print("\n" + "=" * 60)
    print("  STEP 2: Creating/configuring web app")
    print("=" * 60)

    # Check if webapp already exists
    resp = api_get(f"/webapps/{DOMAIN}/")
    if resp.status_code == 200:
        log(f"Webapp {DOMAIN} already exists — will reconfigure", "WARN")
    else:
        # Create new webapp
        log("Creating new webapp...")
        resp = api_post(
            "/webapps/",
            data={
                "domain_name": DOMAIN,
                "python_version": "python310",
            }
        )
        if resp.status_code in (200, 201):
            log(f"Webapp created: https://{DOMAIN}", "OK")
        else:
            log(f"Create webapp failed ({resp.status_code}): {resp.text[:200]}", "ERR")
            return False

    # Update source code directory
    resp = api_patch(
        f"/webapps/{DOMAIN}/",
        data={"source_directory": REMOTE_DIR}
    )
    if resp.status_code == 200:
        log(f"Source directory set to {REMOTE_DIR}", "OK")
    else:
        log(f"Set source dir failed: {resp.text[:100]}", "WARN")

    return True


# ============================================
# STEP 3: Configure WSGI file
# ============================================
def configure_wsgi():
    print("\n" + "=" * 60)
    print("  STEP 3: Configuring WSGI")
    print("=" * 60)

    # Get the WSGI file path from webapp config
    resp = api_get(f"/webapps/{DOMAIN}/")
    if resp.status_code != 200:
        log("Cannot get webapp config", "ERR")
        return False

    webapp_info = resp.json()
    wsgi_path = webapp_info.get("wsgi_file_path", "")

    if not wsgi_path:
        # Default path for PythonAnywhere
        wsgi_path = f"/var/www/{USERNAME.replace('.', '_')}_pythonanywhere_com_wsgi.py"
        log(f"Using default WSGI path: {wsgi_path}", "WARN")
    else:
        log(f"WSGI file path: {wsgi_path}", "INFO")

    # Write WSGI content
    resp = api_post(
        f"/files/path{wsgi_path}",
        files={"content": ("wsgi.py", WSGI_CONTENT.encode())}
    )
    if resp.status_code in (200, 201):
        log("WSGI file configured", "OK")
    else:
        log(f"WSGI write failed ({resp.status_code}): {resp.text[:200]}", "ERR")
        return False

    return True


# ============================================
# STEP 4: Install dependencies via console
# ============================================
def install_dependencies():
    print("\n" + "=" * 60)
    print("  STEP 4: Installing dependencies")
    print("=" * 60)

    log("Creating Bash console to install dependencies...")

    # Create a bash console
    resp = api_post(
        "/consoles/",
        data={"executable": "bash", "arguments": "", "working_directory": REMOTE_DIR}
    )
    if resp.status_code not in (200, 201):
        log(f"Console creation failed ({resp.status_code}): {resp.text[:200]}", "ERR")
        return False

    console_id = resp.json().get("id")
    log(f"Console created (ID: {console_id})", "OK")

    # Send pip install command
    install_cmd = f"cd {REMOTE_DIR} && pip3.10 install --user -r requirements.txt\n"
    time.sleep(3)  # Wait for console to initialize

    resp = api_post(
        f"/consoles/{console_id}/send_input/",
        data={"input": install_cmd}
    )
    if resp.status_code == 200:
        log("pip install command sent", "OK")
    else:
        log(f"Send command failed ({resp.status_code}): {resp.text[:100]}", "ERR")

    # Wait for installation to complete
    log("Waiting for installation (this may take 30-60 seconds)...")
    time.sleep(45)

    # Check console output
    resp = api_get(f"/consoles/{console_id}/get_latest_output/")
    if resp.status_code == 200:
        output = resp.json().get("output", "")
        if "Successfully installed" in output or "already satisfied" in output:
            log("Dependencies installed successfully!", "OK")
        else:
            log("Install output (check for errors):", "WARN")
            # Print last 500 chars of output
            print(f"    {output[-500:]}")
    else:
        log("Could not read console output — check manually", "WARN")

    return True


# ============================================
# STEP 5: Reload webapp
# ============================================
def reload_webapp():
    print("\n" + "=" * 60)
    print("  STEP 5: Reloading web app")
    print("=" * 60)

    resp = api_post(f"/webapps/{DOMAIN}/reload/")
    if resp.status_code == 200:
        log("Webapp reloaded!", "OK")
        return True
    else:
        log(f"Reload failed ({resp.status_code}): {resp.text[:200]}", "ERR")
        return False


# ============================================
# STEP 6: Verify health endpoint
# ============================================
def verify_deployment():
    print("\n" + "=" * 60)
    print("  STEP 6: Verifying deployment")
    print("=" * 60)

    log("Waiting 10 seconds for app to start...")
    time.sleep(10)

    url = f"https://{DOMAIN}/health"
    log(f"Testing: {url}")

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            log(f"Status: {data.get('status', 'unknown')}", "OK")
            log(f"Version: {data.get('version', 'unknown')}", "OK")
            log(f"System: {data.get('system', 'unknown')}", "OK")
            print("\n" + "=" * 60)
            print(f"  🎉 DEPLOYMENT SUCCESSFUL!")
            print(f"  🌐 Backend URL: https://{DOMAIN}")
            print(f"  ❤️  Health:     https://{DOMAIN}/health")
            print("=" * 60)
            return True
        else:
            log(f"Health check returned {resp.status_code}", "ERR")
            log(f"Response: {resp.text[:300]}", "ERR")
    except Exception as e:
        log(f"Health check failed: {e}", "ERR")

    print("\n  ⚠️  Deployment may need manual review on PythonAnywhere dashboard.")
    print(f"  🔗 https://www.pythonanywhere.com/user/{USERNAME}/webapps/")
    return False


# ============================================
# MAIN
# ============================================
def main():
    print("=" * 60)
    print("  S.A.V.E. — PythonAnywhere Deployment")
    print(f"  Target: https://{DOMAIN}")
    print("=" * 60)

    # Verify API access first
    log("Verifying API access...")
    resp = api_get("/cpu/")
    if resp.status_code != 200:
        log(f"API authentication failed ({resp.status_code}): {resp.text}", "ERR")
        log("Check your username and API token.", "ERR")
        return

    log("API access verified", "OK")

    steps = [
        ("Upload files", upload_files),
        ("Setup webapp", setup_webapp),
        ("Configure WSGI", configure_wsgi),
        ("Install dependencies", install_dependencies),
        ("Reload webapp", reload_webapp),
        ("Verify deployment", verify_deployment),
    ]

    for name, func in steps:
        if not func():
            log(f"Step '{name}' reported issues — continuing anyway...", "WARN")

    print("\n  Done! Check https://www.pythonanywhere.com/user/"
          f"{USERNAME}/webapps/ for details.\n")


if __name__ == "__main__":
    main()
