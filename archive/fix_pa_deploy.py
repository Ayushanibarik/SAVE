"""
Download and upload missing packages directly to PythonAnywhere project folder.
This avoids the console API entirely by extracting wheel contents and uploading via files API.
"""
import requests
import os
import io
import zipfile
import time
import tempfile

USERNAME = "ayushanimesh"
TOKEN = "23d702e079ad2623ddf91ec16cd57dcec39f1b5c"
DOMAIN = USERNAME + ".pythonanywhere.com"
API = "https://www.pythonanywhere.com/api/v0/user/" + USERNAME
H = {"Authorization": "Token " + TOKEN}
REMOTE_DIR = "/home/" + USERNAME + "/save-backend"

# PyPI simple API to get wheel URLs
import subprocess


def pip_download(package, dest_dir):
    """Download a wheel for a package."""
    subprocess.check_call([
        "pip", "download", package,
        "--no-deps",
        "-d", dest_dir,
        "--python-version", "3.10",
        "--only-binary=:all:",
        "--platform", "manylinux_2_17_x86_64",
        "--implementation", "cp",
        "--abi", "cp310",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def pip_download_noarch(package, dest_dir):
    """Download a noarch/pure-python wheel."""
    subprocess.check_call([
        "pip", "download", package,
        "--no-deps",
        "-d", dest_dir,
        "--only-binary=:all:",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def upload_file(remote_path, content):
    """Upload file content to PythonAnywhere."""
    r = requests.post(
        API + "/files/path" + remote_path,
        headers=H,
        files={"content": ("file", content)}
    )
    return r.status_code


def upload_wheel_contents(wheel_path, remote_base):
    """Extract wheel and upload all Python files/dirs to remote."""
    uploaded = 0
    skipped = 0
    with zipfile.ZipFile(wheel_path, 'r') as zf:
        for info in zf.infolist():
            # Skip .dist-info directories
            if ".dist-info/" in info.filename:
                continue
            # Skip directories
            if info.is_dir():
                continue

            content = zf.read(info.filename)
            remote_path = remote_base + "/" + info.filename
            status = upload_file(remote_path, content)
            if status in (200, 201):
                uploaded += 1
            else:
                skipped += 1
    return uploaded, skipped


def main():
    # Packages we need that aren't pre-installed on PythonAnywhere
    # flask IS pre-installed, but flask-cors is not
    packages_noarch = ["flask-cors"]

    work_dir = tempfile.mkdtemp()
    print("Working dir:", work_dir)

    # 1. Download wheels
    print("\n=== Step 1: Downloading wheels ===")
    for pkg in packages_noarch:
        print(f"  Downloading {pkg}...")
        try:
            pip_download_noarch(pkg, work_dir)
            print(f"  OK: {pkg}")
        except Exception as e:
            print(f"  FAILED: {pkg} - {e}")

    # List downloaded wheels
    wheels = [f for f in os.listdir(work_dir) if f.endswith(".whl")]
    print(f"\n  Downloaded {len(wheels)} wheels:")
    for w in wheels:
        print(f"    {w}")

    # 2. Upload wheel contents to save-backend directory
    print("\n=== Step 2: Uploading package files ===")
    for wheel_file in wheels:
        wheel_path = os.path.join(work_dir, wheel_file)
        pkg_name = wheel_file.split("-")[0]
        print(f"  Extracting and uploading {pkg_name}...")
        uploaded, skipped = upload_wheel_contents(wheel_path, REMOTE_DIR)
        print(f"    Uploaded {uploaded} files, skipped {skipped}")

    # 3. Update WSGI to a clean version (no subprocess hacks)
    print("\n=== Step 3: Updating WSGI file ===")
    wsgi_content = """import sys
import os

# Add project directory (this also contains extracted package files)
project_home = '/home/ayushanimesh/save-backend'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Add user site-packages
user_site = '/home/ayushanimesh/.local/lib/python3.10/site-packages'
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# Set environment variables
os.environ['TWILIO_ACCOUNT_SID'] = 'AC3d8ad8a35d6bfa2f9560d194b115121c'
os.environ['TWILIO_AUTH_TOKEN'] = '60e4ccc50ba0c94f596311fbfae1dc06'
os.environ['TWILIO_PHONE_NUMBER'] = '+12523903034'

# Import Flask app
from optimize import app as application
"""
    r = requests.get(API + "/webapps/" + DOMAIN + "/", headers=H)
    wsgi_path = r.json().get("wsgi_file_path", "")
    if not wsgi_path:
        wsgi_path = "/var/www/" + USERNAME + "_pythonanywhere_com_wsgi.py"

    status = upload_file(wsgi_path, wsgi_content.encode())
    print(f"  WSGI upload: {status}")

    # 4. Reload webapp
    print("\n=== Step 4: Reloading webapp ===")
    r = requests.post(API + "/webapps/" + DOMAIN + "/reload/", headers=H)
    print(f"  Reload: {r.status_code}")

    # 5. Test
    print("\n=== Step 5: Testing health ===")
    time.sleep(15)
    try:
        r = requests.get("https://" + DOMAIN + "/health", timeout=30)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print("  SUCCESS!")
            print(r.text[:500])
        else:
            print("  Failed!")
            # Check error log
            r2 = requests.get(
                API + "/files/path/var/log/" + DOMAIN + ".error.log",
                headers=H
            )
            if r2.status_code == 200:
                lines = r2.text.strip().split("\n")
                print("\n  --- Last 10 error log lines ---")
                for line in lines[-10:]:
                    print("  ", line)
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
