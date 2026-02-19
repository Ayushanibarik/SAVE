"""Check PythonAnywhere error log and install status."""
import requests

USERNAME = "ayushanimesh"
TOKEN = "23d702e079ad2623ddf91ec16cd57dcec39f1b5c"
DOMAIN = USERNAME + ".pythonanywhere.com"
API = "https://www.pythonanywhere.com/api/v0/user/" + USERNAME
H = {"Authorization": "Token " + TOKEN}
REMOTE_DIR = "/home/" + USERNAME + "/save-backend"

# Check error log
print("=== ERROR LOG ===")
r = requests.get(API + "/files/path/var/log/" + DOMAIN + ".error.log", headers=H)
if r.status_code == 200:
    with open("pa_error.log", "w", encoding="utf-8") as f:
        f.write(r.text)
    print(r.text[-2000:])

# Check install error log
print("\n=== INSTALL ERROR LOG ===")
r = requests.get(API + "/files/path" + REMOTE_DIR + "/install_error.log", headers=H)
print("Status:", r.status_code)
if r.status_code == 200:
    print(r.text[:500])
else:
    print("No install_error.log (may mean install succeeded or wasn't attempted)")

# Check deps marker
print("\n=== DEPS MARKER ===")
r = requests.get(API + "/files/path" + REMOTE_DIR + "/.deps_installed", headers=H)
print("Status:", r.status_code)
if r.status_code == 200:
    print("Content:", r.text)

# Check what files exist in save-backend
print("\n=== FILES IN save-backend ===")
r = requests.get(API + "/files/path" + REMOTE_DIR + "/", headers=H)
if r.status_code == 200:
    for item in r.json():
        print(" ", item)
