import requests
import json

def test_geocode():
    print("Testing Geocode Endpoint...")
    try:
        r = requests.get("http://localhost:5000/geocode?q=Paris", timeout=10)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("Response:", json.dumps(r.json(), indent=2))
        else:
            print("Error:", r.text)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_geocode()
