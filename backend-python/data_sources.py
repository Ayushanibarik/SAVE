"""
Multi-Source Data Layer for Disaster Response System
Supports: MongoDB, Google Sheets, OpenStreetMap (Overpass API)
Enhanced with caching, better error handling, and realistic capacity estimation
"""

import os
import json
import requests
import sys
import random
from datetime import datetime
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from config import Config
from cache_manager import cache


# ============================================
# DATA MODELS
# ============================================

class Hospital:
    def __init__(self, name: str, beds: int, lat: float, lng: float,
                 address: str = "", phone: str = "", specialty: str = "General",
                 emergency: str = "yes", icu_beds: int = 0, trauma_level: str = ""):
        self.name = name
        self.beds = beds
        self.lat = lat
        self.lng = lng
        self.address = address
        self.phone = phone
        self.specialty = specialty
        self.emergency = emergency
        self.icu_beds = icu_beds
        self.trauma_level = trauma_level

    def to_dict(self) -> dict:
        return {
            "Hospital": self.name,
            "Beds": self.beds,
            "lat": self.lat,
            "lng": self.lng,
            "address": self.address,
            "phone": self.phone,
            "specialty": self.specialty,
            "emergency": self.emergency,
            "icu_beds": self.icu_beds,
            "trauma_level": self.trauma_level
        }


class Ambulance:
    def __init__(self, ambulance_id: str, lat: float, lng: float,
                 status: str = "available", driver: str = "",
                 vehicle_type: str = "BLS", equipment: list = None):
        self.id = ambulance_id
        self.lat = lat
        self.lng = lng
        self.status = status
        self.driver = driver
        self.vehicle_type = vehicle_type  # BLS (Basic) or ALS (Advanced)
        self.equipment = equipment or []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "lat": self.lat,
            "lng": self.lng,
            "status": self.status,
            "driver": self.driver,
            "vehicle_type": self.vehicle_type,
            "equipment": self.equipment
        }


# ============================================
# ABSTRACT DATA SOURCE
# ============================================

class DataSource(ABC):
    @abstractmethod
    def get_hospitals(self, lat: float, lng: float, radius_km: float = 10) -> List[Dict]:
        pass

    @abstractmethod
    def get_ambulances(self, lat: float, lng: float, radius_km: float = 10) -> List[Dict]:
        pass


# ============================================
# 1. STATIC DATA SOURCE (Default/Fallback)
# ============================================

class StaticDataSource(DataSource):
    """Static hardcoded data for demo/fallback purposes"""

    def __init__(self):
        self.hospitals = [
            Hospital("City Hospital A", 25, 22.722, 88.481, "123 Main St",
                     "+91-9876543210", "Emergency", "yes", 5, "Level II"),
            Hospital("General Hospital B", 50, 22.726, 88.490, "456 Central Ave",
                     "+91-9876543211", "Trauma", "yes", 10, "Level I"),
            Hospital("Medical Center C", 15, 22.718, 88.478, "789 Health Rd",
                     "+91-9876543212", "General", "yes", 3, "Level III"),
            Hospital("District Hospital D", 80, 22.730, 88.495, "321 District Rd",
                     "+91-9876543213", "Multi-specialty", "yes", 15, "Level I"),
            Hospital("Community Health Center E", 10, 22.715, 88.475, "654 Community Ln",
                     "+91-9876543214", "General", "yes", 2, "Level IV"),
        ]
        self.ambulances = [
            Ambulance("AMB-001", 22.720, 88.480, "available", "Raj Kumar", "ALS",
                     ["Defibrillator", "Ventilator", "Cardiac Monitor"]),
            Ambulance("AMB-002", 22.724, 88.487, "available", "Amit Singh", "BLS",
                     ["Oxygen", "Stretcher", "First Aid Kit"]),
            Ambulance("AMB-003", 22.718, 88.492, "available", "Priya Sharma", "ALS",
                     ["Defibrillator", "Ventilator", "Drug Kit"]),
            Ambulance("AMB-004", 22.728, 88.483, "available", "Deepak Roy", "BLS",
                     ["Oxygen", "Stretcher", "Splints"]),
            Ambulance("AMB-005", 22.716, 88.489, "en_route", "Sunita Devi", "ALS",
                     ["Defibrillator", "Ventilator", "Blood Products"]),
        ]

    def get_hospitals(self, lat: float, lng: float, radius_km: float = 10) -> List[Dict]:
        return [h.to_dict() for h in self.hospitals]

    def get_ambulances(self, lat: float, lng: float, radius_km: float = 10) -> List[Dict]:
        return [a.to_dict() for a in self.ambulances if a.status == "available"]


# ============================================
# 2. MONGODB DATA SOURCE
# ============================================

class MongoDBDataSource(DataSource):
    """MongoDB database connection for hospital/ambulance data"""

    def __init__(self, connection_string: str = None, db_name: str = None):
        self.connection_string = connection_string or Config.MONGODB_URI
        self.db_name = db_name or Config.MONGODB_DB
        self.client = None
        self.db = None
        self._connect()

    def _connect(self):
        try:
            from pymongo import MongoClient
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,  # 5s timeout
                connectTimeoutMS=5000
            )
            # Test connection
            self.client.server_info()
            self.db = self.client[self.db_name]
            print(f"[MongoDB] Connected to {self.db_name}")
        except ImportError:
            print("[MongoDB] pymongo not installed. Run: pip install pymongo")
        except Exception as e:
            print(f"[MongoDB] Connection error: {e}")
            self.client = None
            self.db = None

    def get_hospitals(self, lat: float, lng: float, radius_km: float = 10) -> List[Dict]:
        if not self.db:
            return []

        try:
            # Use geospatial query if 2dsphere index exists
            hospitals = list(self.db.hospitals.find({
                "location": {
                    "$nearSphere": {
                        "$geometry": {"type": "Point", "coordinates": [lng, lat]},
                        "$maxDistance": radius_km * 1000
                    }
                }
            }).limit(15))

            return [{
                "Hospital": h.get("name", "Unknown"),
                "Beds": h.get("beds", Config.DEFAULT_MEDIUM_HOSPITAL_BEDS),
                "lat": h.get("location", {}).get("coordinates", [0, 0])[1],
                "lng": h.get("location", {}).get("coordinates", [0, 0])[0],
                "address": h.get("address", ""),
                "phone": h.get("phone", ""),
                "specialty": h.get("specialty", "General"),
                "emergency": h.get("emergency", "yes"),
                "icu_beds": h.get("icu_beds", 0),
                "trauma_level": h.get("trauma_level", "")
            } for h in hospitals]
        except Exception as e:
            print(f"[MongoDB] Hospital query error: {e}")
            # Fallback to simple query
            try:
                hospitals = list(self.db.hospitals.find().limit(15))
                return [{
                    "Hospital": h.get("name", "Unknown"),
                    "Beds": h.get("beds", Config.DEFAULT_MEDIUM_HOSPITAL_BEDS),
                    "lat": h.get("lat", 0),
                    "lng": h.get("lng", 0),
                    "address": h.get("address", ""),
                    "phone": h.get("phone", ""),
                } for h in hospitals]
            except Exception as e2:
                print(f"[MongoDB] Fallback query also failed: {e2}")
                return []

    def get_ambulances(self, lat: float, lng: float, radius_km: float = 10) -> List[Dict]:
        if not self.db:
            return []

        try:
            ambulances = list(self.db.ambulances.find({"status": "available"}).limit(10))
            return [{
                "id": a.get("ambulance_id", str(a.get("_id", "UNKNOWN"))),
                "lat": a.get("lat", 0),
                "lng": a.get("lng", 0),
                "status": a.get("status", "unknown"),
                "driver": a.get("driver", ""),
                "vehicle_type": a.get("vehicle_type", "BLS")
            } for a in ambulances]
        except Exception as e:
            print(f"[MongoDB] Ambulance query error: {e}")
            return []

    def seed_sample_data(self):
        """Seed MongoDB with enhanced sample data"""
        if not self.db:
            return

        # Clear existing
        self.db.hospitals.delete_many({})
        self.db.ambulances.delete_many({})

        # Add hospitals with enhanced data
        hospitals = [
            {"name": "City Hospital A", "beds": 25, "icu_beds": 5,
             "lat": 22.722, "lng": 88.481,
             "location": {"type": "Point", "coordinates": [88.481, 22.722]},
             "address": "123 Main St", "phone": "+91-9876543210",
             "specialty": "Emergency", "trauma_level": "Level II"},
            {"name": "General Hospital B", "beds": 50, "icu_beds": 10,
             "lat": 22.726, "lng": 88.490,
             "location": {"type": "Point", "coordinates": [88.490, 22.726]},
             "address": "456 Central Ave", "phone": "+91-9876543211",
             "specialty": "Trauma", "trauma_level": "Level I"},
            {"name": "Medical Center C", "beds": 15, "icu_beds": 3,
             "lat": 22.718, "lng": 88.478,
             "location": {"type": "Point", "coordinates": [88.478, 22.718]},
             "address": "789 Health Rd", "phone": "+91-9876543212",
             "specialty": "General", "trauma_level": "Level III"},
        ]
        self.db.hospitals.insert_many(hospitals)

        # Create geospatial index
        try:
            self.db.hospitals.create_index([("location", "2dsphere")])
        except Exception:
            pass

        # Add ambulances with enhanced data
        ambulances = [
            {"ambulance_id": "AMB-001", "lat": 22.720, "lng": 88.480,
             "status": "available", "driver": "Raj Kumar", "vehicle_type": "ALS"},
            {"ambulance_id": "AMB-002", "lat": 22.724, "lng": 88.487,
             "status": "available", "driver": "Amit Singh", "vehicle_type": "BLS"},
            {"ambulance_id": "AMB-003", "lat": 22.718, "lng": 88.492,
             "status": "available", "driver": "Priya Sharma", "vehicle_type": "ALS"},
        ]
        self.db.ambulances.insert_many(ambulances)

        print("[MongoDB] Enhanced sample data seeded successfully!")


# ============================================
# 3. GOOGLE SHEETS DATA SOURCE
# ============================================

class GoogleSheetsDataSource(DataSource):
    """
    Google Sheets integration for easy data editing.
    Uses published CSV URL (no API key needed for public sheets)
    """

    def __init__(self, hospitals_sheet_url: str = None, ambulances_sheet_url: str = None):
        self.hospitals_url = hospitals_sheet_url or Config.GOOGLE_SHEETS_HOSPITALS_URL
        self.ambulances_url = ambulances_sheet_url or Config.GOOGLE_SHEETS_AMBULANCES_URL

    def _fetch_csv(self, url: str) -> List[Dict]:
        """Fetch and parse CSV from Google Sheets"""
        if not url:
            return []

        try:
            import csv
            from io import StringIO

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            reader = csv.DictReader(StringIO(response.text))
            return list(reader)
        except Exception as e:
            print(f"[Google Sheets] Error fetching data: {e}")
            return []

    def get_hospitals(self, lat: float, lng: float, radius_km: float = 10) -> List[Dict]:
        data = self._fetch_csv(self.hospitals_url)

        hospitals = []
        for row in data:
            try:
                hospitals.append({
                    "Hospital": row.get("name", row.get("Hospital", "Unknown")),
                    "Beds": int(row.get("beds", row.get("Beds", Config.DEFAULT_MEDIUM_HOSPITAL_BEDS))),
                    "lat": float(row.get("lat", row.get("latitude", 0))),
                    "lng": float(row.get("lng", row.get("longitude", 0))),
                    "address": row.get("address", ""),
                    "phone": row.get("phone", ""),
                    "specialty": row.get("specialty", "General"),
                    "emergency": row.get("emergency", "yes"),
                })
            except (ValueError, TypeError):
                continue

        return hospitals

    def get_ambulances(self, lat: float, lng: float, radius_km: float = 10) -> List[Dict]:
        data = self._fetch_csv(self.ambulances_url)

        ambulances = []
        for row in data:
            try:
                ambulances.append({
                    "id": row.get("id", row.get("ambulance_id", "UNKNOWN")),
                    "lat": float(row.get("lat", row.get("latitude", 0))),
                    "lng": float(row.get("lng", row.get("longitude", 0))),
                    "status": row.get("status", "available"),
                    "driver": row.get("driver", ""),
                    "vehicle_type": row.get("vehicle_type", "BLS")
                })
            except (ValueError, TypeError):
                continue

        return ambulances


# ============================================
# 4. OPENSTREETMAP / OVERPASS API DATA SOURCE
# ============================================

class OpenStreetMapDataSource(DataSource):
    """
    Fetch real hospital data from OpenStreetMap via Overpass API.
    Enhanced with caching, better retry logic, and realistic capacity estimation.
    """

    # Multi-node fallback for Overpass API
    OVERPASS_URLS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter"
    ]

    def _estimate_bed_capacity(self, tags: dict) -> int:
        """
        Estimate hospital bed capacity from OSM tags.
        Uses beds tag if available, otherwise estimates from hospital type.
        """
        # Direct beds tag
        beds_str = tags.get("beds", "")
        if beds_str:
            try:
                return max(1, int(beds_str))
            except (ValueError, TypeError):
                pass

        # Estimate from hospital type/name
        name = (tags.get("name", "") or "").lower()
        healthcare = tags.get("healthcare", "")
        building = tags.get("building", "")

        # Large indicators
        if any(kw in name for kw in ["medical college", "teaching", "university",
                                       "district", "government general", "super"]):
            return Config.DEFAULT_LARGE_HOSPITAL_BEDS

        # Medium indicators
        if any(kw in name for kw in ["general", "multi", "corporate", "private",
                                       "memorial", "city"]):
            return Config.DEFAULT_MEDIUM_HOSPITAL_BEDS

        # Small indicators (clinics, nursing homes)
        if any(kw in name for kw in ["clinic", "nursing", "maternity", "eye",
                                       "dental", "primary"]):
            return Config.DEFAULT_SMALL_HOSPITAL_BEDS

        if healthcare == "hospital":
            return Config.DEFAULT_MEDIUM_HOSPITAL_BEDS

        return Config.DEFAULT_SMALL_HOSPITAL_BEDS

    def get_hospitals(self, lat: float, lng: float, radius_km: float = 10) -> List[Dict]:
        """Query OpenStreetMap for hospitals near the location with caching and fallback"""

        # Check cache first
        cache_key = f"osm_hospitals_{lat:.3f}_{lng:.3f}_{radius_km}"
        cached = cache.get(cache_key)
        if cached is not None:
            print(f"[OSM] Cache HIT: {len(cached)} hospitals")
            return cached

        radius_m = int(radius_km * 1000)

        # Overpass QL query for hospitals
        query = f"""
        [out:json][timeout:{Config.OSM_TIMEOUT}];
        (
          node["amenity"="hospital"](around:{radius_m},{lat},{lng});
          way["amenity"="hospital"](around:{radius_m},{lat},{lng});
          relation["amenity"="hospital"](around:{radius_m},{lat},{lng});
        );
        out center;
        """

        last_error = ""
        for url in self.OVERPASS_URLS:
            try:
                print(f"[OSM] Trying Overpass API: {url}...")
                sys.stdout.flush()
                response = requests.post(
                    url,
                    data={"data": query},
                    timeout=Config.OSM_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()

                hospitals = []
                for element in data.get("elements", []):
                    tags = element.get("tags", {})

                    # Get coordinates (handle ways/relations with center)
                    if element["type"] == "node":
                        h_lat, h_lng = element["lat"], element["lon"]
                    else:
                        center = element.get("center", {})
                        h_lat = center.get("lat", lat)
                        h_lng = center.get("lon", lng)

                    # Estimate realistic bed capacity
                    estimated_beds = self._estimate_bed_capacity(tags)

                    hospitals.append({
                        "Hospital": tags.get("name", f"Hospital {element['id']}"),
                        "Beds": estimated_beds,
                        "lat": h_lat,
                        "lng": h_lng,
                        "address": tags.get("addr:full", tags.get("addr:street", "")),
                        "phone": tags.get("phone", tags.get("contact:phone", "")),
                        "emergency": tags.get("emergency", "yes"),
                        "specialty": tags.get("healthcare:speciality",
                                    tags.get("medical_system:western", "General")),
                        "osm_id": element["id"],
                        "operator": tags.get("operator", ""),
                    })

                print(f"[OSM] Success! Found {len(hospitals)} hospitals using {url}")
                sys.stdout.flush()

                # Cache the result
                if hospitals:
                    cache.set(cache_key, hospitals)

                return hospitals

            except Exception as e:
                last_error = str(e)
                print(f"[OSM] Error with {url}: {last_error}")
                sys.stdout.flush()
                continue  # Try next URL

        print(f"[OSM] All Overpass nodes failed. Last error: {last_error}")
        sys.stdout.flush()
        return []

    def get_ambulances(self, lat: float, lng: float, radius_km: float = 10) -> List[Dict]:
        """
        Query OSM for fire/ambulance stations and generate ambulance positions.
        Note: OSM doesn't have real-time ambulance tracking.
        """

        # Check cache first
        cache_key = f"osm_ambulances_{lat:.3f}_{lng:.3f}_{radius_km}"
        cached = cache.get(cache_key)
        if cached is not None:
            print(f"[OSM] Cache HIT: {len(cached)} ambulances")
            return cached

        radius_m = int(radius_km * 1000)

        # Query for fire/ambulance stations
        query = f"""
        [out:json][timeout:{Config.OSM_TIMEOUT}];
        (
          node["amenity"="fire_station"](around:{radius_m},{lat},{lng});
          node["emergency"="ambulance_station"](around:{radius_m},{lat},{lng});
          way["amenity"="fire_station"](around:{radius_m},{lat},{lng});
          way["emergency"="ambulance_station"](around:{radius_m},{lat},{lng});
        );
        out center;
        """

        # Try each Overpass URL (FIXED: was using undefined self.OVERPASS_URL)
        last_error = ""
        for url in self.OVERPASS_URLS:
            try:
                response = requests.post(
                    url,
                    data={"data": query},
                    timeout=Config.OSM_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()

                ambulances = []
                for i, element in enumerate(data.get("elements", [])[:8]):
                    # Handle way/relation center coordinates
                    if element["type"] == "node":
                        a_lat, a_lng = element["lat"], element["lon"]
                    else:
                        center = element.get("center", {})
                        a_lat = center.get("lat", lat)
                        a_lng = center.get("lon", lng)

                    station_name = element.get("tags", {}).get("name", "Station")
                    ambulances.append({
                        "id": f"AMB-{str(element['id'])[-4:]}",
                        "lat": a_lat,
                        "lng": a_lng,
                        "status": "available",
                        "station": station_name,
                        "vehicle_type": "ALS" if i % 2 == 0 else "BLS"
                    })

                if ambulances:
                    print(f"[OSM] Found {len(ambulances)} ambulance stations")
                    cache.set(cache_key, ambulances)
                    return ambulances

                break  # Query succeeded but no results, don't retry
            except Exception as e:
                last_error = str(e)
                print(f"[OSM] Ambulance query error with {url}: {last_error}")
                continue

        # If no stations found via any node, generate context-aware simulated ambulances
        print(f"[OSM] No ambulance stations found, generating dynamic simulated units")
        ambulances = self._generate_dynamic_ambulances(lat, lng)
        # Do NOT cache dynamic ambulances — they should vary each time
        return ambulances

    def _generate_dynamic_ambulances(self, lat: float, lng: float,
                                      disaster_type: str = "FIRE",
                                      severity: str = "HIGH",
                                      patient_count: int = 10) -> List[Dict]:
        """
        Generate context-aware ambulance fleet that varies on every call.
        Factors: disaster type, severity, patient count, time of day, randomness.
        """
        import math

        # Base count varies by severity
        severity_multiplier = {"CRITICAL": 1.5, "HIGH": 1.2, "MEDIUM": 1.0, "LOW": 0.7}
        base_count = max(3, int(patient_count * 0.4 * severity_multiplier.get(severity, 1.0)))
        # Add random jitter (±2)
        count = max(2, min(12, base_count + random.randint(-2, 2)))

        # Vehicle type distribution depends on disaster
        disaster_vehicle_profiles = {
            "FIRE":              {"ALS": 0.5, "BLS": 0.2, "Hazmat": 0.2, "Rescue": 0.1},
            "FLOOD":             {"ALS": 0.3, "BLS": 0.2, "Water Rescue": 0.4, "Hovercraft": 0.1},
            "EARTHQUAKE":        {"ALS": 0.4, "BLS": 0.2, "Heavy Rescue": 0.3, "K9 Unit": 0.1},
            "ACCIDENT":          {"ALS": 0.6, "BLS": 0.3, "Motorcycle": 0.1},
            "CHEMICAL_SPILL":    {"ALS": 0.3, "BLS": 0.1, "Hazmat": 0.5, "Decon Unit": 0.1},
            "BUILDING_COLLAPSE": {"ALS": 0.4, "BLS": 0.1, "Heavy Rescue": 0.3, "K9 Unit": 0.2},
        }
        profile = disaster_vehicle_profiles.get(disaster_type, {"ALS": 0.5, "BLS": 0.5})
        vehicle_types = list(profile.keys())
        vehicle_weights = list(profile.values())

        # Equipment pools by vehicle type
        equipment_map = {
            "ALS": ["Defibrillator", "Ventilator", "Cardiac Monitor", "IV Pump", "Drug Kit", "Intubation Kit"],
            "BLS": ["Oxygen", "Stretcher", "First Aid Kit", "Splints", "Bandages", "AED"],
            "Hazmat": ["SCBA", "Chemical Suits", "Decon Shower", "Gas Detector", "Neutralizer"],
            "Rescue": ["Jaws of Life", "Thermal Camera", "Rope Kit", "Cutting Tools"],
            "Heavy Rescue": ["Hydraulic Jack", "Concrete Saw", "Pneumatic Shoring", "Search Camera"],
            "Water Rescue": ["Life Jackets", "Inflatable Boat", "Throw Rope", "Dry Suits"],
            "Hovercraft": ["Rescue Sled", "Life Jackets", "Medical Kit"],
            "K9 Unit": ["Search Dog", "Handler Kit", "GPS Tracker", "Water Supply"],
            "Decon Unit": ["Decon Tent", "PPE Kits", "Shower System", "Waste Containers"],
            "Motorcycle": ["First Aid", "AED", "Radio", "Emergency Blanket"],
        }

        # Driver name pool (randomly select)
        driver_pool = [
            "Raj Kumar", "Amit Singh", "Priya Sharma", "Deepak Roy", "Sunita Devi",
            "Vikram Patel", "Anita Gupta", "Rahul Verma", "Meera Nair", "Arjun Das",
            "Kavita Joshi", "Suresh Iyer", "Pooja Reddy", "Manoj Tiwari", "Neha Kapoor",
            "Sanjay Mishra", "Ritu Agarwal", "Prakash Rao", "Divya Sen", "Karan Malhotra",
        ]
        random.shuffle(driver_pool)

        # Possible statuses (most should be available/dispatched)
        status_choices = ["available"] * 6 + ["dispatched"] * 3 + ["en_route"]

        ambulances = []
        for i in range(count):
            # Pick vehicle type based on disaster profile weights
            vtype = random.choices(vehicle_types, weights=vehicle_weights, k=1)[0]
            equip_pool = equipment_map.get(vtype, ["First Aid Kit"])
            # Random subset of equipment (3-5 items)
            equip = random.sample(equip_pool, k=min(len(equip_pool), random.randint(3, 5)))

            # Scatter positions around disaster location (0.5-3 km)
            angle = random.uniform(0, 2 * math.pi)
            dist_km = random.uniform(0.5, 3.0)
            d_lat = dist_km / 111.0 * math.cos(angle)
            d_lng = dist_km / (111.0 * math.cos(math.radians(lat))) * math.sin(angle)

            ambulances.append({
                "id": f"AMB-{random.randint(1000, 9999)}",
                "lat": round(lat + d_lat, 6),
                "lng": round(lng + d_lng, 6),
                "status": random.choice(status_choices),
                "driver": driver_pool[i % len(driver_pool)],
                "vehicle_type": vtype,
                "equipment": equip,
            })

        return ambulances


# ============================================
# DATA SOURCE MANAGER (Factory Pattern)
# ============================================

class DataSourceManager:
    """
    Manages multiple data sources with fallback logic.
    Priority: Requested -> Primary -> Static (always available)
    """

    def __init__(self):
        self.sources = {}
        self.primary_source = "static"

        # Always have static as fallback
        self.sources["static"] = StaticDataSource()

        # Try to initialize other sources
        self._init_mongodb()
        self._init_google_sheets()
        self._init_osm()

    def _init_mongodb(self):
        try:
            mongo = MongoDBDataSource()
            if mongo.db:
                self.sources["mongodb"] = mongo
                print("[DataManager] MongoDB source available")
        except Exception as e:
            print(f"[DataManager] MongoDB not available: {e}")

    def _init_google_sheets(self):
        sheets = GoogleSheetsDataSource()
        if sheets.hospitals_url or sheets.ambulances_url:
            self.sources["google_sheets"] = sheets
            print("[DataManager] Google Sheets source available")

    def _init_osm(self):
        self.sources["osm"] = OpenStreetMapDataSource()
        print("[DataManager] OpenStreetMap source available")

    def set_primary_source(self, source_name: str):
        """Set the primary data source"""
        if source_name in self.sources:
            self.primary_source = source_name
            print(f"[DataManager] Primary source set to: {source_name}")
        else:
            print(f"[DataManager] Unknown source: {source_name}, keeping: {self.primary_source}")

    def get_hospitals(self, lat: float, lng: float, radius_km: float = 10,
                      source: str = None) -> List[Dict]:
        """Get hospitals with fallback logic"""
        source = source or self.primary_source

        # Try requested source
        if source in self.sources:
            try:
                hospitals = self.sources[source].get_hospitals(lat, lng, radius_km)
                if hospitals:
                    return hospitals
                print(f"[DataManager] No hospitals from {source}, falling back to static")
            except Exception as e:
                print(f"[DataManager] Error from {source}: {e}, falling back to static")

        # Fallback to static
        return self.sources["static"].get_hospitals(lat, lng, radius_km)

    def get_ambulances(self, lat: float, lng: float, radius_km: float = 10,
                       source: str = None) -> List[Dict]:
        """Get ambulances with fallback logic"""
        source = source or self.primary_source

        if source in self.sources:
            try:
                ambulances = self.sources[source].get_ambulances(lat, lng, radius_km)
                if ambulances:
                    return ambulances
                print(f"[DataManager] No ambulances from {source}, falling back to static")
            except Exception as e:
                print(f"[DataManager] Error from {source}: {e}, falling back to static")

        return self.sources["static"].get_ambulances(lat, lng, radius_km)

    def list_sources(self) -> List[str]:
        """List available data sources"""
        return list(self.sources.keys())

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        return cache.get_stats()


# ============================================
# GLOBAL INSTANCE
# ============================================

data_manager = DataSourceManager()

# ============================================
# TEST / DEMO
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("Enhanced Data Source Manager Test")
    print("=" * 50)

    print(f"\nAvailable sources: {data_manager.list_sources()}")

    # Test location (Kolkata area)
    test_lat, test_lng = 22.721, 88.485

    print(f"\n--- Testing Static Source ---")
    data_manager.set_primary_source("static")
    hospitals = data_manager.get_hospitals(test_lat, test_lng)
    print(f"Hospitals: {len(hospitals)}")
    for h in hospitals:
        print(f"  {h['Hospital']}: {h['Beds']} beds, ICU: {h.get('icu_beds', 'N/A')}")

    ambulances = data_manager.get_ambulances(test_lat, test_lng)
    print(f"\nAmbulances: {len(ambulances)}")
    for a in ambulances:
        print(f"  {a['id']}: {a.get('vehicle_type', 'BLS')} - {a['status']}")

    print(f"\n--- Testing OpenStreetMap Source ---")
    data_manager.set_primary_source("osm")
    hospitals = data_manager.get_hospitals(test_lat, test_lng, radius_km=5)
    print(f"Found {len(hospitals)} hospitals from OSM")
    if hospitals:
        for h in hospitals[:3]:
            print(f"  {h['Hospital']}: {h['Beds']} beds (estimated)")

    print(f"\n--- Cache Stats ---")
    print(json.dumps(data_manager.get_cache_stats(), indent=2))
