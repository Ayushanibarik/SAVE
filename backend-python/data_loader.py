"""
S.A.V.E. Disaster Datasets Loader — FEMA / WHO / NDMA

Simulates real-world disaster scenarios based on public datasets.
- FEMA (USA): Hurricane/Flood data (coastal impact zones)
- WHO (Global): Pandemic/Disease outbreak data (contagion spread)
- NDMA (India): Earthquake/Cyclone data (population density factors)
"""

import math
import random
import numpy as np
from datetime import datetime, timedelta

class DisasterDataLoader:
    
    def __init__(self):
        self.datasets = {
            "FEMA": self._generate_fema_hurricane,
            "WHO": self._generate_who_outbreak,
            "NDMA": self._generate_ndma_earthquake
        }

    def load_scenario(self, source="FEMA", center_lat=28.61, center_lng=77.23, radius_km=15):
        """
        Generates a realistic disaster scenario based on the source profile.
        Returns a list of patient objects with locations and severities.
        """
        generator = self.datasets.get(source, self._generate_fema_hurricane)
        return generator(center_lat, center_lng, radius_km)

    def _generate_fema_hurricane(self, lat, lng, radius_km):
        """
        Simulates FEMA Hurricane Impact Data.
        Characteristics:
        - High volume of injuries (trauma, drowning)
        - Clustered in low-lying areas (simulated by random clusters)
        - Severity skew: Moderate (ESI 3) to High (ESI 2)
        """
        patients = []
        num_clusters = random.randint(3, 6)
        
        for _ in range(num_clusters):
            # Create a localized impact zone
            cluster_lat = lat + random.uniform(-0.05, 0.05)
            cluster_lng = lng + random.uniform(-0.05, 0.05)
            cluster_size = random.randint(20, 50)
            
            for _ in range(cluster_size):
                p_lat = cluster_lat + random.gauss(0, 0.005)
                p_lng = cluster_lng + random.gauss(0, 0.005)
                
                # Severity distribution for Hurricane: 10% Critical, 30% High, 40% Moderate, 20% Low
                if random.random() < 0.1: severity = "CRITICAL" # ESI 1
                elif random.random() < 0.4: severity = "HIGH"   # ESI 2
                elif random.random() < 0.8: severity = "MEDIUM" # ESI 3
                else: severity = "LOW"                          # ESI 4/5
                
                patients.append({
                    "id": f"FEMA-{len(patients)+1}",
                    "lat": round(p_lat, 6),
                    "lng": round(p_lng, 6),
                    "severity": severity,
                    "condition": random.choice(["Trauma", "Drowning", "Debris Injury", "Hypothermia"]),
                    "age": random.randint(5, 90)
                })
        return patients

    def _generate_who_outbreak(self, lat, lng, radius_km):
        """
        Simulates WHO Disease Outbreak Data.
        Characteristics:
        - High volume of respiratory/fever cases
        - Diffuse spread (less clustering, more uniform)
        - Severity skew: High (ESI 2) to Critical (ESI 1) for severe pathogens
        """
        patients = []
        num_cases = random.randint(80, 150)
        
        for i in range(num_cases):
            # Diffuse spread
            p_lat = lat + random.uniform(-0.08, 0.08)
            p_lng = lng + random.uniform(-0.08, 0.08)
            
            # Severity for pandemic: 15% Critical, 40% High, 30% Medium, 15% Low
            r = random.random()
            if r < 0.15: severity = "CRITICAL"
            elif r < 0.55: severity = "HIGH"
            elif r < 0.85: severity = "MEDIUM"
            else: severity = "LOW"
            
            patients.append({
                "id": f"WHO-{i+1}",
                "lat": round(p_lat, 6),
                "lng": round(p_lng, 6),
                "severity": severity,
                "condition": random.choice(["Severe Respiratory", "High Fever", "Septic Shock", "Dehydration"]),
                "age": random.randint(1, 85)
            })
        return patients

    def _generate_ndma_earthquake(self, lat, lng, radius_km):
        """
        Simulates NDMA Earthquake Data.
        Characteristics:
        - Extremely high trauma volume (crush injuries, fractures)
        - Highly clustered around building collapse sites
        - Severity skew: Critical (ESI 1) and High (ESI 2) dominate
        """
        patients = []
        num_collapse_sites = random.randint(2, 4)
        
        for _ in range(num_collapse_sites):
            site_lat = lat + random.uniform(-0.03, 0.03)
            site_lng = lng + random.uniform(-0.03, 0.03)
            site_casualties = random.randint(30, 80)
            
            for _ in range(site_casualties):
                p_lat = site_lat + random.gauss(0, 0.002) # Tighter clusters
                p_lng = site_lng + random.gauss(0, 0.002)
                
                # Severity for Earthquake: 25% Critical, 40% High, 25% Medium, 10% Low
                r = random.random()
                if r < 0.25: severity = "CRITICAL"
                elif r < 0.65: severity = "HIGH"
                elif r < 0.90: severity = "MEDIUM"
                else: severity = "LOW"
                
                patients.append({
                    "id": f"NDMA-{len(patients)+1}",
                    "lat": round(p_lat, 6),
                    "lng": round(p_lng, 6),
                    "severity": severity,
                    "condition": random.choice(["Crush Injury", "Fracture", "Head Trauma", "Laceration"]),
                    "age": random.randint(1, 95)
                })
        return patients

# Global instance
data_loader = DisasterDataLoader()
