"""
Situation-Based Medicine & Medical Supply Requirements
Enhanced with ESI Triage System, Oxygen/Staffing Calculations,
and Clinical Decision Rationale
"""

from typing import Dict, List
from config import Config

# ============================================
# ESI (Emergency Severity Index) TRIAGE SYSTEM
# ============================================
# ESI 1: Immediate - Life-threatening (e.g., cardiac arrest, severe burns)
# ESI 2: Emergent  - High risk, confused/disoriented, severe pain
# ESI 3: Urgent    - Stable vitals but needs multiple resources
# ESI 4: Less Urgent - Needs one resource (e.g., X-ray, sutures)
# ESI 5: Non-Urgent  - No resources needed (e.g., minor cuts)

ESI_LEVELS = {
    1: {
        "name": "Immediate",
        "color": "red",
        "description": "Life-threatening, requires immediate intervention",
        "max_wait_minutes": 0,
        "staff_per_patient": 3.0,   # 1 physician + 2 nurses minimum
        "oxygen_liters_per_min": 15.0,  # High-flow O2
        "icu_required": True,
        "expected_los_hours": 72     # Length of stay
    },
    2: {
        "name": "Emergent",
        "color": "orange",
        "description": "High risk, severe pain, or altered mental status",
        "max_wait_minutes": 10,
        "staff_per_patient": 2.0,   # 1 physician + 1 nurse
        "oxygen_liters_per_min": 6.0,   # Moderate O2
        "icu_required": False,
        "expected_los_hours": 24
    },
    3: {
        "name": "Urgent",
        "color": "yellow",
        "description": "Stable vitals, needs multiple diagnostic resources",
        "max_wait_minutes": 30,
        "staff_per_patient": 1.0,   # 1 nurse + shared physician
        "oxygen_liters_per_min": 2.0,   # Low-flow if needed
        "icu_required": False,
        "expected_los_hours": 8
    },
    4: {
        "name": "Less Urgent",
        "color": "green",
        "description": "Needs one resource (e.g., X-ray, lab work, sutures)",
        "max_wait_minutes": 60,
        "staff_per_patient": 0.5,   # Shared nursing
        "oxygen_liters_per_min": 0.0,
        "icu_required": False,
        "expected_los_hours": 4
    },
    5: {
        "name": "Non-Urgent",
        "color": "blue",
        "description": "No resources needed, minor complaints",
        "max_wait_minutes": 120,
        "staff_per_patient": 0.25,
        "oxygen_liters_per_min": 0.0,
        "icu_required": False,
        "expected_los_hours": 1
    }
}


# ============================================
# ESI DISTRIBUTION BY DISASTER TYPE
# ============================================
# Defines expected patient distribution across ESI levels

DISASTER_ESI_DISTRIBUTION = {
    "FIRE": {
        1: 0.15,   # 15% critical (severe burns, smoke inhalation arrest)
        2: 0.25,   # 25% emergent (significant burns, respiratory distress)
        3: 0.30,   # 30% urgent (moderate burns, inhalation exposure)
        4: 0.20,   # 20% less urgent (minor burns, smoke exposure)
        5: 0.10    # 10% non-urgent (anxiety, minor irritation)
    },
    "FLOOD": {
        1: 0.05,
        2: 0.15,
        3: 0.30,
        4: 0.30,
        5: 0.20
    },
    "EARTHQUAKE": {
        1: 0.20,   # Crush injuries are frequently critical
        2: 0.25,
        3: 0.25,
        4: 0.20,
        5: 0.10
    },
    "ACCIDENT": {
        1: 0.10,
        2: 0.20,
        3: 0.30,
        4: 0.25,
        5: 0.15
    },
    "CHEMICAL_SPILL": {
        1: 0.25,   # Chemical exposure can be immediately life-threatening
        2: 0.30,
        3: 0.25,
        4: 0.15,
        5: 0.05
    },
    "BUILDING_COLLAPSE": {
        1: 0.20,
        2: 0.25,
        3: 0.25,
        4: 0.20,
        5: 0.10
    }
}

DEFAULT_ESI_DISTRIBUTION = {
    1: 0.10, 2: 0.20, 3: 0.30, 4: 0.25, 5: 0.15
}


# ============================================
# DISASTER TYPES & MEDICINE REQUIREMENTS
# ============================================

DISASTER_MEDICINE_MAP = {
    "FIRE": {
        "priority": "CRITICAL",
        "description": "Burn injuries, smoke inhalation, thermal trauma",
        "icd10_codes": ["T20-T32", "T58", "J68.0"],
        "medicines": [
            {"name": "Silver Sulfadiazine Cream (1%)", "quantity_per_patient": 2, "unit": "tubes (50g)", "priority": "HIGH",
             "clinical_note": "Topical antimicrobial for partial-thickness burns"},
            {"name": "Morphine Sulfate IV (10mg/mL)", "quantity_per_patient": 3, "unit": "vials", "priority": "HIGH",
             "clinical_note": "Titrate to pain score <4/10, monitor respiratory rate"},
            {"name": "Lactated Ringer's Solution", "quantity_per_patient": 5, "unit": "bags (1L)", "priority": "CRITICAL",
             "clinical_note": "Parkland formula: 4mL × kg × %TBSA burn over 24h"},
            {"name": "Oxygen Cylinders (Type D)", "quantity_per_patient": 1, "unit": "cylinder", "priority": "CRITICAL",
             "clinical_note": "100% O2 for suspected CO poisoning via non-rebreather"},
            {"name": "Albuterol Nebulizer Solution", "quantity_per_patient": 2, "unit": "vials (2.5mg)", "priority": "HIGH",
             "clinical_note": "For bronchospasm from inhalation injury"},
            {"name": "Burn Dressings (Hydrogel)", "quantity_per_patient": 10, "unit": "sheets", "priority": "HIGH",
             "clinical_note": "Non-adherent, cooling, for partial-thickness burns"},
            {"name": "Tetanus Toxoid (Td/Tdap)", "quantity_per_patient": 1, "unit": "dose (0.5mL IM)", "priority": "MEDIUM",
             "clinical_note": "If >5 years since last booster or unknown status"},
            {"name": "Ciprofloxacin 500mg", "quantity_per_patient": 14, "unit": "tablets", "priority": "HIGH",
             "clinical_note": "Prophylaxis for burn wound infection, 7-day course"},
        ],
        "equipment": [
            {"name": "Mechanical Ventilators", "quantity": 2, "priority": "CRITICAL",
             "clinical_note": "For patients with inhalation injury requiring intubation"},
            {"name": "Specialized Burn Beds (Clinitron)", "quantity": 5, "priority": "HIGH"},
            {"name": "Pulse Oximeters + CO-Oximetry", "quantity": 10, "priority": "HIGH",
             "clinical_note": "Standard SpO2 unreliable with CO poisoning"},
            {"name": "Laryngoscope Kit", "quantity": 3, "priority": "CRITICAL",
             "clinical_note": "Assess for airway edema in inhalation injury"},
        ]
    },

    "FLOOD": {
        "priority": "HIGH",
        "description": "Waterborne diseases, near-drowning, hypothermia, leptospirosis",
        "icd10_codes": ["T75.1", "A27", "A09", "T68"],
        "medicines": [
            {"name": "ORS Packets (WHO Formula)", "quantity_per_patient": 10, "unit": "packets", "priority": "CRITICAL",
             "clinical_note": "200mL per kg for moderate dehydration"},
            {"name": "Ciprofloxacin 500mg", "quantity_per_patient": 14, "unit": "tablets", "priority": "HIGH",
             "clinical_note": "Empiric for traveler's diarrhea / waterborne GI infection"},
            {"name": "Metronidazole 400mg", "quantity_per_patient": 21, "unit": "tablets", "priority": "HIGH",
             "clinical_note": "For suspected Giardia/amoebic dysentery, 7-day course"},
            {"name": "Chlorine Tablets (67mg NaDCC)", "quantity_per_patient": 50, "unit": "tablets", "priority": "MEDIUM",
             "clinical_note": "1 tablet per 20L for field water purification"},
            {"name": "Clotrimazole Cream 1%", "quantity_per_patient": 1, "unit": "tube (30g)", "priority": "MEDIUM",
             "clinical_note": "Fungal dermatitis from prolonged water exposure"},
            {"name": "Doxycycline 100mg", "quantity_per_patient": 14, "unit": "capsules", "priority": "HIGH",
             "clinical_note": "Leptospirosis prophylaxis/treatment, 7-day course"},
            {"name": "Normal Saline IV (0.9%)", "quantity_per_patient": 3, "unit": "bags (1L)", "priority": "HIGH",
             "clinical_note": "Volume resuscitation for dehydration/near-drowning"},
            {"name": "Loperamide 2mg", "quantity_per_patient": 10, "unit": "tablets", "priority": "HIGH",
             "clinical_note": "Symptomatic relief, not for bloody diarrhea"},
        ],
        "equipment": [
            {"name": "Warming Blankets (Bair Hugger)", "quantity": 20, "priority": "HIGH",
             "clinical_note": "Active rewarming for hypothermia (target 36°C core)"},
            {"name": "Portable Water Purification Units", "quantity": 2, "priority": "MEDIUM"},
            {"name": "Rectal Thermometers", "quantity": 10, "priority": "HIGH",
             "clinical_note": "Accurate core temp for hypothermia staging"},
        ]
    },

    "EARTHQUAKE": {
        "priority": "CRITICAL",
        "description": "Crush syndrome, fractures, rhabdomyolysis, dust inhalation",
        "icd10_codes": ["T79.5", "T14.2", "M62.89", "T79.6"],
        "medicines": [
            {"name": "Morphine Sulfate IV (10mg/mL)", "quantity_per_patient": 5, "unit": "vials", "priority": "CRITICAL",
             "clinical_note": "Administer BEFORE release of crush injury to prevent reperfusion shock"},
            {"name": "Packed Red Blood Cells (O-neg)", "quantity_per_patient": 2, "unit": "units (300mL)", "priority": "CRITICAL",
             "clinical_note": "For hemorrhagic shock, target Hb >7g/dL"},
            {"name": "Lactated Ringer's Solution", "quantity_per_patient": 6, "unit": "bags (1L)", "priority": "CRITICAL",
             "clinical_note": "1.5L/hr IV before extrication for crush injury"},
            {"name": "Mannitol 20%", "quantity_per_patient": 4, "unit": "bottles (250mL)", "priority": "CRITICAL",
             "clinical_note": "Osmotic diuretic for crush syndrome, 1g/kg over 20min"},
            {"name": "Tetanus Immunoglobulin (TIG)", "quantity_per_patient": 1, "unit": "dose (250 IU IM)", "priority": "HIGH",
             "clinical_note": "For contaminated wounds with unknown vaccination status"},
            {"name": "Ceftriaxone 1g IV", "quantity_per_patient": 7, "unit": "vials", "priority": "HIGH",
             "clinical_note": "Broad-spectrum prophylaxis for open fractures/wounds"},
            {"name": "Splints & Fiberglass Casts", "quantity_per_patient": 2, "unit": "sets", "priority": "HIGH"},
            {"name": "Surgical Sutures (3-0 Nylon)", "quantity_per_patient": 5, "unit": "packets", "priority": "HIGH"},
        ],
        "equipment": [
            {"name": "Portable X-Ray (C-Arm)", "quantity": 2, "priority": "CRITICAL"},
            {"name": "Minor Surgery Kits", "quantity": 10, "priority": "CRITICAL"},
            {"name": "Continuous Renal Replacement (CRRT)", "quantity": 2, "priority": "HIGH",
             "clinical_note": "For rhabdomyolysis-induced acute kidney injury"},
            {"name": "Blood Bank Refrigerator", "quantity": 1, "priority": "CRITICAL"},
        ]
    },

    "ACCIDENT": {
        "priority": "HIGH",
        "description": "Polytrauma, hemorrhage, TBI, spinal injuries",
        "icd10_codes": ["S06", "T07", "S22", "S32"],
        "medicines": [
            {"name": "Packed Red Blood Cells (O-neg)", "quantity_per_patient": 3, "unit": "units (300mL)", "priority": "CRITICAL",
             "clinical_note": "Massive transfusion protocol: 1:1:1 ratio with FFP, platelets"},
            {"name": "Tranexamic Acid (TXA) 1g IV", "quantity_per_patient": 2, "unit": "vials", "priority": "CRITICAL",
             "clinical_note": "Within 3 hours of injury for hemorrhage control (CRASH-2 trial)"},
            {"name": "Fentanyl IV (50mcg/mL)", "quantity_per_patient": 4, "unit": "vials (2mL)", "priority": "HIGH",
             "clinical_note": "Short-acting analgesic, titrate in 25-50mcg increments"},
            {"name": "Normal Saline IV (0.9%)", "quantity_per_patient": 4, "unit": "bags (1L)", "priority": "HIGH",
             "clinical_note": "Balanced resuscitation, avoid >2L crystalloid (permissive hypotension)"},
            {"name": "Tetanus Toxoid (Td/Tdap)", "quantity_per_patient": 1, "unit": "dose (0.5mL IM)", "priority": "MEDIUM"},
            {"name": "Chromic Sutures (2-0)", "quantity_per_patient": 3, "unit": "packets", "priority": "HIGH"},
            {"name": "Amoxicillin-Clavulanate 625mg", "quantity_per_patient": 14, "unit": "tablets", "priority": "MEDIUM",
             "clinical_note": "Prophylaxis for contaminated wounds, 7-day course"},
            {"name": "Ibuprofen 400mg", "quantity_per_patient": 20, "unit": "tablets", "priority": "MEDIUM",
             "clinical_note": "Anti-inflammatory, avoid in suspected internal bleeding"},
        ],
        "equipment": [
            {"name": "CT Scanner Access (Priority)", "quantity": 1, "priority": "CRITICAL",
             "clinical_note": "Pan-scan for polytrauma (head, C-spine, chest, abdomen, pelvis)"},
            {"name": "Operating Theater (Trauma)", "quantity": 2, "priority": "CRITICAL"},
            {"name": "Cervical Collars (Philadelphia)", "quantity": 10, "priority": "HIGH"},
            {"name": "Spine Boards + Head Blocks", "quantity": 5, "priority": "HIGH"},
            {"name": "FAST Ultrasound", "quantity": 2, "priority": "HIGH",
             "clinical_note": "Focused Assessment with Sonography for Trauma"},
        ]
    },

    "CHEMICAL_SPILL": {
        "priority": "CRITICAL",
        "description": "Chemical burns, organophosphate poisoning, inhalation injury",
        "icd10_codes": ["T54", "T60.0", "T59", "J68.0"],
        "medicines": [
            {"name": "Atropine Sulfate 1mg/mL IV", "quantity_per_patient": 10, "unit": "vials", "priority": "CRITICAL",
             "clinical_note": "For organophosphate/nerve agent: 2mg IV q5min until secretions dry"},
            {"name": "Pralidoxime (2-PAM) 1g IV", "quantity_per_patient": 5, "unit": "vials", "priority": "CRITICAL",
             "clinical_note": "Within 36h of organophosphate exposure, reactivates AChE"},
            {"name": "Activated Charcoal (50g)", "quantity_per_patient": 2, "unit": "bottles", "priority": "HIGH",
             "clinical_note": "Only if ingestion <1h and airway protected, NOT for caustics"},
            {"name": "Calcium Gluconate 10% IV", "quantity_per_patient": 4, "unit": "vials (10mL)", "priority": "HIGH",
             "clinical_note": "For hydrofluoric acid burns: topical gel + IV for systemic toxicity"},
            {"name": "Sodium Bicarbonate 8.4%", "quantity_per_patient": 3, "unit": "vials (50mL)", "priority": "MEDIUM",
             "clinical_note": "Urinary alkalinization for certain chemical exposures"},
            {"name": "Oxygen (High-Flow)", "quantity_per_patient": 2, "unit": "cylinders", "priority": "CRITICAL",
             "clinical_note": "15L/min via non-rebreather for all chemical inhalation"},
            {"name": "Albuterol Nebulizer", "quantity_per_patient": 2, "unit": "vials (2.5mg)", "priority": "HIGH"},
            {"name": "Sterile Eye Wash (500mL)", "quantity_per_patient": 2, "unit": "bottles", "priority": "HIGH",
             "clinical_note": "Continuous irrigation for 15-20 min for chemical eye exposure"},
        ],
        "equipment": [
            {"name": "Decontamination Showers", "quantity": 2, "priority": "CRITICAL",
             "clinical_note": "Decon BEFORE entering ED; remove all clothing"},
            {"name": "Level C PPE (Staff)", "quantity": 20, "priority": "CRITICAL"},
            {"name": "Mechanical Ventilators", "quantity": 5, "priority": "CRITICAL"},
            {"name": "Poison Control Hotline Access", "quantity": 1, "priority": "HIGH"},
        ]
    },

    "BUILDING_COLLAPSE": {
        "priority": "CRITICAL",
        "description": "Crush injuries, compartment syndrome, asphyxiation",
        "icd10_codes": ["W20", "T79.5", "T79.6", "T71"],
        "medicines": [
            {"name": "Morphine Sulfate IV (10mg/mL)", "quantity_per_patient": 5, "unit": "vials", "priority": "CRITICAL",
             "clinical_note": "Pre-extrication analgesia critical for crush injury management"},
            {"name": "Packed Red Blood Cells (O-neg)", "quantity_per_patient": 3, "unit": "units (300mL)", "priority": "CRITICAL"},
            {"name": "Lactated Ringer's Solution", "quantity_per_patient": 5, "unit": "bags (1L)", "priority": "CRITICAL",
             "clinical_note": "Aggressive fluid resuscitation before and during extrication"},
            {"name": "Mannitol 20%", "quantity_per_patient": 4, "unit": "bottles (250mL)", "priority": "CRITICAL",
             "clinical_note": "Renal protection from myoglobin-induced AKI"},
            {"name": "Tetanus Immunoglobulin (TIG)", "quantity_per_patient": 1, "unit": "dose (250 IU IM)", "priority": "HIGH"},
            {"name": "Ceftriaxone 1g IV", "quantity_per_patient": 7, "unit": "vials", "priority": "HIGH",
             "clinical_note": "Broad-spectrum coverage for contaminated crush wounds"},
        ],
        "equipment": [
            {"name": "Minor Surgery Kits", "quantity": 10, "priority": "CRITICAL"},
            {"name": "Portable X-Ray (C-Arm)", "quantity": 2, "priority": "HIGH"},
            {"name": "Fasciotomy Kit", "quantity": 3, "priority": "CRITICAL",
             "clinical_note": "For emergent compartment syndrome release"},
        ]
    }
}

# Default/Generic disaster requirements
DEFAULT_REQUIREMENTS = {
    "priority": "HIGH",
    "description": "General mass casualty incident",
    "icd10_codes": ["T07"],
    "medicines": [
        {"name": "Normal Saline IV (0.9%)", "quantity_per_patient": 3, "unit": "bags (1L)", "priority": "HIGH",
         "clinical_note": "Initial volume resuscitation"},
        {"name": "Morphine Sulfate IV", "quantity_per_patient": 3, "unit": "vials", "priority": "HIGH",
         "clinical_note": "Pain management, monitor RR and sedation level"},
        {"name": "Sterile Gauze & Dressings", "quantity_per_patient": 5, "unit": "packets", "priority": "MEDIUM"},
        {"name": "Amoxicillin-Clavulanate 625mg", "quantity_per_patient": 14, "unit": "tablets", "priority": "MEDIUM",
         "clinical_note": "Wound infection prophylaxis"},
        {"name": "Tetanus Toxoid (Td/Tdap)", "quantity_per_patient": 1, "unit": "dose (0.5mL IM)", "priority": "MEDIUM"},
    ],
    "equipment": [
        {"name": "Trauma First Aid Kits", "quantity": 10, "priority": "HIGH"},
        {"name": "Folding Stretchers", "quantity": 5, "priority": "HIGH"},
    ]
}


# ============================================
# ESI TRIAGE CALCULATIONS
# ============================================

def calculate_esi_distribution(disaster_type: str, patient_count: int) -> Dict:
    """
    Calculate expected ESI triage distribution for a disaster.
    
    Returns patient counts per ESI level with clinical details.
    """
    disaster_type = disaster_type.upper()
    distribution = DISASTER_ESI_DISTRIBUTION.get(disaster_type, DEFAULT_ESI_DISTRIBUTION)

    esi_breakdown = {}
    assigned_total = 0

    for level in sorted(distribution.keys()):
        if level == max(distribution.keys()):
            # Last level gets remainder to ensure total matches
            count = patient_count - assigned_total
        else:
            count = max(1, round(distribution[level] * patient_count))
            if assigned_total + count > patient_count:
                count = patient_count - assigned_total

        assigned_total += count
        level_info = ESI_LEVELS[level]

        esi_breakdown[f"ESI-{level}"] = {
            "level": level,
            "name": level_info["name"],
            "color": level_info["color"],
            "patient_count": max(0, count),
            "percentage": round(distribution[level] * 100, 1),
            "max_wait_minutes": level_info["max_wait_minutes"],
            "description": level_info["description"],
            "icu_required": level_info["icu_required"]
        }

    return {
        "total_patients": patient_count,
        "disaster_type": disaster_type,
        "distribution": esi_breakdown,
        "icu_patients": sum(
            v["patient_count"] for v in esi_breakdown.values()
            if v["icu_required"]
        ),
        "critical_patients": sum(
            v["patient_count"] for k, v in esi_breakdown.items()
            if v["level"] <= 2
        )
    }


def get_oxygen_requirements(esi_distribution: Dict) -> Dict:
    """
    Calculate oxygen requirements based on ESI distribution.
    
    Returns O2 needs in liters per minute and cylinder estimates.
    """
    total_o2_lpm = 0.0
    breakdown = []

    for key, data in esi_distribution.get("distribution", {}).items():
        level = data["level"]
        count = data["patient_count"]
        level_info = ESI_LEVELS[level]
        o2_per_patient = level_info["oxygen_liters_per_min"]
        level_total = o2_per_patient * count

        total_o2_lpm += level_total

        if o2_per_patient > 0:
            breakdown.append({
                "esi_level": key,
                "patients": count,
                "o2_per_patient_lpm": o2_per_patient,
                "total_lpm": round(level_total, 1),
                "delivery_method": "Non-rebreather mask" if o2_per_patient >= 10
                    else "Simple face mask" if o2_per_patient >= 5
                    else "Nasal cannula"
            })

    # Type D cylinder = 415L, lasts ~28 min at 15LPM
    cylinders_per_hour = (total_o2_lpm * 60) / 415 if total_o2_lpm > 0 else 0
    # For 24h supply
    cylinders_24h = int(cylinders_per_hour * 24) + 1

    return {
        "total_o2_lpm": round(total_o2_lpm, 1),
        "cylinders_per_hour": round(cylinders_per_hour, 1),
        "cylinders_24h_supply": cylinders_24h,
        "breakdown": breakdown,
        "clinical_note": (
            f"Total continuous O2 demand: {round(total_o2_lpm, 1)} L/min. "
            f"Recommend {cylinders_24h} Type D cylinders for 24h supply. "
            "Consider piped O2 if available."
        )
    }


def get_staffing_requirements(patient_count: int, esi_distribution: Dict) -> Dict:
    """
    Calculate medical staffing needs based on ESI distribution.
    """
    total_staff_needed = 0.0
    staffing_breakdown = []

    for key, data in esi_distribution.get("distribution", {}).items():
        level = data["level"]
        count = data["patient_count"]
        level_info = ESI_LEVELS[level]
        staff_ratio = level_info["staff_per_patient"]
        staff_needed = staff_ratio * count
        total_staff_needed += staff_needed

        staffing_breakdown.append({
            "esi_level": key,
            "patients": count,
            "staff_ratio": f"1:{round(1 / staff_ratio, 1)}" if staff_ratio > 0
                else "minimal",
            "staff_needed": round(staff_needed, 1)
        })

    # Role breakdown (approximate)
    physicians = max(1, int(total_staff_needed * 0.3))
    nurses = max(2, int(total_staff_needed * 0.5))
    paramedics = max(1, int(total_staff_needed * 0.2))

    return {
        "total_staff_needed": round(total_staff_needed),
        "physicians": physicians,
        "nurses": nurses,
        "paramedics": paramedics,
        "breakdown": staffing_breakdown,
        "additional_roles": [
            {"role": "Triage Officer", "count": 1,
             "note": "Experienced EP to manage START/ESI triage at scene"},
            {"role": "Trauma Surgeon", "count": max(1, int(patient_count * 0.1)),
             "note": "On-call for ESI-1 and ESI-2 patients requiring OR"},
            {"role": "Anesthesiologist", "count": max(1, int(patient_count * 0.05)),
             "note": "Airway management and sedation for procedures"},
            {"role": "Pharmacist", "count": 1,
             "note": "Medication verification and antidote preparation"},
            {"role": "Social Worker", "count": 1,
             "note": "Family notification and psychological first aid"},
        ],
        "clinical_note": (
            f"Minimum {round(total_staff_needed)} clinical staff required. "
            f"Call in off-duty staff if current roster < {round(total_staff_needed)}. "
            "Activate mutual aid if staffing gap >30%."
        )
    }


def get_clinical_rationale(disaster_type: str, allocation: list) -> Dict:
    """
    Generate clinical decision rationale for hospital allocation.
    Explains why each hospital was chosen and any concerns.
    """
    disaster_type = disaster_type.upper()
    rationale = {
        "disaster_type": disaster_type,
        "decision_factors": [
            "Geographic proximity (golden hour compliance)",
            "Available bed capacity vs. patient volume",
            "Hospital specialty capabilities match",
            "ESI acuity distribution alignment",
        ],
        "allocation_rationale": [],
        "clinical_concerns": [],
        "recommendations": []
    }

    total_assigned = 0
    total_patients = sum(a.get("assigned", 0) for a in allocation)

    for alloc in allocation:
        hospital = alloc.get("hospital", "Unknown")
        assigned = alloc.get("assigned", 0)
        distance = alloc.get("distance", 0)
        beds = alloc.get("available_beds", 0)
        total_assigned += assigned

        # Distance-based assessment
        if distance <= 3:
            transport_risk = "LOW"
            eta_note = "Within golden hour for critical patients"
        elif distance <= 10:
            transport_risk = "MODERATE"
            eta_note = "Acceptable for ESI 2-5, marginal for ESI-1"
        else:
            transport_risk = "HIGH"
            eta_note = "Consider helicopter transport for ESI-1 patients"

        # Capacity assessment
        utilization = (assigned / beds * 100) if beds > 0 else 100
        if utilization > 90:
            capacity_note = "NEAR CAPACITY — may need to divert overflow"
        elif utilization > 70:
            capacity_note = "Moderate load — manageable with current staff"
        else:
            capacity_note = "Adequate capacity for surge"

        rationale["allocation_rationale"].append({
            "hospital": hospital,
            "patients_assigned": assigned,
            "distance_km": distance,
            "transport_risk": transport_risk,
            "eta_note": eta_note,
            "capacity_utilization_pct": round(utilization, 1),
            "capacity_note": capacity_note
        })

    # Clinical concerns
    if total_assigned < total_patients:
        unallocated = total_patients - total_assigned
        rationale["clinical_concerns"].append(
            f"CRITICAL: {unallocated} patients remain unallocated. "
            "Activate regional mutual aid and consider field hospital deployment."
        )

    if any(a.get("distance", 0) > 10 for a in allocation if a.get("assigned", 0) > 0):
        rationale["clinical_concerns"].append(
            "Some hospitals exceed 10km — ESI-1 patients may exceed golden hour. "
            "Request aeromedical transport for critical cases."
        )

    # Recommendations
    rationale["recommendations"] = [
        "Activate Hospital Incident Command System (HICS) at all receiving facilities",
        "Establish unified communication channel between all receiving hospitals",
        f"Pre-alert blood banks: anticipate {total_patients * 2} units O-negative",
        "Deploy mobile triage unit to disaster site if not already present",
    ]

    return rationale


# ============================================
# MAIN MEDICINE REQUIREMENTS FUNCTION
# ============================================

def get_medicine_requirements(disaster_type: str, patient_count: int) -> Dict:
    """
    Get medicine requirements based on disaster type and patient count.
    Enhanced with ESI triage, oxygen, and staffing data.
    """
    disaster_type = disaster_type.upper()

    # Get requirements for disaster type or use default
    requirements = DISASTER_MEDICINE_MAP.get(disaster_type, DEFAULT_REQUIREMENTS)

    # Scale medicines by patient count
    scaled_medicines = []
    for med in requirements["medicines"]:
        scaled_medicines.append({
            "name": med["name"],
            "required_quantity": med["quantity_per_patient"] * patient_count,
            "unit": med["unit"],
            "priority": med["priority"],
            "per_patient": med["quantity_per_patient"],
            "clinical_note": med.get("clinical_note", "")
        })

    # Equipment (base requirements, scale some by patient count)
    equipment = []
    for eq in requirements.get("equipment", []):
        equipment.append({
            "name": eq["name"],
            "quantity": eq["quantity"],
            "priority": eq["priority"],
            "clinical_note": eq.get("clinical_note", "")
        })

    # ESI Triage calculations
    esi_data = calculate_esi_distribution(disaster_type, patient_count)
    oxygen_data = get_oxygen_requirements(esi_data)
    staffing_data = get_staffing_requirements(patient_count, esi_data)

    return {
        "disaster_type": disaster_type,
        "priority": requirements["priority"],
        "description": requirements["description"],
        "icd10_codes": requirements.get("icd10_codes", []),
        "patient_count": patient_count,
        "medicines": scaled_medicines,
        "equipment": equipment,
        "critical_items": [m for m in scaled_medicines if m["priority"] == "CRITICAL"],
        "high_priority_items": [m for m in scaled_medicines if m["priority"] == "HIGH"],
        "esi_triage": esi_data,
        "oxygen_requirements": oxygen_data,
        "staffing_requirements": staffing_data
    }


def get_hospital_preparation_checklist(disaster_type: str, patient_count: int, hospital_name: str) -> Dict:
    """
    Generate a comprehensive preparation checklist for a specific hospital.
    Now includes ESI-based staffing and resource preparation.
    """
    requirements = get_medicine_requirements(disaster_type, patient_count)
    esi_data = requirements["esi_triage"]

    checklist = {
        "hospital": hospital_name,
        "alert_level": requirements["priority"],
        "expected_patients": patient_count,
        "disaster_type": disaster_type,
        "esi_summary": {
            "icu_patients": esi_data["icu_patients"],
            "critical_patients": esi_data["critical_patients"],
            "distribution": esi_data["distribution"],
        },
        "immediate_actions": [],
        "medicine_checklist": [],
        "equipment_checklist": [],
        "staffing": requirements["staffing_requirements"],
        "oxygen": requirements["oxygen_requirements"]
    }

    # Immediate actions based on priority and ESI
    if requirements["priority"] == "CRITICAL":
        checklist["immediate_actions"] = [
            "🔴 ACTIVATE Hospital Incident Command System (HICS)",
            "📞 Call in ALL off-duty emergency and surgical staff",
            "🏥 Clear non-critical patients from Emergency Department",
            "🩸 Alert blood bank: prepare O-negative units for MTP",
            "🔪 Prepare operating theaters for emergency surgery",
            "🏷️ Set up triage area with ESI color-coded zones",
            f"🛏️ Prepare {esi_data['icu_patients']} ICU beds for ESI-1 patients",
            f"💨 Ensure {requirements['oxygen_requirements']['cylinders_24h_supply']} oxygen cylinders available",
            "📋 Brief all staff on disaster-specific protocols",
            "🚨 Notify regional trauma center of expected surge",
        ]
    elif requirements["priority"] == "HIGH":
        checklist["immediate_actions"] = [
            "🟡 ACTIVATE Emergency Response Protocol (Level 2)",
            "📞 Notify on-call staff, consider calling additional staff",
            "🏥 Prepare Emergency Department for patient surge",
            "💊 Stock up on required medicines from pharmacy",
            "🏷️ Set up triage area",
            "📋 Brief medical staff on incoming patients",
            f"💨 Ensure {requirements['oxygen_requirements']['cylinders_24h_supply']} oxygen cylinders available",
        ]
    else:
        checklist["immediate_actions"] = [
            "🟢 STANDBY: Monitor situation for escalation",
            "📞 Ensure on-call staff are available",
            "💊 Verify medication stock levels",
            "📋 Brief ED charge nurse",
        ]

    # Medicine checklist with status
    for med in requirements["medicines"]:
        checklist["medicine_checklist"].append({
            "item": med["name"],
            "required": f"{med['required_quantity']} {med['unit']}",
            "priority": med["priority"],
            "clinical_note": med.get("clinical_note", ""),
            "status": "PENDING"
        })

    # Equipment checklist
    for eq in requirements["equipment"]:
        checklist["equipment_checklist"].append({
            "item": eq["name"],
            "required": eq["quantity"],
            "priority": eq["priority"],
            "clinical_note": eq.get("clinical_note", ""),
            "status": "PENDING"
        })

    return checklist


def get_all_disaster_types() -> List[str]:
    """Return list of all supported disaster types."""
    return list(DISASTER_MEDICINE_MAP.keys())


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Enhanced Medicine Requirements + ESI Triage Test")
    print("=" * 60)

    # Test EARTHQUAKE with 15 patients
    result = get_medicine_requirements("EARTHQUAKE", 15)
    print(f"\n📋 Disaster: EARTHQUAKE | Patients: 15")
    print(f"   Priority: {result['priority']}")

    print(f"\n🏷️ ESI Triage Distribution:")
    for level, data in result["esi_triage"]["distribution"].items():
        print(f"   {level} ({data['name']}): {data['patient_count']} patients ({data['percentage']}%)")

    print(f"\n   ICU Required: {result['esi_triage']['icu_patients']} patients")
    print(f"   Critical (ESI 1-2): {result['esi_triage']['critical_patients']} patients")

    print(f"\n💨 Oxygen Requirements:")
    print(f"   Total O2: {result['oxygen_requirements']['total_o2_lpm']} L/min")
    print(f"   24h Supply: {result['oxygen_requirements']['cylinders_24h_supply']} cylinders")

    print(f"\n👩‍⚕️ Staffing Requirements:")
    print(f"   Total Staff: {result['staffing_requirements']['total_staff_needed']}")
    print(f"   Physicians: {result['staffing_requirements']['physicians']}")
    print(f"   Nurses: {result['staffing_requirements']['nurses']}")
    print(f"   Paramedics: {result['staffing_requirements']['paramedics']}")

    print(f"\n💊 Critical Medicines:")
    for item in result["critical_items"]:
        print(f"   ⚠️ {item['name']}: {item['required_quantity']} {item['unit']}")
        if item.get("clinical_note"):
            print(f"      Note: {item['clinical_note']}")

    print("\n" + "=" * 60)

    # Test clinical rationale
    sample_allocation = [
        {"hospital": "City Hospital A", "assigned": 5, "distance": 2.3, "available_beds": 8},
        {"hospital": "General Hospital B", "assigned": 7, "distance": 5.1, "available_beds": 15},
        {"hospital": "Medical Center C", "assigned": 3, "distance": 12.0, "available_beds": 5},
    ]
    rationale = get_clinical_rationale("EARTHQUAKE", sample_allocation)
    print("\n🧠 Clinical Rationale:")
    for r in rationale["allocation_rationale"]:
        print(f"   {r['hospital']}: {r['patients_assigned']}pts, {r['distance_km']}km, "
              f"Transport Risk: {r['transport_risk']}, Load: {r['capacity_utilization_pct']}%")

    if rationale["clinical_concerns"]:
        print(f"\n⚠️ Concerns:")
        for c in rationale["clinical_concerns"]:
            print(f"   - {c}")
