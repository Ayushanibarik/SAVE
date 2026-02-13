"""
Input Validation for Disaster Response API
Validates and sanitizes all incoming request data
"""

from exceptions import ValidationError
from config import Config

VALID_DISASTER_TYPES = [
    "FIRE", "FLOOD", "EARTHQUAKE", "ACCIDENT",
    "CHEMICAL_SPILL", "BUILDING_COLLAPSE"
]

VALID_SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

VALID_DATA_SOURCES = ["static", "mongodb", "google_sheets", "osm"]


def validate_coordinates(lat, lng, field_prefix=""):
    """Validate latitude and longitude values"""
    prefix = f"{field_prefix}." if field_prefix else ""

    if lat is None or lng is None:
        raise ValidationError(
            f"Missing coordinates: {prefix}lat and {prefix}lng are required"
        )

    try:
        lat = float(lat)
        lng = float(lng)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Invalid coordinates: {prefix}lat and {prefix}lng must be numbers"
        )

    if not (-90 <= lat <= 90):
        raise ValidationError(
            f"Invalid latitude: {prefix}lat must be between -90 and 90, got {lat}"
        )

    if not (-180 <= lng <= 180):
        raise ValidationError(
            f"Invalid longitude: {prefix}lng must be between -180 and 180, got {lng}"
        )

    return lat, lng


def validate_patients(patients):
    """Validate patient count"""
    if patients is None:
        raise ValidationError("Missing required field: patients")

    try:
        patients = int(patients)
    except (ValueError, TypeError):
        raise ValidationError(f"Invalid patients value: must be an integer, got {type(patients).__name__}")

    if patients < 1:
        raise ValidationError(f"Invalid patients count: must be at least 1, got {patients}")

    if patients > Config.MAX_PATIENTS:
        raise ValidationError(
            f"Patient count {patients} exceeds maximum of {Config.MAX_PATIENTS}"
        )

    return patients


def validate_radius(radius_km):
    """Validate search radius"""
    if radius_km is None:
        return Config.DEFAULT_RADIUS_KM

    try:
        radius_km = float(radius_km)
    except (ValueError, TypeError):
        raise ValidationError(f"Invalid radius: must be a number, got {type(radius_km).__name__}")

    if radius_km <= 0:
        raise ValidationError(f"Invalid radius: must be positive, got {radius_km}")

    if radius_km > Config.MAX_RADIUS_KM:
        raise ValidationError(
            f"Radius {radius_km}km exceeds maximum of {Config.MAX_RADIUS_KM}km"
        )

    return radius_km


def validate_disaster_type(disaster_type):
    """Validate disaster type string"""
    if not disaster_type:
        return "FIRE"  # default

    disaster_type = disaster_type.upper().strip()

    if disaster_type not in VALID_DISASTER_TYPES:
        raise ValidationError(
            f"Invalid disaster type: '{disaster_type}'. "
            f"Valid types: {', '.join(VALID_DISASTER_TYPES)}"
        )

    return disaster_type


def validate_severity(severity):
    """Validate severity level"""
    if not severity:
        return "HIGH"  # default

    severity = severity.upper().strip()

    if severity not in VALID_SEVERITY_LEVELS:
        raise ValidationError(
            f"Invalid severity level: '{severity}'. "
            f"Valid levels: {', '.join(VALID_SEVERITY_LEVELS)}"
        )

    return severity


def validate_data_source(source):
    """Validate data source name"""
    if not source:
        return Config.DATA_SOURCE

    source = source.lower().strip()

    if source not in VALID_DATA_SOURCES:
        raise ValidationError(
            f"Invalid data source: '{source}'. "
            f"Valid sources: {', '.join(VALID_DATA_SOURCES)}"
        )

    return source


def validate_optimize_request(data):
    """Validate full optimization request"""
    if not data:
        raise ValidationError("Request body is required (JSON)")

    patients = validate_patients(data.get("patients"))
    lat, lng = validate_coordinates(data.get("lat"), data.get("lng"))
    disaster_type = validate_disaster_type(data.get("disaster_type", "FIRE"))
    source = validate_data_source(data.get("source"))

    return {
        "patients": patients,
        "lat": lat,
        "lng": lng,
        "disaster_type": disaster_type,
        "source": source,
        "hospitals": data.get("hospitals")
    }


def validate_medicine_request(data):
    """Validate medicine requirements request"""
    if not data:
        raise ValidationError("Request body is required (JSON)")

    patients = validate_patients(data.get("patients", 6))
    disaster_type = validate_disaster_type(data.get("disaster_type", "FIRE"))
    severity = validate_severity(data.get("severity", "HIGH"))

    return {
        "patients": patients,
        "disaster_type": disaster_type,
        "severity": severity
    }


def validate_ambulance_request(data):
    """Validate ambulance dispatch request"""
    if not data:
        raise ValidationError("Request body is required (JSON)")

    lat, lng = validate_coordinates(
        data.get("lat", 22.721),
        data.get("lng", 88.485)
    )
    source = validate_data_source(data.get("source"))

    return {
        "lat": lat,
        "lng": lng,
        "source": source
    }
