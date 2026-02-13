"""
Custom Exception Classes for Disaster Response System
Provides structured error handling with proper HTTP status codes
"""


class DisasterResponseError(Exception):
    """Base exception for all disaster response errors"""
    status_code = 500

    def __init__(self, message, status_code=None, details=None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}

    def to_dict(self):
        return {
            "error": self.message,
            "status_code": self.status_code,
            "details": self.details,
            "error_type": self.__class__.__name__
        }


class ValidationError(DisasterResponseError):
    """Input validation failed"""
    status_code = 400


class DataSourceError(DisasterResponseError):
    """Data source unavailable or returned invalid data"""
    status_code = 503


class OptimizationError(DisasterResponseError):
    """Optimization algorithm failed"""
    status_code = 500


class CacheError(DisasterResponseError):
    """Cache operation failed (non-critical)"""
    status_code = 500


class RateLimitError(DisasterResponseError):
    """Too many requests"""
    status_code = 429
