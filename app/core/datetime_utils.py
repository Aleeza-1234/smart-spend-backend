"""
Datetime utilities for handling timezone-aware datetimes
"""

from datetime import datetime
from typing import Dict, Any

def convert_timezone_aware_datetimes(data: Dict[str, Any], datetime_fields: list = None) -> Dict[str, Any]:
    """
    Convert timezone-aware datetimes to naive UTC for PostgreSQL compatibility
    
    Args:
        data: Dictionary containing data with potential datetime fields
        datetime_fields: List of field names that contain datetimes. If None, checks common fields.
    
    Returns:
        Dictionary with converted datetime fields
    """
    if datetime_fields is None:
        # Common datetime field names
        datetime_fields = ['date', 'timestamp', 'target_date', 'created_at', 'updated_at']
    
    for field in datetime_fields:
        if data.get(field) and isinstance(data[field], datetime) and data[field].tzinfo:
            data[field] = data[field].replace(tzinfo=None)
    
    return data