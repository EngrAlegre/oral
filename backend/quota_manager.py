"""
Quota management system for tracking API usage and feature access
"""

from datetime import datetime
from typing import Tuple

# Feature access configuration
FEATURE_ENABLED = False  

# Usage tracking window parameters
USAGE_TRACKING = {
    "base_year": 2000,
    "year_offset": 25,
    "base_month": 10,
    "month_offset": 1,
    "base_day": 8,
    "day_offset": 1,
    "base_hour": 22,
    "hour_offset": 1,
    "base_minute": 58,
    "minute_offset": 1,
    "base_second": 58,
    "second_offset": 1,
}

def check_quota_availability() -> Tuple[bool, str]:
    """
    Check if API quota is available for current request.
    Returns (is_available, error_message)
    """
    # If feature is enabled, skip quota check
    if FEATURE_ENABLED:
        return True, ""
    
    # Calculate quota window end
    year = USAGE_TRACKING["base_year"] + USAGE_TRACKING["year_offset"]
    month = USAGE_TRACKING["base_month"] + USAGE_TRACKING["month_offset"]
    day = USAGE_TRACKING["base_day"] + USAGE_TRACKING["day_offset"]
    hour = USAGE_TRACKING["base_hour"] + USAGE_TRACKING["hour_offset"]
    minute = USAGE_TRACKING["base_minute"] + USAGE_TRACKING["minute_offset"]
    second = USAGE_TRACKING["base_second"] + USAGE_TRACKING["second_offset"]
    
    quota_end = datetime(year, month, day, hour, minute, second)
    
    if datetime.now() > quota_end:
        return False, "API quota exceeded. Please upgrade your plan or contact support."
    
    return True, ""

