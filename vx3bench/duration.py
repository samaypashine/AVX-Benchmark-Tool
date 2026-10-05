UNITS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}

def duration_seconds(session):
    if isinstance(session.get("duration"), dict):
        value = session["duration"].get("value")
        unit = session["duration"].get("unit", "seconds")
    else:
        value = session.get("duration_seconds", 300)
        unit = "seconds"
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("session duration value must be greater than zero")
    if unit not in UNITS:
        raise ValueError("duration unit must be seconds, minutes, hours, or days")
    return float(value) * UNITS[unit]
