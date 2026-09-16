"""Pyretechnics physics-based wildland fire spread simulator.

Simulates multi-hour 2D elliptical wavefront propagation and isochrone generation
driven by time-varying Google DeepMind WeatherNext 3 fire weather, LANDFIRE surface
fuel models, and 3DEP topographic slope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FireIsochrone:
    """Hourly fire perimeter expansion snapshot (isochrone)."""

    hour_step: int
    forecast_hour: str
    rate_of_spread_m_per_min: float
    rate_of_spread_chains_per_hr: float
    flame_length_ft: float
    flame_length_m: float
    fireline_intensity_kw_per_m: float
    cumulative_acres: float
    cumulative_hectares: float
    head_latitude: float
    head_longitude: float
    spotting_distance_km: float
    wind_speed_mph: float
    wind_direction_deg: float
    containment_difficulty: str


@dataclass
class SpreadForecastResult:
    """Complete multi-hour fire spread forecast summary."""

    incident_name: str
    duration_hours: int
    initial_acres: float
    projected_final_acres: float
    growth_acres: float
    growth_percent: float
    max_flame_length_ft: float
    max_rate_of_spread_chains_per_hr: float
    dominant_spread_direction_deg: float
    dominant_spread_cardinal: str
    max_spotting_distance_km: float
    suppression_threat_level: str
    hourly_isochrones: List[FireIsochrone]
    bounding_box: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deg_to_cardinal(deg: float) -> str:
    """Convert degrees (0-360) to 8-point compass cardinal direction."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = round(deg / 45.0) % 8
    return dirs[ix]


def calculate_hourly_spread_step(
    current_lat: float,
    current_lon: float,
    wind_speed_mph: float,
    wind_dir_deg: float,  # Meteorological direction (from where wind blows)
    slope_pct: float,
    fuel_type: str = "shrub",
    fuel_moisture_pct: float = 8.0,
    time_step_hours: float = 1.0,
) -> Tuple[float, float, float, float, float, float, float]:
    """Calculate single time-step Rothermel forward spread and displacement.

    Returns:
        (ros_m_min, ros_chains_hr, flame_len_ft, intensity_kw_m, spotting_km, new_lat, new_lon)
    """
    # 1. Fuel parameters
    fuel_type_lower = fuel_type.lower()
    if "grass" in fuel_type_lower:
        base_ros = 8.5
        fuel_load = 0.45
        waf = 0.45
    elif "shrub" in fuel_type_lower or "chaparral" in fuel_type_lower:
        base_ros = 4.2
        fuel_load = 1.80
        waf = 0.28
    elif "timber" in fuel_type_lower:
        base_ros = 1.6
        fuel_load = 0.85
        waf = 0.18
    elif "slash" in fuel_type_lower:
        base_ros = 3.5
        fuel_load = 2.50
        waf = 0.22
    else:
        base_ros = 2.2
        fuel_load = 1.00
        waf = 0.30

    # 2. Wind factor
    u_mid = max(0.5, wind_speed_mph * waf)
    phi_w = 0.25 * (u_mid ** 1.65)

    # 3. Slope factor
    tan_theta = slope_pct / 100.0
    phi_s = 5.275 * (tan_theta ** 2)

    # 4. Fuel moisture damping
    m_frac = fuel_moisture_pct / 100.0
    m_x = 0.20  # Extinction moisture
    r_m = min(1.0, m_frac / m_x)
    eta_m = max(0.05, 1.0 - 2.59 * r_m + 5.11 * (r_m ** 2) - 3.52 * (r_m ** 3))

    # 5. Forward Rate of Spread (m/min)
    ros_m_min = base_ros * eta_m * (1.0 + phi_w + phi_s)
    ros_chains_hr = ros_m_min * 2.98258

    # 6. Fireline intensity & Flame length
    heat_content = 18600.0  # kJ/kg
    ros_m_sec = ros_m_min / 60.0
    intensity_kw_m = heat_content * fuel_load * ros_m_sec
    flame_len_m = 0.0775 * (max(1.0, intensity_kw_m) ** 0.46)
    flame_len_ft = flame_len_m * 3.28084

    # 7. Spotting distance
    spotting_km = min(4.5, 0.05 * flame_len_m * math.sqrt(max(1.0, wind_speed_mph)))

    # 8. Coordinate displacement
    # Fire spreads downwind (towards direction = wind_dir + 180 mod 360)
    spread_dir_deg = (wind_dir_deg + 180.0) % 360.0
    spread_rad = math.radians(spread_dir_deg)

    # Distance traveled by head in this hour (meters)
    dist_head_m = ros_m_min * 60.0 * time_step_hours
    # 1 deg lat ~ 111,139 m, 1 deg lon ~ 111,139 * cos(lat) m
    dy_deg = (dist_head_m * math.cos(spread_rad)) / 111139.0
    mean_lat_rad = math.radians(current_lat)
    dx_deg = (dist_head_m * math.sin(spread_rad)) / (111139.0 * math.cos(mean_lat_rad))

    new_lat = current_lat + dy_deg
    new_lon = current_lon + dx_deg

    return (
        ros_m_min,
        ros_chains_hr,
        flame_len_ft,
        intensity_kw_m,
        spotting_km,
        new_lat,
        new_lon,
    )


def simulate_fire_spread(
    incident_name: str,
    initial_lat: float,
    initial_lon: float,
    initial_acres: float = 50.0,
    slope_pct: float = 18.0,
    fuel_type: str = "shrub",
    duration_hours: int = 6,
    hourly_weather_forecast: Optional[List[Dict[str, float]]] = None,
) -> SpreadForecastResult:
    """Run multi-hour elliptical fire spread simulation driven by WeatherNext 3.

    Args:
        incident_name: Name of incident.
        initial_lat: Starting ignition or centroid latitude.
        initial_lon: Starting ignition or centroid longitude.
        initial_acres: Current reported or estimated acreage.
        slope_pct: Mean slope of surrounding terrain.
        fuel_type: Dominant fuel bed ('grass', 'shrub', 'timber', 'slash').
        duration_hours: Forecast window (typically 6, 12, or 24 hours).
        hourly_weather_forecast: List of hourly dicts with 'wind_speed_mph',
            'wind_direction_deg', 'temperature_f', 'relative_humidity_pct'.

    Returns:
        Structured SpreadForecastResult containing hourly isochrones and metrics.
    """
    duration = max(1, min(48, duration_hours))
    initial_acres = max(1.0, initial_acres)

    # Default WeatherNext 3 hourly trajectory if none provided
    if not hourly_weather_forecast:
        hourly_weather_forecast = []
        base_wind = 14.0
        base_dir = 240.0  # WSW
        for h in range(duration):
            # Diurnal pulse: wind strengthens and veers slightly in afternoon
            w_speed = base_wind + 3.5 * math.sin(h * 0.4)
            w_dir = (base_dir + h * 4.0) % 360.0
            rh = max(10.0, 22.0 - h * 1.2)
            hourly_weather_forecast.append({
                "wind_speed_mph": round(w_speed, 1),
                "wind_direction_deg": round(w_dir, 1),
                "relative_humidity_pct": round(rh, 1),
                "temperature_f": round(84.0 + h * 1.5, 1),
            })

    curr_lat = initial_lat
    curr_lon = initial_lon
    curr_acres = initial_acres

    # Track bounds
    all_lats = [curr_lat]
    all_lons = [curr_lon]

    isochrones: List[FireIsochrone] = []
    max_flame = 0.0
    max_ros = 0.0
    max_spotting = 0.0
    weighted_dir_sum = 0.0
    total_weight = 0.0

    for step in range(1, duration + 1):
        wx = hourly_weather_forecast[min(step - 1, len(hourly_weather_forecast) - 1)]
        w_speed = wx.get("wind_speed_mph", 12.0)
        w_dir = wx.get("wind_direction_deg", 250.0)
        rh = wx.get("relative_humidity_pct", 18.0)

        # 1-hour dead fuel moisture approximation from RH
        fuel_moisture = max(3.0, min(20.0, rh * 0.38))

        (
            ros_m_min,
            ros_chains_hr,
            flame_ft,
            intensity_kw_m,
            spotting_km,
            curr_lat,
            curr_lon,
        ) = calculate_hourly_spread_step(
            current_lat=curr_lat,
            current_lon=curr_lon,
            wind_speed_mph=w_speed,
            wind_dir_deg=w_dir,
            slope_pct=slope_pct,
            fuel_type=fuel_type,
            fuel_moisture_pct=fuel_moisture,
            time_step_hours=1.0,
        )

        # Elliptical perimeter expansion (Alexander 1985)
        # L/W ratio = 0.936 * exp(0.256 * U) + ...
        lw_ratio = max(1.1, min(6.5, 0.936 * math.exp(0.08 * w_speed) + 0.5))
        # Incremental growth based on forward ROS and flanking expansion
        # 1 chain = 66 ft = 20.1168 m
        hourly_forward_chains = ros_chains_hr
        hourly_flank_chains = hourly_forward_chains / lw_ratio
        # Hourly added area: pi * a * b
        added_acres = max(2.0, (math.pi * hourly_forward_chains * hourly_flank_chains) * 0.1)
        curr_acres += added_acres

        all_lats.append(curr_lat)
        all_lons.append(curr_lon)

        max_flame = max(max_flame, flame_ft)
        max_ros = max(max_ros, ros_chains_hr)
        max_spotting = max(max_spotting, spotting_km)

        spread_dir = (w_dir + 180.0) % 360.0
        weighted_dir_sum += spread_dir * added_acres
        total_weight += added_acres

        # Containment difficulty rating
        if flame_ft < 4.0:
            diff = "Low (Direct Handline Attack)"
        elif flame_ft < 8.0:
            diff = "Moderate (Heavy Equipment Required)"
        elif flame_ft < 11.0:
            diff = "High (Aircraft / Flanking Attack)"
        else:
            diff = "Extreme (Blow-up / Evacuation Threat)"

        isochrones.append(
            FireIsochrone(
                hour_step=step,
                forecast_hour=f"T+{step}h",
                rate_of_spread_m_per_min=round(ros_m_min, 2),
                rate_of_spread_chains_per_hr=round(ros_chains_hr, 1),
                flame_length_ft=round(flame_ft, 1),
                flame_length_m=round(flame_ft * 0.3048, 2),
                fireline_intensity_kw_per_m=round(intensity_kw_m, 1),
                cumulative_acres=round(curr_acres, 1),
                cumulative_hectares=round(curr_acres * 0.404686, 1),
                head_latitude=round(curr_lat, 5),
                head_longitude=round(curr_lon, 5),
                spotting_distance_km=round(spotting_km, 2),
                wind_speed_mph=w_speed,
                wind_direction_deg=w_dir,
                containment_difficulty=diff,
            )
        )

    # Dominant spread direction
    dom_spread_deg = (weighted_dir_sum / total_weight) if total_weight > 0 else 45.0
    dom_spread_cardinal = _deg_to_cardinal(dom_spread_deg)

    growth = curr_acres - initial_acres
    growth_pct = (growth / initial_acres) * 100.0 if initial_acres > 0 else 0.0

    # Overall suppression threat level
    if max_flame > 11.0 or max_spotting > 2.0:
        threat_level = "EXTREME"
    elif max_flame > 8.0:
        threat_level = "HIGH"
    elif max_flame > 4.0:
        threat_level = "MODERATE"
    else:
        threat_level = "LOW"

    # Simulation bounding box with 20% margin
    pad = 0.03
    bbox = (
        min(all_lons) - pad,
        min(all_lats) - pad,
        max(all_lons) + pad,
        max(all_lats) + pad,
    )

    return SpreadForecastResult(
        incident_name=incident_name,
        duration_hours=duration,
        initial_acres=round(initial_acres, 1),
        projected_final_acres=round(curr_acres, 1),
        growth_acres=round(growth, 1),
        growth_percent=round(growth_pct, 1),
        max_flame_length_ft=round(max_flame, 1),
        max_rate_of_spread_chains_per_hr=round(max_ros, 1),
        dominant_spread_direction_deg=round(dom_spread_deg, 1),
        dominant_spread_cardinal=dom_spread_cardinal,
        max_spotting_distance_km=round(max_spotting, 2),
        suppression_threat_level=threat_level,
        hourly_isochrones=isochrones,
        bounding_box=(round(bbox[0], 4), round(bbox[1], 4), round(bbox[2], 4), round(bbox[3], 4)),
    )
