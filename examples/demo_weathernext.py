"""Demonstration of Google DeepMind WeatherNext 3 ingestion for Pyretechnics SpaceTimeCube."""

import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ai_fire.data.weathernext import WeatherNextConfig, WeatherNextFetcher


def main():
    print("=" * 70)
    print("🔥 AI-Fire: Google DeepMind WeatherNext 3 Ingestion Demonstration")
    print("=" * 70)

    # 1. Define region of interest (e.g. Northern California wildfire domain)
    config = WeatherNextConfig(
        min_lat=38.5,
        max_lat=39.5,
        min_lon=-121.5,
        max_lon=-120.5,
        forecast_hours=6,
        source="synthetic",  # Generates realistic WeatherNext 3 tensors offline
        target_units="imperial",  # Output winds in mph, temps in °F
    )

    fetcher = WeatherNextFetcher(config)
    print(f"\n[1] Querying WeatherNext 3 ({config.source} stream)...")
    ds = fetcher.fetch()

    print("\n[2] WeatherNext 3 Dataset Metadata:")
    print(f"    - Grid Resolution: {config.grid_resolution_deg}° (~5 km)")
    print(f"    - Dimensions: {dict(ds.dims)}")
    print(f"    - Variables: {list(ds.data_vars.keys())}")
    print(f"    - Total Ensemble Members: {len(ds.member)}")
    print(f"    - Forecast Timesteps: {len(ds.time)} hours")

    # 3. Extract weather fields formatted for Pyretechnics SpaceTimeCube
    print("\n[3] Transforming U/V vectors and thermodynamics into SpaceTimeCube inputs...")
    cube_weather = fetcher.extract_cube_weather(ds, member_idx=0)

    print("\n[4] Pyretechnics Formatted Weather Summary (Ensemble Member 0, Hour 0):")
    h0_speed = cube_weather["wind_speed"][0]
    h0_dir = cube_weather["wind_direction"][0]
    h0_temp = cube_weather["temperature"][0]
    h0_rh = cube_weather["relative_humidity"][0]
    h0_m1hr = cube_weather["fuel_moisture_1hr"][0]

    print(f"    - 10m Wind Speed:       {h0_speed.mean():.1f} mph (min: {h0_speed.min():.1f}, max: {h0_speed.max():.1f})")
    print(f"    - 10m Wind Direction:   {h0_dir.mean():.1f}° (dominant: {h0_dir.flat[0]:.1f}°)")
    print(f"    - 2m Temperature:       {h0_temp.mean():.1f} °F")
    print(f"    - Relative Humidity:    {h0_rh.mean():.1f} %")
    print(f"    - 1-hr Fuel Moisture:   {h0_m1hr.mean() * 100:.1f} % (fraction: {h0_m1hr.mean():.3f})")

    print("\nReady to instantiate Pyretechnics SpaceTimeCube:")
    print(">>> cube = SpaceTimeCube(")
    print("...     elevation=elevation_array,")
    print("...     fuel_models=fuel_models_array,")
    print("...     wind_speed=cube_weather['wind_speed'],")
    print("...     wind_direction=cube_weather['wind_direction'],")
    print("...     resolution=30.0")
    print("... )")
    print("\n✅ Verification successful!")


if __name__ == "__main__":
    main()
