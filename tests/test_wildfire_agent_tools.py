"""Unit tests for wildfire risk agent Earth Engine and fire physics tools."""

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ai_fire.agents.wildfire_risk_agent import tools


class TestWildfireRiskAgentTools(unittest.TestCase):
    """Test suite for tools in wildfire_risk_agent/tools.py."""

    def setUp(self):
        # Sample polygon in Northern California wildfire corridor
        self.sample_geojson = json.dumps({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.585, 38.565],
                        [-122.555, 38.565],
                        [-122.555, 38.595],
                        [-122.585, 38.595],
                        [-122.585, 38.565],
                    ]
                ],
            },
            "properties": {"name": "Test Wildfire ROI"},
        })

    def test_get_geometry_area(self):
        """Test polygon area calculation."""
        res = asyncio.run(tools.get_geometry_area(self.sample_geojson))
        self.assertNotIn("error", res)
        self.assertIn("area_hectares", res)
        self.assertIn("area_acres", res)
        self.assertGreater(res["area_hectares"], 0)
        self.assertGreater(res["area_acres"], 0)

    def test_get_topography_risk_stats(self):
        """Test elevation, slope, and aspect calculations."""
        res = asyncio.run(tools.get_topography_risk_stats(self.sample_geojson))
        self.assertNotIn("error", res)
        self.assertIn("slope_mean_degrees", res)
        self.assertIn("dominant_aspect_cardinal", res)
        self.assertIn("slope_spread_multiplier", res)
        self.assertGreaterEqual(res["slope_spread_multiplier"], 1.0)

    def test_get_vegetation_fuel_stats(self):
        """Test vegetation and surface fuel classification."""
        res = asyncio.run(tools.get_vegetation_fuel_stats(self.sample_geojson))
        self.assertNotIn("error", res)
        self.assertIn("dominant_fuel_family", res)
        self.assertIn("surface_fuel_hazard", res)

    def test_get_canopy_structure_stats(self):
        """Test tree canopy cover and bulk density evaluation."""
        res = asyncio.run(tools.get_canopy_structure_stats(self.sample_geojson))
        self.assertNotIn("error", res)
        self.assertIn("canopy_cover_pct", res)
        self.assertIn("canopy_bulk_density_kg_m3", res)
        self.assertIn("crown_fire_susceptibility", res)

    def test_get_historical_burn_stats(self):
        """Test historical burn scar querying."""
        res = asyncio.run(tools.get_historical_burn_stats(self.sample_geojson))
        self.assertNotIn("error", res)
        self.assertIn("burn_scars_detected_15yr", res)
        self.assertIn("fuel_age_status", res)

    def test_get_fire_danger_indices(self):
        """Test GridMET fire danger and fuel moisture indices."""
        res = asyncio.run(tools.get_fire_danger_indices(self.sample_geojson))
        self.assertNotIn("error", res)
        self.assertIn("energy_release_component", res)
        self.assertIn("dead_fuel_moisture_100hr_pct", res)

    def test_get_weathernext_fire_weather(self):
        """Test WeatherNext 3 wind and atmospheric extraction."""
        res = asyncio.run(tools.get_weathernext_fire_weather(self.sample_geojson))
        self.assertNotIn("error", res)
        self.assertIn("wind_speed_10m_mph", res)
        self.assertIn("relative_humidity_pct", res)
        self.assertIn("red_flag_warning_status", res)

    def test_calculate_surface_fire_behavior(self):
        """Test Rothermel rate of spread and flame length physics."""
        # Test shrub fuel under high wind and dry conditions
        res_shrub = tools.calculate_surface_fire_behavior(
            fuel_type="shrub",
            slope_pct=25.0,
            wind_speed_mph=20.0,
            fuel_moisture_pct=4.0,
        )
        self.assertNotIn("error", res_shrub)
        self.assertGreater(res_shrub["rate_of_spread_m_per_min"], 0)
        self.assertGreater(res_shrub["flame_length_meters"], 0)
        self.assertIn("suppression_tactics_guide", res_shrub)

        # Grass spreads faster than timber under same wind
        res_grass = tools.calculate_surface_fire_behavior(
            fuel_type="grass",
            slope_pct=10.0,
            wind_speed_mph=15.0,
            fuel_moisture_pct=5.0,
        )
        res_timber = tools.calculate_surface_fire_behavior(
            fuel_type="timber",
            slope_pct=10.0,
            wind_speed_mph=15.0,
            fuel_moisture_pct=5.0,
        )
        self.assertGreater(
            res_grass["rate_of_spread_m_per_min"],
            res_timber["rate_of_spread_m_per_min"],
        )


if __name__ == "__main__":
    unittest.main()
