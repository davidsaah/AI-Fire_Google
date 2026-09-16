"""Unit tests for SpaceTimeCube builder and automated spread forecast launcher."""

import asyncio
import unittest

from ai_fire.agents.wildfire_risk_agent.tools import (
    find_active_wildfires as agent_find_fires,
    get_wildfire_incident_details as agent_get_details,
    launch_fire_spread_forecast as agent_launch_forecast,
)
from ai_fire.data.cube_builder import build_simulation_cube, execute_end_to_end_forecast
from ai_fire.models.spread_simulator import simulate_fire_spread


class TestCubeBuilderAndSpread(unittest.TestCase):
    """Test suite for SpaceTimeCube assembly and spread simulations."""

    def test_build_simulation_cube(self):
        """Verify 30m landscape and WeatherNext 3 cube construction."""
        cube = build_simulation_cube(
            center_lat=34.1724,
            center_lon=-117.1121,
            incident_name="Line Fire",
            domain_size_km=10.0,
            forecast_hours=6,
            initial_acres=200.0,
        )

        self.assertEqual(cube.incident_name, "Line Fire")
        self.assertEqual(cube.forecast_horizon_hours, 6)
        self.assertEqual(len(cube.hourly_weather), 6)
        self.assertGreater(cube.landscape.mean_slope_pct, 0.0)
        self.assertTrue(len(cube.landscape.dominant_fuel_name) > 0)
        self.assertEqual(len(cube.bounding_box), 4)

    def test_simulate_fire_spread(self):
        """Verify multi-hour elliptical spread calculations and isochrones."""
        result = simulate_fire_spread(
            incident_name="Test Complex",
            initial_lat=34.1724,
            initial_lon=-117.1121,
            initial_acres=50.0,
            slope_pct=22.0,
            fuel_type="shrub",
            duration_hours=4,
        )

        self.assertGreater(result.projected_final_acres, result.initial_acres)
        self.assertGreater(result.growth_acres, 0.0)
        self.assertGreater(result.max_rate_of_spread_chains_per_hr, 0.0)
        self.assertGreater(result.max_flame_length_ft, 0.0)
        self.assertEqual(len(result.hourly_isochrones), 4)

        # Check hourly progression monotonicity
        for i in range(len(result.hourly_isochrones) - 1):
            curr_iso = result.hourly_isochrones[i]
            next_iso = result.hourly_isochrones[i + 1]
            self.assertLessEqual(curr_iso.cumulative_acres, next_iso.cumulative_acres)

    def test_execute_end_to_end_forecast(self):
        """Verify complete pipeline execution."""
        res = execute_end_to_end_forecast(
            center_lat=34.05,
            center_lon=-118.52,
            incident_name="Palisades",
            initial_acres=500.0,
            duration_hours=3,
        )

        self.assertIn("cube_id", res)
        self.assertIn("projected_final_acres", res)
        self.assertIn("hourly_isochrones", res)
        self.assertEqual(len(res["hourly_isochrones"]), 3)
        self.assertIn(res["suppression_threat_level"], ["LOW", "MODERATE", "HIGH", "EXTREME"])

    def test_agent_tools_integration(self):
        """Verify agent tools execute cleanly in asyncio loop."""
        loop = asyncio.new_event_loop()
        try:
            # 1. Test find_active_wildfires
            res_find = loop.run_until_complete(
                agent_find_fires(region_or_state="CA", min_acres=5.0)
            )
            self.assertIn("incidents", res_find)
            self.assertGreaterEqual(res_find["total_incidents_found"], 1)

            # 2. Test get_wildfire_incident_details
            res_det = loop.run_until_complete(
                agent_get_details("Line")
            )
            self.assertTrue("incident" in res_det or "status" in res_det)

            # 3. Test launch_fire_spread_forecast with direct coordinates
            res_launch = loop.run_until_complete(
                agent_launch_forecast("34.1724, -117.1121", duration_hours=3)
            )
            self.assertIn("projected_final_acres", res_launch)
            self.assertIn("hourly_isochrones", res_launch)
            self.assertEqual(len(res_launch["hourly_isochrones"]), 3)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
