import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ai_fire.data.weathernext import WeatherNextConfig, WeatherNextFetcher


class TestWeatherNextFetcher(unittest.TestCase):
    """Test suite for WeatherNext 3 fetcher and calculations."""

    def setUp(self):
        self.config = WeatherNextConfig(
            min_lat=38.0,
            max_lat=39.0,
            min_lon=-121.0,
            max_lon=-120.0,
            forecast_hours=4,
            source="synthetic",
            target_units="imperial",
        )
        self.fetcher = WeatherNextFetcher(self.config)

    def test_uv_to_meteorological_cardinal_directions(self):
        """Test conversion of wind vectors to standard meteorological direction."""
        # West wind (blowing eastward, u > 0, v = 0) -> from 270 deg
        speed, direction = self.fetcher.uv_to_meteorological(np.array([10.0]), np.array([0.0]))
        self.assertAlmostEqual(speed[0], 10.0, places=4)
        self.assertAlmostEqual(direction[0], 270.0, places=4)

        # East wind (blowing westward, u < 0, v = 0) -> from 90 deg
        speed, direction = self.fetcher.uv_to_meteorological(np.array([-10.0]), np.array([0.0]))
        self.assertAlmostEqual(speed[0], 10.0, places=4)
        self.assertAlmostEqual(direction[0], 90.0, places=4)

        # South wind (blowing northward, u = 0, v > 0) -> from 180 deg
        speed, direction = self.fetcher.uv_to_meteorological(np.array([0.0]), np.array([10.0]))
        self.assertAlmostEqual(speed[0], 10.0, places=4)
        self.assertAlmostEqual(direction[0], 180.0, places=4)

        # North wind (blowing southward, u = 0, v < 0) -> from 0 deg / 360 deg
        speed, direction = self.fetcher.uv_to_meteorological(np.array([0.0]), np.array([-10.0]))
        self.assertAlmostEqual(speed[0], 10.0, places=4)
        self.assertTrue(np.isclose(direction[0], 0.0) or np.isclose(direction[0], 360.0))

        # Southwest wind (u > 0, v > 0) -> from 225 deg
        speed, direction = self.fetcher.uv_to_meteorological(np.array([10.0]), np.array([10.0]))
        self.assertAlmostEqual(speed[0], 10.0 * np.sqrt(2), places=4)
        self.assertAlmostEqual(direction[0], 225.0, places=4)

    def test_compute_relative_humidity(self):
        """Test relative humidity calculation using Magnus approximation."""
        # 100% RH when temp == dewpoint
        temp_k = np.array([293.15])  # 20°C
        dewpoint_k = np.array([293.15])  # 20°C
        rh = self.fetcher.compute_relative_humidity(temp_k, dewpoint_k)
        self.assertAlmostEqual(rh[0], 100.0, places=2)

        # 25°C temp and 10°C dewpoint (~39-40% RH)
        temp_k = np.array([298.15])
        dewpoint_k = np.array([283.15])
        rh = self.fetcher.compute_relative_humidity(temp_k, dewpoint_k)
        self.assertTrue(38.0 <= rh[0] <= 41.0, f"Expected ~39.5%, got {rh[0]}")

    def test_estimate_1hr_dead_fuel_moisture(self):
        """Test 1-hour dead fuel moisture formulation."""
        # Hot, dry conditions: 95°F, 12% RH -> very dry fuel (around 4-6%)
        m_dry = self.fetcher.estimate_1hr_dead_fuel_moisture(np.array([95.0]), np.array([12.0]))
        self.assertTrue(0.02 <= m_dry[0] <= 0.07, f"Got {m_dry[0]}")

        # Humid conditions: 65°F, 80% RH -> damp fuel (> 15%)
        m_wet = self.fetcher.estimate_1hr_dead_fuel_moisture(np.array([65.0]), np.array([80.0]))
        self.assertTrue(0.12 <= m_wet[0] <= 0.30, f"Got {m_wet[0]}")

    def test_synthetic_generation_and_cube_weather_extraction(self):
        """Test generating synthetic WeatherNext 3 dataset and extracting SpaceTimeCube inputs."""
        ds = self.fetcher.generate_synthetic(self.config)

        # Verify dataset coordinates and dims
        self.assertIn("time", ds.dims)
        self.assertIn("member", ds.dims)
        self.assertIn("latitude", ds.dims)
        self.assertIn("longitude", ds.dims)
        self.assertEqual(len(ds.time), self.config.forecast_hours)
        self.assertEqual(len(ds.member), 64)

        # Extract weather arrays for Pyretechnics
        cube_weather = self.fetcher.extract_cube_weather(ds, member_idx=0, target_units="imperial")

        required_keys = [
            "wind_speed",
            "wind_direction",
            "wind_speed_100m",
            "temperature",
            "relative_humidity",
            "fuel_moisture_1hr",
            "lats",
            "lons",
            "times",
        ]
        for key in required_keys:
            self.assertIn(key, cube_weather)

        # Verify array shape: (time, lat, lon)
        expected_shape = (
            self.config.forecast_hours,
            len(cube_weather["lats"]),
            len(cube_weather["lons"]),
        )
        self.assertEqual(cube_weather["wind_speed"].shape, expected_shape)
        self.assertEqual(cube_weather["wind_direction"].shape, expected_shape)

        # Values should be within physical ranges
        self.assertTrue(np.all(cube_weather["wind_speed"] >= 0))
        self.assertTrue(np.all((cube_weather["wind_direction"] >= 0) & (cube_weather["wind_direction"] <= 360)))
        self.assertTrue(np.all((cube_weather["relative_humidity"] >= 1) & (cube_weather["relative_humidity"] <= 100)))
        self.assertTrue(np.all((cube_weather["fuel_moisture_1hr"] >= 0.01) & (cube_weather["fuel_moisture_1hr"] <= 0.35)))


if __name__ == "__main__":
    unittest.main()
