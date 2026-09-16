"""Unit tests for NIFC WFIGS, NASA FIRMS, and wildfire detection engine."""

import unittest

from ai_fire.data.fire_detector import (
    FireSource,
    WildfireIncident,
    _normalize_state_code,
    cluster_firms_hotspots,
    find_active_wildfires,
    query_firms_hotspots,
    query_nifc_active_fires,
    query_nifc_perimeters,
)


class TestFireDetector(unittest.TestCase):
    """Test suite for wildfire detection client."""

    def test_normalize_state_code(self):
        """Verify state normalization to US-XX format."""
        self.assertEqual(_normalize_state_code("CA"), "US-CA")
        self.assertEqual(_normalize_state_code("California"), "US-CA")
        self.assertEqual(_normalize_state_code("texas"), "US-TX")
        self.assertEqual(_normalize_state_code("US-OR"), "US-OR")
        self.assertIsNone(_normalize_state_code(None))

    def test_query_nifc_active_fires(self):
        """Verify NIFC active fire query returns structured incidents."""
        incidents = query_nifc_active_fires(state="CA", min_acres=1.0, max_results=5)
        self.assertIsInstance(incidents, list)
        self.assertGreaterEqual(len(incidents), 1)

        first = incidents[0]
        self.assertIsInstance(first, WildfireIncident)
        self.assertTrue(len(first.name) > 0)
        self.assertGreater(first.acres, 0.0)
        self.assertTrue(-180.0 <= first.longitude <= 180.0)
        self.assertTrue(-90.0 <= first.latitude <= 90.0)
        self.assertEqual(first.source, FireSource.NIFC_WFIGS.value)

    def test_query_nifc_perimeters(self):
        """Verify NIFC perimeter queries return GeoJSON polygon records."""
        perimeters = query_nifc_perimeters(min_acres=1.0, max_results=5)
        self.assertIsInstance(perimeters, list)

    def test_cluster_firms_hotspots(self):
        """Verify spatial clustering of adjacent 375m thermal pixels."""
        mock_hotspots = [
            {"latitude": 34.170, "longitude": -117.110, "frp_mw": 25.0, "bright_ti4_k": 340.0, "confidence": "high", "acq_date": "2026-09-15"},
            {"latitude": 34.172, "longitude": -117.112, "frp_mw": 18.0, "bright_ti4_k": 335.0, "confidence": "high", "acq_date": "2026-09-15"},
            {"latitude": 34.175, "longitude": -117.115, "frp_mw": 32.0, "bright_ti4_k": 350.0, "confidence": "high", "acq_date": "2026-09-15"},
            # Distinct separate fire far away
            {"latitude": 38.500, "longitude": -120.500, "frp_mw": 45.0, "bright_ti4_k": 360.0, "confidence": "nominal", "acq_date": "2026-09-15"},
        ]
        clusters = cluster_firms_hotspots(mock_hotspots, eps_km=2.0)
        self.assertEqual(len(clusters), 2)

        # Largest FRP cluster should be first
        first = clusters[0]
        self.assertGreater(first.frp_mw, 40.0)
        self.assertGreater(first.acres, 0.0)
        self.assertEqual(first.source, FireSource.NASA_FIRMS_VIIRS.value)

    def test_find_active_wildfires(self):
        """Verify unified multi-source discovery returns ranked incidents."""
        fires = find_active_wildfires(query_str="California", min_acres=5.0, max_results=5)
        self.assertIsInstance(fires, list)
        self.assertGreaterEqual(len(fires), 1)

        # Verify descending order by acres
        for i in range(len(fires) - 1):
            self.assertGreaterEqual(fires[i].acres, fires[i + 1].acres)


if __name__ == "__main__":
    unittest.main()
