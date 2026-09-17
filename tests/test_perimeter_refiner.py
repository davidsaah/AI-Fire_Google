"""Unit tests for wildfire perimeter identification and boundary refinement engine."""

import unittest

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ai_fire.data.perimeter_refiner import (
    compute_alpha_shape,
    disaggregate_fire_components,
    refine_geometry_morphology,
    refine_wildfire_perimeter,
    smooth_perimeter_chaikin,
)


class TestPerimeterRefiner(unittest.TestCase):
    """Test suite for perimeter identification algorithms."""

    def setUp(self):
        """Set up synthetic crescent/horseshoe wildfire hotspot points."""
        # Non-convex horseshoe cluster (e.g. fire flanking around a mountain ridge)
        self.mock_hotspots = [
            {"latitude": 34.170, "longitude": -117.110, "frp_mw": 20.0},
            {"latitude": 34.173, "longitude": -117.113, "frp_mw": 35.0},
            {"latitude": 34.178, "longitude": -117.117, "frp_mw": 60.0},
            {"latitude": 34.183, "longitude": -117.123, "frp_mw": 75.0},
            {"latitude": 34.185, "longitude": -117.128, "frp_mw": 45.0},
            {"latitude": 34.182, "longitude": -117.133, "frp_mw": 30.0},
            {"latitude": 34.176, "longitude": -117.131, "frp_mw": 18.0},
            {"latitude": 34.171, "longitude": -117.126, "frp_mw": 12.0},
        ]
        self.points_lon_lat = np.array(
            [[h["longitude"], h["latitude"]] for h in self.mock_hotspots]
        )

    def test_compute_alpha_shape(self):
        """Verify alpha-shape generates a valid, watertight concave polygon."""
        poly = compute_alpha_shape(self.points_lon_lat, alpha_km=1.2)
        self.assertTrue(isinstance(poly, (Polygon, MultiPolygon)))
        self.assertTrue(poly.is_valid)
        self.assertGreater(poly.area, 0.0)

    def test_smooth_perimeter_chaikin(self):
        """Verify Chaikin algorithm doubles vertex density and preserves boundary closure."""
        raw_coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        smoothed = smooth_perimeter_chaikin(raw_coords, iterations=1)
        self.assertGreater(len(smoothed), len(raw_coords))
        self.assertEqual(smoothed[0], smoothed[-1])

    def test_refine_geometry_morphology(self):
        """Verify morphological closing and curvature regularization."""
        raw_poly = compute_alpha_shape(self.points_lon_lat, alpha_km=1.0)
        refined = refine_geometry_morphology(raw_poly, closing_dist_deg=0.003, smoothing_iters=1)
        self.assertTrue(refined.is_valid)
        self.assertGreater(refined.area, 0.0)

    def test_disaggregate_fire_components(self):
        """Verify sectors (head, flanks, heel, spots) are disaggregated correctly."""
        poly = compute_alpha_shape(self.points_lon_lat, alpha_km=1.2)
        sectors = disaggregate_fire_components(
            hotspots=self.mock_hotspots,
            refined_perimeter=poly,
            wind_direction_deg=240.0,  # Wind from SW -> Fire spreads to NE
        )

        self.assertIn("total_refined_acres", sectors)
        self.assertIn("spread_heading_deg", sectors)
        self.assertIn("active_head_detections", sectors)
        self.assertIn("flank_detections", sectors)
        self.assertIn("heel_detections", sectors)
        self.assertGreaterEqual(sectors["active_head_detections"], 1)

    def test_refine_wildfire_perimeter_pipeline(self):
        """Verify complete end-to-end perimeter refinement pipeline."""
        res = refine_wildfire_perimeter(
            hotspots=self.mock_hotspots,
            alpha_km=1.2,
            wind_direction_deg=240.0,
        )

        self.assertEqual(res["status"], "success")
        self.assertIn("refined_perimeter_acres", res)
        self.assertIn("naive_convex_hull_acres", res)
        self.assertIn("perimeter_geojson", res)
        self.assertEqual(res["total_hotspots_processed"], len(self.mock_hotspots))


if __name__ == "__main__":
    unittest.main()
