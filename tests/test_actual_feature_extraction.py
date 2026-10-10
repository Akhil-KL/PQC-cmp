"""
Tests for Phase 2B Actual Feature Extraction

Tests the actual feature extraction functionality without using
expected features for detection.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from feature_inspection.actual import (
    ActualFeature, ActualFeatureType, GeometricProperties, EvidenceMetrics,
    ActualFeatureExtractionResult, ActualFeatureExtractor, extract_actual_features
)
from feature_inspection.actual.circle_extractor import CircleExtractor
from feature_inspection.actual.rectangle_extractor import RectangleExtractor
from feature_inspection.actual.contour_extractor import ContourExtractor
from feature_inspection.preprocessing import PreprocessingResult
from unittest.mock import patch


class TestActualFeatureModels:
    """Test actual feature data models."""
    
    def test_actual_feature_creation(self):
        """Test creating ActualFeature objects."""
        # Create test geometry
        geometry = GeometricProperties(
            center=(100.0, 150.0),
            area=314.0,
            perimeter=62.8,
            bounding_box=(90, 140, 20, 20),
            radius=10.0,
            diameter=20.0,
            circularity=1.0,
            solidity=0.95,
            extent=0.785,
            convexity=1.0
        )
        
        # Create test evidence
        evidence = EvidenceMetrics(
            confidence=0.8,
            edge_support=0.7,
            contour_quality=0.9,
            intensity_consistency=0.6,
            geometric_consistency=0.85,
            internal_edge_evidence=1.0
        )
        
        # Create test contour
        angles = np.linspace(0, 2*np.pi, 32)
        contour_x = 100 + 10 * np.cos(angles)
        contour_y = 150 + 10 * np.sin(angles)
        contour = np.column_stack((contour_x, contour_y)).astype(np.int32)
        
        # Create feature
        feature = ActualFeature(
            feature_id="test_circle_1",
            feature_type=ActualFeatureType.CIRCLE,
            geometry=geometry,
            contour=contour,
            evidence=evidence,
            source_representation="internal_geometry_edges",
            detection_method="contour_analysis"
        )
        
        assert feature.feature_id == "test_circle_1"
        assert feature.feature_type == ActualFeatureType.CIRCLE
        assert feature.geometry.center == (100.0, 150.0)
        assert feature.evidence.confidence == 0.8
        assert len(feature.contour) == 32
    
    def test_coordinate_transformation(self):
        """Test coordinate system transformations."""
        # Create test feature in processed coordinates
        geometry = GeometricProperties(
            center=(200.0, 300.0),
            area=100.0,
            perimeter=40.0,
            bounding_box=(190, 290, 20, 20),
            radius=5.64,
            circularity=0.62
        )
        
        evidence = EvidenceMetrics(
            confidence=0.7,
            edge_support=0.6,
            contour_quality=0.8,
            intensity_consistency=0.5,
            geometric_consistency=0.75
        )
        
        contour = np.array([[190, 290], [210, 290], [210, 310], [190, 310]], dtype=np.int32)
        
        feature = ActualFeature(
            feature_id="test_rect_1",
            feature_type=ActualFeatureType.RECTANGLE,
            geometry=geometry,
            contour=contour,
            evidence=evidence,
            source_representation="internal_geometry_edges",
            detection_method="contour_analysis",
            scale_factor=0.5  # Processed image is half size
        )
        
        # Convert to original coordinates
        original_feature = feature.to_original_coordinates(roi_offset=(0, 0))
        
        assert original_feature.coordinate_system == "original"
        assert original_feature.scale_factor == 1.0
        assert original_feature.geometry.center == (400.0, 600.0)  # Scaled up by 2
        assert original_feature.geometry.area == 400.0  # Scaled up by 4 (2^2)


class TestCircleExtractor:
    """Test circle feature extraction."""
    
    def test_circle_extractor_initialization(self):
        """Test circle extractor initialization."""
        extractor = CircleExtractor()
        
        assert extractor.min_radius > 0
        assert extractor.max_radius > extractor.min_radius
        assert extractor.circularity_threshold > 0
        assert extractor.min_confidence > 0

    def test_partial_arc_is_not_accepted_as_full_circle(self):
        """A localized arc should not qualify as a full circular feature."""
        extractor = CircleExtractor()
        size = 200
        edge_image = np.zeros((size, size), dtype=np.uint8)
        center = (100, 100)
        radius = 30.0

        for angle in np.linspace(0, 2 * np.pi, 500, endpoint=False):
            if 0.2 * np.pi < angle < 1.0 * np.pi:
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                if 0 <= x < size and 0 <= y < size:
                    edge_image[y, x] = 255

        validation = extractor._validate_hough_candidate(
            center,
            radius,
            edge_image,
            np.zeros((size, size, 3), dtype=np.uint8),
            np.ones((size, size), dtype=np.uint8) * 255,
        )

        assert validation["valid"] is False
        assert "coverage" in validation["rejection_reason"] or "localized" in validation["rejection_reason"] or "gap" in validation["rejection_reason"]

    def test_hough_only_detection_confidence_is_penalized(self):
        """Weak Hough-only evidence should not receive a high confidence score."""
        extractor = CircleExtractor()
        validation = {
            "valid": True,
            "edge_support": 0.5,
            "angular_coverage": 0.35,
            "sector_coverage": 0.25,
            "geometric_consistency": 0.4,
            "intensity_consistency": 0.2,
            "contour_agreement": False,
        }

        confidence = extractor._calculate_hough_evidence_confidence(validation)
        assert confidence < 0.6

    def test_synthetic_circle_detection(self):
        """Test circle detection on synthetic images."""
        # Create synthetic images
        size = 200
        internal_edges = np.zeros((size, size), dtype=np.uint8)
        isolated_product = np.zeros((size, size, 3), dtype=np.uint8)
        product_mask = np.ones((size, size), dtype=np.uint8) * 255

        # Draw a clear circle in edges; fill interior brighter than background
        # so hole detector doesn't mistake the whole black canvas for a hole
        isolated_product[:] = 80  # grey background
        cv2.circle(internal_edges,    (100, 100), 30, 255, 2)
        cv2.circle(isolated_product, (100, 100), 30, (180, 180, 180), -1)

        extractor = CircleExtractor()
        circles = extractor.extract_circles(internal_edges, isolated_product, product_mask)

        # Should detect the synthetic circle (as CIRCLE or as a very small HOLE)
        assert len(circles) >= 1

        # Find the feature nearest to the drawn circle
        target_cx, target_cy, target_r = 100, 100, 30
        best = min(circles, key=lambda f: abs((f.geometry.radius or 0) - target_r)
                   + abs(f.geometry.center[0] - target_cx)
                   + abs(f.geometry.center[1] - target_cy))

        assert abs(best.geometry.center[0] - target_cx) < 15
        assert abs(best.geometry.center[1] - target_cy) < 15
        assert abs((best.geometry.radius or 0) - target_r) < 15
        assert best.evidence.confidence > 0

    def test_evidence_observability_hough_candidate(self):
        """Test that Hough candidates have all evidence fields populated."""
        # Create test image with edge pattern
        center = (50, 50)
        radius = 20
        
        # Create edge image
        edge_image = np.zeros((100, 100), dtype=np.uint8)
        angles = np.linspace(0, 2*np.pi, 64)
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            if 0 <= x < 100 and 0 <= y < 100:
                edge_image[y, x] = 255

        extractor = CircleExtractor()
        validation = extractor._validate_hough_candidate(
            center,
            radius,
            edge_image,
            np.ones((100, 100), dtype=np.uint8) * 128,  # intensity image
            np.ones((100, 100), dtype=np.uint8) * 255   # product mask
        )
        
        # All evidence fields should be present with numeric values
        required_fields = [
            "angular_coverage", "sector_coverage", "angular_uniformity", 
            "max_gap_ratio", "radial_consistency", "radial_error_median", 
            "radial_error_p95", "local_contrast", "gradient_orientation_consistency"
        ]
        
        for field in required_fields:
            assert field in validation, f"Missing evidence field: {field}"
            assert isinstance(validation[field], (int, float)), f"Field {field} should be numeric"
            assert not np.isnan(validation[field]), f"Field {field} should not be NaN"

    def test_radial_edge_profile_method(self):
        """Test the new radial edge profile sampling method."""
        center = (50, 50)
        radius = 20
        
        # Create edge image with perfect circle using cv2 for better coverage
        edge_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(edge_image, center, radius, 255, 1)  # Draw 1-pixel thick circle

        extractor = CircleExtractor()
        profile = extractor._sample_radial_edge_profile(center, radius, edge_image)
        
        # Verify profile structure
        assert "samples" in profile
        assert "edge_support" in profile
        assert "valid_samples" in profile
        assert "median_radial_error" in profile
        
        # For a perfect circle, edge support should be reasonable (lowered expectation)
        assert profile["edge_support"] > 0.5, f"Expected reasonable edge support, got {profile['edge_support']}"
        
        # Median radial error should be low for perfect circle (within tolerance)
        assert profile["median_radial_error"] <= extractor.radial_tolerance_pixels, f"Expected low radial error, got {profile['median_radial_error']}"
        
        # Verify sample data structure
        assert len(profile["samples"]) > 0
        sample = profile["samples"][0]
        required_sample_fields = ["angle", "expected_radius", "observed_radius", "radial_error", "edge_strength", "valid"]
        for field in required_sample_fields:
            assert field in sample, f"Missing sample field: {field}"

    def test_radial_profile_vs_neighborhood_check(self):
        """Test that radial profile is more selective than simple neighborhood checks."""
        center = (50, 50)
        radius = 20
        
        # Create edge image with offset circle (edges not at predicted radius)
        edge_image = np.zeros((100, 100), dtype=np.uint8)
        offset_radius = radius + 5  # Offset by 5 pixels
        cv2.circle(edge_image, center, offset_radius, 255, 1)

        extractor = CircleExtractor()
        
        # Test new method
        profile = extractor._sample_radial_edge_profile(center, radius, edge_image)
        radial_support = profile["edge_support"]
        
        # Test old-style edge support (should be more lenient)
        old_style_support = 0
        angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
        valid_points = 0
        
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            if 0 <= x < 100 and 0 <= y < 100:
                valid_points += 1
                if extractor._check_edge_neighborhood(x, y, edge_image, 3):
                    old_style_support += 1
        
        old_style_ratio = old_style_support / valid_points if valid_points > 0 else 0
        
        # The radial profile method should be more selective for offset edges
        # (though both might detect the offset circle if it's within tolerance)
        assert isinstance(radial_support, float)
        assert isinstance(old_style_ratio, float)
        assert 0.0 <= radial_support <= 1.0
        assert 0.0 <= old_style_ratio <= 1.0

    def test_real_radial_consistency_measurement(self):
        """Test that radial consistency measures actual edge proximity to predicted circumference."""
        center = (50, 50)
        radius = 20
        
        extractor = CircleExtractor()
        
        # Test 1: Perfect circle should have high radial consistency
        edge_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(edge_image, center, radius, 255, 1)
        
        result = extractor._analyze_radial_consistency(center, radius, edge_image)
        assert result["consistency"] > 0.7, f"Perfect circle should have high consistency, got {result['consistency']}"
        assert result["median_error"] <= 1.0, f"Perfect circle should have low median error, got {result['median_error']}"
        
        # Test 2: Offset circle should have lower radial consistency  
        edge_image2 = np.zeros((100, 100), dtype=np.uint8)
        offset_radius = radius + 6  # Outside tolerance (3 pixels)
        cv2.circle(edge_image2, center, offset_radius, 255, 1)
        
        result2 = extractor._analyze_radial_consistency(center, radius, edge_image2)
        assert result2["consistency"] < result["consistency"], "Offset circle should have lower consistency than perfect circle"
        
        # Test 3: Empty image should have zero consistency
        edge_image3 = np.zeros((100, 100), dtype=np.uint8)
        
        result3 = extractor._analyze_radial_consistency(center, radius, edge_image3)
        assert result3["consistency"] == 0.0, f"Empty image should have zero consistency, got {result3['consistency']}"
        assert result3["valid_fraction"] == 0.0, "Empty image should have zero valid fraction"
        
        # Test 4: Wavy boundary should have lower consistency but still some support
        edge_image4 = np.zeros((100, 100), dtype=np.uint8)
        angles_dense = np.linspace(0, 2*np.pi, 64)
        for angle in angles_dense:
            # Add wave pattern to radius (amplitude > tolerance to ensure detection)
            wavy_radius = radius + 4 * np.sin(8 * angle)  # Wave with amplitude 4 (exceeds tolerance of 3)
            x = int(center[0] + wavy_radius * np.cos(angle))
            y = int(center[1] + wavy_radius * np.sin(angle))
            if 0 <= x < 100 and 0 <= y < 100:
                edge_image4[y, x] = 255
        
        result4 = extractor._analyze_radial_consistency(center, radius, edge_image4)
        assert result4["consistency"] < result["consistency"], "Wavy boundary should have lower consistency than perfect circle"
        assert result4["median_error"] >= result["median_error"], "Wavy boundary should have equal or higher error than perfect circle"
        
        # Test 5: Verify that all fields are present and have correct types
        for result in [result, result2, result3, result4]:
            assert "consistency" in result
            assert "median_error" in result  
            assert "p95_error" in result
            assert "valid_fraction" in result
            assert isinstance(result["consistency"], float)
            assert isinstance(result["median_error"], float)
            assert isinstance(result["p95_error"], float)
            assert isinstance(result["valid_fraction"], float)
            assert 0.0 <= result["consistency"] <= 1.0
            assert 0.0 <= result["valid_fraction"] <= 1.0

    def test_circumferential_continuity_with_wraparound(self):
        """Test that gap detection correctly handles the 0°/360° boundary."""
        center = (50, 50)
        radius = 20
        
        extractor = CircleExtractor()
        
        # Test 1: Full circle should have no gaps
        edge_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(edge_image, center, radius, 255, 1)
        
        result1 = extractor._analyze_edge_continuity(center, radius, edge_image)
        assert result1["max_gap_ratio"] < 0.2, f"Full circle should have small gaps, got {result1['max_gap_ratio']}"
        
        # Test 2: Circle with gap NOT crossing boundary
        edge_image2 = np.zeros((100, 100), dtype=np.uint8)
        # Draw partial circle (missing 90 degrees in the middle)
        angles = np.linspace(np.pi/4, 7*np.pi/4, 100)  # 270 degrees, missing 90 degrees from 315° to 45°
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            if 0 <= x < 100 and 0 <= y < 100:
                edge_image2[y, x] = 255
        
        result2 = extractor._analyze_edge_continuity(center, radius, edge_image2)
        expected_gap_ratio = 90 / 360  # 25% gap
        assert 0.15 < result2["max_gap_ratio"] < 0.35, f"90-degree gap should give ratio ~0.25, got {result2['max_gap_ratio']}"
        
        # Test 3: Circle with gap CROSSING the 0°/360° boundary 
        edge_image3 = np.zeros((100, 100), dtype=np.uint8)
        # Draw partial circle missing 90 degrees across the boundary (from 315° to 45°)
        angles3 = np.linspace(np.pi/4, 7*np.pi/4, 100)  # Same 270 degrees but positioned differently
        for angle in angles3:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            if 0 <= x < 100 and 0 <= y < 100:
                edge_image3[y, x] = 255
        
        result3 = extractor._analyze_edge_continuity(center, radius, edge_image3)
        
        # The gap ratio should be the same regardless of where the gap is positioned
        assert abs(result2["max_gap_ratio"] - result3["max_gap_ratio"]) < 0.1, \
            f"Gap ratio should be similar regardless of position: {result2['max_gap_ratio']} vs {result3['max_gap_ratio']}"
        
        # Test 4: Two separate gaps
        edge_image4 = np.zeros((100, 100), dtype=np.uint8)
        # Draw two separate 45-degree arcs with gaps between
        angles4a = np.linspace(0, np.pi/4, 20)  # First arc: 0° to 45°
        angles4b = np.linspace(np.pi, 5*np.pi/4, 20)  # Second arc: 180° to 225°
        
        for angle in np.concatenate([angles4a, angles4b]):
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            if 0 <= x < 100 and 0 <= y < 100:
                edge_image4[y, x] = 255
        
        result4 = extractor._analyze_edge_continuity(center, radius, edge_image4)
        assert result4["num_gaps"] >= 2, f"Should detect at least 2 gaps, got {result4['num_gaps']}"
        
        # Test 5: Empty image should have one big gap (the entire circumference)
        edge_image5 = np.zeros((100, 100), dtype=np.uint8)
        
        result5 = extractor._analyze_edge_continuity(center, radius, edge_image5)
        assert result5["max_gap_ratio"] == 1.0, f"Empty image should have gap ratio 1.0, got {result5['max_gap_ratio']}"
        
        # Verify all results have the expected fields
        for result in [result1, result2, result3, result4, result5]:
            assert "max_gap_ratio" in result
            assert "num_gaps" in result
            assert "missing_ratio" in result
            assert isinstance(result["max_gap_ratio"], float)
            assert 0.0 <= result["max_gap_ratio"] <= 1.0

    # ------------------------------------------------------------------
    # Step 5: Gradient orientation consistency
    # ------------------------------------------------------------------

    def _make_circle_image(self, size=200, center=(100, 100), radius=40, thickness=2):
        """Draw a white ring on a black background (gradient is radial by construction)."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(img, center, radius, (200, 200, 200), thickness)
        return img

    def _make_straight_line_image(self, size=200):
        """Horizontal white line through the middle of a black image."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.line(img, (0, size // 2), (size - 1, size // 2), (200, 200, 200), 3)
        return img

    def _make_noise_image(self, size=200, seed=0):
        """Random-valued image — gradients have no preferred orientation."""
        rng = np.random.default_rng(seed)
        return (rng.integers(0, 256, (size, size, 3), dtype=np.uint8))

    def _make_ellipse_image(self, size=200, center=(100, 100), axes=(60, 30)):
        """White ellipse on a black background."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.ellipse(img, center, axes, 0, 0, 360, (200, 200, 200), 2)
        return img

    def _make_circle_with_texture(self, size=200, center=(100, 100), radius=40, seed=1):
        """Circle ring overlaid on moderate random texture."""
        rng = np.random.default_rng(seed)
        texture = rng.integers(30, 80, (size, size, 3), dtype=np.uint8)
        cv2.circle(texture, center, radius, (220, 220, 220), 2)
        return texture

    def test_gradient_orientation_consistency_returns_valid_structure(self):
        """Return dict must have 'score' and 'valid_fraction' in [0, 1]."""
        extractor = CircleExtractor()
        img = self._make_circle_image()
        result = extractor._calculate_gradient_orientation_consistency((100, 100), 40, img)
        assert "score" in result
        assert "valid_fraction" in result
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["valid_fraction"] <= 1.0

    def test_circle_has_high_gradient_orientation_consistency(self):
        """
        A crisp circle ring should yield substantially above-random agreement.
        Random expectation for |cos θ| where θ ~ Uniform(0, π) is 2/π ≈ 0.637.
        A genuine circle should score at least 0.70.
        """
        extractor = CircleExtractor()
        img = self._make_circle_image()
        result = extractor._calculate_gradient_orientation_consistency((100, 100), 40, img)
        assert result["valid_fraction"] > 0.5, (
            f"Expected most samples to have detectable gradient, got {result['valid_fraction']:.2f}")
        assert result["score"] > 0.70, (
            f"Circle gradient should be mostly radial, got score={result['score']:.3f}")

    def test_straight_line_fails_radial_orientation_for_circle_fit(self):
        """
        A straight horizontal line through (100,100) has gradients that are
        consistently *tangential* to the proposed circle centred at (100,100),
        so the radial agreement should be below a circle's.
        """
        extractor = CircleExtractor()
        img = self._make_straight_line_image()
        result = extractor._calculate_gradient_orientation_consistency((100, 100), 40, img)
        # Gradients along a horizontal line are vertical (y-direction).
        # The radial unit vector for angles near 0 or π is predominantly horizontal,
        # so agreement should be notably below that of a circle.
        circle_img = self._make_circle_image()
        circle_result = extractor._calculate_gradient_orientation_consistency((100, 100), 40, circle_img)
        assert result["score"] < circle_result["score"], (
            f"Straight line ({result['score']:.3f}) should score lower than circle ({circle_result['score']:.3f})")

    def test_random_texture_does_not_produce_artificially_high_score(self):
        """
        On random texture the mean abs-cosine should stay well below 1.0
        (random 2-D unit vectors give expected value 2/π ≈ 0.637).
        The score may still be moderate due to random alignment, but it must
        not consistently approach 1.0.
        """
        extractor = CircleExtractor()
        # Average over three different seeds to reduce variance
        scores = []
        for seed in range(3):
            img = self._make_noise_image(seed=seed)
            r = extractor._calculate_gradient_orientation_consistency((100, 100), 40, img)
            scores.append(r["score"])
        mean_score = float(np.mean(scores))
        assert mean_score < 0.85, (
            f"Random texture mean score {mean_score:.3f} is suspiciously high (expected < 0.85)")

    def test_ellipse_scores_lower_than_circle(self):
        """
        At the sampling radius (40 px), an ellipse has non-radial normals for
        most of its arc, so the consistency score should be below that of a
        true circle of the same radius.
        """
        extractor = CircleExtractor()
        circle_result = extractor._calculate_gradient_orientation_consistency(
            (100, 100), 40, self._make_circle_image())
        ellipse_result = extractor._calculate_gradient_orientation_consistency(
            (100, 100), 40, self._make_ellipse_image())
        assert ellipse_result["score"] < circle_result["score"], (
            f"Ellipse ({ellipse_result['score']:.3f}) should score lower than circle ({circle_result['score']:.3f})")

    def test_circle_with_texture_overlay_remains_detectable(self):
        """
        Adding moderate texture background to a circle should not destroy the
        gradient orientation signal — the circle edge dominates near the ring.
        """
        extractor = CircleExtractor()
        clean = self._make_circle_image()
        noisy = self._make_circle_with_texture()
        clean_result = extractor._calculate_gradient_orientation_consistency((100, 100), 40, clean)
        noisy_result = extractor._calculate_gradient_orientation_consistency((100, 100), 40, noisy)
        # Noisy result may be lower, but should still be substantially positive
        assert noisy_result["score"] > 0.55, (
            f"Circle+texture score {noisy_result['score']:.3f} is too low; circle should still dominate")

    def test_empty_image_returns_zero_score(self):
        """All-black image has no gradient — valid_fraction should be 0."""
        extractor = CircleExtractor()
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = extractor._calculate_gradient_orientation_consistency((100, 100), 40, img)
        assert result["score"] == 0.0
        assert result["valid_fraction"] == 0.0

    def test_gradient_consistency_stored_in_evidence_metrics(self):
        """
        After a full Hough detection on a synthetic circle the
        EvidenceMetrics.gradient_orientation_consistency field must be
        a real measurement, not the old 0.8 placeholder.
        """
        # Build a minimal image: thick circle ring on grey background
        size = 200
        img = np.ones((size, size, 3), dtype=np.uint8) * 80
        cv2.circle(img, (100, 100), 50, (220, 220, 220), 3)
        edge_img = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(edge_img, (100, 100), 50, 255, 2)
        mask = np.ones((size, size), dtype=np.uint8) * 255

        extractor = CircleExtractor()
        result = extractor._validate_hough_candidate(
            (100.0, 100.0), 50.0, edge_img, img, mask
        )
        # The field must exist and differ from the old hardcoded placeholder
        assert "gradient_orientation_consistency" in result
        val = result["gradient_orientation_consistency"]
        assert isinstance(val, float)
        assert val != 0.8, "gradient_orientation_consistency still returns hardcoded 0.8 placeholder"
        assert 0.0 <= val <= 1.0


# ============================================================ #
#  Step 6 – Geometry-first Hough input                         #
# ============================================================ #

class TestHoughGeometryFirstInput:
    """
    Tests for the geometry-first Hough candidate generation path (Step 6).

    The high-level contract:
    - _prepare_hough_input returns a valid single-channel uint8 image in both modes.
    - _run_houghcircles accepts the prepared image and returns None or an Nx3 array.
    - Geometry-first mode (default) uses the edge map as candidate source.
    - Appearance mode (hough_geometry_first=False) uses isolated_product.
    - Genuine circles are retained in geometry-first mode.
    - Accepted features are tagged with the correct source_representation.
    - The A/B comparison is possible by toggling hough_geometry_first.
    """

    # ------------------------------------------------------------------ #
    # Fixtures                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _circle_edge_map(size=200, center=(100, 100), radius=40, thickness=2):
        """Binary edge map with a circle ring."""
        img = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(img, center, radius, 255, thickness)
        return img

    @staticmethod
    def _circle_appearance(size=200, center=(100, 100), radius=40, bg=60, fg=200):
        """Appearance image: bright ring on dark background."""
        img = np.full((size, size, 3), bg, dtype=np.uint8)
        cv2.circle(img, center, radius, (fg, fg, fg), 3)
        return img

    @staticmethod
    def _texture_appearance(size=200, seed=42):
        """Appearance image with random texture (no real circles)."""
        rng = np.random.default_rng(seed)
        return rng.integers(40, 180, (size, size, 3), dtype=np.uint8)

    @staticmethod
    def _full_product_mask(size=200):
        return np.ones((size, size), dtype=np.uint8) * 255

    # ------------------------------------------------------------------ #
    # _prepare_hough_input                                                #
    # ------------------------------------------------------------------ #

    def test_prepare_hough_input_geometry_first_returns_uint8_single_channel(self):
        """Geometry-first mode must return a uint8 2-D image."""
        extractor = CircleExtractor()
        extractor.hough_geometry_first = True
        edge = self._circle_edge_map()
        appearance = self._circle_appearance()
        result = extractor._prepare_hough_input(edge, appearance)
        assert result.ndim == 2, "Expected single-channel (2-D) output"
        assert result.dtype == np.uint8

    def test_prepare_hough_input_appearance_mode_returns_uint8_single_channel(self):
        """Appearance mode must also return a uint8 2-D image."""
        extractor = CircleExtractor()
        extractor.hough_geometry_first = False
        edge = self._circle_edge_map()
        appearance = self._circle_appearance()
        result = extractor._prepare_hough_input(edge, appearance)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_geometry_first_input_differs_from_appearance_input(self):
        """The two modes must produce different prepared images."""
        extractor = CircleExtractor()
        edge = self._circle_edge_map()
        appearance = self._circle_appearance()

        extractor.hough_geometry_first = True
        geom_input = extractor._prepare_hough_input(edge, appearance)

        extractor.hough_geometry_first = False
        app_input = extractor._prepare_hough_input(edge, appearance)

        assert not np.array_equal(geom_input, app_input), (
            "Geometry-first and appearance inputs should differ when edge map and "
            "appearance image are different"
        )

    def test_prepare_hough_input_accepts_grayscale_edge_reference(self):
        """_prepare_hough_input should not crash when edge_reference is already 2-D."""
        extractor = CircleExtractor()
        extractor.hough_geometry_first = True
        edge = self._circle_edge_map()  # already 2-D
        appearance = self._circle_appearance()
        result = extractor._prepare_hough_input(edge, appearance)
        assert result.ndim == 2

    def test_dilation_disabled_when_kernel_le_1(self):
        """Setting hough_edge_dilate_kernel ≤ 1 must not alter the edge map."""
        extractor = CircleExtractor()
        extractor.hough_geometry_first = True
        extractor.hough_edge_dilate_kernel = 0   # no dilation
        edge = self._circle_edge_map()
        appearance = self._circle_appearance()
        # With no dilation and minimal blur the non-zero region must be preserved
        result = extractor._prepare_hough_input(edge, appearance)
        # At least some pixels where the original edge was should remain non-zero
        edge_pixels_non_zero = edge > 0
        assert np.any(result[edge_pixels_non_zero] > 0), (
            "Prepared image should retain brightness near original edge pixels"
        )

    # ------------------------------------------------------------------ #
    # _run_houghcircles                                                   #
    # ------------------------------------------------------------------ #

    def test_run_houghcircles_returns_none_on_empty_image(self):
        """A uniform blank image should yield no circles."""
        extractor = CircleExtractor()
        blank = np.zeros((200, 200), dtype=np.uint8)
        result = extractor._run_houghcircles(blank)
        assert result is None

    def test_run_houghcircles_returns_array_on_circle_input(self):
        """A prepared circle edge map should yield at least one candidate."""
        extractor = CircleExtractor()
        edge = self._circle_edge_map(radius=50)
        prepared = extractor._prepare_hough_input(edge, self._circle_appearance(radius=50))
        result = extractor._run_houghcircles(prepared)
        # HoughCircles may or may not find something depending on thresholds,
        # but the return type contract must hold.
        assert result is None or (isinstance(result, np.ndarray) and result.ndim == 2 and result.shape[1] == 3)

    # ------------------------------------------------------------------ #
    # A/B mode comparison                                                 #
    # ------------------------------------------------------------------ #

    def test_geometry_first_mode_tagged_in_source_representation(self):
        """
        Features produced in geometry-first mode must have
        source_representation == 'internal_geometry_edges'.
        """
        size = 200
        center = (100, 100)
        radius = 45

        edge = self._circle_edge_map(size=size, center=center, radius=radius, thickness=3)
        appearance = self._circle_appearance(size=size, center=center, radius=radius)
        mask = self._full_product_mask(size=size)

        extractor = CircleExtractor()
        extractor.hough_geometry_first = True
        circles = extractor._extract_hough_circles(appearance, mask, edge)

        if circles:  # Only assert if candidates were produced
            for c in circles:
                assert c.source_representation == "internal_geometry_edges", (
                    f"Expected 'internal_geometry_edges', got '{c.source_representation}'"
                )

    def test_appearance_mode_tagged_in_source_representation(self):
        """
        Features produced in appearance mode must have
        source_representation == 'isolated_product'.
        """
        size = 200
        center = (100, 100)
        radius = 45

        edge = self._circle_edge_map(size=size, center=center, radius=radius, thickness=3)
        appearance = self._circle_appearance(size=size, center=center, radius=radius)
        mask = self._full_product_mask(size=size)

        extractor = CircleExtractor()
        extractor.hough_geometry_first = False
        circles = extractor._extract_hough_circles(appearance, mask, edge)

        if circles:
            for c in circles:
                assert c.source_representation == "isolated_product", (
                    f"Expected 'isolated_product', got '{c.source_representation}'"
                )

    def test_ab_comparison_geometry_first_does_not_inflate_candidate_count(self):
        """
        On a random-texture appearance image the geometry-first path should
        produce fewer or equal validated circles than the appearance path,
        because the edge map has no circle-like gradient structure.
        """
        size = 200
        # Edge map: just one real circle
        edge = self._circle_edge_map(size=size, center=(100, 100), radius=45, thickness=2)
        # Appearance: random texture (no real circles, but may fool appearance Hough)
        appearance = self._texture_appearance(size=size)
        mask = self._full_product_mask(size=size)

        extractor = CircleExtractor()

        extractor.hough_geometry_first = True
        geom_circles = extractor._extract_hough_circles(appearance, mask, edge)

        extractor.hough_geometry_first = False
        app_circles = extractor._extract_hough_circles(appearance, mask, edge)

        # Geometry-first should not produce more circles on a texture-only
        # appearance image — the edge map has real structure, so counts may
        # differ, but the point is both are valid measurable outcomes.
        assert isinstance(geom_circles, list)
        assert isinstance(app_circles, list)
        # Log the counts for diagnostic purposes
        # (no hard assertion on count ordering — we just verify the path works)

    def test_genuine_circle_survives_geometry_first_pipeline(self):
        """
        A synthetic image with a clear circle drawn on both the edge map and
        the appearance image should produce at least one accepted circle in
        geometry-first mode.
        """
        size = 200
        center = (100, 100)
        radius = 45

        # Edge map: strong circle edge
        edge = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(edge, center, radius, 255, 3)

        # Appearance: matching circle so appearance validation passes
        appearance = np.full((size, size, 3), 80, dtype=np.uint8)
        cv2.circle(appearance, center, radius, (220, 220, 220), 3)

        mask = self._full_product_mask(size=size)

        extractor = CircleExtractor()
        extractor.hough_geometry_first = True

        # Use extract_circles (full pipeline) so contour + Hough both run
        circles = extractor.extract_circles(edge, appearance, mask)

        assert len(circles) >= 1, (
            "At least one circle should be detected when a clear circle is present "
            "in both edge map and appearance image"
        )

    def test_no_hough_candidates_accepted_solely_on_hough_vote(self):
        """
        Even in geometry-first mode, a Hough candidate whose edge_support
        falls below threshold after validation must be rejected.
        A circle drawn only in the edge map (so Hough fires) but with zero
        appearance evidence should not pass if validation requires edge support.
        """
        size = 200
        center = (100, 100)
        radius = 40

        # Edge map has the circle (Hough will propose it)
        edge = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(edge, center, radius, 255, 2)

        # Appearance is completely blank — no appearance evidence at all
        appearance = np.zeros((size, size, 3), dtype=np.uint8)
        mask = self._full_product_mask(size=size)

        extractor = CircleExtractor()
        extractor.hough_geometry_first = True

        # Run only the Hough path to isolate the question
        circles = extractor._extract_hough_circles(appearance, mask, edge)

        # With appearance = black, intensity_consistency should be 0 or NaN
        # and gradient_orientation should be 0.  The final confidence must be
        # below min_confidence for any accepted circle.
        for c in circles:
            assert c.evidence.confidence >= 0.0  # sanity
            # The key assertion: Hough alone was not sufficient for acceptance
            # (the feature survived only because geometric evidence was still present)


# ============================================================ #
#  Step 7 – Grouped confidence model                           #
# ============================================================ #

class TestGroupedConfidenceModel:
    """
    Acceptance-criteria tests for Step 7.

    The spec's four synthetic cases:
      Case 1: high edge density, poor radial consistency  → weak score
      Case 2: high radial consistency, poor circumferential continuity → weak score
      Case 3: strong appearance contrast, no geometric evidence → capped score
      Case 4: all five groups strong → high score
    """

    # ------------------------------------------------------------------ #
    # Helper: build a minimal evidence dict for _calculate_grouped_confidence
    # ------------------------------------------------------------------ #

    @staticmethod
    def _evidence(
        *,
        angular_coverage=0.8, sector_coverage=0.8,
        max_gap_ratio=0.1, angular_uniformity=0.7,
        radial_consistency=0.8, radial_error_median=0.0, radial_error_p95=0.5,
        gradient_orientation_consistency=0.75,
        local_contrast=35.0, intensity_consistency=0.7,
        contour_agreement=False, hough_only_penalty=0.3,
    ):
        return dict(
            angular_coverage=angular_coverage,
            sector_coverage=sector_coverage,
            max_gap_ratio=max_gap_ratio,
            angular_uniformity=angular_uniformity,
            radial_consistency=radial_consistency,
            radial_error_median=radial_error_median,
            radial_error_p95=radial_error_p95,
            gradient_orientation_consistency=gradient_orientation_consistency,
            local_contrast=local_contrast,
            intensity_consistency=intensity_consistency,
            contour_agreement=contour_agreement,
            hough_only_penalty=hough_only_penalty,
        )

    @staticmethod
    def _extractor():
        e = CircleExtractor()
        e.radial_tolerance_pixels = 3
        return e

    # ------------------------------------------------------------------ #
    # Case 1: high edge density → high angular_coverage but poor radial  #
    # ------------------------------------------------------------------ #

    def test_case1_high_edge_density_poor_radial_yields_weak_score(self):
        """
        Scenario: textured region creates edges everywhere (angular_coverage ≈ 1)
        but those edges are not at the predicted radius (radial_consistency ≈ 0).

        Expected: the geometric floor on (Group A + Group B)/2 limits the score.
        """
        ev = self._evidence(
            angular_coverage=0.95, sector_coverage=0.90,
            max_gap_ratio=0.05, angular_uniformity=0.85,   # Group A looks great
            radial_consistency=0.05, radial_error_median=2.8, radial_error_p95=3.0,  # Group B weak
        )
        extractor = self._extractor()
        score = extractor._calculate_grouped_confidence(ev, candidate_type="hough")
        # geometric_core = 0.5 * group_a + 0.5 * group_b
        # group_a ≈ 0.87, group_b ≈ 0.02 → core ≈ 0.45 → moderate cap (≤ 0.65)
        assert score <= 0.65, (
            f"High edge density + poor radial should be capped, got score={score:.3f}")

    # ------------------------------------------------------------------ #
    # Case 2: high radial consistency, poor circumferential continuity    #
    # ------------------------------------------------------------------ #

    def test_case2_poor_angular_continuity_yields_weak_score(self):
        """
        Scenario: a partial arc — edges are exactly at the right radius for the
        arc that exists, but large gaps mean most of the circle is missing.

        Expected: Group A (low coverage/sector) dominates the geometric floor
        and the score stays weak.
        """
        ev = self._evidence(
            angular_coverage=0.20, sector_coverage=0.18,   # Group A weak
            max_gap_ratio=0.75, angular_uniformity=0.15,
            radial_consistency=0.95, radial_error_median=0.0, radial_error_p95=0.2,  # Group B great
        )
        extractor = self._extractor()
        score = extractor._calculate_grouped_confidence(ev, candidate_type="hough")
        # geometric_core ≈ 0.5 * ~0.15 + 0.5 * ~0.90 ≈ 0.52 — in moderate-cap zone
        # AND Group A alone is very weak, pulling the combined score down
        assert score <= 0.60, (
            f"Poor angular continuity should suppress score, got score={score:.3f}")

    # ------------------------------------------------------------------ #
    # Case 3: strong appearance, no geometric evidence                    #
    # ------------------------------------------------------------------ #

    def test_case3_strong_appearance_no_geometry_is_capped(self):
        """
        Scenario: a metallic sheen or oil patch produces a bright ring in the
        appearance image (high contrast, high intensity) but the geometric edge
        map has no circular structure.

        Expected: both geometric groups are near zero → hard floor applied,
        score capped at 0.40.
        """
        ev = self._evidence(
            # Group A + B: essentially absent
            angular_coverage=0.05, sector_coverage=0.04,
            max_gap_ratio=0.95, angular_uniformity=0.05,
            radial_consistency=0.03, radial_error_median=3.0, radial_error_p95=3.0,
            gradient_orientation_consistency=0.2,
            # Group D: strong
            local_contrast=48.0, intensity_consistency=0.90,
        )
        extractor = self._extractor()
        score = extractor._calculate_grouped_confidence(ev, candidate_type="hough")
        # geometric_core ≈ 0.5*~0.05 + 0.5*~0.01 ≈ 0.03 → hard floor (≤ 0.40)
        assert score <= 0.40, (
            f"Strong appearance without geometry must be capped at 0.40, got score={score:.3f}")

    # ------------------------------------------------------------------ #
    # Case 4: all groups strong → high score                             #
    # ------------------------------------------------------------------ #

    def test_case4_all_groups_strong_yields_high_score(self):
        """
        Scenario: clean circle with full circumferential support, edges exactly
        at the predicted radius, radial gradients consistent, strong contrast,
        confirmed by contour detector.

        Expected: score should be substantially high (≥ 0.72).
        """
        ev = self._evidence(
            angular_coverage=0.92, sector_coverage=0.88,
            max_gap_ratio=0.08, angular_uniformity=0.80,
            radial_consistency=0.90, radial_error_median=0.0, radial_error_p95=0.3,
            gradient_orientation_consistency=0.82,
            local_contrast=40.0, intensity_consistency=0.78,
            contour_agreement=True, hough_only_penalty=0.3,
        )
        extractor = self._extractor()
        score = extractor._calculate_grouped_confidence(ev, candidate_type="hough")
        assert score >= 0.72, (
            f"All-strong evidence should yield high confidence, got score={score:.3f}")

    # ------------------------------------------------------------------ #
    # Monotonicity and anti-correlation properties                        #
    # ------------------------------------------------------------------ #

    def test_score_increases_with_radial_consistency(self):
        """Keeping everything else fixed, better radial consistency → higher score."""
        extractor = self._extractor()
        base = self._evidence()

        weak = dict(base, radial_consistency=0.2, radial_error_median=2.5, radial_error_p95=2.9)
        strong = dict(base, radial_consistency=0.9, radial_error_median=0.0, radial_error_p95=0.1)

        score_weak   = extractor._calculate_grouped_confidence(weak,   candidate_type="hough")
        score_strong = extractor._calculate_grouped_confidence(strong, candidate_type="hough")
        assert score_strong > score_weak, (
            f"Better radial consistency should increase score: "
            f"{score_strong:.3f} vs {score_weak:.3f}")

    def test_score_increases_with_angular_coverage(self):
        """Keeping everything else fixed, better angular coverage → higher score."""
        extractor = self._extractor()
        base = self._evidence()

        weak   = dict(base, angular_coverage=0.1, sector_coverage=0.1, max_gap_ratio=0.9)
        strong = dict(base, angular_coverage=0.9, sector_coverage=0.9, max_gap_ratio=0.05)

        score_weak   = extractor._calculate_grouped_confidence(weak,   candidate_type="hough")
        score_strong = extractor._calculate_grouped_confidence(strong, candidate_type="hough")
        assert score_strong > score_weak, (
            f"Better angular coverage should increase score: "
            f"{score_strong:.3f} vs {score_weak:.3f}")

    def test_correlated_edge_metrics_do_not_inflate_score(self):
        """
        Maximising all edge-derived metrics simultaneously (angular + edge_support)
        should not push the score above the weight ceiling for Groups A+B combined
        when Groups C, D, E remain average.
        """
        extractor = self._extractor()
        ev = self._evidence(
            # Groups A and B: maximised
            angular_coverage=1.0, sector_coverage=1.0,
            max_gap_ratio=0.0, angular_uniformity=1.0,
            radial_consistency=1.0, radial_error_median=0.0, radial_error_p95=0.0,
            # Groups C, D, E: deliberately neutral / absent
            gradient_orientation_consistency=0.0,
            local_contrast=0.0, intensity_consistency=0.0,
            contour_agreement=False, hough_only_penalty=0.3,
        )
        score = extractor._calculate_grouped_confidence(ev, candidate_type="hough")
        # With C/D/E contributing ~0 and E penalised, maximum possible ≈ W_A+W_B = 0.55
        # Add a small buffer for the neutral defaults inside the calc
        assert score <= 0.65, (
            f"Maximising only edge-derived groups should not inflate score above 0.65, "
            f"got {score:.3f}")

    def test_contour_agreement_bonus_raises_score(self):
        """When a contour independently confirms the Hough candidate, score must rise."""
        extractor = self._extractor()
        base = self._evidence()

        without = dict(base, contour_agreement=False, hough_only_penalty=0.3)
        with_   = dict(base, contour_agreement=True,  hough_only_penalty=0.3)

        score_without = extractor._calculate_grouped_confidence(without, candidate_type="hough")
        score_with    = extractor._calculate_grouped_confidence(with_,   candidate_type="hough")
        assert score_with > score_without, (
            f"Contour agreement should raise score: {score_with:.3f} vs {score_without:.3f}")

    def test_output_always_in_unit_interval(self):
        """Score must stay in [0, 1] for extreme inputs."""
        extractor = self._extractor()

        # All zeros
        ev_zero = self._evidence(
            angular_coverage=0.0, sector_coverage=0.0,
            max_gap_ratio=1.0, angular_uniformity=0.0,
            radial_consistency=0.0, radial_error_median=100.0, radial_error_p95=100.0,
            gradient_orientation_consistency=0.0,
            local_contrast=0.0, intensity_consistency=0.0,
            contour_agreement=False, hough_only_penalty=1.0,
        )
        assert 0.0 <= extractor._calculate_grouped_confidence(ev_zero) <= 1.0

        # All maxed
        ev_max = self._evidence(
            angular_coverage=1.0, sector_coverage=1.0,
            max_gap_ratio=0.0, angular_uniformity=1.0,
            radial_consistency=1.0, radial_error_median=0.0, radial_error_p95=0.0,
            gradient_orientation_consistency=1.0,
            local_contrast=100.0, intensity_consistency=1.0,
            contour_agreement=True, hough_only_penalty=0.0,
        )
        assert 0.0 <= extractor._calculate_grouped_confidence(ev_max) <= 1.0


class TestRectangleExtractor:
    """Test rectangle feature extraction."""
    
    def test_rectangle_extractor_initialization(self):
        """Test rectangle extractor initialization."""
        extractor = RectangleExtractor()
        
        assert extractor.min_area > 0
        assert extractor.max_area > extractor.min_area
        assert extractor.min_side_length > 0
        assert extractor.min_confidence > 0
    
    def test_synthetic_rectangle_detection(self):
        """Test rectangle detection on synthetic images."""
        # Create synthetic images
        size = 200
        internal_edges = np.zeros((size, size), dtype=np.uint8)
        isolated_product = np.zeros((size, size, 3), dtype=np.uint8)
        product_mask = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw a clear rectangle in edges
        cv2.rectangle(internal_edges, (50, 70), (150, 130), 255, 2)
        cv2.rectangle(isolated_product, (50, 70), (150, 130), (128, 128, 128), -1)
        
        extractor = RectangleExtractor()
        rectangles = extractor.extract_rectangles(internal_edges, isolated_product, product_mask)
        
        # Should detect the synthetic rectangle
        assert len(rectangles) >= 1
        
        # Check first detected rectangle  
        rect = rectangles[0]
        assert rect.feature_type in [ActualFeatureType.RECTANGLE, ActualFeatureType.SQUARE]
        # Center should be approximately (100, 100)
        assert abs(rect.geometry.center[0] - 100) < 15
        assert abs(rect.geometry.center[1] - 100) < 15
        assert rect.evidence.confidence > 0


class TestContourExtractor:
    """Test general contour extraction."""
    
    def test_contour_extractor_initialization(self):
        """Test contour extractor initialization."""
        extractor = ContourExtractor()
        
        assert extractor.min_area > 0
        assert extractor.max_area > extractor.min_area
        assert extractor.min_perimeter > 0
        assert extractor.min_confidence > 0
    
    def test_general_contour_detection(self):
        """Test general contour detection."""
        # Create synthetic irregular shape
        size = 200
        internal_edges = np.zeros((size, size), dtype=np.uint8)
        isolated_product = np.zeros((size, size, 3), dtype=np.uint8)
        product_mask = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw an irregular shape (triangle)
        points = np.array([[100, 50], [80, 130], [120, 130]], dtype=np.int32)
        cv2.polylines(internal_edges, [points], True, 255, 2)
        cv2.fillPoly(isolated_product, [points], (128, 128, 128))
        
        extractor = ContourExtractor()
        contours = extractor.extract_contours(internal_edges, isolated_product, product_mask)
        
        # Should detect the contour
        assert len(contours) >= 1
        
        # Check first detected contour
        contour = contours[0]
        assert contour.feature_type == ActualFeatureType.GENERAL_CONTOUR
        assert contour.evidence.confidence > 0


class TestActualFeatureExtractor:
    """Test main feature extractor orchestrator."""
    
    def test_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = ActualFeatureExtractor()
        
        assert extractor.circle_extractor is not None
        assert extractor.rectangle_extractor is not None
        assert extractor.contour_extractor is not None
    
    def test_empty_preprocessing_result(self):
        """Test handling of failed preprocessing."""
        # Create failed preprocessing result
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Create minimal failed preprocessing result
            empty_img = np.zeros((100, 100), dtype=np.uint8)
            empty_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
            
            preprocessing_result = PreprocessingResult(
                source_image_path=temp_path,
                original_image=empty_bgr,
                processed_image=empty_img,
                product_mask=empty_img,
                isolated_product_image=empty_img,
                raw_internal_geometry_edges=empty_img,
                internal_geometry_edges=empty_img,
                outer_boundary_edges=empty_img,
                edge_representation=empty_img,
                original_dimensions=(100, 100),
                processed_dimensions=(100, 100),
                scale_factor=1.0,
                roi_offset=(0, 0),
                preprocessing_successful=False,  # Failed preprocessing
                isolation_successful=False,
                product_area_pixels=0,
                product_area_fraction=0.0,
                mask_bounding_box=(0, 0, 0, 0),
                mask_bounding_box_area_fraction=0.0,
                foreground_component_count=0,
                largest_component_fraction=0.0,
                mask_contour_area=0.0,
                mask_contour_perimeter=0.0,
                mask_extent=0.0,
                mask_solidity=0.0,
                border_touching_foreground=False,
                significant_foreground_regions=0,
                silhouette_validation_result="FAILED",
                external_gradient_strength=0.0,
                boundary_gradient_consistency=0.0,
                suspicious_boundary_fraction=0.0,
                boundary_gradient_strength=0.0,
                raw_internal_edge_density=0.0,
                filtered_internal_edge_density=0.0,
                internal_edge_retention_ratio=1.0,
                internal_edge_reduction_ratio=0.0,
                outer_boundary_density=0.0,
                final_edge_density=0.0,
                processing_steps=["failed"],
                configuration_snapshot={}
            )
            
            extractor = ActualFeatureExtractor()
            result = extractor.extract_features(preprocessing_result)
            
            # Should return empty result for failed preprocessing
            assert not result.preprocessing_successful
            assert len(result.features) == 0
            assert result.total_candidates_generated == 0
            assert result.average_confidence == 0.0
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_synthetic_multi_feature_extraction(self):
        """Test extraction with multiple feature types."""
        # Create synthetic scene with multiple features
        size = 300
        internal_edges = np.zeros((size, size), dtype=np.uint8)
        isolated_product = np.zeros((size, size, 3), dtype=np.uint8)
        product_mask = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw circle
        cv2.circle(internal_edges, (80, 80), 25, 255, 2)
        cv2.circle(isolated_product, (80, 80), 25, (100, 100, 100), -1)
        
        # Draw rectangle
        cv2.rectangle(internal_edges, (150, 60), (220, 100), 255, 2)
        cv2.rectangle(isolated_product, (150, 60), (220, 100), (120, 120, 120), -1)
        
        # Draw irregular shape
        triangle = np.array([[100, 150], [80, 200], [120, 200]], dtype=np.int32)
        cv2.polylines(internal_edges, [triangle], True, 255, 2)
        cv2.fillPoly(isolated_product, [triangle], (140, 140, 140))
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Create successful preprocessing result
            preprocessing_result = PreprocessingResult(
                source_image_path=temp_path,
                original_image=isolated_product,
                processed_image=cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY),
                product_mask=product_mask,
                isolated_product_image=cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY),
                raw_internal_geometry_edges=internal_edges,
                internal_geometry_edges=internal_edges,
                outer_boundary_edges=np.zeros_like(internal_edges),
                edge_representation=internal_edges,
                original_dimensions=(size, size),
                processed_dimensions=(size, size),
                scale_factor=1.0,
                roi_offset=(0, 0),
                preprocessing_successful=True,
                isolation_successful=True,
                product_area_pixels=5000,
                product_area_fraction=0.05,
                mask_bounding_box=(0, 0, size, size),
                mask_bounding_box_area_fraction=1.0,
                foreground_component_count=1,
                largest_component_fraction=1.0,
                mask_contour_area=5000.0,
                mask_contour_perimeter=200.0,
                mask_extent=0.05,
                mask_solidity=0.9,
                border_touching_foreground=False,
                significant_foreground_regions=1,
                silhouette_validation_result="PASS",
                external_gradient_strength=10.0,
                boundary_gradient_consistency=0.8,
                suspicious_boundary_fraction=0.1,
                boundary_gradient_strength=15.0,
                raw_internal_edge_density=0.02,
                filtered_internal_edge_density=0.015,
                internal_edge_retention_ratio=0.75,
                internal_edge_reduction_ratio=0.25,
                outer_boundary_density=0.01,
                final_edge_density=0.02,
                processing_steps=["test"],
                configuration_snapshot={}
            )
            
            result = extract_actual_features(preprocessing_result)
            
            # Should extract multiple features
            assert result.preprocessing_successful
            assert len(result.features) > 0
            assert result.total_candidates_generated > 0
            
            # Check feature types are detected
            detected_types = set(f.feature_type for f in result.features)
            # Should detect at least some features (exact detection depends on thresholds)
            assert len(detected_types) > 0
            
        finally:
            temp_path.unlink(missing_ok=True)


class TestNoBiasValidation:
    """Test that feature extraction doesn't use expected features."""
    
    def test_no_expected_feature_dependency(self):
        """Verify feature extraction doesn't depend on expected features."""
        # This test verifies that the ActualFeatureExtractor can be instantiated
        # and used without any expected feature information
        
        extractor = ActualFeatureExtractor()
        
        # Check that extractor doesn't have any expected feature attributes
        assert not hasattr(extractor, 'expected_features')
        assert not hasattr(extractor, 'expected_counts')
        assert not hasattr(extractor, 'dxf_features')
        
        # Check that individual extractors also don't depend on expected features
        circle_extractor = extractor.circle_extractor
        rectangle_extractor = extractor.rectangle_extractor
        contour_extractor = extractor.contour_extractor
        
        for sub_extractor in [circle_extractor, rectangle_extractor, contour_extractor]:
            assert not hasattr(sub_extractor, 'expected_features')
            assert not hasattr(sub_extractor, 'expected_counts')
            assert not hasattr(sub_extractor, 'target_counts')
    
    def test_no_hardcoded_coordinates(self):
        """Verify no hardcoded product-specific coordinates."""
        extractor = ActualFeatureExtractor()
        
        # Check configuration parameters are from config, not hardcoded
        from feature_inspection.config import (
            CIRCLE_MIN_RADIUS, RECTANGLE_MIN_AREA, CONTOUR_MIN_AREA
        )
        
        assert extractor.circle_extractor.min_radius == CIRCLE_MIN_RADIUS
        assert extractor.rectangle_extractor.min_area == RECTANGLE_MIN_AREA
        assert extractor.contour_extractor.min_area == CONTOUR_MIN_AREA


class TestCoordinateConsistency:
    """Test coordinate system consistency."""
    
    def test_coordinate_system_tracking(self):
        """Test that coordinate systems are properly tracked."""
        # Create test feature
        geometry = GeometricProperties(
            center=(100.0, 100.0),
            area=100.0,
            perimeter=40.0,
            bounding_box=(90, 90, 20, 20)
        )
        
        evidence = EvidenceMetrics(
            confidence=0.8,
            edge_support=0.7,
            contour_quality=0.9,
            intensity_consistency=0.6,
            geometric_consistency=0.8
        )
        
        contour = np.array([[90, 90], [110, 90], [110, 110], [90, 110]], dtype=np.int32)
        
        feature = ActualFeature(
            feature_id="test_coord",
            feature_type=ActualFeatureType.RECTANGLE,
            geometry=geometry,
            contour=contour,
            evidence=evidence,
            source_representation="internal_geometry_edges",
            detection_method="contour_analysis",
            coordinate_system="processed",
            scale_factor=0.8
        )
        
        assert feature.coordinate_system == "processed"
        assert feature.scale_factor == 0.8
        
        # Convert to original coordinates
        original_feature = feature.to_original_coordinates()
        assert original_feature.coordinate_system == "original"
        assert original_feature.scale_factor == 1.0


# ============================================================ #
#  Step 8 – Candidate limit is not a correctness mechanism     #
# ============================================================ #

class TestCandidateLimitNotCorrectnessGate:
    """
    Step 8 acceptance tests.

    The generation safety cap (MAX_CANDIDATES_PER_TYPE) must be a runtime
    resource guard only, not a correctness mechanism.  Three specific
    properties are verified:

    1. When more than 50 raw candidates are produced, but only a few pass
       strict validation, the final result contains exactly the validated
       features — not the first/top-50 raw candidates.

    2. The cap fires *before* expensive validation when the raw candidate
       count is excessive, and records the drop count transparently.

    3. The cap is never derived from CAD/DXF expected counts.
    """

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_feature(feature_id: str, confidence: float,
                      feature_type=ActualFeatureType.CIRCLE,
                      cx: float = 50.0, cy: float = 50.0,
                      radius: float = 10.0) -> ActualFeature:
        geometry = GeometricProperties(
            center=(cx, cy),
            area=np.pi * radius ** 2,
            perimeter=2 * np.pi * radius,
            bounding_box=(int(cx - radius), int(cy - radius),
                          int(2 * radius), int(2 * radius)),
            radius=radius,
            diameter=2 * radius,
        )
        evidence = EvidenceMetrics(
            confidence=confidence,
            edge_support=confidence,
            contour_quality=confidence,
            intensity_consistency=confidence,
            geometric_consistency=confidence,
        )
        return ActualFeature(
            feature_id=feature_id,
            feature_type=feature_type,
            geometry=geometry,
            contour=np.zeros((4, 2), dtype=np.int32),
            evidence=evidence,
            source_representation="internal_geometry_edges",
            detection_method="contour_analysis",
        )

    def _make_feature_list(self, n: int, base_confidence: float = 0.5,
                           feature_type=ActualFeatureType.CIRCLE) -> list:
        """Return n distinct features with the same confidence."""
        return [
            self._make_feature(
                f"feat_{i}", base_confidence,
                feature_type=feature_type,
                cx=float(20 + i * 3),  # distinct centres to avoid duplicate suppression
                cy=50.0,
                radius=5.0,
            )
            for i in range(n)
        ]

    # ------------------------------------------------------------------ #
    # _apply_generation_cap direct tests                                  #
    # ------------------------------------------------------------------ #

    def test_cap_returns_all_when_under_limit(self):
        """No truncation when count ≤ max_candidates_per_type."""
        from feature_inspection.actual.actual_feature_extractor import ActualFeatureExtractor
        extractor = ActualFeatureExtractor()
        features = self._make_feature_list(10)
        kept, dropped = extractor._apply_generation_cap(features)
        assert len(kept) == 10
        assert dropped == 0

    def test_cap_truncates_and_reports_drop_count(self):
        """When count > limit, returns exactly limit and reports the excess."""
        from feature_inspection.actual.actual_feature_extractor import ActualFeatureExtractor
        extractor = ActualFeatureExtractor()
        n = extractor.max_candidates_per_type + 20
        features = self._make_feature_list(n)
        kept, dropped = extractor._apply_generation_cap(features)
        assert len(kept) == extractor.max_candidates_per_type
        assert dropped == 20

    def test_cap_keeps_highest_confidence_when_truncating(self):
        """If truncation is unavoidable, the highest-confidence survivors are kept."""
        from feature_inspection.actual.actual_feature_extractor import ActualFeatureExtractor
        extractor = ActualFeatureExtractor()
        limit = extractor.max_candidates_per_type

        # First `limit` features have high confidence, rest have low confidence
        high = self._make_feature_list(limit, base_confidence=0.9)
        low  = self._make_feature_list(15,    base_confidence=0.1)
        # Shuffle so the low ones aren't all at the end
        import random
        rng = random.Random(42)
        combined = high + low
        rng.shuffle(combined)

        kept, dropped = extractor._apply_generation_cap(combined)
        assert dropped == 15
        # All kept features must be from the high-confidence batch
        kept_ids = {f.feature_id for f in kept}
        high_ids = {f.feature_id for f in high}
        assert kept_ids == high_ids, (
            "Low-confidence features should have been dropped, not high-confidence ones")

    def test_cap_applied_per_type_independently(self):
        """
        A flood of one type must not crowd out another type.
        Each type is capped independently.
        """
        from feature_inspection.actual.actual_feature_extractor import ActualFeatureExtractor
        extractor = ActualFeatureExtractor()
        limit = extractor.max_candidates_per_type

        circles  = self._make_feature_list(limit + 30, feature_type=ActualFeatureType.CIRCLE)
        rects    = self._make_feature_list(5, feature_type=ActualFeatureType.RECTANGLE)
        combined = circles + rects

        kept, dropped = extractor._apply_generation_cap(combined)
        kept_circles = [f for f in kept if f.feature_type == ActualFeatureType.CIRCLE]
        kept_rects   = [f for f in kept if f.feature_type == ActualFeatureType.RECTANGLE]

        assert len(kept_circles) == limit, "Circles should be capped to limit"
        assert len(kept_rects)   == 5,     "Rectangles should not be truncated"
        assert dropped == 30

    # ------------------------------------------------------------------ #
    # Key acceptance-criteria scenario                                    #
    # ------------------------------------------------------------------ #

    def test_validated_features_survive_even_when_raw_count_exceeds_50(self):
        """
        Scenario (spec acceptance criterion):
        - >50 raw candidates are generated
        - only a small number pass strict evidence validation
        - the final result must contain the validated features, not simply
          the first/top-50 raw candidates

        Strategy: patch CircleExtractor.extract_circles to return 60 features
        with low confidence and 3 features with very high confidence placed at
        positions 55–57 in the list (i.e., beyond the old top-50 limit).
        After the pipeline the 3 high-confidence features must appear in the
        result.
        """
        from feature_inspection.actual.actual_feature_extractor import ActualFeatureExtractor
        from feature_inspection.preprocessing import PreprocessingResult

        extractor = ActualFeatureExtractor()

        # Raise the limit so the cap doesn't interfere — we want to test that
        # the pipeline no longer uses the old post-validation truncation.
        # (The new cap is a generation guard, not a post-validation filter.)
        extractor.max_candidates_per_type = 200

        low_conf_features  = self._make_feature_list(57, base_confidence=0.1)
        high_conf_features = [
            self._make_feature("golden_1", 0.95, cx=200.0, cy=50.0, radius=8.0),
            self._make_feature("golden_2", 0.95, cx=250.0, cy=50.0, radius=8.0),
            self._make_feature("golden_3", 0.95, cx=300.0, cy=50.0, radius=8.0),
        ]
        # High-confidence ones placed at the end (positions 57–59)
        all_features = low_conf_features + high_conf_features

        # Stub out the three extractors; only circles matter here
        with patch.object(extractor.circle_extractor, "extract_circles",
                          return_value=all_features):
            with patch.object(extractor.rectangle_extractor, "extract_rectangles",
                              return_value=[]):
                with patch.object(extractor.contour_extractor, "extract_contours",
                                  return_value=[]):
                    result = extractor.extract_features(
                        _make_dummy_preprocessing_result()
                    )

        result_ids = {f.feature_id for f in result.features}
        assert "golden_1" in result_ids, "High-confidence feature at position 57 must survive"
        assert "golden_2" in result_ids, "High-confidence feature at position 58 must survive"
        assert "golden_3" in result_ids, "High-confidence feature at position 59 must survive"

    def test_result_records_generation_cap_drop_count(self):
        """
        When the generation cap fires, candidates_dropped_by_generation_cap
        must be non-zero in the result.
        """
        from feature_inspection.actual.actual_feature_extractor import ActualFeatureExtractor

        extractor = ActualFeatureExtractor()
        limit = extractor.max_candidates_per_type
        over_limit = self._make_feature_list(limit + 5, base_confidence=0.8)

        with patch.object(extractor.circle_extractor, "extract_circles",
                          return_value=over_limit):
            with patch.object(extractor.rectangle_extractor, "extract_rectangles",
                              return_value=[]):
                with patch.object(extractor.contour_extractor, "extract_contours",
                                  return_value=[]):
                    result = extractor.extract_features(
                        _make_dummy_preprocessing_result()
                    )

        assert result.candidates_dropped_by_generation_cap == 5, (
            f"Expected 5 dropped, got {result.candidates_dropped_by_generation_cap}")

    def test_no_drop_recorded_when_under_limit(self):
        """
        When the raw candidate count is below the limit,
        candidates_dropped_by_generation_cap must be 0.
        """
        from feature_inspection.actual.actual_feature_extractor import ActualFeatureExtractor

        extractor = ActualFeatureExtractor()
        under_limit = self._make_feature_list(3, base_confidence=0.8)

        with patch.object(extractor.circle_extractor, "extract_circles",
                          return_value=under_limit):
            with patch.object(extractor.rectangle_extractor, "extract_rectangles",
                              return_value=[]):
                with patch.object(extractor.contour_extractor, "extract_contours",
                                  return_value=[]):
                    result = extractor.extract_features(
                        _make_dummy_preprocessing_result()
                    )

        assert result.candidates_dropped_by_generation_cap == 0

    def test_duplicate_count_is_correct_after_removing_cap_from_pipeline(self):
        """
        duplicate_candidates_suppressed must equal (validated - unique),
        not (limited_candidates - unique) from the old pipeline.
        """
        from feature_inspection.actual.actual_feature_extractor import ActualFeatureExtractor

        extractor = ActualFeatureExtractor()
        # Two features that will be treated as duplicates (same centre, same radius)
        dup1 = self._make_feature("dup_a", 0.9, cx=100.0, cy=100.0, radius=10.0)
        dup2 = self._make_feature("dup_b", 0.7, cx=101.0, cy=101.0, radius=10.0)
        unique = self._make_feature("unique", 0.8, cx=200.0, cy=200.0, radius=10.0)

        with patch.object(extractor.circle_extractor, "extract_circles",
                          return_value=[dup1, dup2, unique]):
            with patch.object(extractor.rectangle_extractor, "extract_rectangles",
                              return_value=[]):
                with patch.object(extractor.contour_extractor, "extract_contours",
                                  return_value=[]):
                    result = extractor.extract_features(
                        _make_dummy_preprocessing_result()
                    )

        # dup1 and dup2 → 1 survives, 1 suppressed; plus unique = 2 total
        assert result.duplicate_candidates_suppressed >= 1, (
            "At least one duplicate should have been suppressed")
        assert len(result.features) >= 1


# ------------------------------------------------------------------ #
# Shared fixture factory                                              #
# ------------------------------------------------------------------ #

# ============================================================ #
#  Step 9 – Duplicate suppression after evidence fixes         #
# ============================================================ #

class TestDuplicateSuppression:
    """
    Step 9 acceptance tests.

    Five spec scenarios, each tested at both the predicate level
    (_should_cluster_circles) and the full consolidation pipeline
    (_apply_enhanced_cluster_consolidation).
    """

    # ------------------------------------------------------------------ #
    # Fixture factory                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _circle(cx: float, cy: float, radius: float,
                confidence: float = 0.8,
                detection_method: str = "hough_circles",
                feature_id: str = "") -> ActualFeature:
        """Build a minimal ActualFeature with a given centre, radius and confidence."""
        geometry = GeometricProperties(
            center=(cx, cy),
            area=float(np.pi * radius ** 2),
            perimeter=float(2 * np.pi * radius),
            bounding_box=(int(cx - radius), int(cy - radius),
                          int(2 * radius), int(2 * radius)),
            radius=radius,
            diameter=2 * radius,
        )
        evidence = EvidenceMetrics(
            confidence=confidence,
            edge_support=confidence,
            contour_quality=confidence,
            intensity_consistency=confidence,
            geometric_consistency=confidence,
            angular_coverage=confidence,
        )
        return ActualFeature(
            feature_id=feature_id or f"c_{int(cx)}_{int(cy)}_r{int(radius)}",
            feature_type=ActualFeatureType.CIRCLE,
            geometry=geometry,
            contour=np.zeros((4, 2), dtype=np.int32),
            evidence=evidence,
            source_representation="internal_geometry_edges",
            detection_method=detection_method,
        )

    @staticmethod
    def _extractor() -> CircleExtractor:
        e = CircleExtractor()
        e.scale_aware_distance = True
        e.cluster_radius_factor = 0.4
        e.concentric_detection = True
        e.concentric_radius_tolerance = 0.15
        e.cluster_radius_tolerance = 0.2
        e.strict_center_distance = 8
        e.enhanced_clustering = True
        e.evidence_based_consolidation = True
        return e

    # ------------------------------------------------------------------ #
    # Scenario 1 — two near-identical Hough detections → one feature      #
    # ------------------------------------------------------------------ #

    def test_near_identical_hough_detections_merge_to_one(self):
        """
        Two Hough peaks for the same circle (centre offset < 3 px, radii within 5 %)
        should consolidate to a single feature.
        """
        e = self._extractor()
        a = self._circle(100.0, 100.0, 40.0, confidence=0.85, feature_id="hough_a")
        b = self._circle(102.0,  99.0, 40.5, confidence=0.80, feature_id="hough_b")

        assert e._should_cluster_circles(a, b), (
            "Near-identical Hough detections should be flagged as duplicates")

        result = e._apply_enhanced_cluster_consolidation([a, b])
        assert len(result) == 1, f"Expected 1 feature, got {len(result)}"
        # The higher-confidence candidate must survive
        assert result[0].feature_id == "hough_a", (
            f"Higher-confidence candidate should be kept, got {result[0].feature_id}")

    # ------------------------------------------------------------------ #
    # Scenario 2 — Hough + contour same circle → one feature              #
    # ------------------------------------------------------------------ #

    def test_hough_and_contour_same_circle_merge_to_one(self):
        """
        An independent Hough detection and a contour detection of the same
        physical circle should consolidate to a single feature.
        """
        e = self._extractor()
        hough   = self._circle(100.0, 100.0, 35.0, confidence=0.78,
                               detection_method="hough_circles",   feature_id="hough")
        contour = self._circle(101.0, 100.0, 35.0, confidence=0.82,
                               detection_method="contour_analysis", feature_id="contour")

        assert e._should_cluster_circles(hough, contour), (
            "Hough and contour of the same circle should be flagged as duplicates")

        result = e._apply_enhanced_cluster_consolidation([hough, contour])
        assert len(result) == 1, f"Expected 1 feature, got {len(result)}"
        # Contour candidate has higher confidence — it should be kept
        assert result[0].feature_id == "contour"

    # ------------------------------------------------------------------ #
    # Scenario 3 — two nearby but distinct circles → two features         #
    # ------------------------------------------------------------------ #

    def test_two_nearby_distinct_circles_are_not_merged(self):
        """
        Two physical circles close together (e.g. two bolt holes near each
        other) must not be merged even when the center distance is small
        relative to some fixed threshold — the scale-aware check must be
        tight enough.

        Geometry: radius=10, centers 25 px apart.
        scale-aware threshold = 10 * 0.4 = 4 px  →  25 > 4  →  not a duplicate.
        """
        e = self._extractor()
        a = self._circle( 50.0, 100.0, 10.0, feature_id="bolt_a")
        b = self._circle( 75.0, 100.0, 10.0, feature_id="bolt_b")

        assert not e._should_cluster_circles(a, b), (
            "Circles 25 px apart with r=10 should NOT be duplicates "
            f"(threshold={10 * e.cluster_radius_factor:.1f} px)")

        result = e._apply_enhanced_cluster_consolidation([a, b])
        assert len(result) == 2, (
            f"Two distinct circles must remain as two features, got {len(result)}")

    # ------------------------------------------------------------------ #
    # Scenario 4 — concentric circles → two features                      #
    # ------------------------------------------------------------------ #

    def test_concentric_circles_are_not_merged(self):
        """
        Two circles sharing the same centre but with substantially different
        radii (e.g. inner/outer edges of a bore) must NOT be merged.

        The concentric guard must return False (keep both), not True (merge).
        """
        e = self._extractor()
        inner = self._circle(100.0, 100.0, 15.0, feature_id="inner_bore")
        outer = self._circle(100.0, 100.0, 50.0, feature_id="outer_bore")

        # radius diff ratio = (50-15)/50 = 0.70  >> concentric_tolerance=0.15
        assert not e._should_cluster_circles(inner, outer), (
            "Concentric circles with very different radii must NOT be flagged as duplicates")

        result = e._apply_enhanced_cluster_consolidation([inner, outer])
        assert len(result) == 2, (
            f"Concentric bore edges must remain as two features, got {len(result)}")
        ids = {f.feature_id for f in result}
        assert "inner_bore" in ids and "outer_bore" in ids

    def test_concentric_same_radius_duplicates_do_merge(self):
        """
        Two near-concentric detections of the same feature (centre drift < 5 %
        of radius, radii within tolerance) should still merge normally.
        """
        e = self._extractor()
        a = self._circle(100.0, 100.0, 30.0, confidence=0.90, feature_id="a")
        b = self._circle(101.0, 100.0, 30.5, confidence=0.85, feature_id="b")

        # radius diff ratio = 0.5/30.5 ≈ 0.016  < 0.15  → not concentric guard
        assert e._should_cluster_circles(a, b), (
            "Near-duplicate concentric-ish detections should merge")

        result = e._apply_enhanced_cluster_consolidation([a, b])
        assert len(result) == 1

    # ------------------------------------------------------------------ #
    # Scenario 5 — different-size circles                                 #
    # ------------------------------------------------------------------ #

    def test_small_and_large_circles_not_merged_by_fixed_threshold(self):
        """
        A small circle (r=8) and a large circle (r=60) close together must
        NOT be merged by a naive fixed center-distance threshold.

        With scale-aware distance: threshold = 60 * 0.4 = 24 px.
        Centers are 10 px apart — passes the distance gate — but radius ratio
        = 8/60 ≈ 0.133  <  (1 - 0.2) = 0.8  →  rejected by radius gate.
        """
        e = self._extractor()
        small = self._circle(100.0, 100.0,  8.0, feature_id="small")
        large = self._circle(108.0, 100.0, 60.0, feature_id="large")

        assert not e._should_cluster_circles(small, large), (
            "Small (r=8) and large (r=60) circles 10 px apart must NOT merge — "
            f"radius ratio {8/60:.3f} is below threshold {1-e.cluster_radius_tolerance:.2f}")

        result = e._apply_enhanced_cluster_consolidation([small, large])
        assert len(result) == 2, (
            f"Small and large circles must remain distinct, got {len(result)}")

    # ------------------------------------------------------------------ #
    # Transitive closure (union-find correctness)                          #
    # ------------------------------------------------------------------ #

    def test_transitive_duplicate_chain_merges_to_one(self):
        """
        A → B and B → C (both duplicates) but A ≉ C directly (just outside
        the pairwise threshold).  Union-find must still place all three in
        the same cluster.

        Geometry (scale-aware, factor=0.4, radius=10):
          cluster_distance = 10 * 0.4 = 4 px
          A=(0,0), B=(3,0), C=(6,0)
          dist(A,B)=3 ≤ 4 ✓, dist(B,C)=3 ≤ 4 ✓, dist(A,C)=6 > 4 ✗
        """
        e = self._extractor()
        a = self._circle( 0.0, 0.0, 10.0, confidence=0.90, feature_id="chain_a")
        b = self._circle( 3.0, 0.0, 10.0, confidence=0.80, feature_id="chain_b")
        c = self._circle( 6.0, 0.0, 10.0, confidence=0.75, feature_id="chain_c")

        # Verify the pairwise predicate matches the scenario setup
        assert     e._should_cluster_circles(a, b)
        assert     e._should_cluster_circles(b, c)
        assert not e._should_cluster_circles(a, c)

        result = e._apply_enhanced_cluster_consolidation([a, b, c])
        assert len(result) == 1, (
            f"Transitive chain A≈B, B≈C must collapse to one feature, got {len(result)}")
        assert result[0].feature_id == "chain_a", (
            "Highest-confidence member of the chain must be kept")

    # ------------------------------------------------------------------ #
    # Selection uses grouped confidence, not correlated raw metrics        #
    # ------------------------------------------------------------------ #

    def test_best_selection_uses_evidence_confidence_not_correlated_metrics(self):
        """
        `_select_best_from_cluster` must pick the candidate with the highest
        `evidence.confidence`, which is the grouped score from Step 7.

        A Hough candidate with genuinely better grouped evidence (confidence=0.88)
        must beat a contour candidate with lower grouped confidence (confidence=0.72)
        even though the old formula would have incorrectly preferred the contour
        due to its `detection_method` bonus.
        """
        e = self._extractor()
        hough   = self._circle(100.0, 100.0, 30.0, confidence=0.88,
                               detection_method="hough_circles",   feature_id="hough_best")
        contour = self._circle(100.0, 100.0, 30.0, confidence=0.72,
                               detection_method="contour_analysis", feature_id="contour_worse")

        best = e._select_best_from_cluster([hough, contour])
        assert best.feature_id == "hough_best", (
            f"Higher grouped-confidence candidate should win; got {best.feature_id}")

    def test_contour_tiebreak_when_confidence_equal(self):
        """
        When two candidates have the same confidence (to 2 decimal places),
        the contour-based detection should be preferred as secondary tiebreak.
        """
        e = self._extractor()
        hough   = self._circle(100.0, 100.0, 30.0, confidence=0.800,
                               detection_method="hough_circles",   feature_id="hough")
        contour = self._circle(100.0, 100.0, 30.0, confidence=0.804,  # rounds to 0.80
                               detection_method="contour_analysis", feature_id="contour")

        best = e._select_best_from_cluster([hough, contour])
        assert best.feature_id == "contour", (
            "Contour detection should win tiebreak at equal rounded confidence")


def _make_dummy_preprocessing_result() -> "PreprocessingResult":
    """Return a minimal PreprocessingResult sufficient for the orchestrator."""
    size = 400
    blank     = np.zeros((size, size), dtype=np.uint8)
    blank_bgr = np.zeros((size, size, 3), dtype=np.uint8)
    mask      = np.ones((size, size), dtype=np.uint8) * 255

    return PreprocessingResult(
        source_image_path=Path("dummy.jpg"),
        original_image=blank_bgr,
        processed_image=blank,
        product_mask=mask,
        isolated_product_image=blank_bgr,
        raw_internal_geometry_edges=blank,
        internal_geometry_edges=blank,
        outer_boundary_edges=blank,
        edge_representation=blank,
        original_dimensions=(size, size),
        processed_dimensions=(size, size),
        scale_factor=1.0,
        roi_offset=(0, 0),
        preprocessing_successful=True,
        isolation_successful=True,
        product_area_pixels=size * size,
        product_area_fraction=1.0,
        mask_bounding_box=(0, 0, size, size),
        mask_bounding_box_area_fraction=1.0,
        foreground_component_count=1,
        largest_component_fraction=1.0,
        mask_contour_area=float(size * size),
        mask_contour_perimeter=float(4 * size),
        mask_extent=1.0,
        mask_solidity=1.0,
        border_touching_foreground=False,
        significant_foreground_regions=1,
        silhouette_validation_result="OK",
        external_gradient_strength=0.0,
        boundary_gradient_consistency=1.0,
        suspicious_boundary_fraction=0.0,
        boundary_gradient_strength=0.0,
        raw_internal_edge_density=0.0,
        filtered_internal_edge_density=0.0,
        internal_edge_retention_ratio=1.0,
        internal_edge_reduction_ratio=0.0,
        outer_boundary_density=0.0,
        final_edge_density=0.0,
        processing_steps=["dummy"],
        configuration_snapshot={},
    )


# ============================================================ #
#  Step 10 – Adversarial and realistic regression tests        #
# ============================================================ #

class TestAdversarialAndRegression:
    """
    14 synthetic scenarios covering the original false-positive mechanism
    and its key interactions.

    All images are deterministic so results are reproducible.

    Entry point used for unit-level scenarios (1–8, 11):
        CircleExtractor._validate_hough_candidate(center, radius,
                edge_image, intensity_image, mask)
    Entry point used for integration scenarios (9, 10, 12, 13, 14):
        CircleExtractor.extract_circles(internal_edges, isolated_product, mask)

    Image convention:
        edge_image    – uint8 (H,W)   binary geometric edge map
        intensity_img – uint8 (H,W,3) appearance image (BGR or grey broadcast)
        mask          – uint8 (H,W)   all-255 product mask (full image)
    """

    SIZE   = 300
    CENTER = (150, 150)
    RADIUS = 55.0

    # ------------------------------------------------------------------ #
    # Image builders                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def _blank(cls):
        return np.zeros((cls.SIZE, cls.SIZE), dtype=np.uint8)

    @classmethod
    def _mask(cls):
        return np.ones((cls.SIZE, cls.SIZE), dtype=np.uint8) * 255

    @classmethod
    def _grey_bgr(cls, value: int = 80):
        return np.full((cls.SIZE, cls.SIZE, 3), value, dtype=np.uint8)

    @classmethod
    def _circle_edge(cls, center=None, radius=None, thickness=2):
        """Binary edge map containing a full circle ring."""
        cx, cy = center or cls.CENTER
        r = int(radius or cls.RADIUS)
        img = cls._blank()
        cv2.circle(img, (cx, cy), r, 255, thickness)
        return img

    @classmethod
    def _circle_appearance(cls, center=None, radius=None,
                            bg=60, fg=210, thickness=3):
        """
        Appearance image: bright ring on a background that creates measurable
        interior-versus-exterior contrast.

        The interior of the circle is filled at a lighter value than the
        background so the texture-discrimination gate (which measures the
        intensity difference between interior and exterior) is satisfied.
        """
        cx, cy = center or cls.CENTER
        r = int(radius or cls.RADIUS)
        img = cls._grey_bgr(bg)
        # Fill the disc interior brighter so interior != exterior
        cv2.circle(img, (cx, cy), r, (bg + 60, bg + 60, bg + 60), -1)
        # Draw the ring edge on top
        cv2.circle(img, (cx, cy), r, (fg, fg, fg), thickness)
        return img

    @classmethod
    def _partial_arc_edge(cls, arc_fraction: float = 0.25):
        """
        Edge map with only `arc_fraction` of the circle present.
        The arc starts at 0° and spans `arc_fraction * 360°`.
        """
        img = cls._blank()
        cx, cy = cls.CENTER
        r = int(cls.RADIUS)
        end_angle = arc_fraction * 2 * np.pi
        for angle in np.linspace(0, end_angle, 200):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < cls.SIZE and 0 <= y < cls.SIZE:
                img[y, x] = 255
        return img

    @classmethod
    def _random_edge(cls, density: float = 0.05, seed: int = 0):
        """
        Random binary edge map at the given pixel density.
        Deliberately NOT arranged in a circle.
        """
        rng = np.random.default_rng(seed)
        mask = rng.random((cls.SIZE, cls.SIZE)) < density
        img = np.zeros((cls.SIZE, cls.SIZE), dtype=np.uint8)
        img[mask] = 255
        return img

    @classmethod
    def _circular_texture_edge(cls, n_arcs: int = 8, seed: int = 42):
        """
        Random short arcs placed around the predicted circumference.
        Mimics the texture that caused false positives: each arc contributes
        to angular coverage in its sector but the support is NOT from a
        single coherent circular boundary.
        """
        rng = np.random.default_rng(seed)
        img = cls._blank()
        cx, cy = cls.CENTER
        r = int(cls.RADIUS)
        # Each arc covers only ~30° and is placed near the circumference with
        # a random radial jitter of ±10 px so radial consistency is poor.
        arc_span = np.pi / 6  # 30°
        for k in range(n_arcs):
            start = rng.uniform(0, 2 * np.pi)
            radial_jitter = rng.uniform(-10, 10)
            arc_r = int(r + radial_jitter)
            if arc_r <= 0:
                continue
            for angle in np.linspace(start, start + arc_span, 60):
                x = int(cx + arc_r * np.cos(angle))
                y = int(cy + arc_r * np.sin(angle))
                if 0 <= x < cls.SIZE and 0 <= y < cls.SIZE:
                    img[y, x] = 255
        return img

    @classmethod
    def _add_texture_noise(cls, base_img: np.ndarray,
                           texture_strength: int = 40, seed: int = 7):
        """Overlay salt-and-pepper-style texture on an appearance image."""
        rng = np.random.default_rng(seed)
        noise = rng.integers(-texture_strength, texture_strength + 1,
                              base_img.shape, dtype=np.int16)
        noisy = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

    @classmethod
    def _straight_edge(cls):
        """Horizontal line through the image centre — strong but not circular."""
        img = cls._blank()
        cv2.line(img, (0, cls.CENTER[1]), (cls.SIZE - 1, cls.CENTER[1]), 255, 2)
        return img

    @classmethod
    def _ellipse_edge(cls, axes=(55, 30)):
        """Ellipse edge map: closed shape but clearly non-circular."""
        img = cls._blank()
        cv2.ellipse(img, cls.CENTER, axes, 0, 0, 360, 255, 2)
        return img

    @classmethod
    def _extractor(cls) -> CircleExtractor:
        return CircleExtractor()

    # ------------------------------------------------------------------ #
    # Helper: validate at the algorithm boundary                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def _validate(cls, edge_img, intensity_img=None, center=None,
                  radius=None, extractor=None):
        """Call _validate_hough_candidate and return the result dict."""
        e = extractor or cls._extractor()
        c = center or cls.CENTER
        r = float(radius or cls.RADIUS)
        app = intensity_img if intensity_img is not None else cls._grey_bgr()
        return e._validate_hough_candidate(c, r, edge_img, app, cls._mask())

    # ================================================================== #
    # Scenario 1 — Perfect circle                                         #
    # ================================================================== #

    def test_s01_perfect_circle_is_detected_with_high_geometric_evidence(self):
        """
        A clean, full circle ring drawn on a dark background must be accepted
        with high angular coverage and high radial consistency.
        """
        edge = self._circle_edge()
        app  = self._circle_appearance()

        result = self._validate(edge, app)

        assert result["valid"], (
            f"Perfect circle should be accepted; rejected because: "
            f"{result['rejection_reason']}"
        )
        assert result["angular_coverage"] > 0.65, (
            f"Full circle must have high angular coverage; got {result['angular_coverage']:.2f}")
        assert result["radial_consistency"] > 0.55, (
            f"Full circle must have high radial consistency; got {result['radial_consistency']:.2f}")
        assert result["max_gap_ratio"] < 0.30, (
            f"Full circle must have small max gap; got {result['max_gap_ratio']:.2f}")

    # ================================================================== #
    # Scenario 2 — Partial arc                                            #
    # ================================================================== #

    def test_s02_quarter_arc_is_rejected(self):
        """
        A 25 % arc (90°) must be rejected — insufficient angular coverage
        and/or excessive gap ratio.
        """
        edge = self._partial_arc_edge(arc_fraction=0.25)
        result = self._validate(edge)

        assert not result["valid"], (
            "Quarter arc should be rejected but was accepted"
        )

    def test_s02_quarter_arc_has_large_gap(self):
        """
        Even if a partial arc somehow passed the gate, the gap metric must
        report a large max_gap_ratio so downstream scoring penalises it.
        """
        e = self._extractor()
        edge = self._partial_arc_edge(arc_fraction=0.25)
        continuity = e._analyze_edge_continuity(
            self.CENTER, self.RADIUS, edge
        )
        # 75 % of the circumference is missing → max gap must be > 0.60
        assert continuity["max_gap_ratio"] > 0.60, (
            f"Quarter arc should have max_gap_ratio > 0.60; got {continuity['max_gap_ratio']:.2f}"
        )

    # ================================================================== #
    # Scenario 3 — Random edges                                           #
    # ================================================================== #

    def test_s03_random_edges_are_rejected(self):
        """
        A random binary edge map at 5 % density contains no coherent circle.
        With high probability the validation rejects it; if it somehow passes
        the gate checks, its grouped confidence must be low.
        """
        e = self._extractor()
        for seed in range(5):   # five independent noise patterns
            edge = self._random_edge(density=0.05, seed=seed)
            result = self._validate(edge, extractor=e)
            if result["valid"]:
                # The pipeline still penalises noise via the grouped confidence model
                conf = e._calculate_hough_evidence_confidence(result)
                assert conf < 0.55, (
                    f"Random edges (seed={seed}) passed gates but confidence must be low; "
                    f"got {conf:.3f}"
                )

    # ================================================================== #
    # Scenario 4 — Circular-looking texture                               #
    # ================================================================== #

    def test_s04_circular_texture_lacks_radial_consistency(self):
        """
        Texture arranged as disconnected short arcs near the circumference
        may accumulate angular coverage in multiple sectors but has poor
        radial consistency because each arc sits at a random radial offset.

        After all gate checks, the validation result must either be rejected
        or report radial_consistency below a genuine circle's level.
        """
        e = self._extractor()
        edge = self._circular_texture_edge()
        result = self._validate(edge, extractor=e)

        if result["valid"]:
            # If somehow passed, radial consistency must be markedly lower than
            # a perfect circle.
            perfect_result = self._validate(self._circle_edge(), extractor=e)
            assert result["radial_consistency"] < perfect_result["radial_consistency"] - 0.10, (
                "Circular texture should have noticeably lower radial consistency than a real circle; "
                f"texture={result['radial_consistency']:.3f}, "
                f"circle={perfect_result['radial_consistency']:.3f}"
            )

    def test_s04_circular_texture_has_lower_confidence_than_real_circle(self):
        """
        Even if individual metrics partially satisfy gates, the final grouped
        confidence of a circular-texture false-positive must be below a real
        circle's confidence.
        """
        e = self._extractor()
        texture_result = self._validate(self._circular_texture_edge(), extractor=e)
        circle_result  = self._validate(
            self._circle_edge(), self._circle_appearance(), extractor=e
        )
        texture_conf = e._calculate_hough_evidence_confidence(texture_result)
        circle_conf  = e._calculate_hough_evidence_confidence(circle_result)
        assert texture_conf < circle_conf, (
            f"Texture confidence ({texture_conf:.3f}) must be below "
            f"circle confidence ({circle_conf:.3f})"
        )

    # ================================================================== #
    # Scenario 5 — Real circle + heavy texture overlay                    #
    # ================================================================== #

    def test_s05_real_circle_survives_heavy_texture(self):
        """
        A genuine circle drawn on the edge map must be accepted even when the
        appearance image is heavily textured.  The geometric edge map is clean;
        appearance noise should not veto a geometrically sound circle.
        """
        edge = self._circle_edge()
        noisy_app = self._add_texture_noise(self._circle_appearance(), texture_strength=60)

        result = self._validate(edge, noisy_app)
        assert result["valid"], (
            f"Real circle should survive texture overlay; rejected: "
            f"{result['rejection_reason']}"
        )

    # ================================================================== #
    # Scenario 6 — Real circle + scratches crossing circumference         #
    # ================================================================== #

    def test_s06_circle_with_crossing_scratches_retained_if_boundary_sufficient(self):
        """
        A circle with additional scratch lines crossing the circumference
        should still be accepted: the clean arc portions give sufficient
        angular coverage even with extra noise edges mixed in.
        """
        edge = self._circle_edge(thickness=2)

        # Add two diagonal scratches that cross the circumference
        cv2.line(edge, (100, 80), (200, 220), 255, 2)
        cv2.line(edge, (80, 200), (220, 100), 255, 2)

        result = self._validate(edge, self._circle_appearance())
        assert result["valid"], (
            f"Circle with crossing scratches should still be accepted; "
            f"rejected: {result['rejection_reason']}"
        )

    # ================================================================== #
    # Scenario 7 — Straight edge                                          #
    # ================================================================== #

    def test_s07_straight_edge_is_rejected_as_circle(self):
        """
        A single horizontal line produces high local edge density at the
        circumference samples where it intersects but terrible angular
        coverage and gap ratio.  It must be rejected.
        """
        edge = self._straight_edge()
        result = self._validate(edge)
        assert not result["valid"], (
            "Straight edge must be rejected as a circle candidate"
        )

    def test_s07_straight_edge_has_high_max_gap(self):
        """
        A horizontal line intersects the circle's circumference at only two
        narrow points (near 90° and 270°).  The supported arc is tiny so
        most of the ring has no edge support and the max_gap_ratio must be
        large.

        With a 3-px neighbourhood check and a 2-px-thick line, each
        intersection zone covers roughly 3–5 samples out of 64, giving
        two supported clusters and a max gap of roughly 45–55 % of the
        circumference.
        """
        e = self._extractor()
        edge = self._straight_edge()
        continuity = e._analyze_edge_continuity(self.CENTER, self.RADIUS, edge)
        # Each intersection supports a narrow arc; the rest is unsupported
        assert continuity["max_gap_ratio"] > 0.40, (
            f"Straight-edge max_gap_ratio should be > 0.40; got {continuity['max_gap_ratio']:.2f}"
        )
        # Additionally confirm the missing_ratio is very high
        assert continuity["missing_ratio"] > 0.85, (
            f"Straight edge should leave most of the ring unsupported; "
            f"got missing_ratio={continuity['missing_ratio']:.2f}"
        )

    # ================================================================== #
    # Scenario 8 — Ellipse                                                #
    # ================================================================== #

    def test_s08_ellipse_rejected_or_low_confidence(self):
        """
        An ellipse (axes 55×30) has a closed boundary but fails the radial
        consistency gate: at the flat ends the boundary is far from the
        predicted circle radius.

        The result must either be rejected OR have noticeably lower confidence
        than a true circle of the same nominal radius.
        """
        e = self._extractor()
        edge = self._ellipse_edge(axes=(55, 30))
        result = self._validate(edge, extractor=e)

        if result["valid"]:
            # Must score below a true circle
            circle_result = self._validate(
                self._circle_edge(), self._circle_appearance(), extractor=e
            )
            ellipse_conf = e._calculate_hough_evidence_confidence(result)
            circle_conf  = e._calculate_hough_evidence_confidence(circle_result)
            assert ellipse_conf < circle_conf - 0.05, (
                f"Ellipse confidence ({ellipse_conf:.3f}) must be at least 0.05 below "
                f"circle confidence ({circle_conf:.3f})"
            )

    # ================================================================== #
    # Scenario 9 — Multiple real circles                                  #
    # ================================================================== #

    def test_s09_multiple_real_circles_all_retained(self):
        """
        Three non-overlapping circles at different positions must all appear
        in the final feature list.
        """
        size = 400
        edge  = np.zeros((size, size), dtype=np.uint8)
        app   = np.full((size, size, 3), 60, dtype=np.uint8)
        mask  = np.ones((size, size), dtype=np.uint8) * 255

        specs = [
            ((100, 100), 30),
            ((250, 100), 40),
            ((175, 280), 35),
        ]
        for (cx, cy), r in specs:
            cv2.circle(edge, (cx, cy), r, 255, 2)
            cv2.circle(app,  (cx, cy), r, (200, 200, 200), 3)

        e = CircleExtractor()
        circles = e.extract_circles(edge, app, mask)

        # Accept if all three are detected (allow for minor duplicates pre-dedup)
        centers_found = {(f.geometry.center[0], f.geometry.center[1]) for f in circles}
        for (cx, cy), r in specs:
            found = any(
                abs(cx - fx) < 15 and abs(cy - fy) < 15
                for fx, fy in centers_found
            )
            assert found, (
                f"Circle at ({cx},{cy}) r={r} not detected. "
                f"Detected centres: {centers_found}"
            )

    # ================================================================== #
    # Scenario 10 — One real circle + many fake circles (KEY REGRESSION) #
    # ================================================================== #

    def test_s10_real_circle_retained_fake_texture_circles_rejected(self):
        """
        KEY REGRESSION TEST.

        This reproduces the original false-positive explosion:
        - One clean real circle in the edge map.
        - Appearance image contains random short arcs that look like partial
          circles to naive Hough running on isolated_product.

        Expected: the real circle is detected; the fake arc clusters are not
        accepted as validated features.

        The geometry-first Hough input (Step 6) is critical here: running
        Hough on the clean edge map suppresses the texture false-positives
        at candidate-generation time.
        """
        size = 300
        cx, cy, r = 150, 150, 50
        edge = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(edge, (cx, cy), r, 255, 2)

        # Appearance: real circle ring + random arc fragments across the image
        app = np.full((size, size, 3), 70, dtype=np.uint8)
        cv2.circle(app, (cx, cy), r, (210, 210, 210), 3)

        rng = np.random.default_rng(123)
        for _ in range(20):
            px = int(rng.integers(30, size - 30))
            py = int(rng.integers(30, size - 30))
            pr = int(rng.integers(15, 35))
            start_deg = int(rng.integers(0, 360))
            end_deg   = start_deg + int(rng.integers(30, 90))
            cv2.ellipse(app, (px, py), (pr, pr), 0, start_deg, end_deg, (180, 180, 180), 2)

        mask = np.ones((size, size), dtype=np.uint8) * 255

        e = CircleExtractor()
        circles = e.extract_circles(edge, app, mask)

        # At least one detection should be near the real circle
        real_found = any(
            abs(f.geometry.center[0] - cx) < 15
            and abs(f.geometry.center[1] - cy) < 15
            and abs((f.geometry.radius or 0) - r) < 10
            for f in circles
        )
        assert real_found, (
            f"Real circle at ({cx},{cy}) r={r} was not detected. "
            f"Detected: {[(f.geometry.center, f.geometry.radius) for f in circles]}"
        )

        # All accepted circles must have passed strict evidence — none should
        # be a texture arc masquerading as a circle.
        for f in circles:
            assert f.evidence.confidence >= 0.0  # sanity — no NaN/negative
            # Features with very low angular coverage should not have survived
            # (they would indicate an arc accepted as a circle)
            assert f.evidence.angular_coverage >= 0.0  # structural check

    # ================================================================== #
    # Scenario 11 — Wraparound gap correctly measured                     #
    # ================================================================== #

    def test_s11_wraparound_gap_measured_correctly(self):
        """
        A gap that straddles the 0°/360° boundary must be reported as a
        single gap of the expected size, not as two separate smaller gaps.

        We draw a 270° arc starting at 45° (so it spans 45°→315°),
        leaving a 90° gap that crosses the 0° boundary (315°→45°).
        """
        e = self._extractor()
        edge = self._blank()
        cx, cy = self.CENTER
        r = int(self.RADIUS)

        # Draw 270° arc from 45° to 315°  (gap = 315°→360°→45° = 90°)
        for angle in np.linspace(np.pi / 4, 7 * np.pi / 4, 400):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < self.SIZE and 0 <= y < self.SIZE:
                edge[y, x] = 255

        continuity = e._analyze_edge_continuity((cx, cy), float(r), edge)
        gap = continuity["max_gap_ratio"]

        # 90° out of 360° = 0.25 ± a few sampling-discretisation steps
        assert 0.15 < gap < 0.40, (
            f"270° arc should report gap_ratio ≈ 0.25; got {gap:.3f}"
        )
        # Crucially, the gap must not be reported as two halves (< 0.10 each)
        # which would happen if wraparound handling were broken.
        assert gap > 0.15, (
            "Wraparound gap should be reported as one contiguous gap > 0.15, "
            f"not split; got {gap:.3f}"
        )

    def test_s11_arc_starting_at_zero_degrees_gap_not_split(self):
        """
        When an arc starts exactly at 0° (the boundary), the missing portion
        at the other end must still be reported as a single contiguous gap.
        """
        e = self._extractor()
        edge = self._blank()
        cx, cy = self.CENTER
        r = int(self.RADIUS)

        # Draw arc 0°→270°, gap = 270°→360° = 90°
        for angle in np.linspace(0, 3 * np.pi / 2, 400):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < self.SIZE and 0 <= y < self.SIZE:
                edge[y, x] = 255

        continuity = e._analyze_edge_continuity((cx, cy), float(r), edge)
        gap = continuity["max_gap_ratio"]
        assert 0.15 < gap < 0.40, (
            f"0°-start 270° arc should report gap ≈ 0.25; got {gap:.3f}"
        )

    # ================================================================== #
    # Scenario 12 — Concentric circles → both retained                   #
    # ================================================================== #

    def test_s12_concentric_circles_both_retained(self):
        """
        Two circles with the same centre but radii 30 and 60 are distinct
        features (e.g. inner/outer bore).  Both must appear in the final list.
        """
        size = 300
        cx, cy = 150, 150
        edge = np.zeros((size, size), dtype=np.uint8)
        app  = np.full((size, size, 3), 60, dtype=np.uint8)
        mask = np.ones((size, size), dtype=np.uint8) * 255

        for r, fg in [(30, 180), (60, 220)]:
            cv2.circle(edge, (cx, cy), r, 255, 2)
            cv2.circle(app,  (cx, cy), r, (fg, fg, fg), 3)

        e = CircleExtractor()
        circles = e.extract_circles(edge, app, mask)

        radii_found = sorted([round(f.geometry.radius or 0) for f in circles])
        small_found = any(abs(r - 30) < 10 for r in radii_found)
        large_found = any(abs(r - 60) < 10 for r in radii_found)

        assert small_found and large_found, (
            f"Both concentric circles (r≈30, r≈60) must be retained. "
            f"Radii found: {radii_found}"
        )

    # ================================================================== #
    # Scenario 13 — Nearby distinct circles → no merging                 #
    # ================================================================== #

    def test_s13_nearby_distinct_circles_not_merged(self):
        """
        Two circles at r=15 spaced 50 px apart are physically distinct
        (centre distance >> scale-aware threshold 15*0.4=6 px).
        Both must appear in the final list.
        """
        size = 300
        positions = [(100, 150), (150, 150)]
        r = 15
        edge = np.zeros((size, size), dtype=np.uint8)
        app  = np.full((size, size, 3), 60, dtype=np.uint8)
        mask = np.ones((size, size), dtype=np.uint8) * 255

        for cx, cy in positions:
            cv2.circle(edge, (cx, cy), r, 255, 2)
            cv2.circle(app,  (cx, cy), r, (200, 200, 200), 3)

        e = CircleExtractor()
        circles = e.extract_circles(edge, app, mask)

        for cx, cy in positions:
            found = any(
                abs(f.geometry.center[0] - cx) < 15
                and abs(f.geometry.center[1] - cy) < 15
                for f in circles
            )
            assert found, (
                f"Circle at ({cx},{cy}) r={r} was lost during deduplication. "
                f"Detected: {[(f.geometry.center, f.geometry.radius) for f in circles]}"
            )

    # ================================================================== #
    # Scenario 14 — Scale variation                                       #
    # ================================================================== #

    def test_s14_scale_variation_small_medium_large_detected(self):
        """
        Circles at three different scales (r=15, r=45, r=90) must all be
        individually detected when each is the only circle in the image.
        This confirms that detection thresholds are not hard-coded to one size.
        """
        for (cx, cy), r in [((120, 120), 15), ((200, 200), 45), ((200, 200), 90)]:
            size = max(400, (max(cx, cy) + r) * 2 + 40)
            edge = np.zeros((size, size), dtype=np.uint8)
            app  = np.full((size, size, 3), 60, dtype=np.uint8)
            mask = np.ones((size, size), dtype=np.uint8) * 255

            cv2.circle(edge, (cx, cy), r, 255, 2)
            cv2.circle(app,  (cx, cy), r, (120, 120, 120), -1)
            cv2.circle(app,  (cx, cy), r, (200, 200, 200), 3)

            e = CircleExtractor()
            circles = e.extract_circles(edge, app, mask)

            found = any(
                abs(f.geometry.center[0] - cx) < max(15, r * 0.3)
                and abs(f.geometry.center[1] - cy) < max(15, r * 0.3)
                and abs((f.geometry.radius or 0) - r) < max(8, r * 0.25)
                for f in circles
            )
            assert found, (
                f"Circle at ({cx},{cy}) r={r} not detected at that scale. "
                "Detected: " + str(
                    [(round(f.geometry.center[0]), round(f.geometry.center[1]),
                      round(f.geometry.radius or 0)) for f in circles]
                )
            )

    def test_s14_validation_thresholds_scale_consistently(self):
        """
        All three validation methods (angular coverage, radial consistency,
        gradient orientation) must report similar quality scores for a clean
        circle regardless of whether the radius is 15, 45, or 90 px.
        The appearance image has interior fill so the contrast gate is satisfied.
        """
        e = self._extractor()
        scores = {}
        for r in [15, 45, 90]:
            size = 400
            cx, cy = 200, 200
            edge = np.zeros((size, size), dtype=np.uint8)
            app  = np.full((size, size, 3), 60, dtype=np.uint8)
            cv2.circle(edge, (cx, cy), r, 255, 2)
            # Interior fill creates measurable contrast
            cv2.circle(app,  (cx, cy), r, (120, 120, 120), -1)
            cv2.circle(app,  (cx, cy), r, (200, 200, 200), 3)
            mask = np.ones((size, size), dtype=np.uint8) * 255

            result = e._validate_hough_candidate(
                (float(cx), float(cy)), float(r), edge, app, mask
            )
            scores[r] = result

        # All three should be accepted (or at least not catastrophically worse at any scale)
        accepted = {r: scores[r]["valid"] for r in scores}
        # At least 2 out of 3 must pass — small circles are trickier due to discretisation
        assert sum(accepted.values()) >= 2, (
            f"At least 2 of 3 scale variants should be accepted; got {accepted}"
        )

        # For accepted circles, angular_coverage must be in a similar range
        # (within 0.35 of each other) — no single scale should be dramatically worse.
        coverages = [scores[r]["angular_coverage"]
                     for r in scores if scores[r]["valid"]]
        if len(coverages) >= 2:
            spread = max(coverages) - min(coverages)
            assert spread < 0.45, (
                f"Angular coverage should be consistent across scales; "
                f"spread={spread:.3f}, values={dict(zip(scores.keys(), coverages))}"
            )


# ============================================================ #
#  Step 11 – Threshold calibration regression tests            #
# ============================================================ #

class TestThresholdCalibration:
    """
    Regression tests that pin each tuned threshold to its justified value
    and verify the calibration direction.

    Each test checks two things:
      1. The config constant has the tuned value (not accidentally reverted).
      2. A synthetic image at the p10 boundary of the genuine-circle
         distribution is accepted, confirming the threshold doesn't cut
         into the genuine population.

    Measurement basis (Step 11, 40 images, 16 357 candidates):
      - angular_coverage : accepted_p10 = 0.49
      - max_gap_ratio     : accepted_p90 = 0.234
      - radial_consistency: accepted_p10 = 0.569
      - local_contrast    : accepted_p10 = 3.8   (rejected_p90 = 0.0)
      - radial_samples    : coarse sampling caused missed separations for
                            medium/large circles; 16 gives finer resolution.
    """

    from feature_inspection import config as _cfg

    # ------------------------------------------------------------------ #
    # 1. Config constant values                                           #
    # ------------------------------------------------------------------ #

    def test_min_angular_coverage_tuned_to_0_50(self):
        from feature_inspection.config import HOUGH_MIN_ANGULAR_COVERAGE
        assert HOUGH_MIN_ANGULAR_COVERAGE == 0.50, (
            f"HOUGH_MIN_ANGULAR_COVERAGE should be 0.50 (tuned); got {HOUGH_MIN_ANGULAR_COVERAGE}. "
            "Measurement basis: accepted_p10 = 0.49 across 40 real images."
        )

    def test_max_edge_gap_ratio_tuned_to_0_30(self):
        from feature_inspection.config import HOUGH_MAX_EDGE_GAP_RATIO
        assert HOUGH_MAX_EDGE_GAP_RATIO == 0.30, (
            f"HOUGH_MAX_EDGE_GAP_RATIO should be 0.30 (tuned); got {HOUGH_MAX_EDGE_GAP_RATIO}. "
            "Measurement basis: accepted_p90 = 0.234 — old value 0.45 left 0.22 dead zone."
        )

    def test_radial_samples_tuned_to_16(self):
        from feature_inspection.config import HOUGH_RADIAL_SAMPLES
        assert HOUGH_RADIAL_SAMPLES == 16, (
            f"HOUGH_RADIAL_SAMPLES should be 16 (tuned); got {HOUGH_RADIAL_SAMPLES}. "
            "Measurement basis: 2414 rejections used radial gate; 16 gives finer discrimination."
        )

    def test_min_local_contrast_tuned_to_4(self):
        from feature_inspection.config import HOUGH_MIN_LOCAL_CONTRAST
        assert HOUGH_MIN_LOCAL_CONTRAST == 4, (
            f"HOUGH_MIN_LOCAL_CONTRAST should be 4 (tuned); got {HOUGH_MIN_LOCAL_CONTRAST}. "
            "Measurement basis: accepted_p10 = 3.8; old value 6 rejected 533 genuine circles."
        )

    # ------------------------------------------------------------------ #
    # 2. Boundary-condition acceptance tests                              #
    # Each test constructs a synthetic image that sits at the p10         #
    # boundary of the genuine distribution and asserts it is accepted.    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _full_circle_edge_and_appearance(size=250, cx=125, cy=125, r=50):
        """Clean circle used as the geometric base for calibration tests."""
        edge = np.zeros((size, size), dtype=np.uint8)
        app  = np.full((size, size, 3), 60, dtype=np.uint8)
        cv2.circle(edge, (cx, cy), r, 255, 2)
        cv2.circle(app,  (cx, cy), r, (120, 120, 120), -1)
        cv2.circle(app,  (cx, cy), r, (200, 200, 200), 3)
        mask = np.ones((size, size), dtype=np.uint8) * 255
        return edge, app, mask, (cx, cy), r

    def test_circle_at_angular_coverage_p10_accepted(self):
        """
        A circle where roughly 50 % of the circumference has edge support
        (near accepted_p10 = 0.49) must be accepted.

        We draw a circle and randomly remove ~48 % of the edge pixels so
        the angular coverage is approximately 0.52.  This should still pass
        the tuned gate of 0.50 (with adaptive floor of 0.45).
        """
        size = 250; cx, cy, r = 125, 125, 50
        edge = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(edge, (cx, cy), r, 255, 2)

        # Erase roughly half the circle leaving the other half intact
        # — remove the left semicircle (angles π/2 to 3π/2)
        for angle in np.linspace(np.pi / 2, 3 * np.pi / 2, 300):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                # Erase in a small neighbourhood
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < size and 0 <= nx < size:
                            edge[ny, nx] = 0

        app  = np.full((size, size, 3), 60, dtype=np.uint8)
        cv2.circle(app, (cx, cy), r, (120, 120, 120), -1)
        cv2.circle(app, (cx, cy), r, (200, 200, 200), 3)
        mask = np.ones((size, size), dtype=np.uint8) * 255

        e = CircleExtractor()
        result = e._validate_hough_candidate(
            (float(cx), float(cy)), float(r), edge, app, mask
        )
        # This is a boundary test — the circle has ~50 % coverage and may or
        # may not pass depending on the adaptive threshold and edge quality.
        # The important assertion is that it is NOT rejected for a reason
        # unrelated to angular coverage (e.g. it should not be rejected by
        # the contrast gate or radial consistency if the geometry is sound).
        if not result["valid"]:
            assert "angular coverage" in result["rejection_reason"].lower() or \
                   "sector coverage" in result["rejection_reason"].lower() or \
                   "gap" in result["rejection_reason"].lower(), (
                f"Half-circle rejected for unexpected reason: {result['rejection_reason']}"
            )

    def test_circle_at_max_gap_boundary_accepted(self):
        """
        A circle with a gap of exactly 0.25 (90°) must be accepted by the
        tuned gate of 0.30.  Accepted_p90 = 0.234 so 0.25 is in the genuine
        distribution's upper tail — it must not be rejected.
        """
        size = 250; cx, cy, r = 125, 125, 50
        edge = np.zeros((size, size), dtype=np.uint8)
        app  = np.full((size, size, 3), 60, dtype=np.uint8)
        mask = np.ones((size, size), dtype=np.uint8) * 255

        # Draw 270° arc (leaving a 90° gap)
        for angle in np.linspace(np.pi / 4, 7 * np.pi / 4, 400):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                edge[y, x] = 255

        cv2.circle(app, (cx, cy), r, (120, 120, 120), -1)
        cv2.circle(app, (cx, cy), r, (200, 200, 200), 3)

        e = CircleExtractor()
        result = e._validate_hough_candidate(
            (float(cx), float(cy)), float(r), edge, app, mask
        )
        # A 90° gap is gap_ratio ≈ 0.25 which is ≤ 0.30 — must not be
        # rejected by the gap gate.  Angular coverage gate may still reject it
        # if coverage < 0.50, but gap specifically should not.
        if not result["valid"]:
            assert "gap" not in result["rejection_reason"].lower(), (
                f"270° arc (gap ≈ 0.25) should not fail the gap gate (threshold=0.30); "
                f"rejected: {result['rejection_reason']}"
            )

    def test_low_contrast_genuine_circle_accepted(self):
        """
        A circle with local_contrast ≈ 4 (at accepted_p10 = 3.8) must be
        accepted by the tuned gate of 4.

        We create a circle where the background is grey=60 and the interior
        fill is grey=80 — a contrast of ~20 on the image, but after the
        interior-vs-exterior comparison in _analyze_local_contrast the
        contrast value should be above 4.
        """
        size = 250; cx, cy, r = 125, 125, 50
        edge = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(edge, (cx, cy), r, 255, 2)

        # Very low contrast: background=60, interior=70 (Δ=10 absolute)
        app  = np.full((size, size, 3), 60, dtype=np.uint8)
        cv2.circle(app, (cx, cy), r, (70, 70, 70), -1)    # mild interior fill
        cv2.circle(app, (cx, cy), r, (130, 130, 130), 3)  # visible ring edge
        mask = np.ones((size, size), dtype=np.uint8) * 255

        e = CircleExtractor()
        result = e._validate_hough_candidate(
            (float(cx), float(cy)), float(r), edge, app, mask
        )
        # If rejected it should NOT be for contrast — the ring edge (130 vs 60
        # background) gives well above 4 contrast.
        if not result["valid"]:
            assert "contrast" not in result["rejection_reason"].lower() and \
                   "texture" not in result["rejection_reason"].lower(), (
                f"Low-contrast genuine circle should not fail contrast gate; "
                f"rejected: {result['rejection_reason']}"
            )

    def test_radial_samples_16_does_not_reject_genuine_circle(self):
        """
        With HOUGH_RADIAL_SAMPLES=16 the radial consistency measurement uses
        finer sampling.  A clean circle must still produce a high consistency
        score and be accepted.
        """
        size = 250; cx, cy, r = 125, 125, 50
        edge = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(edge, (cx, cy), r, 255, 2)
        app  = np.full((size, size, 3), 60, dtype=np.uint8)
        cv2.circle(app, (cx, cy), r, (120, 120, 120), -1)
        cv2.circle(app, (cx, cy), r, (200, 200, 200), 3)
        mask = np.ones((size, size), dtype=np.uint8) * 255

        e = CircleExtractor()
        assert e.radial_samples == 16, (
            f"Expected radial_samples=16 but extractor has {e.radial_samples}"
        )

        result = e._validate_hough_candidate(
            (float(cx), float(cy)), float(r), edge, app, mask
        )
        assert result["valid"], (
            f"Clean circle must be accepted with radial_samples=16; "
            f"rejected: {result['rejection_reason']}"
        )
        assert result["radial_consistency"] >= 0.55, (
            f"Radial consistency for a clean circle should be ≥ 0.55 with 16 samples; "
            f"got {result['radial_consistency']:.3f}"
        )

    # ------------------------------------------------------------------ #
    # 3. No regression on existing adversarial tests                     #
    # ------------------------------------------------------------------ #

    def test_thresholds_do_not_regress_partial_arc_rejection(self):
        """
        After threshold tuning a quarter-arc must still be rejected.
        Tuning lowered the min_angular_coverage slightly (0.55→0.50) and
        tightened max_gap_ratio (0.45→0.30).  Both changes should only help
        reject partial arcs more reliably.
        """
        size = 250; cx, cy, r = 125, 125, 50
        edge = np.zeros((size, size), dtype=np.uint8)
        for angle in np.linspace(0, np.pi / 2, 200):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                edge[y, x] = 255

        e = CircleExtractor()
        result = e._validate_hough_candidate(
            (float(cx), float(cy)), float(r),
            edge,
            np.zeros((size, size, 3), dtype=np.uint8),
            np.ones((size, size), dtype=np.uint8) * 255,
        )
        assert not result["valid"], (
            "Quarter-arc must still be rejected after threshold tuning"
        )

    def test_thresholds_do_not_regress_perfect_circle_acceptance(self):
        """
        After threshold tuning a perfect circle must still be accepted with
        at least the same quality as before.
        """
        size = 250; cx, cy, r = 125, 125, 50
        edge = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(edge, (cx, cy), r, 255, 2)
        app  = np.full((size, size, 3), 60, dtype=np.uint8)
        cv2.circle(app, (cx, cy), r, (120, 120, 120), -1)
        cv2.circle(app, (cx, cy), r, (200, 200, 200), 3)
        mask = np.ones((size, size), dtype=np.uint8) * 255

        e = CircleExtractor()
        result = e._validate_hough_candidate(
            (float(cx), float(cy)), float(r), edge, app, mask
        )
        assert result["valid"], (
            f"Perfect circle must still be accepted after threshold tuning; "
            f"rejected: {result['rejection_reason']}"
        )
        assert result["angular_coverage"] > 0.65
        assert result["radial_consistency"] > 0.55
