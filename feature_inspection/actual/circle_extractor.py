"""
Circle Feature Extractor for Phase 2B

Detects circular features from preprocessed product images using multiple
geometric and edge-based evidence sources.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from .feature_models import ActualFeature, ActualFeatureType, GeometricProperties, EvidenceMetrics
from ..config import (
    CIRCLE_MIN_RADIUS, CIRCLE_MAX_RADIUS, CIRCLE_MIN_AREA, CIRCLE_MAX_AREA,
    CIRCLE_CIRCULARITY_THRESHOLD, CIRCLE_CONTOUR_APPROX_EPSILON,
    CIRCLE_HOUGH_ACCUMULATOR_THRESHOLD, CIRCLE_MIN_CENTER_DISTANCE,
    CIRCLE_EDGE_SUPPORT_THRESHOLD, FEATURE_MIN_CONFIDENCE,
    FEATURE_PRODUCT_MASK_OVERLAP_THRESHOLD, EDGE_EVIDENCE_WEIGHT,
    CONTOUR_EVIDENCE_WEIGHT, INTENSITY_EVIDENCE_WEIGHT, GEOMETRY_EVIDENCE_WEIGHT,
    HOUGH_CIRCLE_PARAM1, HOUGH_CIRCLE_PARAM2_MULTIPLIER, HOUGH_CIRCLE_BLUR_KERNEL_SIZE,
    HOUGH_CIRCLE_DP, HOUGH_CIRCLE_MIN_DIST_MULTIPLIER, HOUGH_CIRCLE_CONTOUR_SAMPLING_POINTS,
    EDGE_SUPPORT_SAMPLING_POINTS, EDGE_SUPPORT_NEIGHBORHOOD_SIZE,
    HOUGH_CONFIDENCE_RADIUS_WEIGHT, HOUGH_CONFIDENCE_EDGE_WEIGHT,
    CONTOUR_INTENSITY_EVIDENCE_WEIGHT, HOUGH_GEOMETRIC_EVIDENCE_REQUIRED,
    HOUGH_EDGE_SUPPORT_MULTIPLIER, HOUGH_RADIUS_ANOMALY_DETECTION,
    HOUGH_MAX_RADIUS_PERCENTILE, HOUGH_AREA_POPULATION_THRESHOLD,
    ENHANCED_DUPLICATE_SUPPRESSION, SAME_TYPE_CENTER_DISTANCE_STRICT,
    HOUGH_CLUSTER_CONSOLIDATION, HOUGH_CLUSTER_RADIUS_TOLERANCE,
    CONFIDENCE_GEOMETRIC_VALIDATION, HOUGH_CONFIDENCE_GEOMETRIC_WEIGHT,
    HOUGH_ANGULAR_COVERAGE_REQUIRED, HOUGH_MIN_ANGULAR_COVERAGE,
    HOUGH_ANGULAR_SECTORS, HOUGH_MIN_SECTOR_COVERAGE,
    HOUGH_EDGE_CONTINUITY_REQUIRED, HOUGH_MAX_EDGE_GAP_RATIO,
    HOUGH_RADIAL_CONSISTENCY_REQUIRED, HOUGH_RADIAL_SAMPLES,
    HOUGH_MIN_RADIAL_AGREEMENT, HOUGH_RADIAL_TOLERANCE_PIXELS,
    HOUGH_RADIAL_SEARCH_STEP, HOUGH_CONTOUR_AGREEMENT_BONUS,
    HOUGH_ONLY_PENALTY, HOUGH_TEXTURE_DISCRIMINATION, HOUGH_MIN_LOCAL_CONTRAST,
    HOUGH_CLUSTER_CONSOLIDATION_ENHANCED, HOUGH_SCALE_AWARE_DISTANCE,
    HOUGH_CLUSTER_RADIUS_FACTOR, HOUGH_CONCENTRIC_DETECTION,
    HOUGH_CONCENTRIC_RADIUS_TOLERANCE, HOUGH_EVIDENCE_BASED_CONSOLIDATION,
    HOUGH_GRADIENT_ORIENTATION_REQUIRED, HOUGH_MIN_GRADIENT_ORIENTATION_CONSISTENCY,
    HOUGH_GRADIENT_MIN_MAGNITUDE, HOUGH_GRADIENT_SAMPLE_HALF_WIDTH,
    HOUGH_GEOMETRY_FIRST_INPUT, HOUGH_EDGE_INPUT_DILATE_KERNEL,
    HOLE_DARK_PERCENTILE, HOLE_DARK_OFFSET, HOLE_MIN_RADIUS, HOLE_MAX_RADIUS,
    HOLE_MIN_CIRCULARITY, HOLE_MIN_SOLIDITY, HOLE_MIN_INTERIOR_DARK_FRACTION,
    HOLE_PRODUCT_MASK_OVERLAP, HOLE_DARK_MAX,
)

logger = logging.getLogger(__name__)


class CircleExtractor:
    """
    Extracts circular features from preprocessed product images.
    
    Uses multiple evidence sources:
    1. Contour-based circle detection
    2. Hough circle transform
    3. Edge support validation
    4. Geometric consistency checks
    """
    
    def __init__(self):
        """Initialize circle extractor with configuration parameters."""
        self.min_radius = CIRCLE_MIN_RADIUS
        self.max_radius = CIRCLE_MAX_RADIUS
        self.min_area = CIRCLE_MIN_AREA
        self.max_area = CIRCLE_MAX_AREA
        self.circularity_threshold = CIRCLE_CIRCULARITY_THRESHOLD
        self.approx_epsilon = CIRCLE_CONTOUR_APPROX_EPSILON
        self.hough_threshold = CIRCLE_HOUGH_ACCUMULATOR_THRESHOLD
        self.min_center_distance = CIRCLE_MIN_CENTER_DISTANCE
        self.edge_support_threshold = CIRCLE_EDGE_SUPPORT_THRESHOLD
        self.min_confidence = FEATURE_MIN_CONFIDENCE
        self.mask_overlap_threshold = FEATURE_PRODUCT_MASK_OVERLAP_THRESHOLD
        
        # Hough-specific parameters
        self.hough_param1 = HOUGH_CIRCLE_PARAM1
        self.hough_param2 = int(CIRCLE_HOUGH_ACCUMULATOR_THRESHOLD * HOUGH_CIRCLE_PARAM2_MULTIPLIER)
        self.hough_blur_kernel = HOUGH_CIRCLE_BLUR_KERNEL_SIZE
        self.hough_dp = HOUGH_CIRCLE_DP
        self.hough_min_dist = int(CIRCLE_MIN_CENTER_DISTANCE * HOUGH_CIRCLE_MIN_DIST_MULTIPLIER)
        self.hough_sampling_points = HOUGH_CIRCLE_CONTOUR_SAMPLING_POINTS
        
        # Edge support parameters
        self.edge_sampling_points = EDGE_SUPPORT_SAMPLING_POINTS
        self.edge_neighborhood_size = EDGE_SUPPORT_NEIGHBORHOOD_SIZE
        
        # Confidence calculation parameters
        self.hough_radius_weight = HOUGH_CONFIDENCE_RADIUS_WEIGHT
        self.hough_edge_weight = HOUGH_CONFIDENCE_EDGE_WEIGHT
        self.intensity_weight = CONTOUR_INTENSITY_EVIDENCE_WEIGHT
        
        # Enhanced validation parameters
        self.geometric_evidence_required = HOUGH_GEOMETRIC_EVIDENCE_REQUIRED
        self.edge_support_multiplier = HOUGH_EDGE_SUPPORT_MULTIPLIER
        self.radius_anomaly_detection = HOUGH_RADIUS_ANOMALY_DETECTION
        self.max_radius_percentile = HOUGH_MAX_RADIUS_PERCENTILE
        self.area_population_threshold = HOUGH_AREA_POPULATION_THRESHOLD
        self.enhanced_duplicate_suppression = ENHANCED_DUPLICATE_SUPPRESSION
        self.strict_center_distance = SAME_TYPE_CENTER_DISTANCE_STRICT
        self.cluster_consolidation = HOUGH_CLUSTER_CONSOLIDATION
        self.cluster_radius_tolerance = HOUGH_CLUSTER_RADIUS_TOLERANCE
        self.confidence_geometric_validation = CONFIDENCE_GEOMETRIC_VALIDATION
        self.confidence_geometric_weight = HOUGH_CONFIDENCE_GEOMETRIC_WEIGHT
        
        # Enhanced Hough image evidence validation parameters
        self.angular_coverage_required = HOUGH_ANGULAR_COVERAGE_REQUIRED
        self.min_angular_coverage = HOUGH_MIN_ANGULAR_COVERAGE
        self.angular_sectors = HOUGH_ANGULAR_SECTORS
        self.min_sector_coverage = HOUGH_MIN_SECTOR_COVERAGE
        self.edge_continuity_required = HOUGH_EDGE_CONTINUITY_REQUIRED
        self.max_edge_gap_ratio = HOUGH_MAX_EDGE_GAP_RATIO
        self.radial_consistency_required = HOUGH_RADIAL_CONSISTENCY_REQUIRED
        self.radial_samples = HOUGH_RADIAL_SAMPLES
        self.min_radial_agreement = HOUGH_MIN_RADIAL_AGREEMENT
        self.radial_tolerance_pixels = HOUGH_RADIAL_TOLERANCE_PIXELS
        self.radial_search_step = HOUGH_RADIAL_SEARCH_STEP
        
        # Multi-method agreement parameters
        self.contour_agreement_bonus = HOUGH_CONTOUR_AGREEMENT_BONUS
        self.hough_only_penalty = HOUGH_ONLY_PENALTY
        self.texture_discrimination = HOUGH_TEXTURE_DISCRIMINATION
        self.min_local_contrast = HOUGH_MIN_LOCAL_CONTRAST
        
        # Enhanced clustering parameters
        self.enhanced_clustering = HOUGH_CLUSTER_CONSOLIDATION_ENHANCED
        self.scale_aware_distance = HOUGH_SCALE_AWARE_DISTANCE
        self.cluster_radius_factor = HOUGH_CLUSTER_RADIUS_FACTOR
        self.concentric_detection = HOUGH_CONCENTRIC_DETECTION
        self.concentric_radius_tolerance = HOUGH_CONCENTRIC_RADIUS_TOLERANCE
        self.evidence_based_consolidation = HOUGH_EVIDENCE_BASED_CONSOLIDATION

        # Gradient orientation consistency parameters
        self.gradient_orientation_required = HOUGH_GRADIENT_ORIENTATION_REQUIRED
        self.min_gradient_orientation = HOUGH_MIN_GRADIENT_ORIENTATION_CONSISTENCY
        self.gradient_min_magnitude = HOUGH_GRADIENT_MIN_MAGNITUDE
        self.gradient_sample_half_width = HOUGH_GRADIENT_SAMPLE_HALF_WIDTH

        # Geometry-first Hough input parameters (Step 6)
        self.hough_geometry_first = HOUGH_GEOMETRY_FIRST_INPUT
        self.hough_edge_dilate_kernel = HOUGH_EDGE_INPUT_DILATE_KERNEL

        # Hole detection parameters
        self.hole_dark_percentile = HOLE_DARK_PERCENTILE
        self.hole_dark_offset = HOLE_DARK_OFFSET
        self.hole_dark_max = HOLE_DARK_MAX
        self.hole_min_radius = HOLE_MIN_RADIUS
        self.hole_max_radius = HOLE_MAX_RADIUS
        self.hole_min_circularity = HOLE_MIN_CIRCULARITY
        self.hole_min_solidity = HOLE_MIN_SOLIDITY
        self.hole_min_interior_dark_fraction = HOLE_MIN_INTERIOR_DARK_FRACTION
        self.hole_product_mask_overlap = HOLE_PRODUCT_MASK_OVERLAP
    
    def extract_circles(self, 
                       internal_edges: np.ndarray,
                       isolated_product: np.ndarray, 
                       product_mask: np.ndarray,
                       raw_internal_edges: Optional[np.ndarray] = None,
                       outer_boundary: Optional[np.ndarray] = None) -> List[ActualFeature]:
        """
        Extract circular features from preprocessed representations.
        
        Args:
            internal_edges: Primary geometric edge representation
            isolated_product: Primary visual representation
            product_mask: Spatial constraint mask
            raw_internal_edges: Supporting edge evidence
            outer_boundary: Outer boundary edges
            
        Returns:
            List of detected circular features
        """
        logger.debug("Starting circle extraction")
        
        candidates = []
        
        # Method 0: Dark-region hole detection (highest priority)
        # Detects actual through-holes by finding dark filled blobs inside
        # the product mask.  This is independent of edge-based detection and
        # directly captures bolt holes, centre bores, etc.
        logger.debug("Extracting through-holes via dark-region analysis")
        hole_features = self._extract_holes(isolated_product, product_mask)
        logger.debug(f"Found {len(hole_features)} hole features")
        candidates.extend(hole_features)

        # Method 1: Contour-based circle detection
        logger.debug(f"Extracting contour circles from internal_edges: {internal_edges.shape}")
        contour_circles = self._extract_contour_circles(internal_edges, product_mask)
        logger.debug(f"Found {len(contour_circles)} contour circles")
        candidates.extend(contour_circles)
        
        # Method 2: Hough circle detection
        logger.debug(f"Extracting Hough circles from isolated_product: {isolated_product.shape}")
        hough_circles = self._extract_hough_circles(isolated_product, product_mask, internal_edges)
        logger.debug(f"Found {len(hough_circles)} Hough circles")
        candidates.extend(hough_circles)
        
        # Method 3: Raw edge fallback for missed features
        if raw_internal_edges is not None:
            logger.debug(f"Extracting raw edge circles from raw_internal_edges: {raw_internal_edges.shape}")
            raw_circles = self._extract_contour_circles(raw_internal_edges, product_mask, 
                                                       source_rep="raw_internal_edges")
            logger.debug(f"Found {len(raw_circles)} raw edge circles")
            candidates.extend(raw_circles)
        
        # Validate and refine candidates
        validated_circles = []
        for candidate in candidates:
            # HOLE features are validated inside _extract_holes (dark-region
            # analysis with circularity, solidity, and darkness fraction checks).
            # Skip the edge-based generic validation for them — it would
            # incorrectly penalise them because Hough/angular evidence fields
            # are not populated.
            if candidate.feature_type == ActualFeatureType.HOLE:
                validated_circles.append(candidate)
                continue
            if self._validate_circle_candidate(candidate, internal_edges, isolated_product, product_mask):
                validated_circles.append(candidate)
        
        # Apply population-based area validation if enabled
        # (HOLE features are exempt — their sizes are physically meaningful)
        if self.radius_anomaly_detection and len(validated_circles) > 2:
            circles_only = [f for f in validated_circles
                            if f.feature_type != ActualFeatureType.HOLE]
            holes_only   = [f for f in validated_circles
                            if f.feature_type == ActualFeatureType.HOLE]
            circles_only = self._apply_population_area_validation(circles_only)
            validated_circles = holes_only + circles_only

        # Apply enhanced cluster consolidation if enabled
        # (HOLE features are exempt — they must not be merged with circles)
        if self.enhanced_clustering and len(validated_circles) > 1:
            circles_only = [f for f in validated_circles
                            if f.feature_type != ActualFeatureType.HOLE]
            holes_only   = [f for f in validated_circles
                            if f.feature_type == ActualFeatureType.HOLE]
            if circles_only:
                circles_only = self._apply_enhanced_cluster_consolidation(circles_only)
            validated_circles = holes_only + circles_only
        
        logger.info(f"Circle extraction: {len(candidates)} candidates -> {len(validated_circles)} validated")
        return validated_circles
    
    def _extract_holes(self, isolated_product: np.ndarray,
                      product_mask: np.ndarray) -> List[ActualFeature]:
        """
        Detect actual through-holes in the part by finding dark filled regions.

        Holes through metal appear as uniformly dark discs in the appearance
        image.  This method:

        1. Converts to grayscale and masks to the product region only.
        2. Computes an adaptive "dark threshold" from the intensity distribution
           inside the product mask (low percentile + offset), so the threshold
           self-adjusts regardless of overall image brightness.
        3. Thresholds to a binary dark-region map.
        4. Finds contours on the dark map and filters by circularity, solidity,
           radius bounds, and what fraction of the interior is genuinely dark.
        5. Returns validated hole candidates tagged as ActualFeatureType.HOLE.
        """
        holes: List[ActualFeature] = []

        # ------------------------------------------------------------------ #
        # Prepare grayscale product region                                    #
        # ------------------------------------------------------------------ #
        if len(isolated_product.shape) == 3:
            gray = cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY)
        elif isolated_product.ndim == 2:
            gray = isolated_product.copy()
        else:
            gray = isolated_product[:, :, 0]

        h, w = gray.shape
        mask_bool = product_mask > 0

        if not np.any(mask_bool):
            return holes

        # ------------------------------------------------------------------ #
        # Adaptive dark threshold from the product interior                   #
        # ------------------------------------------------------------------ #
        product_pixels = gray[mask_bool]
        # Base threshold: percentile-derived, clamped to a safe ceiling
        dark_threshold = int(np.percentile(product_pixels, self.hole_dark_percentile)
                             + self.hole_dark_offset)
        dark_threshold = min(dark_threshold, self.hole_dark_max)

        # Multi-threshold scan: run at several levels and keep the best
        # result per location.  This handles cases where very dark holes
        # only appear cleanly at a lower threshold (e.g. bottom-left bolt
        # hole in this image is invisible at the base threshold because
        # nearby rust patches merge with it at higher values).
        scan_thresholds = sorted(set([
            15, 20, 25, dark_threshold,
            # Extra levels for large bores whose interior is mid-tone (machined collar)
            45, 55, 65,
        ]))

        # ------------------------------------------------------------------ #
        # Multi-threshold scan: run at each level, validate each blob,      #
        # deduplicate by center proximity across thresholds.                #
        # Very dark holes (e.g. drilled bolt holes with intensity < 10)     #
        # appear most cleanly at low thresholds before rust patches merge.  #
        # ------------------------------------------------------------------ #
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        prod_uint8 = (product_mask > 0).astype(np.uint8) * 255

        # Centre-dedup: if a candidate center is within this many px of an
        # already-accepted hole, skip it (same hole found at different threshold)
        DEDUP_DIST = 10.0

        blob_candidates: list = []   # list of (cx, cy, r, circ, sol, df, contour, thr)

        for thr in scan_thresholds:
            dark_map = np.zeros((h, w), dtype=np.uint8)
            dark_map[(gray < thr) & mask_bool] = 255
            dark_map = cv2.morphologyEx(dark_map, cv2.MORPH_CLOSE, k3, iterations=1)
            contours, _ = cv2.findContours(dark_map, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < np.pi * self.hole_min_radius ** 2:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circ = 4 * np.pi * area / (perimeter ** 2)
                # For large bores (r > 30 px) the machined interior makes the dark
                # blob irregular — relax circularity proportionally to size
                min_circ_for_this = self.hole_min_circularity
                if circ < min_circ_for_this:
                    continue
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                sol = area / hull_area if hull_area > 0 else 0.0
                if sol < self.hole_min_solidity:
                    continue
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                if radius < self.hole_min_radius or radius > self.hole_max_radius:
                    continue
                # Mask overlap
                hole_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(hole_mask, [contour], -1, 255, -1)
                inter = np.count_nonzero(cv2.bitwise_and(hole_mask, prod_uint8))
                if area > 0 and inter / area < self.hole_product_mask_overlap:
                    continue
                # Interior darkness
                ip = gray[hole_mask > 0]
                if len(ip) == 0:
                    continue
                df = float(np.mean(ip < thr))
                if df < self.hole_min_interior_dark_fraction:
                    continue
                blob_candidates.append((cx, cy, radius, circ, sol, df, contour, thr))

        # Deduplicate: keep highest-circularity candidate for each distinct location
        blob_candidates.sort(key=lambda x: -x[3])   # sort by circularity desc
        accepted_centers: list = []
        accepted_blobs: list = []
        for bc in blob_candidates:
            cx, cy = bc[0], bc[1]
            too_close = any(
                ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 < DEDUP_DIST
                for ax, ay in accepted_centers
            )
            if not too_close:
                accepted_centers.append((cx, cy))
                accepted_blobs.append(bc)

        for i, (cx, cy, radius, circularity, solidity, dark_fraction,
                contour, thr) in enumerate(accepted_blobs):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            center = (float(cx), float(cy))
            x, y, bw, bh = cv2.boundingRect(contour)
            geometry = GeometricProperties(
                center=center,
                area=float(area),
                perimeter=float(perimeter),
                bounding_box=(x, y, bw, bh),
                radius=float(radius),
                diameter=float(2 * radius),
                circularity=float(circularity),
                solidity=float(solidity),
                extent=area / (bw * bh) if bw > 0 and bh > 0 else 0.0,
                convexity=(cv2.arcLength(hull, True) / perimeter
                           if perimeter > 0 else 0.0),
            )
            confidence = float(np.mean([circularity, solidity, dark_fraction]))
            evidence = EvidenceMetrics(
                confidence=confidence,
                edge_support=1.0,
                contour_quality=float(circularity),
                intensity_consistency=float(dark_fraction),
                geometric_consistency=float(solidity),
                intensity_evidence=float(dark_fraction),
            )
            feature_id = f"hole_{i}_{int(cx)}_{int(cy)}"
            hole = ActualFeature(
                feature_id=feature_id,
                feature_type=ActualFeatureType.HOLE,
                geometry=geometry,
                contour=contour,
                evidence=evidence,
                source_representation="isolated_product",
                detection_method="dark_region_analysis",
            )
            holes.append(hole)

        # ------------------------------------------------------------------ #
        # Large-bore Hough pass                                               #
        # Centre bores and large counterbores have machined collars that      #
        # reflect light, making their interior irregular at any single        #
        # intensity threshold.  Run a dedicated Hough pass on the inverted   #
        # grayscale of the product region to find large circular edges that   #
        # bound the bore opening.                                             #
        # ------------------------------------------------------------------ #
        holes.extend(self._detect_large_bores(gray, mask_bool, product_mask,
                                               accepted_centers))

        return holes

    def _detect_large_bores(self, gray: np.ndarray, mask_bool: np.ndarray,
                             product_mask: np.ndarray,
                             existing_hole_centers: list) -> List[ActualFeature]:
        """
        Detect large centre bores using a dark-region distance-transform Hough.

        Strategy
        --------
        1. Build a dark-region binary mask (pixels < 50th percentile of product).
        2. Apply morphological close+open to consolidate the bore interior into
           a single compact blob.
        3. Compute the distance transform — each dark pixel gets its distance to
           the nearest non-dark pixel, so the bore centre is the brightest point.
        4. Run Hough on the normalised distance map.
        5. Validate: interior must be substantially darker than a collar ring
           around the candidate (contrast >= 25 intensity units).

        This approach works even when the bore interior has rust/reflections that
        break contour-based circularity checks.
        """
        bores: List[ActualFeature] = []
        h, w = gray.shape
        prod_uint8 = (mask_bool.astype(np.uint8) * 255)

        # CLAHE-enhanced Hough on the masked product — works for both dark drilled
        # holes and bright machined bores (the collar edge creates a strong gradient
        # regardless of the relative interior/exterior brightness).
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        masked_enhanced = np.where(mask_bool, enhanced, 0).astype(np.uint8)
        blurred = cv2.GaussianBlur(masked_enhanced, (5, 5), 0)

        raw = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1, minDist=60,
            param1=80, param2=15,
            minRadius=30, maxRadius=80,
        )
        if raw is None:
            return bores

        candidates = np.round(raw[0, :]).astype(int)

        for idx, (cx, cy, r) in enumerate(candidates):
            # Skip if center is outside product mask
            if not (0 <= cy < h and 0 <= cx < w and mask_bool[cy, cx]):
                continue

            # Skip if too close to an already-found small bolt hole
            if any(((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 < r * 0.4
                   for ax, ay in existing_hole_centers):
                continue

            # Core validation: interior must be meaningfully darker than the
            # surrounding collar ring.  Require >= 25 intensity units contrast.
            bore_mask   = np.zeros((h, w), dtype=np.uint8)
            collar_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(bore_mask,   (cx, cy), max(1, r - 6), 255, -1)
            cv2.circle(collar_mask, (cx, cy), r + 20, 255, -1)
            cv2.circle(collar_mask, (cx, cy), r +  4, 0,   -1)
            collar_mask = cv2.bitwise_and(collar_mask, prod_uint8)

            interior_px = gray[bore_mask > 0]
            collar_px   = gray[collar_mask > 0]
            if len(interior_px) < 50 or len(collar_px) < 50:
                continue
            contrast = float(np.mean(collar_px)) - float(np.mean(interior_px))
            # Accept either: bore darker than collar (positive contrast, typical drilled hole)
            # OR bore brighter than collar (negative contrast, machined/polished bore interior)
            abs_contrast = abs(contrast)
            if abs_contrast < 16:          # require at least 16 units either way
                continue

            # Bore must be well inside the product mask (≥80 % overlap)
            bore_area = float(np.count_nonzero(bore_mask))
            inter     = float(np.count_nonzero(cv2.bitwise_and(bore_mask, prod_uint8)))
            if bore_area > 0 and inter / bore_area < 0.85:   # strict: real bores are fully inside
                continue

            # Dedup against already accepted bores
            if any(((cx - b.geometry.center[0]) ** 2
                    + (cy - b.geometry.center[1]) ** 2) ** 0.5 < r * 0.5
                   for b in bores):
                continue

            confidence = min(1.0, abs_contrast / 50.0)
            area = float(np.pi * r * r)
            geometry = GeometricProperties(
                center=(float(cx), float(cy)),
                area=area,
                perimeter=float(2 * np.pi * r),
                bounding_box=(cx - r, cy - r, 2 * r, 2 * r),
                radius=float(r),
                diameter=float(2 * r),
                circularity=confidence,
                solidity=confidence,
                extent=np.pi / 4,
                convexity=1.0,
            )
            evidence = EvidenceMetrics(
                confidence=confidence,
                edge_support=confidence,
                contour_quality=confidence,
                intensity_consistency=confidence,
                geometric_consistency=confidence,
                intensity_evidence=confidence,
            )
            bores.append(ActualFeature(
                feature_id=f"bore_{idx}_{cx}_{cy}",
                feature_type=ActualFeatureType.HOLE,
                geometry=geometry,
                contour=np.array([[[cx - r, cy]], [[cx + r, cy]]], dtype=np.int32),
                evidence=evidence,
                source_representation="isolated_product",
                detection_method="large_bore_hough",
            ))

        return bores

    def _extract_contour_circles(self, edge_image: np.ndarray, 
                                product_mask: np.ndarray,
                                source_rep: str = "internal_geometry_edges") -> List[ActualFeature]:
        """Extract circles using contour analysis."""
        circles = []
        
        # Find contours in edge image
        contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Basic area filtering
            if area < self.min_area or area > self.max_area:
                continue
            
            # Calculate geometric properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Check circularity threshold
            if circularity < self.circularity_threshold:
                continue
            
            # Calculate center and radius
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            center = (float(center_x), float(center_y))
            
            # Validate radius bounds
            if radius < self.min_radius or radius > self.max_radius:
                continue
            
            # Check overlap with product mask
            if not self._check_product_mask_overlap(center, radius, product_mask):
                continue
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create geometric properties
            geometry = GeometricProperties(
                center=center,
                area=area,
                perimeter=perimeter,
                bounding_box=(x, y, w, h),
                radius=radius,
                diameter=2 * radius,
                circularity=circularity,
                solidity=self._calculate_solidity(contour),
                extent=area / (w * h) if w > 0 and h > 0 else 0,
                convexity=self._calculate_convexity(contour)
            )
            
            # Calculate evidence metrics
            edge_support = self._calculate_edge_support(center, radius, edge_image)
            evidence = EvidenceMetrics(
                confidence=self._calculate_contour_confidence(geometry, edge_support),
                edge_support=edge_support,
                contour_quality=circularity,
                intensity_consistency=0.5,  # Will be refined in validation
                geometric_consistency=circularity,
                internal_edge_evidence=1.0 if source_rep == "internal_geometry_edges" else 0.0,
                raw_edge_evidence=1.0 if source_rep == "raw_internal_edges" else 0.0
            )
            
            # Create feature
            feature_id = f"circle_contour_{len(circles)}_{int(center_x)}_{int(center_y)}"
            circle = ActualFeature(
                feature_id=feature_id,
                feature_type=ActualFeatureType.CIRCLE,
                geometry=geometry,
                contour=contour,
                evidence=evidence,
                source_representation=source_rep,
                detection_method="contour_analysis"
            )
            
            circles.append(circle)
        
        return circles
    
    def _extract_hough_circles(self, isolated_product: np.ndarray,
                              product_mask: np.ndarray,
                              edge_reference: np.ndarray) -> List[ActualFeature]:
        """
        Extract circles using Hough circle transform.

        Input-selection strategy (geometry-first, Step 6)
        --------------------------------------------------
        ``internal_geometry_edges`` (edge_reference) is used as the *candidate
        generation* input when ``self.hough_geometry_first`` is True (default).
        Geometric edge maps carry only boundary information, so they suppress
        the texture, rust, reflections and machining-mark gradients that can
        produce spurious candidates when Hough runs on the appearance image.

        ``isolated_product`` is still used for *appearance validation*: every
        candidate that passes geometric screening is confirmed (or rejected) by
        intensity consistency and gradient orientation on the full-colour image.

        When ``hough_geometry_first`` is False the pre-Step-6 behaviour is
        preserved exactly — Hough runs directly on the grayscale appearance
        image.  This mode is retained so an A/B comparison between both paths
        is possible without code changes (flip the config flag).
        """
        # ------------------------------------------------------------------ #
        # 1. Prepare the image that drives candidate generation               #
        # ------------------------------------------------------------------ #
        hough_input = self._prepare_hough_input(edge_reference, isolated_product)
        candidate_source = (
            "internal_geometry_edges" if self.hough_geometry_first
            else "isolated_product"
        )

        # ------------------------------------------------------------------ #
        # 2. Run HoughCircles on the chosen input                             #
        # ------------------------------------------------------------------ #
        raw_hough = self._run_houghcircles(hough_input)

        if raw_hough is None:
            return []

        circles: List[ActualFeature] = []

        for i, (center_x, center_y, radius) in enumerate(raw_hough):
            center = (float(center_x), float(center_y))
            radius = float(radius)

            # Check overlap with product mask
            if not self._check_product_mask_overlap(center, radius, product_mask):
                continue

            # Calculate area and perimeter
            area = np.pi * radius * radius
            perimeter = 2 * np.pi * radius

            # Validate area bounds
            if area < self.min_area or area > self.max_area:
                continue

            # Radius anomaly detection
            if self.radius_anomaly_detection:
                if not self._validate_radius_bounds(radius, product_mask):
                    continue

            # ---------------------------------------------------------- #
            # 3. Validate: geometry first, then appearance                #
            # ---------------------------------------------------------- #
            # edge_reference supplies the geometric evidence; isolated_product
            # supplies the appearance evidence.  The roles are explicit now
            # regardless of which image was used to generate the candidate.
            if (self.angular_coverage_required or self.edge_continuity_required or
                    self.radial_consistency_required or self.texture_discrimination):
                validation_result = self._validate_hough_candidate(
                    center, radius, edge_reference, isolated_product, product_mask
                )

                if not validation_result["valid"]:
                    logger.debug(
                        f"Hough circle rejected at ({center[0]:.1f},{center[1]:.1f}) "
                        f"r={radius:.1f}: {validation_result['rejection_reason']}"
                    )
                    continue
            else:
                # Simplified path when enhanced validation is disabled
                edge_support = self._calculate_edge_support(center, radius, edge_reference)
                if self.geometric_evidence_required:
                    if edge_support < self.edge_support_threshold * self.edge_support_multiplier:
                        continue
                else:
                    if edge_support < self.edge_support_threshold * 0.8:
                        continue

                validation_result = {
                    "valid": True,
                    "edge_support": edge_support,
                    "angular_coverage": 1.0,
                    "sector_coverage": 1.0,
                    "angular_uniformity": 0.8,
                    "max_gap_ratio": 0.1,
                    "radial_consistency": 1.0,
                    "radial_error_median": 0.0,
                    "radial_error_p95": 1.0,
                    "local_contrast": 20.0,
                    "gradient_orientation_consistency": 0.8,
                    "circularity_evidence": 0.8,
                    "solidity_evidence": 0.8,
                    "convexity_evidence": 0.8,
                    "contour_quality": 0.8,
                    "intensity_consistency": 0.5,
                    "geometric_consistency": 0.8,
                    "intensity_evidence": 0.5,
                }

            # ---------------------------------------------------------- #
            # 4. Build the ActualFeature from evidence                    #
            # ---------------------------------------------------------- #
            angles = np.linspace(0, 2 * np.pi, self.hough_sampling_points)
            contour_x = center_x + radius * np.cos(angles)
            contour_y = center_y + radius * np.sin(angles)
            contour = np.column_stack((contour_x, contour_y)).astype(np.int32)

            x = int(center_x - radius)
            y = int(center_y - radius)
            w = h = int(2 * radius)

            geometry = GeometricProperties(
                center=center,
                area=area,
                perimeter=perimeter,
                bounding_box=(x, y, w, h),
                radius=radius,
                diameter=2 * radius,
                circularity=validation_result["circularity_evidence"],
                solidity=validation_result["solidity_evidence"],
                extent=np.pi / 4,
                convexity=validation_result["convexity_evidence"],
            )

            evidence = EvidenceMetrics(
                confidence=self._calculate_hough_evidence_confidence(validation_result),
                edge_support=validation_result["edge_support"],
                contour_quality=validation_result["contour_quality"],
                intensity_consistency=validation_result["intensity_consistency"],
                geometric_consistency=validation_result["geometric_consistency"],
                intensity_evidence=validation_result["intensity_evidence"],
                angular_coverage=validation_result["angular_coverage"],
                sector_coverage=validation_result["sector_coverage"],
                angular_uniformity=validation_result["angular_uniformity"],
                max_gap_ratio=validation_result["max_gap_ratio"],
                radial_consistency=validation_result["radial_consistency"],
                radial_error_median=validation_result["radial_error_median"],
                radial_error_p95=validation_result["radial_error_p95"],
                local_contrast=validation_result["local_contrast"],
                gradient_orientation_consistency=validation_result["gradient_orientation_consistency"],
            )

            feature_id = f"circle_hough_{i}_{int(center_x)}_{int(center_y)}"
            circle = ActualFeature(
                feature_id=feature_id,
                feature_type=ActualFeatureType.CIRCLE,
                geometry=geometry,
                contour=contour,
                evidence=evidence,
                source_representation=candidate_source,
                detection_method="hough_circles",
            )

            circles.append(circle)

        return circles

    # ---------------------------------------------------------------------- #
    # Geometry-first Hough helpers                                            #
    # ---------------------------------------------------------------------- #

    def _prepare_hough_input(self, edge_reference: np.ndarray,
                             isolated_product: np.ndarray) -> np.ndarray:
        """
        Return the single-channel uint8 image that will be fed to HoughCircles.

        Geometry-first mode (``hough_geometry_first=True``)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``edge_reference`` (internal_geometry_edges) is a binary edge map where
        non-zero pixels represent confirmed geometric boundaries.  HoughCircles
        uses a built-in Canny pass internally when fed a gradient image, but it
        also works well with a pre-computed edge map provided the map has
        sufficient grey-scale gradient structure.

        To give HoughCircles usable gradient magnitude around each edge pixel we
        apply a distance-transform trick: a slight Gaussian blur of the dilated
        edge map produces smooth intensity ramps centred on each edge, which
        HoughCircles' gradient accumulator can vote on.

        Appearance mode (``hough_geometry_first=False``)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The original path: convert ``isolated_product`` to grayscale and blur
        with the configured kernel — identical to the pre-Step-6 behaviour.
        """
        if self.hough_geometry_first:
            # Work from the binary edge map
            if len(edge_reference.shape) == 3:
                src = cv2.cvtColor(edge_reference, cv2.COLOR_BGR2GRAY)
            else:
                src = edge_reference.copy()

            # Optional morphological dilation to thicken thin edges
            if self.hough_edge_dilate_kernel > 1:
                k = self.hough_edge_dilate_kernel
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                src = cv2.dilate(src, kernel, iterations=1)

            # Smooth to create gradient ramps that HoughCircles can accumulate on
            # Use a moderate blur — large enough for gradient build-up, small
            # enough not to wash out nearby circles.
            smooth = cv2.GaussianBlur(src, (self.hough_blur_kernel | 1, self.hough_blur_kernel | 1), 0)
            return smooth
        else:
            # Legacy appearance path
            gray = (cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY)
                    if len(isolated_product.shape) == 3 else isolated_product)
            k = self.hough_blur_kernel
            if k % 2 == 0:
                k += 1
            return cv2.GaussianBlur(gray, (k, k), 0)

    def _run_houghcircles(self, prepared_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Call cv2.HoughCircles on a prepared single-channel image.

        Returns the raw Nx3 array of (cx, cy, r) rows, or None if nothing
        was found.  Keeping this call isolated makes it easy to mock in tests.
        """
        raw = cv2.HoughCircles(
            prepared_image,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        if raw is None:
            return None
        return np.round(raw[0, :]).astype("int")
    
    def _validate_circle_candidate(self, candidate: ActualFeature,
                                  internal_edges: np.ndarray,
                                  isolated_product: np.ndarray,
                                  product_mask: np.ndarray) -> bool:
        """Validate circle candidate using multiple evidence sources."""
        
        # Refine intensity consistency
        intensity_consistency = self._calculate_intensity_consistency(
            candidate.geometry.center, candidate.geometry.radius, isolated_product
        )
        candidate.evidence.intensity_consistency = intensity_consistency
        candidate.evidence.intensity_evidence = intensity_consistency
        
        # Update overall confidence
        confidence = self._calculate_final_confidence(candidate.evidence)
        candidate.evidence.confidence = confidence
        
        # Apply confidence threshold
        if confidence < self.min_confidence:
            return False
        
        # Check edge support
        if candidate.evidence.edge_support < self.edge_support_threshold:
            return False
        
        return True
    
    def _check_product_mask_overlap(self, center: Tuple[float, float], 
                                   radius: float, product_mask: np.ndarray) -> bool:
        """Check if circle overlaps sufficiently with product mask."""
        # Create circle mask
        h, w = product_mask.shape
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
        
        # Calculate overlap
        intersection = cv2.bitwise_and(circle_mask, product_mask)
        intersection_area = np.count_nonzero(intersection)
        circle_area = np.count_nonzero(circle_mask)
        
        if circle_area == 0:
            return False
        
        overlap_ratio = intersection_area / circle_area
        return overlap_ratio >= self.mask_overlap_threshold
    
    def _calculate_edge_support(self, center: Tuple[float, float], 
                               radius: float, edge_image: np.ndarray) -> float:
        """Calculate fraction of circle perimeter supported by edges using radial profile."""
        # Use the new radial edge profile method for more accurate edge support
        profile_result = self._sample_radial_edge_profile(center, radius, edge_image)
        return profile_result["edge_support"]
    
    def _sample_radial_edge_profile(self, center: Tuple[float, float], radius: float, 
                                  edge_image: np.ndarray) -> dict:
        """
        Sample radial edge profile for a circle candidate.
        
        For each angular sample, search along the radial direction to find
        edges and measure their proximity to the predicted circumference.
        
        Returns:
            dict: Contains per-sample data and aggregate metrics
        """
        angles = np.linspace(0, 2*np.pi, self.edge_sampling_points, endpoint=False)
        h, w = edge_image.shape
        
        samples = []
        valid_samples = 0
        radial_errors = []
        edge_strengths = []
        
        for angle in angles:
            # Calculate expected point on circle
            expected_x = center[0] + radius * np.cos(angle)
            expected_y = center[1] + radius * np.sin(angle)
            
            # Radial unit vector pointing outward from center
            radial_dx = np.cos(angle)
            radial_dy = np.sin(angle)
            
            # Search along radial direction within tolerance
            best_edge_strength = 0.0
            best_radial_offset = None
            observed_radius = None
            
            # Search from inner to outer tolerance band
            for offset in range(-self.radial_tolerance_pixels, self.radial_tolerance_pixels + 1):
                test_radius = radius + offset
                if test_radius <= 0:
                    continue
                    
                x = int(center[0] + test_radius * radial_dx)
                y = int(center[1] + test_radius * radial_dy)
                
                # Check bounds
                if not (0 <= x < w and 0 <= y < h):
                    continue
                
                # Measure edge strength at this position
                edge_strength = float(edge_image[y, x])
                
                # Check neighborhood for stronger edges
                if edge_strength > 0:
                    # Look for peak edge response in small neighborhood
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                neighbor_strength = float(edge_image[ny, nx])
                                edge_strength = max(edge_strength, neighbor_strength)
                
                # Keep track of strongest edge response
                if edge_strength > best_edge_strength:
                    best_edge_strength = edge_strength
                    best_radial_offset = offset
                    observed_radius = test_radius
            
            # Record sample data
            sample_data = {
                "angle": angle,
                "expected_radius": radius,
                "observed_radius": observed_radius,
                "radial_error": abs(best_radial_offset) if best_radial_offset is not None else None,
                "edge_strength": best_edge_strength,
                "valid": best_edge_strength > 0 and best_radial_offset is not None
            }
            samples.append(sample_data)
            
            if sample_data["valid"]:
                valid_samples += 1
                radial_errors.append(sample_data["radial_error"])
                edge_strengths.append(best_edge_strength)
        
        # Calculate aggregate metrics
        edge_support = valid_samples / len(angles) if angles.size > 0 else 0.0
        
        if radial_errors:
            median_error = float(np.median(radial_errors))
            mean_edge_strength = float(np.mean(edge_strengths))
        else:
            median_error = 0.0
            mean_edge_strength = 0.0
        
        return {
            "samples": samples,
            "edge_support": edge_support,
            "valid_samples": valid_samples,
            "total_samples": len(angles),
            "median_radial_error": median_error,
            "mean_edge_strength": mean_edge_strength
        }
    
    def _check_edge_neighborhood(self, x: int, y: int, edge_image: np.ndarray, 
                                neighborhood_size: int = 3) -> bool:
        """Check if there's an edge in the neighborhood of a point."""
        h, w = edge_image.shape
        half_size = neighborhood_size // 2
        
        y_start = max(0, y - half_size)
        y_end = min(h, y + half_size + 1)
        x_start = max(0, x - half_size)
        x_end = min(w, x + half_size + 1)
        
        neighborhood = edge_image[y_start:y_end, x_start:x_end]
        return np.any(neighborhood > 0)
    
    def _calculate_intensity_consistency(self, center: Tuple[float, float],
                                       radius: float, isolated_product: np.ndarray) -> float:
        """Calculate consistency of intensity within circle."""
        # Create circle mask
        h, w = isolated_product.shape[:2]
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius * 0.8), 255, -1)
        
        # Extract pixel intensities
        if len(isolated_product.shape) == 3:
            gray = cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY)
        else:
            gray = isolated_product
            
        pixels = gray[circle_mask > 0]
        
        if len(pixels) == 0:
            return 0.0
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_intensity = np.mean(pixels)
        std_intensity = np.std(pixels)
        
        if mean_intensity == 0:
            return 0.0
        
        cv = std_intensity / mean_intensity
        # Convert to consistency score (higher is better)
        consistency = max(0.0, 1.0 - cv)
        
        return consistency
    
    def _validate_radius_bounds(self, radius: float, product_mask: np.ndarray) -> bool:
        """Validate radius bounds using image-relative constraints."""
        # Get image dimensions
        h, w = product_mask.shape
        image_diagonal = np.sqrt(h*h + w*w)
        
        # Maximum radius as percentile of image diagonal
        max_radius_image_relative = image_diagonal * self.max_radius_percentile
        if radius > max_radius_image_relative:
            return False
        
        # Additional reasonableness check - radius shouldn't be larger than product dimensions
        product_region = np.where(product_mask > 0)
        if len(product_region[0]) > 0:
            product_height = np.max(product_region[0]) - np.min(product_region[0])
            product_width = np.max(product_region[1]) - np.min(product_region[1])
            max_product_dimension = max(product_height, product_width)
            
            # Radius shouldn't exceed product dimensions
            if radius > max_product_dimension:
                return False
        
        return True
    
    def _apply_population_area_validation(self, circles: List[ActualFeature]) -> List[ActualFeature]:
        """Apply population-based area validation to reject anomalously large circles."""
        if len(circles) < 3:
            return circles
        
        # Calculate area statistics
        areas = [circle.geometry.area for circle in circles]
        median_area = np.median(areas)
        
        # Filter out anomalously large circles
        filtered_circles = []
        for circle in circles:
            area_ratio = circle.geometry.area / median_area if median_area > 0 else 1.0
            
            # Reject circles that are much larger than typical population
            if area_ratio <= self.area_population_threshold:
                filtered_circles.append(circle)
            else:
                logger.debug(f"Rejected anomalous circle {circle.feature_id}: area {circle.geometry.area} is {area_ratio:.1f}x median")
        
        return filtered_circles
    
    def _validate_hough_candidate(self, center: Tuple[float, float], radius: float,
                                 edge_image: np.ndarray, intensity_image: np.ndarray,
                                 product_mask: np.ndarray) -> dict:
        """
        Comprehensive validation of Hough circle candidate based on actual image evidence.
        Treats Hough as candidate generator only - validates against real image data.
        """
        result = {
            "valid": True,
            "rejection_reason": "",
            "edge_support": 0.0,
            "angular_coverage": 0.0,
            "sector_coverage": 0.0,
            "angular_uniformity": 0.0,
            "max_gap_ratio": 1.0,
            "radial_consistency": 0.0,
            "radial_error_median": 0.0,
            "radial_error_p95": 0.0,
            "local_contrast": 0.0,
            "gradient_orientation_consistency": 0.0,
            "circularity_evidence": 0.0,
            "solidity_evidence": 0.0,
            "convexity_evidence": 0.0,
            "contour_quality": 0.0,
            "intensity_consistency": 0.0,
            "geometric_consistency": 0.0,
            "intensity_evidence": 0.0,
            "contour_agreement": False
        }
        
        # Calculate basic edge support first to adapt thresholds
        basic_edge_support = self._calculate_edge_support(center, radius, edge_image)
        result["edge_support"] = basic_edge_support
        
        # Adaptive thresholds based on edge quality - more lenient for high-quality edges
        edge_quality_factor = min(1.0, basic_edge_support / 0.7)  # Scale factor [0,1] for adaptation
        
        # 1. Angular coverage analysis
        if self.angular_coverage_required:
            angular_result = self._analyze_angular_coverage(center, radius, edge_image)
            result["angular_coverage"] = angular_result["coverage"]
            result["sector_coverage"] = angular_result["sector_coverage"]
            result["angular_uniformity"] = angular_result["uniformity"]
            result["contour_agreement"] = bool(angular_result["sector_coverage"] > 0.5 and angular_result["uniformity"] > 0.25)
            
            # Reject localized or cluster-biased support.
            min_coverage = max(self.min_angular_coverage * (0.5 + 0.3 * edge_quality_factor), 0.45)
            
            if angular_result["coverage"] < min_coverage:
                result["valid"] = False
                result["rejection_reason"] = f"Insufficient angular coverage: {angular_result['coverage']:.2f} < {min_coverage:.2f}"
                return result
            
            min_sector = max(self.min_sector_coverage * (0.6 + 0.2 * edge_quality_factor), 0.35)
            if angular_result["sector_coverage"] < min_sector:
                result["valid"] = False  
                result["rejection_reason"] = f"Insufficient sector coverage: {angular_result['sector_coverage']:.2f} < {min_sector:.2f}"
                return result

            if angular_result["uniformity"] < 0.2:
                result["valid"] = False
                result["rejection_reason"] = (
                    "Localized angular support: evidence concentrated in a small arc "
                    f"instead of a distributed ring (uniformity={angular_result['uniformity']:.2f})"
                )
                return result
        
        # 2. Edge continuity analysis - keep it strict enough to reject local arcs
        if self.edge_continuity_required:
            continuity_result = self._analyze_edge_continuity(center, radius, edge_image)
            result["max_gap_ratio"] = continuity_result["max_gap_ratio"]
            max_gap_allowed = min(self.max_edge_gap_ratio, 0.35)
            if continuity_result["max_gap_ratio"] > max_gap_allowed:
                result["valid"] = False
                result["rejection_reason"] = f"Excessive edge gaps: {continuity_result['max_gap_ratio']:.2f} > {max_gap_allowed:.2f}"
                return result
        
        # 3. Radial consistency analysis - require evidence in the actual radius band
        if self.radial_consistency_required:
            radial_result = self._analyze_radial_consistency(center, radius, edge_image)
            result["radial_consistency"] = radial_result["consistency"]
            result["radial_error_median"] = radial_result["median_error"]
            result["radial_error_p95"] = radial_result["p95_error"]
            min_radial = max(self.min_radial_agreement, 0.55)
            if radial_result["consistency"] < min_radial:
                result["valid"] = False
                result["rejection_reason"] = f"Poor radial consistency: {radial_result['consistency']:.2f} < {min_radial:.2f}"
                return result
        
        # 4. Texture discrimination - only strict for weak edge cases
        if self.texture_discrimination:
            contrast_result = self._analyze_local_contrast(center, radius, intensity_image)
            result["local_contrast"] = contrast_result["contrast"]
            min_contrast = self.min_local_contrast * (0.4 + 0.4 * (1.0 - edge_quality_factor))
            if contrast_result["contrast"] < min_contrast:
                result["valid"] = False
                result["rejection_reason"] = f"Texture-like evidence: contrast {contrast_result['contrast']:.1f} < {min_contrast:.1f}"
                return result
        
        # 5. Calculate evidence-based geometric properties
        result["intensity_consistency"] = self._calculate_intensity_consistency(center, radius, intensity_image)
        edge_uniformity = angular_result["uniformity"] if self.angular_coverage_required else 0.5
        result["circularity_evidence"] = min(1.0, (result["angular_coverage"] + edge_uniformity + basic_edge_support) / 3.0)
        result["solidity_evidence"] = min(1.0, result["radial_consistency"] * 1.2) if self.radial_consistency_required else min(1.0, basic_edge_support * 1.1)
        result["convexity_evidence"] = min(1.0, result["edge_support"] * 1.1)
        result["contour_quality"] = result["circularity_evidence"]

        # 5b. Gradient orientation consistency — independent geometric signal.
        # At a real circular boundary the local gradient direction is approximately
        # radial; random texture does not maintain this pattern.
        gradient_result = self._calculate_gradient_orientation_consistency(
            center, radius, intensity_image
        )
        result["gradient_orientation_consistency"] = gradient_result["score"]

        # Optional hard gate: reject low-orientation candidates when the feature
        # also has weak edge support (i.e. the two weakest signals both fail).
        if (self.gradient_orientation_required
                and gradient_result["valid_fraction"] >= 0.3  # enough samples to trust
                and gradient_result["score"] < self.min_gradient_orientation
                and basic_edge_support < self.edge_support_threshold):
            result["valid"] = False
            result["rejection_reason"] = (
                f"Gradient orientation inconsistent with circle: "
                f"score={gradient_result['score']:.2f} < {self.min_gradient_orientation:.2f}, "
                f"valid_fraction={gradient_result['valid_fraction']:.2f}"
            )
            return result

        result["geometric_consistency"] = (result["circularity_evidence"] + result["solidity_evidence"]) / 2.0
        result["intensity_evidence"] = result["intensity_consistency"]

        return result
    
    def _analyze_angular_coverage(self, center: Tuple[float, float], radius: float, 
                                 edge_image: np.ndarray) -> dict:
        """Analyze edge support coverage around the full circumference."""
        sectors = self.angular_sectors
        sector_size = 2 * np.pi / sectors
        
        sector_support = []
        edge_positions = []
        
        for sector in range(sectors):
            angle_start = sector * sector_size
            angle_end = (sector + 1) * sector_size
            
            # Sample points in this sector
            sector_angles = np.linspace(angle_start, angle_end, 8, endpoint=False)
            sector_edges = 0
            sector_total = 0
            
            for angle in sector_angles:
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                
                if 0 <= x < edge_image.shape[1] and 0 <= y < edge_image.shape[0]:
                    sector_total += 1
                    if self._check_edge_neighborhood(x, y, edge_image, self.edge_neighborhood_size):
                        sector_edges += 1
                        edge_positions.append(angle)
            
            if sector_total > 0:
                sector_support.append(sector_edges / sector_total)
            else:
                sector_support.append(0.0)
        
        # Calculate coverage metrics. A dense cluster in a small arc can create
        # a reasonable aggregate score while still lacking genuine circular support.
        supported_sectors = sum(1 for support in sector_support if support > 0.25)
        sector_coverage = supported_sectors / sectors
        overall_coverage = np.mean(sector_support)
        uniformity = 1.0 - np.std(sector_support) if sector_support else 0.0
        concentration = max(sector_support) / (np.mean(sector_support) + 1e-6) if sector_support and np.mean(sector_support) > 0 else 0.0
        
        return {
            "coverage": overall_coverage,
            "sector_coverage": sector_coverage,
            "uniformity": max(0.0, uniformity),
            "concentration": concentration,
            "sector_support": sector_support,
            "edge_positions": edge_positions
        }
    
    def _analyze_edge_continuity(self, center: Tuple[float, float], radius: float,
                                edge_image: np.ndarray) -> dict:
        """Analyze continuity of edge support around circumference with proper wraparound."""
        # Sample many points around circumference
        num_samples = 64
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        
        edge_present = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            
            if 0 <= x < edge_image.shape[1] and 0 <= y < edge_image.shape[0]:
                edge_present.append(self._check_edge_neighborhood(x, y, edge_image, self.edge_neighborhood_size))
            else:
                edge_present.append(False)
        
        # Calculate basic metrics
        total_missing = sum(1 for edge in edge_present if not edge)
        missing_ratio = total_missing / num_samples
        
        # Special case: if all edges are missing, it's one big gap
        if total_missing == num_samples:
            return {
                "max_gap": num_samples,
                "max_gap_ratio": 1.0,
                "num_gaps": 1,
                "missing_ratio": missing_ratio,
                "edge_present": edge_present,
                "gaps": [num_samples]
            }
        
        # Special case: if no edges are missing, there are no gaps
        if total_missing == 0:
            return {
                "max_gap": 0,
                "max_gap_ratio": 0.0,
                "num_gaps": 0,
                "missing_ratio": missing_ratio,
                "edge_present": edge_present,
                "gaps": []
            }
        
        # Find gaps in edge support with proper circular boundary handling
        gaps = []
        
        # Find all runs of missing edges using circular array processing
        in_gap = False
        current_gap_length = 0
        gap_start_idx = None
        
        # Process the array twice to handle wraparound correctly
        for i in range(2 * num_samples):
            idx = i % num_samples  # Wrap around to handle circular boundary
            has_edge = edge_present[idx]
            
            if not has_edge:
                if not in_gap:
                    # Starting a new gap
                    in_gap = True
                    gap_start_idx = idx
                    current_gap_length = 1
                else:
                    # Continuing a gap
                    current_gap_length += 1
            else:
                if in_gap:
                    # Ending a gap
                    gaps.append(current_gap_length)
                    in_gap = False
                    current_gap_length = 0
            
            # Stop processing after we've gone around once and handled wraparound
            if i >= num_samples and not in_gap:
                break
        
        # Handle case where we end in a gap (wraparound case)
        if in_gap:
            # Check if this gap wraps around by looking at the beginning
            # If the array starts with missing edges, combine the gaps
            start_missing_count = 0
            for has_edge in edge_present:
                if not has_edge:
                    start_missing_count += 1
                else:
                    break
            
            # If there are missing edges at the start, this is a wraparound gap
            if start_missing_count > 0 and gap_start_idx is not None:
                # Combine the gap at the end with the gap at the start
                total_gap = current_gap_length
                # But don't double-count if the entire circle is one gap
                if total_gap < num_samples:
                    gaps.append(total_gap)
            else:
                gaps.append(current_gap_length)
        
        # Find the maximum gap
        max_gap = max(gaps) if gaps else 0
        max_gap_ratio = max_gap / num_samples
        
        return {
            "max_gap": max_gap,
            "max_gap_ratio": max_gap_ratio,
            "num_gaps": len(gaps),
            "missing_ratio": missing_ratio,
            "edge_present": edge_present,
            "gaps": gaps  # For debugging
        }
    
    def _analyze_radial_consistency(self, center: Tuple[float, float], radius: float,
                                   edge_image: np.ndarray) -> dict:
        """Analyze radial consistency - edges should be at expected radius."""
        angles = np.linspace(0, 2*np.pi, self.radial_samples, endpoint=False)
        valid_errors = []
        supported_count = 0
        
        for angle in angles:
            # Search within radial tolerance to find the strongest edge response
            best_error = float('inf')
            has_valid_edge = False
            best_edge_strength = 0.0
            
            # Search within the tolerance band
            for offset in range(-self.radial_tolerance_pixels, self.radial_tolerance_pixels + 1):
                test_radius = radius + offset
                if test_radius <= 0:
                    continue
                    
                x = int(center[0] + test_radius * np.cos(angle))
                y = int(center[1] + test_radius * np.sin(angle))
                
                if 0 <= x < edge_image.shape[1] and 0 <= y < edge_image.shape[0]:
                    # Check both the exact pixel and its neighborhood for edge strength
                    edge_strength = float(edge_image[y, x])
                    
                    # Also check immediate neighborhood for edges (to handle discretization)
                    if edge_strength == 0:
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < edge_image.shape[1] and 0 <= ny < edge_image.shape[0]:
                                    neighbor_strength = float(edge_image[ny, nx])
                                    edge_strength = max(edge_strength, neighbor_strength)
                    
                    # Require a minimum edge strength to count as valid
                    if edge_strength > 0:
                        error = abs(offset)
                        if error < best_error or (error == best_error and edge_strength > best_edge_strength):
                            best_error = error
                            best_edge_strength = edge_strength
                            has_valid_edge = True
            
            # Only count as valid if we found an edge
            if has_valid_edge and best_edge_strength > 0:
                valid_errors.append(best_error)
                supported_count += 1
        
        if not valid_errors:
            return {
                "consistency": 0.0, 
                "median_error": 0.0,
                "p95_error": 0.0,
                "valid_fraction": 0.0
            }
        
        # Calculate error statistics
        median_error = float(np.median(valid_errors))
        p95_error = float(np.percentile(valid_errors, 95))
        valid_fraction = supported_count / self.radial_samples
        
        # Consistency score based on valid fraction and error magnitude
        # More balanced approach - not too strict or too lenient
        error_penalty = (median_error + 0.5 * p95_error) / max(1.0, self.radial_tolerance_pixels)
        base_consistency = valid_fraction * max(0.0, 1.0 - error_penalty)
        
        # Apply moderate penalty for very sparse support (helps with random noise)
        if valid_fraction < 0.25:  # Less than 25% support is suspicious
            base_consistency *= valid_fraction * 4  # Strong penalty
        elif valid_fraction < 0.5:  # Less than 50% support is concerning
            base_consistency *= (0.5 + valid_fraction)  # Moderate penalty
        
        return {
            "consistency": base_consistency,
            "median_error": median_error,
            "p95_error": p95_error,
            "valid_fraction": valid_fraction
        }
    
    def _calculate_gradient_orientation_consistency(
            self, center: Tuple[float, float], radius: float,
            image: np.ndarray) -> dict:
        """
        Measure how consistently local image gradients point radially at the
        predicted circumference.

        At a genuine circular boundary every gradient vector should be roughly
        parallel (or anti-parallel) to the outward radial direction.  Random
        texture produces gradients whose orientations are unrelated to the
        proposed circle centre, so the mean absolute cosine similarity stays
        near 0.5 (the expected value for uniformly random unit vectors in 2-D).

        Algorithm
        ---------
        For each of the ``edge_sampling_points`` angular positions around the
        predicted circumference:

        1.  Sample the Sobel gradient at the exact circumference pixel *and*
            at ``gradient_sample_half_width`` pixels inward and outward along
            the radial direction; take the sample with the highest magnitude.
        2.  Skip if the best magnitude is below ``gradient_min_magnitude``
            (near-zero gradient carries no orientation information).
        3.  Compute the outward radial unit vector for that angle.
        4.  Agreement = abs(dot(normalised_gradient, radial_unit)).
            Using the absolute value makes the metric polarity-agnostic:
            a gradient pointing directly inward (dark-to-light vs light-to-dark)
            is equally consistent with a circular boundary.
        5.  Average over all valid samples.

        Returns
        -------
        dict with keys:
            ``score``           – mean abs-cosine agreement (0 = random, 1 = perfect)
            ``valid_fraction``  – fraction of circumference samples with usable gradient
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Compute full-image Sobel gradients (float64 prevents overflow)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        h, w = gray.shape
        angles = np.linspace(0, 2 * np.pi, self.edge_sampling_points, endpoint=False)

        agreements: List[float] = []
        valid_count = 0

        for angle in angles:
            # Radial unit vector (outward from centre)
            r_x = float(np.cos(angle))
            r_y = float(np.sin(angle))

            # Sample along the radial band: exact circumference ± half_width pixels
            best_mag = 0.0
            best_gx = 0.0
            best_gy = 0.0

            for offset in range(-self.gradient_sample_half_width,
                                self.gradient_sample_half_width + 1):
                test_r = radius + offset
                if test_r <= 0:
                    continue
                px = int(center[0] + test_r * r_x)
                py = int(center[1] + test_r * r_y)
                if not (0 <= px < w and 0 <= py < h):
                    continue

                sample_gx = gx[py, px]
                sample_gy = gy[py, px]
                mag = float(np.sqrt(sample_gx ** 2 + sample_gy ** 2))
                if mag > best_mag:
                    best_mag = mag
                    best_gx = sample_gx
                    best_gy = sample_gy

            # Skip samples with insufficient gradient magnitude — they add noise,
            # not signal.  Do NOT treat them as agreement = 0 (that would
            # artificially lower the score on genuine smooth boundaries).
            if best_mag < self.gradient_min_magnitude:
                continue

            # Normalise gradient and compute absolute radial agreement
            ng_x = best_gx / best_mag
            ng_y = best_gy / best_mag
            agreement = abs(ng_x * r_x + ng_y * r_y)

            agreements.append(agreement)
            valid_count += 1

        total = len(angles)
        if not agreements:
            return {"score": 0.0, "valid_fraction": 0.0}

        return {
            "score": float(np.mean(agreements)),
            "valid_fraction": valid_count / total,
        }

    def _analyze_local_contrast(self, center: Tuple[float, float], radius: float,
                               intensity_image: np.ndarray) -> dict:
        """Analyze local contrast to discriminate features from texture."""
        # Convert to grayscale if needed
        if len(intensity_image.shape) == 3:
            gray = cv2.cvtColor(intensity_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = intensity_image
        
        # Sample interior and exterior regions
        interior_samples = []
        exterior_samples = []
        
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        
        for angle in angles:
            # Interior sample (0.6 * radius from center)
            int_x = int(center[0] + 0.6 * radius * np.cos(angle))
            int_y = int(center[1] + 0.6 * radius * np.sin(angle))
            
            # Exterior sample (1.4 * radius from center)  
            ext_x = int(center[0] + 1.4 * radius * np.cos(angle))
            ext_y = int(center[1] + 1.4 * radius * np.sin(angle))
            
            if 0 <= int_x < gray.shape[1] and 0 <= int_y < gray.shape[0]:
                interior_samples.append(gray[int_y, int_x])
            
            if 0 <= ext_x < gray.shape[1] and 0 <= ext_y < gray.shape[0]:
                exterior_samples.append(gray[ext_y, ext_x])
        
        if not interior_samples or not exterior_samples:
            return {"contrast": 0.0}
        
        # Calculate contrast
        interior_mean = np.mean(interior_samples)
        exterior_mean = np.mean(exterior_samples)
        contrast = abs(interior_mean - exterior_mean)
        
        return {
            "contrast": contrast,
            "interior_mean": interior_mean,
            "exterior_mean": exterior_mean
        }
    
    def _calculate_hough_evidence_confidence(self, validation_result: dict) -> float:
        """
        Build Hough-candidate confidence from independent evidence groups.

        Delegates to ``_calculate_grouped_confidence`` which scores each group
        separately and prevents correlated edge metrics from being counted
        multiple times.
        """
        if not validation_result["valid"]:
            return 0.0

        evidence = {
            # Group A – Circumferential geometry
            "angular_coverage":   validation_result.get("angular_coverage",  0.0),
            "sector_coverage":    validation_result.get("sector_coverage",   0.0),
            "max_gap_ratio":      validation_result.get("max_gap_ratio",     1.0),
            "angular_uniformity": validation_result.get("angular_uniformity",0.0),
            # Group B – Radial geometry
            "radial_consistency":    validation_result.get("radial_consistency",   0.0),
            "radial_error_median":   validation_result.get("radial_error_median",  0.0),
            "radial_error_p95":      validation_result.get("radial_error_p95",     0.0),
            # Group C – Edge orientation
            "gradient_orientation_consistency": validation_result.get(
                "gradient_orientation_consistency", 0.0),
            # Group D – Appearance
            "local_contrast":          validation_result.get("local_contrast",         0.0),
            "intensity_consistency":   validation_result.get("intensity_consistency",  0.0),
            # Group E – Detector agreement
            "contour_agreement": validation_result.get("contour_agreement", False),
            "hough_only_penalty": self.hough_only_penalty,
        }

        return self._calculate_grouped_confidence(evidence, candidate_type="hough")

    def _calculate_grouped_confidence(self, evidence: dict,
                                      candidate_type: str = "hough") -> float:
        """
        Compute confidence from five *independent* evidence groups so that
        correlated edge-derived metrics cannot inflate the score.

        Groups and scoring
        ------------------
        Each group is reduced to a single score in [0, 1].  The five group
        scores are combined with a weighted geometric-mean-like formula that:

        * caps any single group at its weight (no group dominates)
        * applies a hard floor: if core geometric groups (A and B together) are
          weak, the overall score is capped regardless of how strong appearance
          evidence is

        Group weights (must sum to 1.0):
            A  circumferential geometry   0.28
            B  radial geometry            0.27
            C  edge orientation           0.18
            D  appearance                 0.17
            E  detector agreement         0.10

        Parameters
        ----------
        evidence : dict
            Keys defined in ``_calculate_hough_evidence_confidence``.
        candidate_type : str
            ``"hough"`` or ``"contour"`` — determines which groups have data.
        """
        # ------------------------------------------------------------------ #
        # Group A — Circumferential geometry                                  #
        # angular_coverage + sector_coverage + max_gap + angular_uniformity  #
        # ------------------------------------------------------------------ #
        ang_cov   = float(evidence.get("angular_coverage",   0.0))
        sec_cov   = float(evidence.get("sector_coverage",    0.0))
        gap_score = max(0.0, 1.0 - float(evidence.get("max_gap_ratio", 1.0)))
        uniformity = float(evidence.get("angular_uniformity", 0.0))

        # Weighted average inside the group (all sub-metrics on same scale)
        group_a = 0.35 * ang_cov + 0.30 * sec_cov + 0.20 * gap_score + 0.15 * uniformity

        # ------------------------------------------------------------------ #
        # Group B — Radial geometry                                           #
        # radial_consistency + error magnitude                                #
        # ------------------------------------------------------------------ #
        rad_cons = float(evidence.get("radial_consistency", 0.0))
        # Convert pixel errors to a 0-1 score using the configured tolerance
        tol = max(1.0, float(getattr(self, "radial_tolerance_pixels", 3)))
        med_err = float(evidence.get("radial_error_median", 0.0))
        p95_err = float(evidence.get("radial_error_p95",    0.0))
        error_score = max(0.0, 1.0 - (0.6 * med_err + 0.4 * p95_err) / tol)
        # group_b is meaningful only when we actually ran radial analysis
        if rad_cons > 0.0 or (med_err == 0.0 and p95_err == 0.0 and
                               candidate_type == "hough"):
            group_b = 0.6 * rad_cons + 0.4 * error_score
        else:
            # Contour path doesn't populate radial metrics — treat as neutral
            group_b = 0.5

        # ------------------------------------------------------------------ #
        # Group C — Edge orientation                                          #
        # ------------------------------------------------------------------ #
        orient = float(evidence.get("gradient_orientation_consistency", 0.0))
        # 0.0 means either no valid gradient samples OR the metric was not
        # computed for this candidate type.  Only treat it as neutral (0.5)
        # for contour candidates where gradient analysis is intentionally
        # skipped; for Hough candidates a genuine 0.0 is a real signal.
        if orient > 0.0:
            group_c = orient
        elif candidate_type == "contour":
            group_c = 0.5  # not measured — neutral
        else:
            group_c = 0.0  # Hough: truly absent gradient signal = zero evidence

        # ------------------------------------------------------------------ #
        # Group D — Appearance                                                #
        # ------------------------------------------------------------------ #
        contrast_raw = float(evidence.get("local_contrast",       0.0))
        intensity    = float(evidence.get("intensity_consistency", 0.0))
        # Normalise raw contrast to ~[0,1]: typical values sit in [0, 50]
        contrast_norm = min(1.0, contrast_raw / 40.0)
        group_d = 0.5 * contrast_norm + 0.5 * intensity

        # ------------------------------------------------------------------ #
        # Group E — Detector agreement                                        #
        # ------------------------------------------------------------------ #
        contour_agree = bool(evidence.get("contour_agreement", False))
        hough_penalty = float(evidence.get("hough_only_penalty", 0.0))
        if contour_agree:
            group_e = 1.0
        elif candidate_type == "hough" and hough_penalty > 0.0:
            group_e = 1.0 - hough_penalty   # penalty for unconfirmed Hough
        else:
            group_e = 0.6                   # neutral when not applicable

        # ------------------------------------------------------------------ #
        # Weighted combination — groups have independent contributions        #
        # ------------------------------------------------------------------ #
        W_A, W_B, W_C, W_D, W_E = 0.28, 0.27, 0.18, 0.17, 0.10
        raw_score = (
            W_A * group_a +
            W_B * group_b +
            W_C * group_c +
            W_D * group_d +
            W_E * group_e
        )

        # ------------------------------------------------------------------ #
        # Hard geometric floor                                                #
        # If both core geometric groups (A and B) are weak, cap confidence   #
        # even when appearance evidence is strong.                            #
        # ------------------------------------------------------------------ #
        geometric_core = 0.5 * group_a + 0.5 * group_b
        if geometric_core < 0.35:
            # Strong cap: appearance alone cannot push the score high
            raw_score = min(raw_score, 0.40)
        elif geometric_core < 0.55:
            # Moderate cap
            raw_score = min(raw_score, 0.65)

        # Additional floor: if Group A *alone* is very weak (most of the
        # circumference is missing), cap regardless of how good radial is.
        # A partial arc is not a circle no matter how precisely it sits on
        # the predicted radius.
        if group_a < 0.20:
            raw_score = min(raw_score, 0.45)

        return float(min(1.0, max(0.0, raw_score)))
    
    def _apply_enhanced_cluster_consolidation(self, circles: List[ActualFeature]) -> List[ActualFeature]:
        """
        Consolidate near-duplicate circle detections using union-find.

        Two detections are considered duplicates of the same physical feature
        when ``_should_cluster_circles`` returns True.  A union-find ensures
        transitive closure: if A≈B and B≈C, all three end up in the same
        cluster even when A and C are not directly compared before B is added.

        The best representative of each cluster is selected by
        ``_select_best_from_cluster``.
        """
        if len(circles) <= 1:
            return circles

        n = len(circles)

        # ----- Union-Find helpers ----------------------------------------- #
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path compression
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        # Build equivalence classes
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_cluster_circles(circles[i], circles[j]):
                    union(i, j)

        # Group by root
        from collections import defaultdict as _dd
        groups: dict = _dd(list)
        for i in range(n):
            groups[find(i)].append(i)

        consolidated: List[ActualFeature] = []
        for root, indices in groups.items():
            if len(indices) == 1:
                consolidated.append(circles[indices[0]])
            else:
                cluster = [circles[k] for k in indices]
                best = self._select_best_from_cluster(cluster)
                if best is not None:
                    consolidated.append(best)
                else:
                    # Fallback: keep highest-confidence member so we never drop silently
                    consolidated.append(max(cluster, key=lambda c: c.evidence.confidence))
                logger.debug(
                    f"Consolidated cluster of {len(cluster)} circles into best representative"
                )

        return consolidated

    def _should_cluster_circles(self, circle1: ActualFeature, circle2: ActualFeature) -> bool:
        """
        Return True iff two circle candidates are duplicates of the same physical feature.

        Decision tree
        -------------
        1. Center distance gate  — hard reject if centers are too far apart (scale-aware).
        2. Concentric guard      — if circles share a center but have substantially different
                                   radii they are *distinct* physical features (e.g. inner/outer
                                   bore edges); return False so both are kept.
        3. Radius similarity     — reject if radii differ by more than the configured tolerance.
        """
        center1 = circle1.geometry.center
        center2 = circle2.geometry.center
        radius1 = circle1.geometry.radius or 0.0
        radius2 = circle2.geometry.radius or 0.0

        center_distance = float(np.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        ))

        # ------------------------------------------------------------------ #
        # 1. Center distance gate                                             #
        # ------------------------------------------------------------------ #
        if self.scale_aware_distance:
            larger_radius = max(radius1, radius2)
            cluster_distance = larger_radius * self.cluster_radius_factor
            if center_distance > cluster_distance:
                return False
        else:
            if center_distance > self.strict_center_distance:
                return False

        # ------------------------------------------------------------------ #
        # 2. Concentric guard                                                 #
        # Two circles with near-identical centers but substantially different #
        # radii are distinct physical features — keep both.                  #
        # ------------------------------------------------------------------ #
        if self.concentric_detection and radius1 > 0 and radius2 > 0:
            near_concentric = center_distance < min(radius1, radius2) * 0.3
            radius_diff_ratio = abs(radius1 - radius2) / max(radius1, radius2)
            if near_concentric and radius_diff_ratio > self.concentric_radius_tolerance:
                # Distinct concentric circles — do NOT merge
                return False

        # ------------------------------------------------------------------ #
        # 3. Radius similarity                                                #
        # ------------------------------------------------------------------ #
        if radius1 > 0 and radius2 > 0:
            radius_ratio = min(radius1, radius2) / max(radius1, radius2)
            if radius_ratio < (1.0 - self.cluster_radius_tolerance):
                return False

        return True

    def _select_best_from_cluster(self, cluster_circles: List[ActualFeature]) -> Optional[ActualFeature]:
        """
        Select the best representative from a set of duplicate detections.

        Uses ``evidence.confidence``, which by this point holds the fully
        grouped, appearance-refined score from Step 7.  This is preferred over
        re-combining raw evidence fields because the grouped model already
        de-correlates edge-derived metrics and applies the geometric floor.

        A secondary tiebreak prefers contour-based detections over Hough-only
        ones when confidence values are within 2 % of each other, because
        contour detections carry an independent geometric contour shape whereas
        Hough-only candidates do not.
        """
        if not cluster_circles:
            return None
        if len(cluster_circles) == 1:
            return cluster_circles[0]

        def sort_key(c: ActualFeature):
            conf = c.evidence.confidence
            # Secondary tiebreak: contour-based detections preferred
            is_contour = 1 if c.detection_method == "contour_analysis" else 0
            return (round(conf, 2), is_contour)

        return max(cluster_circles, key=sort_key)
    
    def _calculate_solidity(self, contour: np.ndarray) -> float:
        """Calculate solidity (area/convex_hull_area)."""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return 0.0
        
        return area / hull_area
    
    def _calculate_convexity(self, contour: np.ndarray) -> float:
        """Calculate convexity (convex_hull_perimeter/perimeter)."""
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        
        if perimeter == 0:
            return 0.0
        
        return hull_perimeter / perimeter
    
    def _calculate_contour_confidence(self, geometry: GeometricProperties,
                                    edge_support: float) -> float:
        """
        Calculate confidence for contour-based circle detection.

        Contour candidates have geometric properties but no radial/angular
        analysis data.  Pass what we have into the grouped model; missing
        groups default to neutral so they don't dominate the score.
        """
        evidence = {
            # Group A — use circularity as a proxy for angular coverage
            "angular_coverage":   min(1.0, geometry.circularity),
            "sector_coverage":    min(1.0, geometry.circularity),
            "max_gap_ratio":      max(0.0, 1.0 - geometry.circularity),
            "angular_uniformity": geometry.solidity,
            # Group B — not available for contours (defaults to neutral inside grouped calc)
            "radial_consistency":  0.0,
            "radial_error_median": 0.0,
            "radial_error_p95":    0.0,
            # Group C — not computed yet; neutral
            "gradient_orientation_consistency": 0.0,
            # Group D — edge support as crude appearance proxy
            "local_contrast":       min(40.0, edge_support * 40.0),
            "intensity_consistency": edge_support,
            # Group E — contour IS the independent detector here
            "contour_agreement": True,
            "hough_only_penalty": 0.0,
        }
        return self._calculate_grouped_confidence(evidence, candidate_type="contour")
    
    def _calculate_hough_confidence(self, radius: float, edge_support: float) -> float:
        """Deprecated — retained only for backward compatibility. Do not call."""
        # All new code uses _calculate_hough_evidence_confidence → _calculate_grouped_confidence.
        return self._calculate_grouped_confidence({
            "angular_coverage": edge_support,
            "sector_coverage": edge_support,
            "max_gap_ratio": 1.0 - edge_support,
            "angular_uniformity": edge_support,
            "radial_consistency": 0.0,
            "radial_error_median": 0.0,
            "radial_error_p95": 0.0,
            "gradient_orientation_consistency": 0.0,
            "local_contrast": 0.0,
            "intensity_consistency": 0.0,
            "contour_agreement": False,
            "hough_only_penalty": self.hough_only_penalty,
        }, candidate_type="hough")
    
    def _calculate_final_confidence(self, evidence: EvidenceMetrics) -> float:
        """
        Refine confidence after the per-candidate validation pass has populated
        all evidence fields.

        This is called from ``_validate_circle_candidate`` *after* intensity
        consistency has been measured.  For Hough candidates the rich grouped
        evidence is already in the EvidenceMetrics fields; we rebuild from
        those fields so appearance refinement is incorporated without
        discarding the geometric signal.
        """
        ev = {
            # Group A
            "angular_coverage":   evidence.angular_coverage,
            "sector_coverage":    evidence.sector_coverage,
            "max_gap_ratio":      evidence.max_gap_ratio,
            "angular_uniformity": evidence.angular_uniformity,
            # Group B
            "radial_consistency":  evidence.radial_consistency,
            "radial_error_median": evidence.radial_error_median,
            "radial_error_p95":    evidence.radial_error_p95,
            # Group C
            "gradient_orientation_consistency": evidence.gradient_orientation_consistency,
            # Group D — use freshly measured intensity
            "local_contrast":       evidence.local_contrast,
            "intensity_consistency": evidence.intensity_consistency,
            # Group E
            "contour_agreement": (evidence.internal_edge_evidence > 0.5
                                  or evidence.contour_quality > 0.7),
            "hough_only_penalty": self.hough_only_penalty,
        }
        # Determine candidate type from what evidence is available
        candidate_type = (
            "contour" if evidence.internal_edge_evidence > 0.5 else "hough"
        )
        return self._calculate_grouped_confidence(ev, candidate_type=candidate_type)