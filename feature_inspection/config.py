"""
Feature Inspection Configuration

Centralized configuration for Phase 2A preprocessing parameters.
All tunable preprocessing parameters are defined here to avoid hardcoding.
"""

# =============================================================================
# Phase 2A Preprocessing Configuration
# =============================================================================

# Edge combination weights for final representation
PREPROCESSING_INTERNAL_EDGE_COMBINATION_WEIGHT = 0.7     # Weight for internal geometry edges
PREPROCESSING_OUTER_BOUNDARY_COMBINATION_WEIGHT = 0.8   # Weight for outer boundary edges

# Noise reduction and texture suppression
PREPROCESSING_NOISE_REDUCTION_KERNEL_SIZE = 3           # Kernel size for noise reduction morphology
PREPROCESSING_MIN_EDGE_STRENGTH = 20                    # Minimum edge strength to retain
PREPROCESSING_TEXTURE_SUPPRESSION_THRESHOLD = 15        # Threshold for texture vs geometry distinction

# Geometry preservation parameters - CRITICAL FOR AVOIDING GEOMETRY LOSS
PREPROCESSING_MASK_MORPHOLOGY_ENABLED = False           # Whether to use aggressive morphology on mask
PREPROCESSING_CONSERVATIVE_MASK_KERNEL_SIZE = 3         # Small kernel size for minimal geometry impact  
PREPROCESSING_CONSERVATIVE_MASK_ITERATIONS = 1          # Minimal iterations to preserve thin structures
PREPROCESSING_CONTOUR_BASED_MASK_REFINEMENT = True      # Use contour-based mask refinement instead of morphology
PREPROCESSING_BACKGROUND_ESTIMATION_ENABLED = True      # Enable background estimation for better segmentation
PREPROCESSING_MULTI_THRESHOLD_FUSION = True             # Combine multiple threshold methods for robust segmentation

# Multi-threshold fusion parameters - CRITICAL for robust segmentation without hardcoded weights
PREPROCESSING_OTSU_EVIDENCE_WEIGHT = 0.4                # Weight for Otsu thresholding evidence
PREPROCESSING_BACKGROUND_EVIDENCE_WEIGHT = 0.3          # Weight for background estimation evidence  
PREPROCESSING_GRADIENT_EVIDENCE_WEIGHT = 0.3            # Weight for gradient-based evidence
PREPROCESSING_EVIDENCE_NORMALIZATION_THRESHOLD = 127    # Threshold for evidence normalization (0-255)
PREPROCESSING_GRADIENT_DILATION_KERNEL_SIZE = 5         # Kernel size for gradient evidence dilation
PREPROCESSING_GRADIENT_DILATION_ITERATIONS = 2          # Iterations for gradient evidence dilation

# Contour-based mask refinement parameters
PREPROCESSING_CONTOUR_REFINEMENT_AREA_FACTOR = 0.1      # Multiplier for min component area in contour refinement

# Silhouette validation parameters - CRITICAL for geometry loss detection
PREPROCESSING_SILHOUETTE_ANALYSIS_KERNEL_SIZE = 7       # Kernel size for external region analysis
PREPROCESSING_SILHOUETTE_ANALYSIS_ITERATIONS = 1        # Iterations for external region dilation  
PREPROCESSING_SILHOUETTE_GRADIENT_PERCENTILE = 75       # Percentile for strong gradient threshold (0-100)
PREPROCESSING_SUSPICIOUS_BOUNDARY_THRESHOLD = 0.3       # Threshold for suspicious boundary fraction
PREPROCESSING_WEAK_BOUNDARY_THRESHOLD = 0.5             # Threshold for weak boundary consistency
PREPROCESSING_EXTERNAL_VS_BOUNDARY_RATIO = 1.5          # Ratio threshold for external vs boundary gradient strength

# Texture suppression parameters - CRITICAL for structure preservation  
PREPROCESSING_TEXTURE_SCALE_EVIDENCE_THRESHOLD = 0.3    # Threshold for scale evidence in texture suppression
PREPROCESSING_TEXTURE_COMPACTNESS_THRESHOLD = 5.0       # Compactness threshold for connectivity evidence
PREPROCESSING_TEXTURE_EVIDENCE_COMBINATION_THRESHOLD = 2.0  # Threshold for combining evidence sources (out of 3)
PREPROCESSING_TEXTURE_CLEANUP_KERNEL_SIZE = 3           # Kernel size for final texture cleanup morphology
PREPROCESSING_TEXTURE_STRONG_GRADIENT_MULTIPLIER = 1.5  # Multiplier for strong gradient threshold

# Final edge representation parameters
PREPROCESSING_FINAL_EDGE_CLEANUP_THRESHOLD = 50         # Threshold for final weak edge removal

# Additional algorithm parameters
PREPROCESSING_BACKGROUND_BORDER_FRACTION = 0.05         # Fraction of image size for border region
PREPROCESSING_BACKGROUND_BORDER_MIN_PIXELS = 5          # Minimum border width in pixels
PREPROCESSING_ADAPTIVE_THRESHOLD_BLOCK_SIZE = 15        # Block size for adaptive thresholding
PREPROCESSING_ADAPTIVE_THRESHOLD_C = 5                  # Constant subtracted from mean
PREPROCESSING_GRADIENT_SOBEL_KERNEL_SIZE = 3            # Sobel kernel size for gradient
PREPROCESSING_HOLE_AREA_FRACTION_THRESHOLD = 0.1        # Fraction for internal hole detection
PREPROCESSING_SIGNIFICANT_REGION_AREA_FRACTION = 0.1    # Fraction for significant region detection
PREPROCESSING_STRUCTURE_AREA_FRACTION_THRESHOLD = 0.05  # Fraction for structure preservation
PREPROCESSING_CONTOUR_PERIMETER_THRESHOLD = 20          # Minimum perimeter for contour preservation
PREPROCESSING_CONTOUR_CIRCULARITY_THRESHOLD = 0.3       # Minimum circularity for circular features

# Image processing parameters
PREPROCESSING_MAX_RESOLUTION = 1600         # Maximum long edge dimension
PREPROCESSING_GAUSSIAN_BLUR_KERNEL = 5      # Gaussian blur kernel size for noise reduction

# Product isolation thresholds  
PREPROCESSING_GRADIENT_THRESHOLD = 12       # Morphological gradient threshold for internal edges
PREPROCESSING_MIN_COMPONENT_AREA_PX = 200   # Minimum component area in pixels
PREPROCESSING_BORDER_MARGIN_FRACTION = 0.01 # Border margin as fraction of image size
PREPROCESSING_BORDER_MARGIN_MIN_PX = 3      # Minimum border margin in pixels

# Canny edge detection for outer boundary
PREPROCESSING_CANNY_OUTER_LOW = 50          # Lower Canny threshold
PREPROCESSING_CANNY_OUTER_HIGH = 150        # Upper Canny threshold

# Morphological operations for mask cleanup
PREPROCESSING_MASK_MORPH_KERNEL_SIZE = 9    # Kernel size for morphological operations
PREPROCESSING_MASK_MORPH_CLOSE_ITERS = 3    # Closing iterations
PREPROCESSING_MASK_MORPH_OPEN_ITERS = 2     # Opening iterations

# CLAHE contrast enhancement
PREPROCESSING_CLAHE_CLIP_LIMIT = 2.0        # CLAHE clip limit
PREPROCESSING_CLAHE_TILE_GRID_SIZE = (8, 8) # CLAHE tile grid size

# Internal edge extraction
PREPROCESSING_INTERNAL_EDGE_MAX_DENSITY = 0.08    # Maximum edge density before threshold raising
PREPROCESSING_GRADIENT_STEP_MULTIPLIER = 1.5      # Multiplier for gradient threshold adjustment
PREPROCESSING_GRADIENT_MAX_RETRIES = 4            # Maximum retries for gradient threshold
PREPROCESSING_GRADIENT_KERNEL_SIZE = 3            # Kernel size for morphological gradient
PREPROCESSING_EDGE_CLOSE_KERNEL_SIZE = 2          # Kernel size for edge closing

# Component scoring weights
PREPROCESSING_COMPONENT_AREA_WEIGHT = 0.95        # Weight for area in component scoring
PREPROCESSING_COMPONENT_CENTRALITY_WEIGHT = 0.05  # Weight for centrality in component scoring

# Multi-scale texture suppression parameters
PREPROCESSING_TEXTURE_MULTI_SCALE_LEVELS = [1, 2, 3]  # Different blur levels for scale consistency
PREPROCESSING_TEXTURE_FINAL_CLEANUP_ITERATIONS = 1     # Morphological open iterations for noise removal


# =============================================================================
# Phase 2B Actual Feature Extraction Configuration
# =============================================================================

# Circle detection parameters
CIRCLE_MIN_RADIUS = 5                      # Minimum circle radius in pixels
CIRCLE_MAX_RADIUS = 300                    # Maximum circle radius in pixels  
CIRCLE_MIN_AREA = 50                       # Minimum circle area in pixels
CIRCLE_MAX_AREA = 100000                   # Maximum circle area in pixels
CIRCLE_CIRCULARITY_THRESHOLD = 0.3         # Minimum circularity for circle candidates
CIRCLE_CONTOUR_APPROX_EPSILON = 0.02       # Contour approximation epsilon for circles
CIRCLE_HOUGH_ACCUMULATOR_THRESHOLD = 20    # Hough circle accumulator threshold
CIRCLE_MIN_CENTER_DISTANCE = 10            # Minimum distance between circle centers
CIRCLE_EDGE_SUPPORT_THRESHOLD = 0.6        # Minimum edge support ratio for circles

# Rectangle/square detection parameters  
RECTANGLE_MIN_AREA = 100                   # Minimum rectangle area in pixels
RECTANGLE_MAX_AREA = 50000                 # Maximum rectangle area in pixels
RECTANGLE_MIN_SIDE_LENGTH = 10             # Minimum rectangle side length in pixels
RECTANGLE_CONTOUR_APPROX_EPSILON = 0.02    # Contour approximation epsilon for rectangles
RECTANGLE_ASPECT_RATIO_TOLERANCE = 0.3     # Tolerance for rectangle aspect ratio validation
RECTANGLE_ANGLE_TOLERANCE = 15             # Angle tolerance for rectangle corners (degrees)
RECTANGLE_SIDE_RATIO_TOLERANCE = 0.2       # Tolerance for opposite side length ratios

# General contour extraction parameters
CONTOUR_MIN_AREA = 25                      # Minimum contour area for preservation
CONTOUR_MAX_AREA = 200000                  # Maximum contour area to consider
CONTOUR_MIN_PERIMETER = 20                 # Minimum contour perimeter
CONTOUR_APPROX_EPSILON = 0.01              # General contour approximation epsilon
CONTOUR_CONVEXITY_THRESHOLD = 0.7          # Minimum convexity for contour validation
CONTOUR_SOLIDITY_THRESHOLD = 0.5           # Minimum solidity for contour validation

# Candidate validation parameters
FEATURE_MIN_CONFIDENCE = 0.3               # Minimum confidence for feature acceptance
FEATURE_MIN_EDGE_SUPPORT = 0.4             # Minimum edge support for feature validation
FEATURE_PRODUCT_MASK_OVERLAP_THRESHOLD = 0.7  # Required overlap with product mask
FEATURE_MAX_BORDER_DISTANCE = 5            # Maximum distance from image border for rejection

# Duplicate suppression parameters
DUPLICATE_CENTER_DISTANCE_THRESHOLD = 15   # Maximum center distance for duplicate detection
DUPLICATE_SIZE_RATIO_THRESHOLD = 0.3       # Maximum size ratio difference for duplicates
DUPLICATE_OVERLAP_THRESHOLD = 0.5          # Minimum overlap ratio for duplicate detection
DUPLICATE_CONFIDENCE_PREFERENCE = True     # Prefer higher confidence candidates in duplicate resolution

# Evidence weighting parameters
EDGE_EVIDENCE_WEIGHT = 0.4                 # Weight for edge-based evidence
CONTOUR_EVIDENCE_WEIGHT = 0.3              # Weight for contour-based evidence
INTENSITY_EVIDENCE_WEIGHT = 0.2            # Weight for intensity-based evidence
GEOMETRY_EVIDENCE_WEIGHT = 0.1             # Weight for geometric consistency evidence

# Feature extraction processing parameters
MAX_CANDIDATES_PER_TYPE = 200              # Generation safety cap: maximum RAW candidates
                                           # (pre-validation) to forward per extractor call.
                                           # This is a runtime/memory guard, NOT a correctness
                                           # mechanism. It must never be set from CAD/DXF counts.
                                           # Applied BEFORE expensive validation passes.
                                           # TUNED: raised from 50 to 200 (Step 12 validation showed
                                           # some images produce 60–140 validated circles — the old
                                           # value of 50 was silently acting as a feature-count gate).
CANDIDATE_REFINEMENT_ITERATIONS = 3        # Iterations for candidate refinement
MULTI_SCALE_DETECTION_LEVELS = [1.0, 0.8, 1.2]  # Scale levels for multi-scale detection

# Hough circle detection parameters
HOUGH_CIRCLE_PARAM1 = 50                   # Upper Canny threshold for Hough circles
HOUGH_CIRCLE_PARAM2_MULTIPLIER = 1.0       # Multiplier for param2 (uses CIRCLE_HOUGH_ACCUMULATOR_THRESHOLD)
HOUGH_CIRCLE_BLUR_KERNEL_SIZE = 5          # Gaussian blur kernel size for Hough preprocessing
HOUGH_CIRCLE_DP = 1                        # Inverse ratio of accumulator resolution
HOUGH_CIRCLE_MIN_DIST_MULTIPLIER = 1.0     # Multiplier for min distance (uses CIRCLE_MIN_CENTER_DISTANCE)
HOUGH_CIRCLE_CONTOUR_SAMPLING_POINTS = 64  # Number of points to sample for synthetic contour

# Edge support validation parameters
EDGE_SUPPORT_SAMPLING_POINTS = 64          # Number of points to sample around circle perimeter
EDGE_SUPPORT_NEIGHBORHOOD_SIZE = 3         # Neighborhood size for edge validation
CONTOUR_EDGE_SUPPORT_INTERPOLATION_THRESHOLD = 2  # Distance threshold for contour point interpolation

# Confidence calculation parameters
HOUGH_CONFIDENCE_RADIUS_WEIGHT = 0.3       # Weight for radius score in Hough confidence
HOUGH_CONFIDENCE_EDGE_WEIGHT = 0.7         # Weight for edge support in Hough confidence
CONTOUR_INTENSITY_EVIDENCE_WEIGHT = 0.2    # Weight for intensity evidence in contour confidence

# Cross-type duplicate suppression parameters
CROSS_TYPE_DUPLICATE_ENABLED = True        # Enable cross-type duplicate suppression
CROSS_TYPE_CENTER_THRESHOLD = 12           # Center distance threshold for cross-type duplicates
CROSS_TYPE_SIZE_THRESHOLD = 0.4            # Size ratio threshold for cross-type duplicates
CROSS_TYPE_OVERLAP_THRESHOLD = 0.4         # Overlap threshold for cross-type duplicates

# Contour hierarchy parameters
CONTOUR_HIERARCHY_FILTERING_ENABLED = True # Enable contour hierarchy filtering
CONTOUR_MIN_AREA_RATIO = 0.1               # Minimum area ratio vs parent for nested contours
CONTOUR_MAX_ENCLOSING_RATIO = 0.8          # Maximum area ratio vs product mask for outer contours

# General contour validation enhancement
CONTOUR_MULTIPLE_PROPERTY_VALIDATION = True # Enable multiple property validation for contours
CONTOUR_MIN_SOLIDITY_ENHANCED = 0.6        # Enhanced minimum solidity threshold
CONTOUR_MIN_EXTENT = 0.1                   # Minimum extent (area/bounding_box_area)
CONTOUR_MAX_ASPECT_RATIO = 10.0            # Maximum aspect ratio for contour validation

# Enhanced Hough circle validation parameters
HOUGH_GEOMETRIC_EVIDENCE_REQUIRED = True   # Require geometric evidence validation for Hough circles
HOUGH_EDGE_SUPPORT_MULTIPLIER = 1.1        # Stricter edge support for Hough (vs lenient 0.8)
HOUGH_RADIUS_ANOMALY_DETECTION = True      # Enable radius anomaly detection
HOUGH_MAX_RADIUS_PERCENTILE = 0.95         # Maximum radius as percentile of image diagonal
HOUGH_AREA_POPULATION_THRESHOLD = 3.0      # Maximum area as multiple of median detected area

# Enhanced duplicate suppression parameters  
ENHANCED_DUPLICATE_SUPPRESSION = True      # Enable enhanced duplicate detection
SAME_TYPE_CENTER_DISTANCE_STRICT = 8       # Stricter center distance for same-type duplicates
HOUGH_CLUSTER_CONSOLIDATION = True         # Enable Hough cluster consolidation
HOUGH_CLUSTER_RADIUS_TOLERANCE = 0.2       # Radius similarity tolerance for cluster consolidation

# Contour significance validation parameters
CONTOUR_SIGNIFICANCE_VALIDATION = True     # Enable geometric significance validation
CONTOUR_MIN_AREA_ENHANCED = 100            # Enhanced minimum area for significance
CONTOUR_MAX_AREA_IMAGE_FRACTION = 0.3      # Maximum contour area as fraction of image
CONTOUR_GEOMETRIC_STABILITY_REQUIRED = True # Require geometric stability validation

# Confidence calculation enhancements
CONFIDENCE_GEOMETRIC_VALIDATION = True     # Require actual geometric validation for confidence
HOUGH_CONFIDENCE_GEOMETRIC_WEIGHT = 0.4    # Weight for geometric validation in Hough confidence
CONFIDENCE_INTENSITY_VALIDATION_REQUIRED = True # Require intensity validation before final confidence

# Enhanced Hough circle image evidence validation
HOUGH_ANGULAR_COVERAGE_REQUIRED = True     # Require angular coverage validation for Hough circles
HOUGH_MIN_ANGULAR_COVERAGE = 0.50          # Minimum fraction of circumference with edge support
                                           # TUNED (was 0.55): accepted p10 = 0.49 across 40 images;
                                           # 0.55 cut ~2 % of genuine circles.  Adaptive gate floor
                                           # of 0.45 still protects against very sparse arcs.
HOUGH_ANGULAR_SECTORS = 16                 # Number of angular sectors for coverage analysis
HOUGH_MIN_SECTOR_COVERAGE = 0.5            # Minimum fraction of sectors with edge support
HOUGH_EDGE_CONTINUITY_REQUIRED = True      # Require edge continuity validation
HOUGH_MAX_EDGE_GAP_RATIO = 0.30            # Maximum gap ratio in edge coverage
                                           # TUNED (was 0.45): accepted p90 = 0.234 — the old value
                                           # left a 0.22 dead zone.  0.30 closes it without touching
                                           # any genuine circle in the 40-image sample set.
HOUGH_RADIAL_CONSISTENCY_REQUIRED = True   # Require radial consistency validation
HOUGH_RADIAL_SAMPLES = 16                  # Number of radial samples for consistency check
                                           # TUNED (was 8): finer sampling improves discrimination for
                                           # medium/large circles; 2414 rejections used this gate.
HOUGH_MIN_RADIAL_AGREEMENT = 0.5           # Minimum radial consistency score
HOUGH_RADIAL_TOLERANCE_PIXELS = 3          # Allowed distance from proposed circumference
HOUGH_RADIAL_SEARCH_STEP = 1               # Sampling step within the radial tolerance band

# Multi-method agreement parameters
HOUGH_CONTOUR_AGREEMENT_BONUS = 0.2        # Confidence bonus for Hough+contour agreement  
HOUGH_ONLY_PENALTY = 0.3                   # Confidence penalty for Hough-only detections (increased)
HOUGH_TEXTURE_DISCRIMINATION = True        # Enable texture vs feature discrimination
HOUGH_MIN_LOCAL_CONTRAST = 4               # Minimum local contrast for non-texture features
                                           # TUNED (was 6): accepted p10 = 3.8 across 40 images;
                                           # 533 genuine-circle candidates were rejected by the old
                                           # value.  4 sits just below the observed accepted p10
                                           # while keeping the guard effective against zero-gradient
                                           # regions (rejected p90 = 0.0).

# Advanced cluster consolidation parameters
HOUGH_CLUSTER_CONSOLIDATION_ENHANCED = True # Enable enhanced cluster consolidation
HOUGH_SCALE_AWARE_DISTANCE = True          # Use scale-aware distance for clustering
HOUGH_CLUSTER_RADIUS_FACTOR = 0.4          # Cluster distance as factor of larger radius (tighter from 0.5)
HOUGH_CONCENTRIC_DETECTION = True          # Enable concentric circle detection
HOUGH_CONCENTRIC_RADIUS_TOLERANCE = 0.15   # Tolerance for concentric circle detection
HOUGH_EVIDENCE_BASED_CONSOLIDATION = True  # Use evidence quality for cluster consolidation

# Gradient orientation consistency parameters
HOUGH_GRADIENT_ORIENTATION_REQUIRED = True       # Enable gradient direction validation
HOUGH_MIN_GRADIENT_ORIENTATION_CONSISTENCY = 0.4 # Minimum mean radial gradient agreement (0–1)
HOUGH_GRADIENT_MIN_MAGNITUDE = 8.0               # Minimum Sobel magnitude to consider a sample valid
HOUGH_GRADIENT_SAMPLE_HALF_WIDTH = 1             # Radial pixels either side of circumference to sample

# Geometry-first Hough input parameters (Step 6)
HOUGH_GEOMETRY_FIRST_INPUT = True   # Use internal_geometry_edges as Hough input (geometry-first mode)
                                    # Set False to revert to isolated_product input (appearance mode)
HOUGH_EDGE_INPUT_DILATE_KERNEL = 3  # Kernel size for dilating edge map before Hough (0 = no dilation)
                                    # Slight dilation helps HoughCircles pick up thin-edge circles

# Hole detection parameters (dark-region method — detects actual through-holes in the part)
HOLE_DARK_PERCENTILE = 15           # Intensity percentile inside product mask used to set the
                                    # "dark region" threshold.  Pixels darker than this value
                                    # are candidates for through-holes.
HOLE_DARK_OFFSET = 5                # Additional offset added to the percentile threshold so
                                    # shallow shadows are not classified as holes.
HOLE_DARK_MAX = 55                  # Hard ceiling on the dark threshold — prevents merging with
                                    # mid-tone product surface at higher thresholds.
HOLE_MIN_RADIUS = 6                 # Minimum hole radius in pixels — excludes surface marks
                                    # and pits smaller than ~12px diameter
HOLE_MAX_RADIUS = 150               # Maximum hole radius in pixels (large centre bores allowed)
HOLE_MIN_CIRCULARITY = 0.45         # Minimum circularity for a dark blob to be classified HOLE
HOLE_MIN_SOLIDITY = 0.65            # Minimum solidity — holes are compact, filled regions
HOLE_MIN_INTERIOR_DARK_FRACTION = 0.80  # Fraction of pixels inside the candidate that must be
                                        # darker than the threshold (rejects shallow depressions)
HOLE_PRODUCT_MASK_OVERLAP = 0.75    # Minimum fraction of hole area that must be inside product mask