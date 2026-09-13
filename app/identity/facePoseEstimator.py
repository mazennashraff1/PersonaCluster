"""
Coarse facial pose estimation from InsightFace's 5-point landmarks.

The estimator is kept as a class for compatibility with the original
architecture, while the ``estimate`` method also supports the keyword
arguments used by the current association layer.

Supported calls:

    estimator = FacePoseEstimator()
    estimator.estimate(landmarks, image_shape)

or:

    FacePoseEstimator.estimate(
        landmarks=landmarks,
        frontal_yaw_degrees=20.0,
        profile_yaw_degrees=55.0,
    )

The result is intentionally coarse. It is suitable for pose-aware identity
clustering and representative selection, not precise 3D head-pose analysis.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app import configuration as config


class FacePoseEstimator:
    """Estimate coarse head pose from five facial landmarks."""

    # Generic canonical 3D face model corresponding to:
    # left eye, right eye, nose, left mouth, right mouth.
    _MODEL_POINTS = np.asarray(
        [
            (-30.0, 35.0, 30.0),
            (30.0, 35.0, 30.0),
            (0.0, 0.0, 0.0),
            (-25.0, -30.0, 20.0),
            (25.0, -30.0, 20.0),
        ],
        dtype=np.float64,
    )

    @staticmethod
    def _camera_matrix_from_image_shape(
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        focal_length = float(max(image_width, image_height))
        center = (
            float(image_width) / 2.0,
            float(image_height) / 2.0,
        )

        return np.asarray(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _camera_matrix_from_landmarks(
        image_points: np.ndarray,
    ) -> np.ndarray:
        """
        Build an approximate camera matrix when no image shape is supplied.

        This keeps compatibility with the current association layer, which
        only passes InsightFace's five landmarks.
        """
        min_xy = image_points.min(axis=0)
        max_xy = image_points.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        center = image_points.mean(axis=0)

        focal = max(float(span[0]), float(span[1])) * 4.0

        return np.asarray(
            [
                [focal, 0.0, float(center[0])],
                [0.0, focal, float(center[1])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wrap an Euler angle to the conventional [-90, 90] range."""
        normalized = ((float(angle) + 90.0) % 180.0) - 90.0
        return float(normalized)

    @staticmethod
    def _classify_yaw(
        yaw: float,
        frontal_yaw_degrees: float,
        profile_yaw_degrees: float,
    ) -> str:
        """Convert yaw into the project's coarse pose labels."""
        abs_yaw = abs(float(yaw))

        if abs_yaw <= float(frontal_yaw_degrees):
            return "frontal"

        if abs_yaw >= float(profile_yaw_degrees):
            return "profile"

        if yaw < 0.0:
            return "left"

        return "right"

    @staticmethod
    def estimate(
        landmarks: Optional[object],
        image_shape: Optional[tuple[int, ...]] = None,
        frontal_yaw_degrees: Optional[float] = None,
        profile_yaw_degrees: Optional[float] = None,
    ) -> tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[str],
    ]:
        """
        Estimate yaw, pitch, roll and a coarse pose label.

        ``estimate`` is intentionally a static method so it supports both the
        original instance-style interface and the current association-layer
        call ``FacePoseEstimator.estimate(...)``.

        Args:
            landmarks:
                InsightFace's five landmarks in left-eye, right-eye, nose,
                left-mouth, right-mouth order.
            image_shape:
                Optional original image shape. When omitted, a camera matrix
                is approximated from the landmark span.
            frontal_yaw_degrees:
                Optional frontal threshold. Defaults to configuration.
            profile_yaw_degrees:
                Optional profile threshold. Defaults to configuration.

        Returns:
            ``(yaw, pitch, roll, pose)``. All values are None when the input
            is invalid or solvePnP fails.
        """
        if landmarks is None:
            return None, None, None, None

        if frontal_yaw_degrees is None:
            frontal_yaw_degrees = config.FACE_POSE_FRONTAL_YAW_DEGREES

        if profile_yaw_degrees is None:
            profile_yaw_degrees = config.FACE_POSE_PROFILE_YAW_DEGREES

        try:
            image_points = np.asarray(
                landmarks,
                dtype=np.float64,
            ).reshape(-1, 2)
        except (TypeError, ValueError):
            return None, None, None, None

        if image_points.shape != (5, 2):
            return None, None, None, None

        if not np.all(np.isfinite(image_points)):
            return None, None, None, None

        # Prefer the original camera approximation when image dimensions are
        # available. Otherwise use the landmark-only approximation used by
        # the newer estimator API.
        if image_shape is not None:
            try:
                image_height = int(image_shape[0])
                image_width = int(image_shape[1])
            except (TypeError, ValueError, IndexError):
                image_height = 0
                image_width = 0

            if image_width > 0 and image_height > 0:
                camera_matrix = FacePoseEstimator._camera_matrix_from_image_shape(
                    image_width=image_width,
                    image_height=image_height,
                )
            else:
                camera_matrix = FacePoseEstimator._camera_matrix_from_landmarks(
                    image_points,
                )
        else:
            camera_matrix = FacePoseEstimator._camera_matrix_from_landmarks(
                image_points,
            )

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                FacePoseEstimator._MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except (cv2.error, ValueError, FloatingPointError):
            return None, None, None, None

        if not success or rotation_vector is None or translation_vector is None:
            return None, None, None, None

        try:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(
                rotation_matrix,
            )

            pitch = FacePoseEstimator._normalize_angle(
                float(euler_angles[0]),
            )
            yaw = FacePoseEstimator._normalize_angle(
                float(euler_angles[1]),
            )
            roll = FacePoseEstimator._normalize_angle(
                float(euler_angles[2]),
            )
        except (cv2.error, ValueError, FloatingPointError, IndexError):
            return None, None, None, None

        if not all(np.isfinite(value) for value in (yaw, pitch, roll)):
            return None, None, None, None

        pose = FacePoseEstimator._classify_yaw(
            yaw=yaw,
            frontal_yaw_degrees=float(frontal_yaw_degrees),
            profile_yaw_degrees=float(profile_yaw_degrees),
        )

        return yaw, pitch, roll, pose
