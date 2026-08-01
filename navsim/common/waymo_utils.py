# Copyright 2025 The Waymo Open Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""WOD E2E Rater Feedback Score."""

from typing import Dict, List, Tuple

import numpy as np


_THRESHOLD_TIME_SECONDS = np.array([3, 5], dtype=np.int64)
_BASE_THRESHOLDS = np.array([1.0, 1.8], dtype=np.float64)
_MINIMUM_SCORE_OUTSIDE_TRUST_REGION = 4.0


def get_lat_lng_thresholds(
    init_speed: np.ndarray,
    lat_lng_threshold_multipliers: Tuple[float, float],
    base_thresholds: np.ndarray = _BASE_THRESHOLDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get lateral and longitudinal thresholds."""
    lat_threshold_multiplier, lng_threshold_multiplier = lat_lng_threshold_multipliers
    lat_thresholds = base_thresholds * lat_threshold_multiplier
    lng_thresholds = base_thresholds * lng_threshold_multiplier
    scale_by_init_speed = np.clip(
        0.5 + 0.5 * (init_speed - 1.4) / (11 - 1.4), 0.5, 1.0
    )
    lat_thresholds = scale_by_init_speed[..., None] * lat_thresholds
    lng_thresholds = scale_by_init_speed[..., None] * lng_thresholds
    return lat_thresholds, lng_thresholds


def process_rater_specified_trajectories(
    trajectory_batches: List[List[np.ndarray]],
    trajectory_labels_batches: List[np.ndarray],
    target_num_waypoints: int,
    target_num_trajectories_per_batch: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate or pad rated trajectories to fixed batch and waypoint counts."""
    if len(trajectory_batches) != len(trajectory_labels_batches):
        raise ValueError("The number of trajectory batches and label batches must be the same.")

    processed_trajectory_batches = []
    processed_label_batches = []
    for trajectory_batch, label_batch in zip(trajectory_batches, trajectory_labels_batches):
        if len(trajectory_batch) != len(label_batch):
            raise ValueError(
                "In each batch, the number of trajectories and labels must be the same."
            )

        trajectory_batch = list(trajectory_batch[:target_num_trajectories_per_batch])
        label_batch = list(label_batch[:target_num_trajectories_per_batch])
        num_to_pad = target_num_trajectories_per_batch - len(trajectory_batch)
        trajectory_batch.extend([trajectory_batch[-1]] * num_to_pad)
        label_batch.extend([label_batch[-1]] * num_to_pad)

        processed_trajectories = []
        for trajectory in trajectory_batch:
            trajectory = trajectory[:target_num_waypoints]
            num_waypoints_to_pad = target_num_waypoints - len(trajectory)
            if num_waypoints_to_pad:
                padding = np.repeat(trajectory[-1][None], num_waypoints_to_pad, axis=0)
                trajectory = np.concatenate([trajectory, padding], axis=0)
            processed_trajectories.append(trajectory)

        processed_trajectory_batches.append(np.asarray(processed_trajectories))
        processed_label_batches.append(np.asarray(label_batch))

    return (
        np.asarray(processed_trajectory_batches),
        np.asarray(processed_label_batches),
    )


def get_rater_feedback_score(
    inference_trajectories: np.ndarray,
    inference_probs: np.ndarray,
    rater_specified_trajectories: List[List[np.ndarray]],
    rater_feedback_labels: List[np.ndarray],
    init_speed: np.ndarray,
    lat_lng_threshold_multipliers: Tuple[float, float] = (1.0, 4.0),
    decay_factor: float = 0.1,
    frequency: int = 4,
    length_seconds: int = 5,
    default_num_of_rater_specified_trajectories: int = 3,
    output_trust_region_visualization: bool = False,
    minimum_score_outside_trust_region: float = _MINIMUM_SCORE_OUTSIDE_TRUST_REGION,
) -> Dict[str, np.ndarray]:
    """Compute the Waymo E2E rater feedback score."""
    selected_threshold_seconds = _THRESHOLD_TIME_SECONDS[
        _THRESHOLD_TIME_SECONDS <= length_seconds
    ]
    if len(selected_threshold_seconds) == 0:
        selected_threshold_seconds = np.array([length_seconds], dtype=np.int64)
    elif selected_threshold_seconds[-1] != length_seconds:
        selected_threshold_seconds = np.concatenate(
            [selected_threshold_seconds, np.array([length_seconds], dtype=np.int64)]
        )
    selected_base_thresholds = np.interp(
        selected_threshold_seconds,
        _THRESHOLD_TIME_SECONDS,
        _BASE_THRESHOLDS,
    )

    rater_specified_trajectories, rater_feedback_labels = (
        process_rater_specified_trajectories(
            rater_specified_trajectories,
            rater_feedback_labels,
            target_num_waypoints=length_seconds * frequency,
            target_num_trajectories_per_batch=default_num_of_rater_specified_trajectories,
        )
    )

    if inference_trajectories.shape[-2] != rater_specified_trajectories.shape[-2]:
        raise ValueError(
            "Inference and rater-specified trajectories must have the same number of timesteps."
        )
    if inference_trajectories.shape[-2] < selected_threshold_seconds.max() * frequency:
        raise ValueError(
            "Inference trajectories must have at least "
            f"{selected_threshold_seconds.max()} seconds of timesteps."
        )

    padded_rater_trajectories = np.pad(
        rater_specified_trajectories,
        ((0, 0), (0, 0), (1, 0), (0, 0)),
        constant_values=0,
    )
    displacement_vectors = (
        padded_rater_trajectories[..., 1:, :]
        - padded_rater_trajectories[..., :-1, :]
    )
    lng_directions = displacement_vectors.copy()
    lng_magnitudes = np.linalg.norm(lng_directions, axis=-1)
    lng_directions[..., 0, 0] = np.where(
        lng_magnitudes[..., 0] == 0, 1, lng_directions[..., 0, 0]
    )
    lng_directions[..., 0, 1] = np.where(
        lng_magnitudes[..., 0] == 0, 0, lng_directions[..., 0, 1]
    )
    for timestep in range(1, lng_directions.shape[2]):
        lng_directions[..., timestep, 0] = np.where(
            lng_magnitudes[..., timestep] == 0,
            lng_directions[..., timestep - 1, 0],
            lng_directions[..., timestep, 0],
        )
        lng_directions[..., timestep, 1] = np.where(
            lng_magnitudes[..., timestep] == 0,
            lng_directions[..., timestep - 1, 1],
            lng_directions[..., timestep, 1],
        )

    lat_directions = np.stack(
        [-lng_directions[..., 1], lng_directions[..., 0]], axis=-1
    )
    lng_directions /= np.linalg.norm(lng_directions, axis=-1, keepdims=True)
    lat_directions /= np.linalg.norm(lat_directions, axis=-1, keepdims=True)

    rater_to_inference_vectors = (
        inference_trajectories[..., None, :, :, :]
        - rater_specified_trajectories[..., None, :, :]
    )
    lng_distances = np.abs(
        np.sum(
            lng_directions[..., None, :, :] * rater_to_inference_vectors,
            axis=-1,
        )
    )
    lat_distances = np.abs(
        np.sum(
            lat_directions[..., None, :, :] * rater_to_inference_vectors,
            axis=-1,
        )
    )

    selected_indices = selected_threshold_seconds * frequency - 1
    lng_distances = lng_distances[..., selected_indices]
    lat_distances = lat_distances[..., selected_indices]
    lat_thresholds, lng_thresholds = get_lat_lng_thresholds(
        init_speed, lat_lng_threshold_multipliers, selected_base_thresholds
    )

    normalized_lng_distances = lng_distances / lng_thresholds[..., None, None, :]
    normalized_lat_distances = lat_distances / lat_thresholds[..., None, None, :]
    normalized_distances = np.maximum(
        normalized_lng_distances, normalized_lat_distances
    )
    is_fully_within_trust_region = np.any(
        np.all(normalized_distances <= 1.0, axis=3), axis=1
    )

    exponent = np.maximum(normalized_distances - 1.0, 0.0)
    decay = decay_factor**exponent
    pairwise_scores = rater_feedback_labels[..., None, None] * decay
    scores_per_axis = np.amax(pairwise_scores, axis=1)
    scores_per_inference = np.mean(scores_per_axis, axis=-1)
    scores_per_inference[~is_fully_within_trust_region] = np.maximum(
        minimum_score_outside_trust_region,
        scores_per_inference[~is_fully_within_trust_region],
    )

    outputs = {
        "is_fully_within_trust_region": is_fully_within_trust_region,
        "rater_feedback_score_per_inference": scores_per_inference,
        "rater_feedback_score": np.sum(
            scores_per_inference * inference_probs, axis=-1
        ),
        "rater_specified_trajectories": rater_specified_trajectories,
        "rater_feedback_labels": rater_feedback_labels,
    }
    if output_trust_region_visualization:
        outputs.update(
            {
                "trust_region_center_x": rater_specified_trajectories[
                    ..., selected_indices, 0
                ],
                "trust_region_center_y": rater_specified_trajectories[
                    ..., selected_indices, 1
                ],
                "trust_region_width": 2 * lng_thresholds,
                "trust_region_height": 2 * lat_thresholds,
                "trust_region_angle": np.degrees(
                    np.arctan2(
                        displacement_vectors[..., selected_indices, 1],
                        displacement_vectors[..., selected_indices, 0],
                    )
                ),
            }
        )
    return outputs


def interpolate_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Linearly interpolate 2 Hz waypoints to 4 Hz, including the ego origin."""
    waypoint_count = trajectory.shape[0]
    even_frames = np.arange(2, 2 * waypoint_count + 1, 2)
    all_frames = np.arange(2 * waypoint_count + 1)
    interp_x = np.interp(
        all_frames, np.concatenate([[0], even_frames]), np.concatenate([[0.0], trajectory[:, 0]])
    )[1:]
    interp_y = np.interp(
        all_frames, np.concatenate([[0], even_frames]), np.concatenate([[0.0], trajectory[:, 1]])
    )[1:]
    return np.stack([interp_x, interp_y], axis=-1)
