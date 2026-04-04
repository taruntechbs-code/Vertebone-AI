"""
models.py — OpenEnv RL Environment for MRI-based Bone Quality and Vertebral Risk Assessment.

Implements:
  • BoneEnv: The core OpenEnv-compatible environment (reset, step, state).
  • Three task graders: BoneDensityGrader, FractureRiskGrader, TreatmentRecommendationGrader.
  • Deterministic reward logic — NO randomness.
"""

import os
import glob
import math
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Feature extraction utilities
# ---------------------------------------------------------------------------

def extract_features(image_path: str) -> Dict[str, float]:
    """Extract deterministic features from a grayscale MRI image.

    Returns
    -------
    dict with keys:
        mean_intensity   – average pixel intensity (bone density proxy, 0–255)
        std_intensity    – pixel standard deviation (structural variation)
        edge_density     – fraction of edge pixels (vertebra clarity)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Resize to a fixed resolution for consistency across images
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))

    # Canny edge detection for vertebra edge clarity
    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    return {
        "mean_intensity": round(mean_intensity, 4),
        "std_intensity": round(std_intensity, 4),
        "edge_density": round(edge_density, 6),
    }


# ---------------------------------------------------------------------------
# Ground-truth risk scoring (deterministic)
# ---------------------------------------------------------------------------

def compute_bone_density_score(state: Dict[str, float]) -> float:
    """Return a ground-truth bone density risk score in [0, 1].

    Higher mean_intensity → denser bone → lower risk.
    """
    # Normalise mean_intensity to [0, 1] then invert so high density = low risk
    norm_mean = min(max(state["mean_intensity"] / 255.0, 0.0), 1.0)
    # Weight structural variation: high std → heterogeneous → higher risk
    norm_std = min(max(state["std_intensity"] / 128.0, 0.0), 1.0)
    score = 0.6 * (1.0 - norm_mean) + 0.4 * norm_std
    return round(min(max(score, 0.0), 1.0), 4)


def compute_fracture_risk_score(state: Dict[str, float]) -> float:
    """Return a ground-truth fracture risk score in [0, 1].

    Combines bone density weakness with edge clarity (poor edges → fragile).
    """
    norm_mean = min(max(state["mean_intensity"] / 255.0, 0.0), 1.0)
    norm_edge = min(max(state["edge_density"] / 0.3, 0.0), 1.0)  # 0.3 is a practical ceiling
    norm_std = min(max(state["std_intensity"] / 128.0, 0.0), 1.0)
    score = 0.4 * (1.0 - norm_mean) + 0.35 * (1.0 - norm_edge) + 0.25 * norm_std
    return round(min(max(score, 0.0), 1.0), 4)


def compute_treatment_score(state: Dict[str, float]) -> float:
    """Return a ground-truth treatment urgency score in [0, 1].

    Aggregates both bone-density and fracture-risk factors to determine
    how urgently treatment is recommended.
    """
    density_risk = compute_bone_density_score(state)
    fracture_risk = compute_fracture_risk_score(state)
    score = 0.5 * density_risk + 0.5 * fracture_risk
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

ACTION_SPACE: List[str] = ["low_risk", "medium_risk", "high_risk"]

ACTION_VALUE_MAP: Dict[str, float] = {
    "low_risk": 0.0,
    "medium_risk": 0.5,
    "high_risk": 1.0,
}


# ---------------------------------------------------------------------------
# BoneEnv — OpenEnv-compatible RL environment
# ---------------------------------------------------------------------------

class BoneEnv:
    """OpenEnv RL environment for MRI bone-quality assessment.

    Lifecycle
    ---------
    env = BoneEnv(dataset_dir)
    state_dict = env.reset()
    for _ in range(env.total_steps):
        obs = env.state()
        action = agent.act(obs)       # one of ACTION_SPACE
        obs, reward, done, info = env.step(action)
    """

    def __init__(self, dataset_dir: str = "Dataset") -> None:
        self.dataset_dir: str = dataset_dir

        if not os.path.exists(dataset_dir):
            raise RuntimeError(f"Dataset directory '{dataset_dir}' not found")

        self.image_paths: List[str] = sorted(
            glob.glob(os.path.join(dataset_dir, "*.jpg"))
        )

        if not self.image_paths:
            raise RuntimeError(f"No .jpg images found in '{dataset_dir}'")

        self.total_steps: int = len(self.image_paths)
        self._current_index: int = 0
        self._current_state: Optional[Dict[str, float]] = None
        self._current_image_path: Optional[str] = None
        self._done: bool = False

        # Accumulate per-task scores across all steps
        self._task_scores: Dict[str, List[float]] = {
            "bone_density": [],
            "fracture_risk": [],
            "treatment": [],
        }

    # ----- OpenEnv API: reset -----
    def reset(self) -> Dict[str, Any]:
        """Reset environment to the first image and return initial state."""
        self._current_index = 0
        self._done = False
        self._task_scores = {
            "bone_density": [],
            "fracture_risk": [],
            "treatment": [],
        }
        self._load_current_image()
        return self.state()

    # ----- OpenEnv API: state -----
    def state(self) -> Dict[str, Any]:
        """Return the current observation as a structured dictionary."""
        if self._current_state is None:
            self._load_current_image()
        return {
            "image_file": os.path.basename(self._current_image_path),
            "step": self._current_index,
            "total_steps": self.total_steps,
            "features": dict(self._current_state) if self._current_state else {},
            "done": self._done,
        }

    # ----- OpenEnv API: step -----
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute *action* and advance to the next image.

        Parameters
        ----------
        action : str
            One of "low_risk", "medium_risk", "high_risk".

        Returns
        -------
        observation : dict   — next state
        reward      : float  — aggregate reward ∈ [0, 1]
        done        : bool   — True when all images processed
        info        : dict   — per-task reward breakdown
        """
        if self._done:
            return self.state(), 0.0, True, {}

        if action not in ACTION_SPACE:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {ACTION_SPACE}."
            )

        predicted_value = ACTION_VALUE_MAP[action]
        features = self._current_state

        # Compute per-task rewards
        gt_density = compute_bone_density_score(features)
        gt_fracture = compute_fracture_risk_score(features)
        gt_treatment = compute_treatment_score(features)

        r_density = round(1.0 - abs(predicted_value - gt_density), 4)
        r_fracture = round(1.0 - abs(predicted_value - gt_fracture), 4)
        r_treatment = round(1.0 - abs(predicted_value - gt_treatment), 4)

        # Clamp rewards to [0, 1]
        r_density = min(max(r_density, 0.0), 1.0)
        r_fracture = min(max(r_fracture, 0.0), 1.0)
        r_treatment = min(max(r_treatment, 0.0), 1.0)

        self._task_scores["bone_density"].append(r_density)
        self._task_scores["fracture_risk"].append(r_fracture)
        self._task_scores["treatment"].append(r_treatment)

        # Aggregate reward (mean of 3 task rewards)
        reward = round((r_density + r_fracture + r_treatment) / 3.0, 4)

        info: Dict[str, Any] = {
            "ground_truth": {
                "bone_density": gt_density,
                "fracture_risk": gt_fracture,
                "treatment": gt_treatment,
            },
            "reward_breakdown": {
                "bone_density": r_density,
                "fracture_risk": r_fracture,
                "treatment": r_treatment,
            },
            "predicted_value": predicted_value,
            "action": action,
        }

        # Advance
        self._current_index += 1
        if self._current_index >= self.total_steps:
            self._done = True
        else:
            self._load_current_image()

        return self.state(), reward, self._done, info

    # ----- Task grading (after episode) -----
    def get_task_scores(self) -> Dict[str, float]:
        """Return final mean score per task. Each ∈ [0.0, 1.0]."""
        result: Dict[str, float] = {}
        for task, scores in self._task_scores.items():
            result[task] = round(sum(scores) / len(scores), 4) if scores else 0.0
        return result

    # ----- Internal -----
    def _load_current_image(self) -> None:
        if self._current_index >= len(self.image_paths):
            return
        path = self.image_paths[self._current_index]
        self._current_image_path = path
        self._current_state = extract_features(path)


# ---------------------------------------------------------------------------
# Task Graders
# ---------------------------------------------------------------------------

class BoneDensityGrader:
    """Grades performance on the Bone Density Classification task."""

    task_name: str = "Bone Density Classification"

    @staticmethod
    def grade(env: BoneEnv) -> float:
        scores = env._task_scores.get("bone_density", [])
        if not scores:
            return 0.0
        return round(min(max(sum(scores) / len(scores), 0.0), 1.0), 4)


class FractureRiskGrader:
    """Grades performance on the Fracture Risk Prediction task."""

    task_name: str = "Fracture Risk Prediction"

    @staticmethod
    def grade(env: BoneEnv) -> float:
        scores = env._task_scores.get("fracture_risk", [])
        if not scores:
            return 0.0
        return round(min(max(sum(scores) / len(scores), 0.0), 1.0), 4)


class TreatmentRecommendationGrader:
    """Grades performance on the Treatment Recommendation task."""

    task_name: str = "Treatment Recommendation"

    @staticmethod
    def grade(env: BoneEnv) -> float:
        scores = env._task_scores.get("treatment", [])
        if not scores:
            return 0.0
        return round(min(max(sum(scores) / len(scores), 0.0), 1.0), 4)
