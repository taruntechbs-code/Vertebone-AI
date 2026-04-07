"""
models.py - OpenEnv RL environment for MRI-based bone quality and vertebral risk assessment.

Implements:
  - BoneEnv: a 3-step episode environment (density, risk, treatment).
  - Three task graders with distinct action schemas and reward logic.
  - Deterministic feature extraction and seeded episode context generation.
"""

import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from skimage import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops

load_dotenv()

DEFAULT_HF_ROUTER_URL = "https://router.huggingface.co/v1"
LEGACY_HF_API_URL = "https://api-inference.huggingface.co/v1"


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a value into the inclusive [lower, upper] range."""
    return min(max(value, lower), upper)


def resolve_api_base_url(api_base: Optional[str]) -> Optional[str]:
    """Rewrite deprecated Hugging Face inference URLs to the current router."""
    if api_base == LEGACY_HF_API_URL:
        return DEFAULT_HF_ROUTER_URL
    return api_base


# ---------------------------------------------------------------------------
# Feature extraction utilities
# ---------------------------------------------------------------------------

def extract_features(image_path: str) -> Dict[str, float]:
    """Extract deterministic intensity and GLCM texture features from an MRI image."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    glcm = graycomatrix(
        img_as_ubyte(gray / 255.0),
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
    contrast = float(graycoprops(glcm, "contrast")[0, 0])
    energy = float(graycoprops(glcm, "energy")[0, 0])
    correlation = float(graycoprops(glcm, "correlation")[0, 0])

    return {
        "mean_intensity": round(mean_intensity, 4),
        "std_intensity": round(std_intensity, 4),
        "edge_density": round(edge_density, 6),
        "homogeneity": round(homogeneity, 6),
        "contrast": round(contrast, 6),
        "energy": round(energy, 6),
        "correlation": round(correlation, 6),
    }


# ---------------------------------------------------------------------------
# Task action spaces
# ---------------------------------------------------------------------------

DENSITY_CLASSES: List[str] = ["osteoporotic", "osteopenic", "normal"]
TREATMENT_OPTIONS: List[str] = [
    "no_intervention",
    "lifestyle_modification",
    "physical_therapy",
    "medication",
    "surgical_consultation",
]
FOLLOW_UP_OPTIONS: List[str] = ["3_months", "6_months", "12_months"]
LIFESTYLE_OPTIONS: List[str] = ["calcium_supplement", "exercise", "both", "none"]
ACTION_SPACE: List[str] = [
    "density_class",
    "risk_score",
    "treatment",
    "follow_up_interval",
    "lifestyle_recommendation",
]

DENSITY_INDEX_MAP: Dict[str, int] = {label: idx for idx, label in enumerate(DENSITY_CLASSES)}
TREATMENT_INDEX_MAP: Dict[str, int] = {
    label: idx for idx, label in enumerate(TREATMENT_OPTIONS)
}
DENSITY_TREATMENT_RECOMMENDATIONS: Dict[str, Tuple[str, ...]] = {
    "osteoporotic": ("medication", "surgical_consultation"),
    "osteopenic": ("lifestyle_modification", "physical_therapy"),
    "normal": ("no_intervention",),
}
DENSITY_TREATMENT_SEVERITY: Dict[str, int] = {
    "normal": 0,
    "osteopenic": 1,
    "osteoporotic": 2,
}
TREATMENT_SEVERITY: Dict[str, int] = {
    "no_intervention": 0,
    "lifestyle_modification": 1,
    "physical_therapy": 1,
    "medication": 2,
    "surgical_consultation": 2,
}
FOLLOW_UP_INDEX_MAP: Dict[str, int] = {
    label: idx for idx, label in enumerate(FOLLOW_UP_OPTIONS)
}

TASK_ORDER: List[str] = [
    "BoneDensityClassification",
    "FractureRiskPrediction",
    "TreatmentProtocol",
    "FollowUpInterval",
    "LifestyleRecommendation",
]

STEP_TO_TASK: Dict[int, str] = {
    0: "BoneDensityClassification",
    1: "FractureRiskPrediction",
    2: "TreatmentProtocol",
    3: "FollowUpInterval",
    4: "LifestyleRecommendation",
}


# ---------------------------------------------------------------------------
# Ground-truth task definitions
# ---------------------------------------------------------------------------

def derive_density_class(mean_intensity: float) -> str:
    """Task 1 ground truth from mean intensity."""
    if mean_intensity < 80.0:
        return "osteoporotic"
    if mean_intensity <= 140.0:
        return "osteopenic"
    return "normal"


def derive_treatment_protocol(
    density_class: Optional[str],
    risk_score: Optional[float],
    patient_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Risk-driven treatment logic that is applied consistently for men and women."""
    risk_value = float(risk_score or 0.0)
    previous_fracture = bool((patient_meta or {}).get("previous_fracture"))
    glucocorticoid_use = bool((patient_meta or {}).get("glucocorticoid_use"))

    if density_class == "osteoporotic" or risk_value >= 0.8:
        return "surgical_consultation"
    if density_class == "osteopenic":
        if risk_value >= 0.55 or previous_fracture or glucocorticoid_use:
            return "medication"
        return "physical_therapy"
    if density_class == "normal":
        if risk_value < 0.25 and not previous_fracture and not glucocorticoid_use:
            return "no_intervention"
        return "lifestyle_modification"
    return "medication" if risk_value >= 0.5 else "lifestyle_modification"


def derive_follow_up_interval(
    density_class: Optional[str],
    risk_score: Optional[float],
    treatment: str,
    patient_meta: Dict[str, Any],
) -> str:
    """Risk-based follow-up cadence using fracture_risk and density_result.

    Clinical logic:
      - fracture_risk > 0.6  OR density == osteoporotic  → 3_months
      - fracture_risk 0.3–0.6 OR density == osteopenic   → 6_months
      - fracture_risk < 0.3  AND density == normal        → 12_months
    """
    risk_value = float(risk_score or 0.0)

    # High-risk: short follow-up
    if risk_value > 0.6 or density_class == "osteoporotic":
        return "3_months"

    # Moderate-risk: intermediate follow-up
    if risk_value >= 0.3 or density_class == "osteopenic":
        return "6_months"

    # Low-risk: standard follow-up
    return "12_months"


def derive_lifestyle_recommendation(
    density_class: Optional[str],
    risk_score: Optional[float],
    patient_meta: Dict[str, Any],
    treatment: str,
) -> str:
    """Lifestyle reinforcement proxy based on fracture risk and nutrition status."""
    risk_value = float(risk_score or 0.0)
    bmi = float(patient_meta.get("bmi", 22.0))

    if (
        treatment in {"medication", "surgical_consultation"}
        or density_class == "osteoporotic"
        or risk_value >= 0.6
        or patient_meta.get("previous_fracture")
        or patient_meta.get("glucocorticoid_use")
    ):
        return "both"
    if bmi < 18.5:
        return "calcium_supplement"
    if density_class == "normal" and risk_value < 0.15:
        return "none"
    return "exercise"


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def reward_density_class(predicted_class: Any, ground_truth_class: str) -> float:
    """Exact match = 1.0, one-class-off = 0.5, two-classes-off = 0.0."""
    if predicted_class not in DENSITY_INDEX_MAP:
        return 0.0

    distance = abs(DENSITY_INDEX_MAP[predicted_class] - DENSITY_INDEX_MAP[ground_truth_class])
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.5
    return 0.0


def reward_risk_score(predicted_risk: Any, ground_truth_risk: float) -> Tuple[float, Dict[str, Any]]:
    """Proportional reward for a continuous risk prediction."""
    if isinstance(predicted_risk, bool):
        return 0.0, {"error": "risk_score must be a float"}

    try:
        predicted_float = clamp(float(predicted_risk))
    except (TypeError, ValueError):
        return 0.0, {"error": "risk_score must be a float"}

    score = max(0.0, 1.0 - abs(predicted_float - ground_truth_risk) * 1.5)
    return round(score, 4), {}


def reward_treatment(predicted_treatment: Any, density_result: Optional[str]) -> float:
    """Reward treatment choice using density-driven clinical severity bands."""
    recommended_treatments = DENSITY_TREATMENT_RECOMMENDATIONS.get(
        str(density_result), tuple()
    )
    if predicted_treatment in recommended_treatments:
        return 1.0

    density_severity = DENSITY_TREATMENT_SEVERITY.get(str(density_result))
    predicted_severity = TREATMENT_SEVERITY.get(str(predicted_treatment))
    if density_severity is not None and predicted_severity is not None:
        severity_gap = abs(predicted_severity - density_severity)
        if severity_gap == 1:
            return 0.6
        if severity_gap >= 2:
            return 0.0
    return 0.3


def reward_follow_up_interval(predicted_interval: Any, ground_truth_interval: str) -> float:
    """Exact match = 1.0, adjacent interval = 0.5, otherwise 0.0."""
    if predicted_interval not in FOLLOW_UP_INDEX_MAP:
        return 0.0

    distance = abs(
        FOLLOW_UP_INDEX_MAP[predicted_interval] - FOLLOW_UP_INDEX_MAP[ground_truth_interval]
    )
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.5
    return 0.0


def reward_lifestyle_recommendation(predicted_lifestyle: Any, ground_truth_lifestyle: str) -> float:
    """Reward overlap between recommended lifestyle actions."""
    option_map = {
        "calcium_supplement": {"calcium_supplement"},
        "exercise": {"exercise"},
        "both": {"calcium_supplement", "exercise"},
        "none": set(),
    }
    if predicted_lifestyle not in option_map:
        return 0.0
    if predicted_lifestyle == ground_truth_lifestyle:
        return 1.0

    predicted_set = option_map[predicted_lifestyle]
    ground_truth_set = option_map[ground_truth_lifestyle]
    if predicted_set and ground_truth_set and predicted_set.intersection(ground_truth_set):
        return 0.6 if "both" in {predicted_lifestyle, ground_truth_lifestyle} else 0.3
    return 0.0


def llm_grade_treatment(
    density_result: Any,
    risk_result: Any,
    treatment_chosen: str,
    age_factor: int,
    vertebra_region: str,
) -> float:
    try:
        import openai

        api_base = resolve_api_base_url(
            os.environ.get("HF_API_BASE")
            or os.environ.get("API_BASE_URL")
            or DEFAULT_HF_ROUTER_URL
        )
        model = os.environ.get("MODEL_ID") or os.environ.get("MODEL_NAME")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
        if not all([api_base, model, token]):
            return 0.5

        client = openai.OpenAI(base_url=api_base, api_key=token)
        response = client.chat.completions.create(
            model=model,
            max_tokens=5,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert clinical decision evaluator for bone health "
                        "management. Score the proposed treatment from 0.0 to 1.0 based "
                        "on clinical appropriateness for the provided density_result and "
                        "fracture_risk. Use 1.0 for a clearly appropriate treatment, 0.6 "
                        "for a partially appropriate but suboptimal treatment, and 0.0 "
                        "for a clearly inappropriate treatment. Respond with ONLY a "
                        "decimal number between 0.0 and 1.0."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Evaluate the clinical appropriateness of this treatment.\n"
                        f"- density_result: {density_result}\n"
                        f"- fracture_risk: {risk_result}\n"
                        f"- age_factor: {age_factor}\n"
                        f"- vertebra_region: {vertebra_region}\n"
                        f"- proposed_treatment: {treatment_chosen}\n\n"
                        "Clinical severity guidance:\n"
                        "- osteoporotic -> medication or surgical_consultation\n"
                        "- osteopenic -> lifestyle_modification or physical_therapy\n"
                        "- normal -> no_intervention\n\n"
                        "Score the proposed_treatment from 0.0 to 1.0 based on clinical "
                        "appropriateness in this context. Reply with only the number."
                    ),
                },
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        nums = re.findall(r"\d+\.?\d*", raw)
        if not nums:
            return 0.5
        return float(min(max(float(nums[0]), 0.0), 1.0))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# BoneEnv - OpenEnv-compatible RL environment
# ---------------------------------------------------------------------------

class BoneEnv:
    """OpenEnv RL environment where each episode is a 3-step patient workflow."""

    def __init__(self, dataset_dir: str = "Dataset") -> None:
        self.dataset_dir: str = dataset_dir
        self.data_dir: str = dataset_dir
        if not os.path.exists(dataset_dir):
            raise RuntimeError(f"Dataset directory '{dataset_dir}' not found")

        image_files = [
            name
            for name in os.listdir(dataset_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not image_files:
            raise RuntimeError(f"No image files found in '{dataset_dir}'")

        self.total_steps: int = 5
        self.current_image: Optional[str] = None
        self._current_state: Optional[Dict[str, float]] = None
        self._current_image_path: Optional[str] = None
        self._done: bool = False
        self.episode_step: int = 0
        self.episode_state: Dict[str, Any] = {}
        self.patient_meta: Dict[str, Any] = {}
        self.base_age_factor: int = 60
        self._rng = random.Random()
        self._task_scores: Dict[str, List[float]] = {
            "bone_density": [],
            "fracture_risk": [],
            "treatment": [],
            "follow_up_interval": [],
            "lifestyle_recommendation": [],
        }

    def reset(self) -> Dict[str, Any]:
        """Load the next image and return the Step 0 observation."""
        self._done = False
        self.episode_step = 0
        self.episode_state = {}
        self._task_scores = {
            "bone_density": [],
            "fracture_risk": [],
            "treatment": [],
            "follow_up_interval": [],
            "lifestyle_recommendation": [],
        }
        self.patient_meta = self._sample_patient_meta()
        self.base_age_factor = self.patient_meta["age"]
        image_files = sorted(
            [
                f
                for f in os.listdir(self.dataset_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        chosen = random.choice(image_files)
        img_path = os.path.join(self.dataset_dir, chosen)
        self.current_image = chosen
        self._current_image_path = img_path
        self._current_state = extract_features(img_path)
        observation = self.state()
        observation["prompt"] = self._build_prompt()
        observation["messages"] = []
        return observation

    def state(self) -> Dict[str, Any]:
        """Return the current step-specific observation."""
        if self._current_state is None:
            self._load_current_image()

        features = dict(self._current_state or {})

        if self.episode_step == 0:
            return {
                "mean_intensity": features["mean_intensity"],
                "std_intensity": features["std_intensity"],
                "edge_density": features["edge_density"],
                "homogeneity": features["homogeneity"],
                "contrast": features["contrast"],
                "energy": features["energy"],
                "correlation": features["correlation"],
                "age": self.patient_meta["age"],
                "sex": self.patient_meta["sex"],
                "bmi": self.patient_meta["bmi"],
                "previous_fracture": self.patient_meta["previous_fracture"],
                "glucocorticoid_use": self.patient_meta["glucocorticoid_use"],
                "step": 0,
            }
        if self.episode_step == 1:
            return {
                "mean_intensity": features["mean_intensity"],
                "std_intensity": features["std_intensity"],
                "edge_density": features["edge_density"],
                "homogeneity": features["homogeneity"],
                "contrast": features["contrast"],
                "energy": features["energy"],
                "correlation": features["correlation"],
                "density_result": self.episode_state.get("density_result"),
                "age_factor": self.episode_state["age_factor"],
                "age": self.patient_meta["age"],
                "sex": self.patient_meta["sex"],
                "bmi": self.patient_meta["bmi"],
                "previous_fracture": self.patient_meta["previous_fracture"],
                "glucocorticoid_use": self.patient_meta["glucocorticoid_use"],
                "step": 1,
            }
        if self.episode_step == 2:
            return {
                "density_result": self.episode_state.get("density_result"),
                "risk_result": self.episode_state.get("risk_result"),
                "age_factor": self.episode_state["age_factor"],
                "vertebra_region": self.episode_state["vertebra_region"],
                "age": self.patient_meta["age"],
                "sex": self.patient_meta["sex"],
                "bmi": self.patient_meta["bmi"],
                "previous_fracture": self.patient_meta["previous_fracture"],
                "glucocorticoid_use": self.patient_meta["glucocorticoid_use"],
                "step": 2,
            }
        if self.episode_step == 3:
            return {
                "density_result": self.episode_state.get("density_result"),
                "risk_result": self.episode_state.get("risk_result"),
                "treatment_result": self.episode_state.get("treatment_result"),
                "vertebra_region": self.episode_state["vertebra_region"],
                "age": self.patient_meta["age"],
                "sex": self.patient_meta["sex"],
                "bmi": self.patient_meta["bmi"],
                "previous_fracture": self.patient_meta["previous_fracture"],
                "glucocorticoid_use": self.patient_meta["glucocorticoid_use"],
                "step": 3,
            }
        if self.episode_step == 4:
            return {
                "density_result": self.episode_state.get("density_result"),
                "risk_result": self.episode_state.get("risk_result"),
                "treatment_result": self.episode_state.get("treatment_result"),
                "follow_up_interval_result": self.episode_state.get("follow_up_interval_result"),
                "vertebra_region": self.episode_state["vertebra_region"],
                "age": self.patient_meta["age"],
                "sex": self.patient_meta["sex"],
                "bmi": self.patient_meta["bmi"],
                "previous_fracture": self.patient_meta["previous_fracture"],
                "glucocorticoid_use": self.patient_meta["glucocorticoid_use"],
                "step": 4,
            }
        return {
            "density_result": self.episode_state.get("density_result"),
            "risk_result": self.episode_state.get("risk_result"),
            "treatment_result": self.episode_state.get("treatment_result"),
            "follow_up_interval_result": self.episode_state.get("follow_up_interval_result"),
            "lifestyle_recommendation_result": self.episode_state.get(
                "lifestyle_recommendation_result"
            ),
            "age": self.patient_meta.get("age"),
            "sex": self.patient_meta.get("sex"),
            "bmi": self.patient_meta.get("bmi"),
            "previous_fracture": self.patient_meta.get("previous_fracture"),
            "glucocorticoid_use": self.patient_meta.get("glucocorticoid_use"),
            "step": self.total_steps,
        }

    def step(self, action: str, task: str = "BoneDensityClassification") -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance the episode by selecting the task handler from the provided task name."""
        if self._done:
            return {}, 0.0, True, {}

        expected_task = STEP_TO_TASK[self.episode_step]
        info: Dict[str, Any] = {
            "expected_task": expected_task,
            "step": self.episode_step,
        }
        task_name = {
            "BoneDensityClassification": "BoneDensityClassification",
            "FractureRiskPrediction": "FractureRiskPrediction",
            "TreatmentRecommendation": "TreatmentProtocol",
            "TreatmentProtocol": "TreatmentProtocol",
            "FollowUpInterval": "FollowUpInterval",
            "LifestyleRecommendation": "LifestyleRecommendation",
        }.get(task)
        if task_name is None:
            info["error"] = "unknown task"
            observation = self._augment_step_observation(self.state(), action, 0.0)
            return observation, 0.0, False, info

        if task_name != expected_task:
            info["error"] = f"expected {expected_task} but received {task}"
            observation = self._augment_step_observation(self.state(), action, 0.0)
            return observation, 0.0, False, info

        if task_name == "BoneDensityClassification":
            normalized_action = self._normalize_categorical_action(
                action, DENSITY_CLASSES, default_action="unknown"
            )
            payload = {"density_class": normalized_action}
            reward = self._handle_density_step(task_name, payload, info)
            self.episode_state["age_factor"] = self.base_age_factor
            self.episode_step = 1
            next_obs = self._augment_step_observation(self.state(), normalized_action, reward)
            return next_obs, reward, False, info

        if task_name == "FractureRiskPrediction":
            payload = {"risk_score": action}
            reward = self._handle_risk_step(task_name, payload, info)
            self.episode_state["vertebra_region"] = self._rng.choice(
                ["lumbar", "thoracic", "cervical"]
            )
            self.episode_step = 2
            next_obs = self._augment_step_observation(self.state(), action, reward)
            return next_obs, reward, False, info

        if task_name == "TreatmentProtocol":
            normalized_action = self._normalize_categorical_action(
                action, TREATMENT_OPTIONS, default_action="unknown"
            )
            payload = {"treatment": normalized_action}
            reward = self._handle_treatment_step(task_name, payload, info)
            self.episode_step = 3
            next_obs = self._augment_step_observation(self.state(), normalized_action, reward)
            return next_obs, reward, False, info

        if task_name == "FollowUpInterval":
            normalized_action = self._normalize_categorical_action(
                action, FOLLOW_UP_OPTIONS, default_action="unknown"
            )
            payload = {"follow_up_interval": normalized_action}
            reward = self._handle_follow_up_step(task_name, payload, info)
            self.episode_step = 4
            next_obs = self._augment_step_observation(self.state(), normalized_action, reward)
            return next_obs, reward, False, info

        normalized_action = self._normalize_categorical_action(
            action, LIFESTYLE_OPTIONS, default_action="unknown"
        )
        payload = {"lifestyle_recommendation": normalized_action}
        reward = self._handle_lifestyle_step(task_name, payload, info)
        self._done = True
        self.episode_step = self.total_steps
        info["episode_summary"] = self._build_episode_summary()
        terminal_obs = self._augment_step_observation(self.state(), normalized_action, reward)
        return terminal_obs, reward, True, info

    def get_task_scores(self) -> Dict[str, float]:
        """Return final mean score per task for the current episode."""
        result: Dict[str, float] = {}
        for task, scores in self._task_scores.items():
            result[task] = round(sum(scores) / len(scores), 4) if scores else 0.0
        return result

    def _normalize_payload(self, payload: Any) -> Tuple[Optional[str], Dict[str, Any], bool]:
        """Validate the wrapper format and detect wrong-step action usage."""
        if not isinstance(payload, dict):
            return None, {}, True

        task_name = payload.get("task")
        action = payload.get("action")
        if not isinstance(action, dict):
            return task_name, {}, True

        expected_task = STEP_TO_TASK.get(self.episode_step)
        if expected_task is None:
            return task_name, action, True
        if task_name != expected_task:
            return task_name, action, True

        expected_key = {
            0: "density_class",
            1: "risk_score",
            2: "treatment",
            3: "follow_up_interval",
            4: "lifestyle_recommendation",
        }.get(self.episode_step)
        if expected_key is None:
            return task_name, action, True
        if expected_key not in action:
            return task_name, action, True

        return task_name, action, False

    def _handle_density_step(self, task_name: Optional[str], action: Dict[str, Any], info: Dict[str, Any]) -> float:
        mean_intensity = self._current_state["mean_intensity"]
        if mean_intensity < 80:
            gt_density = "osteoporotic"
        elif mean_intensity <= 140:
            gt_density = "osteopenic"
        else:
            gt_density = "normal"

        predicted_density = action.get("density_class") if task_name == STEP_TO_TASK[0] else None

        # Continuous reward: class-match score + feature-based confidence
        class_order = ["osteoporotic", "osteopenic", "normal"]
        if predicted_density in class_order:
            distance = abs(class_order.index(predicted_density) - class_order.index(gt_density))
            base_score = 1.0 if distance == 0 else (0.5 if distance == 1 else 0.0)
        else:
            base_score = 0.0

        # Continuous modifier from homogeneity — varies per image
        density_signal = clamp(
            (1.0 - mean_intensity / 255.0) * 0.45
            + clamp(self._current_state.get("std_intensity", 0.0) / 80.0) * 0.2
            + clamp(self._current_state.get("edge_density", 0.0) * 3.0) * 0.1
            + self._current_state.get("homogeneity", 0.5) * 0.15
            + self._current_state.get("energy", 0.5) * 0.1
        )
        density_score = round(base_score * 0.6 + density_signal * 0.4, 4)

        self.episode_state["density_result"] = predicted_density
        self._task_scores["bone_density"] = [density_score]
        info["ground_truth"] = {"bone_density": gt_density}
        info["parsed_action"] = {"density_class": predicted_density}
        return density_score

    def _handle_risk_step(self, task_name: Optional[str], action: Dict[str, Any], info: Dict[str, Any]) -> float:
        gt_risk = self._compute_ground_truth_risk(self._current_state, self.patient_meta)
        predicted_risk = action.get("risk_score") if task_name == STEP_TO_TASK[1] else None
        risk_score, risk_info = reward_risk_score(predicted_risk, gt_risk)
        parsed_risk = self._coerce_risk_value(predicted_risk)

        self.episode_state["risk_result"] = parsed_risk
        self._task_scores["fracture_risk"] = [risk_score]
        info["ground_truth"] = {"fracture_risk": gt_risk}
        info["parsed_action"] = {"risk_score": parsed_risk}
        if risk_info:
            info.update(risk_info)
        return risk_score

    def _handle_treatment_step(self, task_name: Optional[str], action: Dict[str, Any], info: Dict[str, Any]) -> float:
        density_for_treatment = self.episode_state.get("density_result")
        risk_for_treatment = self.episode_state.get("risk_result")
        gt_density = derive_density_class(self._current_state["mean_intensity"])
        gt_risk = self._compute_ground_truth_risk(self._current_state, self.patient_meta)
        gt_treatment = derive_treatment_protocol(gt_density, gt_risk, self.patient_meta)
        predicted_treatment = action.get("treatment") if task_name == STEP_TO_TASK[2] else None
        predicted_treatment = predicted_treatment or "no_intervention"
        density_context = (
            density_for_treatment
            if density_for_treatment in DENSITY_TREATMENT_RECOMMENDATIONS
            else gt_density
        )
        risk_context = risk_for_treatment if risk_for_treatment is not None else gt_risk
        rule_score = reward_treatment(predicted_treatment, density_context)
        # Combine density-severity rules with the clinical appropriateness grader.
        llm_score = round(
            llm_grade_treatment(
                density_context,
                risk_context,
                predicted_treatment,
                self.episode_state.get("age_factor", self.base_age_factor),
                self.episode_state.get("vertebra_region", "lumbar"),
            ),
            4,
        )
        final_score = round(rule_score * 0.6 + llm_score * 0.4, 4)

        self.episode_state["density_result"] = density_for_treatment
        self.episode_state["risk_result"] = risk_for_treatment
        self.episode_state["treatment_result"] = predicted_treatment
        self._task_scores["treatment"] = [final_score]
        info["ground_truth"] = {
            "treatment": gt_treatment,
            "bone_density": gt_density,
            "fracture_risk": gt_risk,
            "recommended_treatments": list(
                DENSITY_TREATMENT_RECOMMENDATIONS.get(density_context, tuple())
            ),
        }
        info["parsed_action"] = {"treatment": predicted_treatment}
        info["context"] = {
            "density_result": density_context,
            "fracture_risk": risk_context,
        }
        info["rule_score"] = rule_score
        info["llm_score"] = llm_score
        info["final_score"] = final_score
        reward = final_score
        return reward

    def _handle_follow_up_step(self, task_name: Optional[str], action: Dict[str, Any], info: Dict[str, Any]) -> float:
        gt_density = derive_density_class(self._current_state["mean_intensity"])
        gt_risk = self._compute_ground_truth_risk(self._current_state, self.patient_meta)
        gt_treatment = derive_treatment_protocol(gt_density, gt_risk, self.patient_meta)
        gt_follow_up = derive_follow_up_interval(
            gt_density, gt_risk, gt_treatment, self.patient_meta
        )
        predicted_follow_up = (
            action.get("follow_up_interval") if task_name == STEP_TO_TASK[3] else None
        )
        follow_up_score = reward_follow_up_interval(predicted_follow_up, gt_follow_up)

        self.episode_state["follow_up_interval_result"] = predicted_follow_up
        self._task_scores["follow_up_interval"] = [follow_up_score]
        info["ground_truth"] = {"follow_up_interval": gt_follow_up}
        info["parsed_action"] = {"follow_up_interval": predicted_follow_up}
        return follow_up_score

    def _handle_lifestyle_step(self, task_name: Optional[str], action: Dict[str, Any], info: Dict[str, Any]) -> float:
        gt_density = derive_density_class(self._current_state["mean_intensity"])
        gt_risk = self._compute_ground_truth_risk(self._current_state, self.patient_meta)
        gt_treatment = derive_treatment_protocol(gt_density, gt_risk, self.patient_meta)
        gt_lifestyle = derive_lifestyle_recommendation(
            gt_density, gt_risk, self.patient_meta, gt_treatment
        )
        predicted_lifestyle = (
            action.get("lifestyle_recommendation") if task_name == STEP_TO_TASK[4] else None
        )
        lifestyle_score = reward_lifestyle_recommendation(predicted_lifestyle, gt_lifestyle)

        self.episode_state["lifestyle_recommendation_result"] = predicted_lifestyle
        self._task_scores["lifestyle_recommendation"] = [lifestyle_score]
        info["ground_truth"] = {"lifestyle_recommendation": gt_lifestyle}
        info["parsed_action"] = {"lifestyle_recommendation": predicted_lifestyle}
        return lifestyle_score

    def _build_episode_summary(self) -> Dict[str, float]:
        density_score = self.get_task_scores().get("bone_density", 0.0)
        risk_score = self.get_task_scores().get("fracture_risk", 0.0)
        treatment_score = self.get_task_scores().get("treatment", 0.0)
        follow_up_score = self.get_task_scores().get("follow_up_interval", 0.0)
        lifestyle_score = self.get_task_scores().get("lifestyle_recommendation", 0.0)
        total_score = round(
            (
                density_score
                + risk_score
                + treatment_score
                + follow_up_score
                + lifestyle_score
            )
            / 5.0,
            4,
        )
        return {
            "density_score": density_score,
            "risk_score": risk_score,
            "treatment_score": treatment_score,
            "follow_up_score": follow_up_score,
            "lifestyle_score": lifestyle_score,
            "total_score": total_score,
        }

    def _sample_patient_meta(self) -> Dict[str, Any]:
        return {
            "age": random.randint(40, 85),
            "sex": random.choice(["M", "F"]),
            "bmi": round(random.uniform(17.0, 35.0), 1),
            "previous_fracture": random.choice([0, 1]),
            "glucocorticoid_use": random.choice([0, 1]),
        }

    def _build_prompt(self) -> str:
        return (
            f"You are assessing a vertebral MRI scan. "
            f"Patient profile — Age: {self.patient_meta['age']}, "
            f"Sex: {self.patient_meta['sex']}, BMI: {self.patient_meta['bmi']}, "
            f"Prior fracture: {self.patient_meta['previous_fracture']}, "
            f"Glucocorticoid use: {self.patient_meta['glucocorticoid_use']}. "
            f"Classify bone density risk using the image features provided."
        )

    def _augment_step_observation(
        self, observation: Dict[str, Any], action: Dict[str, Any], reward: float
    ) -> Dict[str, Any]:
        observation["prompt"] = self._build_prompt()
        observation["messages"] = [
            {"category": "FEEDBACK", "content": f"Action taken: {action}. Reward: {reward:.3f}"}
        ]
        return observation

    def _normalize_categorical_action(
        self, action: Any, valid_actions: List[str], default_action: str = "unknown"
    ) -> str:
        if isinstance(action, bool):
            return default_action

        if isinstance(action, (int, float)):
            scaled = clamp(float(action))
            index = min(len(valid_actions) - 1, max(0, round(scaled * (len(valid_actions) - 1))))
            return valid_actions[index]

        if not isinstance(action, str):
            return default_action

        normalized = action.strip().lower()
        if normalized in valid_actions:
            return normalized

        try:
            scaled = clamp(float(normalized))
        except ValueError:
            return default_action

        index = min(len(valid_actions) - 1, max(0, round(scaled * (len(valid_actions) - 1))))
        return valid_actions[index]

    def _coerce_risk_value(self, action: Any) -> Optional[float]:
        try:
            return round(clamp(float(action)), 3)
        except (TypeError, ValueError):
            return None

    def _compute_ground_truth_risk(
        self, features: Dict[str, float], patient_meta: Dict[str, Any]
    ) -> float:
        """
        Compute a FRAX-inspired fracture risk score from image texture and metadata.

        BMI uses shared adult cutoffs for men and women, density thresholds stay sex-neutral,
        and sex only modulates fracture probability modestly rather than changing the imaging baseline.
        """
        bone_quality = (
            features["homogeneity"] * 0.3
            + (1 - features["contrast"] / 100) * 0.2
            + features["energy"] * 0.3
            + features["correlation"] * 0.2
        )
        bone_quality = max(0.0, min(1.0, bone_quality))
        age_factor = clamp((patient_meta["age"] - 40) / 45.0)
        bmi = float(patient_meta["bmi"])
        if bmi < 18.5:
            bmi_factor = 1.0
        elif bmi < 22.0:
            bmi_factor = 0.5
        else:
            bmi_factor = 0.25

        sex_factor = 1.0
        if patient_meta["sex"] == "F" and patient_meta["age"] >= 50:
            sex_factor = 1.08

        fracture_multiplier = 1.5 if patient_meta["previous_fracture"] else 1.0
        steroid_multiplier = 1.2 if patient_meta["glucocorticoid_use"] else 1.0
        raw_risk = (1.0 - bone_quality) * 0.45 + age_factor * 0.3 + bmi_factor * 0.15
        raw_risk *= sex_factor * fracture_multiplier * steroid_multiplier
        return round(max(0.0, min(1.0, raw_risk)), 3)

    def _load_current_image(self) -> None:
        image_files = [
            name
            for name in os.listdir(self.dataset_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not image_files:
            raise RuntimeError(f"No image files found in '{self.dataset_dir}'")

        chosen_file = random.choice(image_files)
        self.current_image = chosen_file
        self._current_image_path = os.path.join(self.dataset_dir, chosen_file)
        self._current_state = extract_features(self._current_image_path)

    def close(self):
        pass  # cleanup hook, required by OpenEnv spec


# ---------------------------------------------------------------------------
# Task graders
# ---------------------------------------------------------------------------

class BoneDensityGrader:
    """Grades performance on the BoneDensityClassification task."""

    task_name: str = "BoneDensityClassification"

    @staticmethod
    def grade(env: BoneEnv) -> float:
        scores = env._task_scores.get("bone_density", [])
        if not scores:
            return 0.0
        return round(clamp(sum(scores) / len(scores)), 4)


class FractureRiskGrader:
    """Grades performance on the FractureRiskPrediction task."""

    task_name: str = "FractureRiskPrediction"

    @staticmethod
    def grade(env: BoneEnv) -> float:
        scores = env._task_scores.get("fracture_risk", [])
        if not scores:
            return 0.0
        return round(clamp(sum(scores) / len(scores)), 4)


class TreatmentRecommendationGrader:
    """Grades performance on the TreatmentProtocol task."""

    task_name: str = "TreatmentProtocol"

    @staticmethod
    def grade(env: BoneEnv) -> float:
        scores = env._task_scores.get("treatment", [])
        if not scores:
            return 0.0
        return round(clamp(sum(scores) / len(scores)), 4)


class FollowUpIntervalGrader:
    """Grades performance on the FollowUpInterval task."""

    task_name: str = "FollowUpInterval"

    @staticmethod
    def grade(env: BoneEnv) -> float:
        scores = env._task_scores.get("follow_up_interval", [])
        if not scores:
            return 0.0
        return round(clamp(sum(scores) / len(scores)), 4)


class LifestyleRecommendationGrader:
    """Grades performance on the LifestyleRecommendation task."""

    task_name: str = "LifestyleRecommendation"

    @staticmethod
    def grade(env: BoneEnv) -> float:
        scores = env._task_scores.get("lifestyle_recommendation", [])
        if not scores:
            return 0.0
        return round(clamp(sum(scores) / len(scores)), 4)
