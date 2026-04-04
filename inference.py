"""
inference.py - End-to-end inference script for the MRI Bone Quality OpenEnv environment.
"""

import os
from typing import Dict

from openai import OpenAI

from models import (
    ACTION_SPACE,
    BoneDensityGrader,
    BoneEnv,
    FractureRiskGrader,
    TreatmentRecommendationGrader,
)


API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
DATASET_DIR: str = os.environ.get("DATASET_DIR", "Dataset")


def build_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )


SYSTEM_PROMPT: str = (
    "You are a medical imaging AI assistant specializing in MRI-based bone quality assessment.\n"
    "Given image features extracted from a spine MRI, you must classify the patient's risk.\n\n"
    "Features provided:\n"
    "  - mean_intensity - average pixel brightness (proxy for bone density; higher = denser bone).\n"
    "  - std_intensity  - pixel standard deviation (structural variation; higher = more heterogeneous).\n"
    "  - edge_density   - fraction of edge pixels (vertebra clarity; higher = clearer edges).\n\n"
    "Based on these features, respond with EXACTLY ONE of the following actions:\n"
    "  low_risk\n"
    "  medium_risk\n"
    "  high_risk\n\n"
    "Guidelines:\n"
    "  - High mean_intensity AND high edge_density -> likely low_risk (strong, clear bones).\n"
    "  - Low mean_intensity OR very high std_intensity -> likely high_risk (weak or degraded bones).\n"
    "  - Otherwise -> medium_risk.\n\n"
    "Reply with ONLY the action string, nothing else."
)


def query_llm(client: OpenAI, features: Dict[str, float]) -> str:
    user_message = (
        f"Image features:\n"
        f"  mean_intensity = {features['mean_intensity']}\n"
        f"  std_intensity  = {features['std_intensity']}\n"
        f"  edge_density   = {features['edge_density']}\n\n"
        "What is the risk classification? Reply with exactly one of: low_risk, medium_risk, high_risk"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        raw = (response.choices[0].message.content or "").strip().lower()

        for action in ACTION_SPACE:
            if action in raw:
                return action

        print("[STEP] Fallback used")
        return improved_fallback(features)
    except Exception:
        print("[STEP] Fallback used")
        return improved_fallback(features)


def improved_fallback(features: dict) -> str:
    def clip(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))

    mean_n = clip(features.get("mean_intensity", 0.0) / 255.0)
    std_n = clip(features.get("std_intensity", 0.0) / 100.0)
    edge_n = clip(features.get("edge_density", 0.0) / 0.3)

    risk_score = (
        0.5 * (1 - mean_n) +
        0.3 * std_n +
        0.2 * (1 - edge_n)
    )

    signal = (mean_n + edge_n - std_n)

    if signal > 0.4:
        return "low_risk"
    elif signal < -0.1:
        return "high_risk"
    else:
        return "medium_risk"


def run_inference() -> None:
    print("[START]")

    env = BoneEnv(dataset_dir=DATASET_DIR)
    print("[STEP] Env initialized")

    client = build_client()

    obs = env.reset()
    print("[STEP] Reset")

    while not obs["done"]:
        features = obs["features"]
        print(f"[STEP] Processing image {obs['step'] + 1}")

        action = query_llm(client, features)
        print(f"[STEP] Action: {action}")

        obs, reward, _, _ = env.step(action)
        print(f"[STEP] Reward: {reward:.4f}")

    density_score = BoneDensityGrader.grade(env)
    fracture_score = FractureRiskGrader.grade(env)
    treatment_score = TreatmentRecommendationGrader.grade(env)
    overall = round((density_score + fracture_score + treatment_score) / 3.0, 4)

    print(f"[STEP] BoneDensity: {density_score:.4f}")
    print(f"[STEP] FractureRisk: {fracture_score:.4f}")
    print(f"[STEP] Treatment: {treatment_score:.4f}")
    print(f"[STEP] Overall: {overall:.4f}")
    print("[END]")


if __name__ == "__main__":
    try:
        run_inference()
    except Exception:
        print("[STEP] Error")
        print("[STEP] BoneDensity: 0.0000")
        print("[STEP] FractureRisk: 0.0000")
        print("[STEP] Treatment: 0.0000")
        print("[STEP] Overall: 0.0000")
        print("[END]")
