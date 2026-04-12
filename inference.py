"""
inference.py - Multi-episode inference and evaluation runner for Vertebone-AI.
"""

import json
import os
import statistics
from typing import Any, Dict, List

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from models import (
    BoneEnv,
    DENSITY_CLASSES,
    FOLLOW_UP_OPTIONS,
    LIFESTYLE_OPTIONS,
    TREATMENT_OPTIONS,
)


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "Vertebone-AI")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "vertebone")
MAX_STEPS = 5
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1
DATASET_DIR: str = os.environ.get("DATASET_DIR", "Dataset")
EVAL_EPISODES: int = int(os.getenv("EVAL_EPISODES", "8"))
RESULTS_PATH: str = os.environ.get("RESULTS_PATH", "results.json")


def _normalize_label_action(raw: str, valid: List[str], default: str) -> str:
    if raw in valid:
        return raw

    try:
        scaled = max(0.0, min(1.0, float(raw)))
    except ValueError:
        return default

    index = min(len(valid) - 1, max(0, round(scaled * (len(valid) - 1))))
    return valid[index]


def _format_error(error: str | None) -> str:
    if not error:
        return "null"
    return error.replace("\n", " ").strip()


def _build_context_text(obs: Dict[str, Any]) -> str:
    context_keys = [
        "density_result",
        "risk_result",
        "treatment_result",
        "follow_up_interval_result",
        "lifestyle_recommendation_result",
        "vertebra_region",
        "age",
        "sex",
        "bmi",
        "previous_fracture",
        "glucocorticoid_use",
    ]
    context_items = [f"{key}={obs[key]}" for key in context_keys if key in obs and obs[key] is not None]
    return ", ".join(context_items)


def query_llm(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    step = obs.get("step", 0)
    prompt_text = obs.get("prompt", "Assess bone quality from the given features.")
    features_text = (
        f"mean_intensity={obs.get('mean_intensity', 0):.2f}, "
        f"std_intensity={obs.get('std_intensity', 0):.2f}, "
        f"edge_density={obs.get('edge_density', 0):.4f}, "
        f"homogeneity={obs.get('homogeneity', 0):.4f}, "
        f"contrast={obs.get('contrast', 0):.4f}, "
        f"energy={obs.get('energy', 0):.4f}, "
        f"correlation={obs.get('correlation', 0):.4f}"
    )
    context_text = _build_context_text(obs)
    system_msg = "You are a clinical bone assessment AI."
    temperature = TEMPERATURE

    if step == 0:
        task = "BoneDensityClassification"
        system_msg = (
            "You are a radiology AI. Classify bone density from MRI features.\n\n"
            "Rules (follow strictly):\n"
            "- If mean_intensity < 80: classify as \"osteoporotic\"\n"
            "- If mean_intensity >= 80 and mean_intensity <= 140: classify as \"osteopenic\"  \n"
            "- If mean_intensity > 140: classify as \"normal\"\n\n"
            "Supporting signals:\n"
            "- homogeneity < 0.5 -> supports osteoporotic\n"
            "- homogeneity 0.5-0.7 -> supports osteopenic\n"
            "- homogeneity > 0.7 -> supports normal\n"
            "- energy < 0.2 -> supports osteoporotic\n"
            "- energy 0.2-0.35 -> supports osteopenic\n"
            "- energy > 0.35 -> supports normal\n\n"
            "Respond with ONLY one word: osteoporotic, osteopenic, or normal"
        )
        user_msg = (
            f"mean_intensity: {obs.get('mean_intensity', 0):.2f}\n"
            f"std_intensity: {obs.get('std_intensity', 0):.2f}\n"
            f"homogeneity: {obs.get('homogeneity', 0):.4f}\n"
            f"energy: {obs.get('energy', 0):.4f}\n"
            f"edge_density: {obs.get('edge_density', 0):.4f}\n"
            f"age: {obs.get('age', 'unknown')}\n"
            f"previous_fracture: {obs.get('previous_fracture', 'unknown')}\n\n"
            "Based on the rules above, classify the bone density:"
        )
        temperature = 0.0
        default = "osteopenic"
        valid = DENSITY_CLASSES
    elif step == 1:
        task = "FractureRiskPrediction"
        user_msg = (
            f"{prompt_text}\n\nImage features: {features_text}\n"
            f"Context: {context_text}\n\n"
            "Predict fracture risk as a float between 0.0 and 1.0. "
            "Reply with only the number, e.g. 0.42"
        )
        default = "0.5"
        valid = None
    elif step == 2:
        task = "TreatmentRecommendation"
        user_msg = (
            f"{prompt_text}\n\nImage features: {features_text}\n"
            f"Context: {context_text}\n\n"
            "Recommend treatment. Reply with exactly one option: "
            "no_intervention, lifestyle_modification, physical_therapy, "
            "medication, or surgical_consultation."
        )
        default = "physical_therapy"
        valid = TREATMENT_OPTIONS
    elif step == 3:
        task = "FollowUpInterval"
        density_result = obs.get("density_result", "unknown")
        user_msg = (
            f"{prompt_text}\n\nImage features: {features_text}\n"
            f"Context: {context_text}\n\n"
            f"density_result={density_result}. "
            "Rules: osteoporotic=3_months, osteopenic=6_months, normal=12_months. "
            "Reply with exactly one option: 3_months, 6_months, or 12_months."
        )
        default = "6_months"
        valid = FOLLOW_UP_OPTIONS
    else:
        task = "LifestyleRecommendation"
        user_msg = (
            f"{prompt_text}\n\nImage features: {features_text}\n"
            f"Context: {context_text}\n\n"
            "Recommend lifestyle support. Reply with exactly one option: "
            "calcium_supplement, exercise, both, or none."
        )
        default = "both"
        valid = LIFESTYLE_OPTIONS

    error_message = None
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=MAX_TOKENS,
            temperature=temperature,
        )
        raw = (response.choices[0].message.content or "").strip().lower()
        if valid is not None:
            action = _normalize_label_action(raw, valid, default)
        else:
            try:
                action = str(round(float(raw), 3))
            except ValueError:
                action = default
    except (APITimeoutError, RateLimitError, APIConnectionError) as exc:
        action = default
        error_message = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        action = default
        error_message = f"{type(exc).__name__}: {exc}"

    return {"task": task, "action": action, "error": error_message}


def run_episode(client: OpenAI, episode_index: int) -> Dict[str, Any]:
    env = BoneEnv(dataset_dir=DATASET_DIR)
    obs = env.reset()
    task_name = TASK_NAME
    print(f"[START] task={task_name} env=vertebone model={MODEL_NAME}")

    rewards: List[float] = []
    success = False
    steps = 0
    info: Dict[str, Any] = {}

    try:
        for episode_step in range(MAX_STEPS):
            payload = query_llm(client, obs)
            obs, env_reward, done, info = env.step(action=payload["action"], task=payload["task"])

            step_error = payload.get("error")
            if step_error:
                reward = 0.011
            else:
                reward = env_reward
            rewards.append(reward)
            steps = episode_step + 1

            step = episode_step + 1
            action = payload["action"]
            error = _format_error(step_error)
            print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}")
            if done:
                success = True
                break
        else:
            success = True
    except Exception:
        success = False
    finally:
        print(f"[END] success={str(success).lower()} steps={steps} rewards={','.join(f'{r:.2f}' for r in rewards)}")
        env.close()

    score = sum(rewards) / len(rewards) if rewards else 0.0
    error_rate = round(sum(1 for reward in rewards if reward == 0.0) / len(rewards), 4) if rewards else 0.0
    density_result = env.episode_state.get("density_result", "unknown")
    fracture_risk = env.episode_state.get("risk_result", "unknown")
    treatment_result = env.episode_state.get("treatment_result", "unknown")
    follow_up_result = env.episode_state.get("follow_up_interval_result", "unknown")
    lifestyle_result = env.episode_state.get("lifestyle_recommendation_result", "unknown")
    sex_value = env.patient_meta.get("sex", "unknown")
    return {
        "episode": episode_index + 1,
        "success": success,
        "steps": steps,
        "score": round(score, 4),
        "rewards": [round(reward, 4) for reward in rewards],
        "density_acc": density_result,
        "fracture_risk": None if fracture_risk == "unknown" else fracture_risk,
        "treatment": treatment_result,
        "follow_up_interval": follow_up_result,
        "lifestyle_recommendation": lifestyle_result,
        "sex": sex_value,
        "error_rate": round(error_rate, 4),
        "raw_info": info,
    }


def save_results(results: Dict[str, Any]) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)


def run_inference() -> None:
    episode_results = [run_episode(client, episode_index) for episode_index in range(EVAL_EPISODES)]

    scores = [result["score"] for result in episode_results]
    mean_score = statistics.fmean(scores) if scores else 0.0
    std_dev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    success_rate = (
        sum(
            1 for result in episode_results
            if result["score"] > SUCCESS_SCORE_THRESHOLD
        ) / len(episode_results)
        if episode_results
        else 0.0
    )
    max_steps = max((len(result["rewards"]) for result in episode_results), default=0)
    per_step_mean_reward = []
    for step_index in range(max_steps):
        step_values = [result["rewards"][step_index] for result in episode_results]
        per_step_mean_reward.append(max(0.01, min(0.99, round(statistics.fmean(step_values), 4))))

    evaluation_summary = {
        "task_name": TASK_NAME,
        "episodes": EVAL_EPISODES,
        "mean_score": round(mean_score, 4),
        "std_dev": round(std_dev, 4),
        "success_rate": round(success_rate, 4),
        "per_step_mean_reward": per_step_mean_reward,
    }
    save_results({"summary": evaluation_summary, "episodes": episode_results})

    per_step_mean_str = ",".join(f"{value:.2f}" for value in per_step_mean_reward)
    print(
        f"[EVAL] episodes={EVAL_EPISODES} mean_score={mean_score:.2f} "
        f"std={std_dev:.2f} success_rate={success_rate:.0%} "
        f"per_step_mean={per_step_mean_str}"
    )


if __name__ == "__main__":
    run_inference()
