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


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "Vertebone-AI")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "vertebone")
MAX_STEPS = 5
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1
HF_API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY") or ""
DATASET_DIR: str = os.environ.get("DATASET_DIR", "Dataset")
EVAL_EPISODES: int = int(os.getenv("EVAL_EPISODES", "8"))
RESULTS_PATH: str = os.environ.get("RESULTS_PATH", "results.json")


def build_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_API_KEY,
        timeout=30.0,
    )


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


def _format_optional_number(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"


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

    if step == 0:
        task = "BoneDensityClassification"
        user_msg = (
            f"{prompt_text}\n\nImage features: {features_text}\n"
            f"Context: {context_text}\n\n"
            "Classify bone density. Reply with exactly one word: "
            "osteoporotic, osteopenic, or normal."
        )
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
        fracture_risk = obs.get("risk_result", "unknown")
        density_result = obs.get("density_result", "unknown")
        user_msg = (
            f"{prompt_text}\n\nImage features: {features_text}\n"
            f"Context: {context_text}\n\n"
            f"The patient's fracture_risk score is {fracture_risk} and "
            f"density_result is {density_result}.\n\n"
            "Use these EXACT clinical rules to choose the follow-up interval:\n"
            "- If fracture_risk > 0.6 OR density_result is osteoporotic → 3_months\n"
            "- If fracture_risk is between 0.3 and 0.6 OR density_result is osteopenic → 6_months\n"
            "- If fracture_risk < 0.3 AND density_result is normal → 12_months\n\n"
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
                {"role": "system", "content": "You are a clinical bone assessment AI."},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
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
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    all_rewards: List[float] = []
    error_count = 0
    success = True
    info: Dict[str, Any] = {}

    try:
        for episode_step in range(MAX_STEPS):
            payload = query_llm(client, obs)
            obs, env_reward, done, info = env.step(action=payload["action"], task=payload["task"])

            step_error = payload.get("error")
            if step_error:
                reward = 0.0
                error_count += 1
            else:
                reward = env_reward
            all_rewards.append(reward)

            print(
                f"[STEP] step={episode_step+1} action={payload['action']} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error={_format_error(step_error)}"
            )
            if done:
                break
    except Exception:
        success = False

    steps = len(all_rewards)
    if steps == 0:
        all_rewards = [0.0 for _ in range(MAX_STEPS)]
        steps = 0

    score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    rewards_str = ",".join(f"{reward:.2f}" for reward in all_rewards)
    error_rate = error_count / len(all_rewards) if all_rewards else 0.0
    density_result = env.episode_state.get("density_result", "unknown")
    fracture_risk = env.episode_state.get("risk_result", "unknown")
    treatment_result = env.episode_state.get("treatment_result", "unknown")
    follow_up_result = env.episode_state.get("follow_up_interval_result", "unknown")
    lifestyle_result = env.episode_state.get("lifestyle_recommendation_result", "unknown")
    sex_value = env.patient_meta.get("sex", "unknown")

    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={len(all_rewards) if success else 0} "
        f"score={score:.2f} rewards={rewards_str} "
        f"density_acc={density_result} fracture_risk={_format_optional_number(fracture_risk)} "
        f"treatment={treatment_result} follow_up_interval={follow_up_result} "
        f"lifestyle_recommendation={lifestyle_result} sex={sex_value} "
        f"error_rate={error_rate:.2f}"
    )

    env.close()
    return {
        "episode": episode_index + 1,
        "success": success,
        "steps": len(all_rewards),
        "score": round(score, 4),
        "rewards": [round(reward, 4) for reward in all_rewards],
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
    client = build_client()
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
        per_step_mean_reward.append(round(statistics.fmean(step_values), 4))

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
