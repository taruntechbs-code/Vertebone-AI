"""Quick validation test for the BoneEnv OpenEnv environment."""
from models import (
    BoneEnv,
    BoneDensityGrader,
    FractureRiskGrader,
    TreatmentRecommendationGrader,
    ACTION_SPACE,
)

print("Running environment validation...")

try:
    env = BoneEnv("Dataset")
except Exception as e:
    print(f"Failed to initialize environment: {e}")
    exit(1)
obs = env.reset()
print("=== RESET OK ===")
print(f"State keys: {list(obs.keys())}")
print(f"First image: {obs['image_file']}")
print(f"Features: {obs['features']}")
print(f"Action space: {ACTION_SPACE}")
print()

total_r = 0.0
step = 0

while not obs["done"]:
    features = obs["features"]
    mean_n = features["mean_intensity"] / 255.0
    edge_n = min(features["edge_density"] / 0.3, 1.0)

    if mean_n > 0.6 and edge_n > 0.5:
        action = "low_risk"
    elif mean_n < 0.35:
        action = "high_risk"
    else:
        action = "medium_risk"

    obs, reward, done, info = env.step(action)
    total_r += reward
    step += 1
    gt = info["ground_truth"]
    print(
        f"Step {step:2d} | {info['action']:12s} | "
        f"reward={reward:.4f} | "
        f"gt_density={gt['bone_density']:.4f} "
        f"gt_fracture={gt['fracture_risk']:.4f} "
        f"gt_treatment={gt['treatment']:.4f}"
    )

print()
print(f"Total reward across {step} steps: {total_r:.4f}")
print(f"Average reward: {total_r / step if step > 0 else 0.0:.4f}")
print()

d = BoneDensityGrader.grade(env)
f = FractureRiskGrader.grade(env)
t = TreatmentRecommendationGrader.grade(env)

print(f"Bone Density Classification : {d:.4f}")
print(f"Fracture Risk Prediction    : {f:.4f}")
print(f"Treatment Recommendation    : {t:.4f}")
print(f"Overall Score               : {round((d + f + t) / 3, 4):.4f}")
print()
print("=== ALL TESTS PASSED ===")
