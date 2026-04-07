"""Quick validation test for the BoneEnv OpenEnv environment."""

from models import BoneDensityGrader, BoneEnv, FractureRiskGrader, TreatmentRecommendationGrader


print("Running environment validation...")

try:
    env = BoneEnv("Dataset")
except Exception as e:
    print(f"Failed to initialize environment: {e}")
    raise SystemExit(1)

obs = env.reset()
print("=== RESET OK ===")
print(f"Step 0 observation: {obs}")
assert obs["step"] == 0
assert set(obs.keys()) == {"mean_intensity", "std_intensity", "edge_density", "step"}

obs, reward, done, info = env.step(
    {
        "task": "BoneDensityClassification",
        "action": {"density_class": "osteopenic"},
    }
)
print(f"Task1 reward: {reward}")
assert done is False
assert obs["step"] == 1
assert "density_result" in obs
assert "age_factor" in obs

obs, reward, done, info = env.step(
    {
        "task": "FractureRiskPrediction",
        "action": {"risk_score": 0.65},
    }
)
print(f"Task2 reward: {reward}")
assert done is False
assert obs["step"] == 2
assert "risk_result" in obs
assert "vertebra_region" in obs

obs, reward, done, info = env.step(
    {
        "task": "TreatmentProtocol",
        "action": {"treatment": "physical_therapy"},
    }
)
print(f"Task3 reward: {reward}")
assert done is True
assert "episode_summary" in info

d = BoneDensityGrader.grade(env)
f = FractureRiskGrader.grade(env)
t = TreatmentRecommendationGrader.grade(env)

print()
print(f"Bone Density Classification : {d:.4f}")
print(f"Fracture Risk Prediction    : {f:.4f}")
print(f"Treatment Recommendation    : {t:.4f}")
print(f"Episode Total Score         : {info['episode_summary']['total_score']:.4f}")
print()
print("=== ALL TESTS PASSED ===")
