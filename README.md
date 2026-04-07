# VerteBone-AI
![Python 3.10](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)
![Qwen2.5-72B](https://img.shields.io/badge/Qwen-2.5--72B-orange)

VerteBone-AI is a reinforcement learning environment for vertebral bone density analysis from MRI-derived texture features. Each episode simulates a clinical decision workflow that starts with density classification, moves through fracture risk estimation, and culminates in treatment planning, while optional extended steps model follow-up cadence and lifestyle counseling for longer-horizon RL evaluation.

## Core Pipeline
The clinical core of VerteBone-AI follows a 3-step pipeline:

1. Density classification
2. Fracture risk prediction
3. Treatment recommendation

The current environment extends that core episode with:

4. Follow-up interval selection
5. Lifestyle recommendation

## Observation Space
Each observation is a structured dictionary built from image-derived features and patient metadata:

- `mean_intensity`
- `std_intensity`
- `edge_density`
- `homogeneity`
- `contrast`
- `energy`
- `correlation`
- `age`
- `sex`
- `bmi`
- `previous_fracture`
- `glucocorticoid_use`

Later steps also include intermediate outputs such as `density_result`, `risk_result`, `treatment_result`, and `follow_up_interval_result`.

## Action Spaces
Actions are task-specific:

- `BoneDensityClassification`: `osteoporotic`, `osteopenic`, `normal`
- `FractureRiskPrediction`: continuous float in `[0.0, 1.0]`
- `TreatmentRecommendation`: `no_intervention`, `lifestyle_modification`, `physical_therapy`, `medication`, `surgical_consultation`
- `FollowUpInterval`: `3_months`, `6_months`, `12_months`
- `LifestyleRecommendation`: `calcium_supplement`, `exercise`, `both`, `none`

## Reward Design
VerteBone-AI uses clinically-inspired local rewards so each decision contributes interpretable signal:

- Density reward blends categorical correctness with image feature confidence, so different scans can produce different partial credit even when the class label is close.
- Fracture risk reward is proportional to numeric error against a FRAX-inspired target that includes age, BMI, fracture history, glucocorticoid exposure, and a modest post-menopausal female risk adjustment.
- Treatment reward combines rule-based category distance with an LLM clinical appropriateness score.
- Follow-up interval reward favors exact matches and gives partial credit for adjacent time windows.
- Lifestyle reward gives credit for overlapping recommendations such as `both` vs `exercise`.

## Example Log Output
```text
[START] task=Vertebone-AI env=vertebone model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=osteoporotic reward=0.88 done=false error=null
[STEP] step=2 action=0.67 reward=0.79 done=false error=null
[STEP] step=3 action=medication reward=0.54 done=false error=null
[STEP] step=4 action=6_months reward=1.00 done=false error=null
[STEP] step=5 action=both reward=1.00 done=true error=null
[END] success=true steps=5 score=0.84 rewards=0.88,0.79,0.54,1.00,1.00 density_acc=osteoporotic fracture_risk=0.67 treatment=medication follow_up_interval=6_months lifestyle_recommendation=both sex=F error_rate=0.00
[EVAL] episodes=50 mean_score=0.61 std=0.12 success_rate=72% per_step_mean=0.58,0.61,0.49,0.66,0.71
```

## Running Locally
```bash
pip install -r requirements.txt
set HF_TOKEN=hf_your_token_here
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```
