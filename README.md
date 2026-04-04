# Bone OpenEnv Environment

This repository contains an OpenEnv-compatible reinforcement learning environment for MRI-based bone quality assessment and vertebral fracture risk prediction.

The project includes:

- `models.py` with the `BoneEnv` environment and task graders
- `inference.py` for end-to-end inference with `[START]`, `[STEP]`, and `[END]` logs
- `server/app.py` exposing the environment through FastAPI endpoints
- `openenv.yaml` describing the environment configuration
- `Dataset/` containing the MRI image inputs

## Project Structure

```text
.
├── Dataset/
├── server/
│   └── app.py
├── Dockerfile
├── inference.py
├── models.py
├── openenv.yaml
├── requirements.txt
└── test_env.py
```

## Requirements

- Python 3.11 recommended
- Dataset available under `Dataset/`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

The project uses the following environment variables:

- `DATASET_DIR` default: `Dataset`
- `API_BASE_URL` default: `https://api.openai.com/v1`
- `MODEL_NAME` default: `meta-llama/Llama-3.1-8B-Instruct`
- `HF_TOKEN` default: empty

## Run Local Validation

Use the quick validation script:

```bash
python test_env.py
```

This checks that the environment initializes, resets correctly, steps through the dataset, and produces final task scores.

## Run Inference

Run the OpenEnv inference script:

```bash
python inference.py
```

The script:

- initializes `BoneEnv`
- processes dataset images one by one
- queries the configured model or uses fallback logic if needed
- prints structured progress logs and final scores

## Run HTTP Server

Start the FastAPI server locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Available endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`

### Example Requests

Reset the environment:

```bash
curl -X POST http://127.0.0.1:7860/reset
```

Take a step:

```bash
curl -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d "{\"action\":\"medium_risk\"}"
```

Fetch current state:

```bash
curl http://127.0.0.1:7860/state
```

## Docker

Build the image:

```bash
docker build -t bone-openenv .
```

Run the container:

```bash
docker run -p 7860:7860 bone-openenv
```

The container starts the FastAPI server with:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## OpenEnv Configuration

The environment is described in `openenv.yaml` with:

- entry point: `models:BoneEnv`
- action space: `low_risk`, `medium_risk`, `high_risk`
- dataset path: `Dataset/`
- three grading tasks:
  - Bone Density Classification
  - Fracture Risk Prediction
  - Treatment Recommendation

## Notes

- `Dataset/` is part of the project and should remain included in the repository.
- The environment logic lives in `models.py`.
- The server layer in `server/app.py` exposes the environment without changing the underlying reward or transition logic.
