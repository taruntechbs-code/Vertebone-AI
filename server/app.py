import os

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from models import BoneEnv


DATASET_DIR = os.environ.get("DATASET_DIR", "Dataset")


class ActionInput(BaseModel):
    action: str


app = FastAPI()

env = None
_env_init_error = None

try:
    env = BoneEnv(dataset_dir=DATASET_DIR)
except Exception as exc:
    _env_init_error = str(exc)


def _get_env() -> BoneEnv:
    if env is None:
        detail = {"error": "Environment not initialized"}
        if _env_init_error:
            detail["detail"] = _env_init_error
        raise HTTPException(status_code=503, detail=detail)
    return env


@app.post("/reset")
def reset():
    observation = _get_env().reset()
    return {
        "status": "ok",
        "observation": jsonable_encoder(observation),
    }


@app.post("/step")
def step(action_input: ActionInput):
    observation, reward, done, info = _get_env().step(action_input.action)
    return {
        "observation": jsonable_encoder(observation),
        "reward": reward,
        "done": done,
        "info": jsonable_encoder(info),
    }


@app.get("/state")
def state():
    return jsonable_encoder(_get_env().state())
