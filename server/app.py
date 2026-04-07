import os

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from models import BoneEnv

app = FastAPI(title="Vertebone-AI OpenEnv")
DATASET_DIR = os.environ.get("DATASET_DIR", "Dataset")


class StepRequest(BaseModel):
    action: Optional[str] = ""
    task: Optional[str] = "BoneDensityClassification"


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def http_reset():
    env = BoneEnv(dataset_dir=DATASET_DIR)
    obs = env.reset()
    if hasattr(obs, "model_dump"):
        obs = obs.model_dump()
    return JSONResponse(content={"observation": obs, "done": False, "reward": 0.0})


@app.post("/step")
async def http_step(request: StepRequest):
    env = BoneEnv(dataset_dir=DATASET_DIR)
    env.reset()
    obs, reward, done, info = env.step(action=request.action, task=request.task)
    if hasattr(obs, "model_dump"):
        obs = obs.model_dump()
    return JSONResponse(content={"observation": obs, "reward": reward, "done": done, "info": info})


@app.get("/state")
async def http_state():
    env = BoneEnv(dataset_dir=DATASET_DIR)
    s = env.state()
    if hasattr(s, "model_dump"):
        s = s.model_dump()
    return JSONResponse(content=s)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env = BoneEnv(dataset_dir=DATASET_DIR)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            if msg_type == "reset":
                obs = env.reset()
                await websocket.send_json({"type": "reset", "observation": obs, "done": False, "reward": 0.0})
            elif msg_type == "step":
                action = data.get("action", "")
                task = data.get("task", "BoneDensityClassification")
                obs, reward, done, info = env.step(action, task=task)
                await websocket.send_json({"type": "step", "observation": obs, "reward": reward, "done": done, "info": info})
            elif msg_type == "state":
                await websocket.send_json({"type": "state", "state": env.state()})
            elif msg_type == "close":
                break
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})
    except WebSocketDisconnect:
        pass
    finally:
        env.close() if hasattr(env, "close") else None


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        workers=int(os.getenv("WORKERS", "1")),
    )


if __name__ == "__main__":
    main()
