import os
import uuid
from models import BoneEnv

DATASET_DIR = os.environ.get("DATASET_DIR", "Dataset")


class BoneEnvironment:
    def __init__(self):
        self._env = None
        self._episode_id = None
        self._step_count = 0
        self._done = False

    def reset(self) -> dict:
        self._env = BoneEnv(dataset_dir=DATASET_DIR)
        obs = self._env.reset()
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        return {"episode_id": self._episode_id, "step": 0, "observation": obs}

    def step(self, action: str, task: str = None) -> dict:
        if self._env is None:
            raise RuntimeError("Call reset() first")
        obs, reward, done, info = self._env.step(action=action, task=task)
        self._step_count += 1
        self._done = done
        return {"observation": obs, "reward": float(reward), "done": done, "info": info, "step": self._step_count}

    @property
    def state(self) -> dict:
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "done": self._done,
            "episode_state": self._env.episode_state if self._env else {}
        }

    def close(self):
        if self._env:
            self._env.close()
