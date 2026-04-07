import json
import requests
import websocket  # websocket-client
from models import BoneEnv
class VerteboneEnvClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.ws_url = self.base_url.replace("http", "ws") + "/ws"
        self._ws = None
    def connect(self):
        self._ws = websocket.create_connection(self.ws_url)
        return self
    def reset(self) -> dict:
        self._ws.send(json.dumps({"type": "reset"}))
        return json.loads(self._ws.recv())
    def step(self, action: str, task: str = "BoneDensityClassification") -> dict:
        self._ws.send(json.dumps({"type": "step", "action": action, "task": task}))
        return json.loads(self._ws.recv())
    def state(self) -> dict:
        self._ws.send(json.dumps({"type": "state"}))
        return json.loads(self._ws.recv())
    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health").json()
    def close(self):
        if self._ws:
            self._ws.send(json.dumps({"type": "close"}))
            self._ws.close()
    def __enter__(self):
        return self.connect()
    def __exit__(self, *args):
        self.close()
__all__ = ["VerteboneEnvClient", "BoneEnv"]
