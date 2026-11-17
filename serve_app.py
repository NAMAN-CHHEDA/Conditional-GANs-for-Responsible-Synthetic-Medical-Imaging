"""Step 7 - deploy the selected generator with Ray Serve.

Python (e.g. from the notebook):
    from serve_app import deploy
    deploy("artifacts/best_generator.pt")          # -> http://127.0.0.1:8000/generate

CLI:
    CHECKPOINT=artifacts/best_generator.pt serve run serve_app:app

Endpoints
    POST /generate     {"label": 0|1|"normal"|"pneumonia", "n": 1..64, "seed": int?, "size": 28|64,
                        "return_grid": bool}
    GET  /health       liveness + model id
    GET  /model_card   intended use, limitations, config

Production concerns handled here
* validation   : pydantic model with extra="forbid", bounded n, enumerated labels -> HTTP 422
* batching     : @serve.batch merges concurrent requests into one generator forward pass
* autoscaling  : 1..4 replicas driven by ongoing requests per replica
* resources    : fractional GPU per replica (0.25 of an A100), 1 CPU
* safety       : every response has synthetic=true + disclaimer, header X-Synthetic-Data,
                 PNG tEXt provenance metadata, per-request cap; no image upload / no
                 patient data ever enters the service.
"""

import os
import secrets
import time
from typing import List, Optional, Tuple

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ray import serve

from gan_lab import SYNTHETIC_DISCLAIMER
from gan_lab.service import GenerateRequest, SyntheticImageService

APP_NAME = "synthetic-cxr"
DEPLOYMENT_NAME = "GeneratorAPI"

api = FastAPI(title="Synthetic PneumoniaMNIST generator (education/research only)",
              description=SYNTHETIC_DISCLAIMER, version="1.0.0")


@serve.deployment(
    name=DEPLOYMENT_NAME,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 4,  # scale up when replicas average > 4 in-flight requests
        "upscale_delay_s": 3,
        "downscale_delay_s": 30,
        "look_back_period_s": 5,
    },
    max_ongoing_requests=16,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
    health_check_period_s=10,
)
@serve.ingress(api)
class GeneratorAPI:
    def __init__(self, checkpoint_path: str):
        self.svc = SyntheticImageService(checkpoint_path)
        ctx = serve.get_replica_context()
        self.replica = getattr(ctx, "replica_tag", None) or str(getattr(ctx, "replica_id", "replica"))
        self.n_requests = 0

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.005)
    async def _generate_batched(self, items: List[Tuple[int, int, int, int]]):
        return self.svc.generate_batch(items)

    @api.post("/generate")
    async def generate(self, req: GenerateRequest):
        t0 = time.perf_counter()
        seed = req.seed if req.seed is not None else secrets.randbelow(2**31 - 1)
        imgs = await self._generate_batched((req.label, req.n, seed, req.size))
        self.n_requests += 1
        body = self.svc.build_response(req, seed, imgs, t0, replica=self.replica)
        return JSONResponse(body, headers={"X-Synthetic-Data": "true", "X-Model-Id": self.svc.model_id})

    @api.get("/health")
    async def health(self):
        return {"status": "ok", "model_id": self.svc.model_id, "device": str(self.svc.device),
                "replica": self.replica, "requests_served": self.n_requests}

    @api.get("/model_card")
    async def model_card(self):
        return self.svc.model_card()


def build_app(checkpoint_path: str, num_gpus: Optional[float] = None, min_replicas: int = 1,
              max_replicas: int = 4, target_ongoing_requests: float = 4, num_cpus: float = 1):
    if num_gpus is None:
        num_gpus = 0.25 if torch.cuda.is_available() else 0.0
    dep = GeneratorAPI.options(
        ray_actor_options={"num_cpus": num_cpus, "num_gpus": num_gpus},
        autoscaling_config={"min_replicas": min_replicas, "initial_replicas": min_replicas,
                            "max_replicas": max_replicas, "target_ongoing_requests": target_ongoing_requests,
                            "upscale_delay_s": 3, "downscale_delay_s": 30, "look_back_period_s": 5},
    )
    return dep.bind(os.path.abspath(checkpoint_path))


def deploy(checkpoint_path: str = "artifacts/best_generator.pt", host: str = "127.0.0.1", port: int = 8000,
           **kw):
    import ray

    if not ray.is_initialized():
        project_dir = os.path.dirname(os.path.abspath(__file__))
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONPATH": project_dir}})
    serve.start(http_options={"host": host, "port": port})
    handle = serve.run(build_app(checkpoint_path, **kw), name=APP_NAME, route_prefix="/")
    return handle


def replica_count(app_name: str = APP_NAME, deployment: str = DEPLOYMENT_NAME) -> dict:
    """Current replica states, e.g. {'RUNNING': 2, 'STARTING': 1}."""
    st = serve.status()
    dep = st.applications[app_name].deployments[deployment]
    return {str(getattr(k, "value", k)): int(v) for k, v in dep.replica_states.items()}


# `serve run serve_app:app` entrypoint (the model is loaded lazily inside replicas)
app = build_app(os.environ.get("CHECKPOINT", "artifacts/best_generator.pt"))
