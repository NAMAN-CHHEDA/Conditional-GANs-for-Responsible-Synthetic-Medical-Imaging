import base64
import io

import pytest
from PIL import Image
from pydantic import ValidationError

from gan_lab.models import Generator, save_generator
from gan_lab.service import GenerateRequest, SyntheticImageService
from gan_lab.trainer import DEFAULT_CONFIG


@pytest.fixture(scope="module")
def svc(tmp_path_factory):
    cfg = {**DEFAULT_CONFIG, "latent_dim": 16, "ngf": 8, "ndf": 8}
    p = tmp_path_factory.mktemp("ckpt") / "g.pt"
    G = Generator(latent_dim=16, ngf=8)
    save_generator(str(p), G, cfg, {"epoch": 1})
    return SyntheticImageService(str(p), device="cpu")


@pytest.mark.parametrize("payload", [
    {"label": 2}, {"label": "covid"}, {"label": True}, {"label": 0, "n": 0}, {"label": 0, "n": 65},
    {"label": 0, "seed": -1}, {"label": 0, "size": 128}, {"label": 0, "patient_id": "x"},
])
def test_invalid_requests_rejected(payload):
    with pytest.raises(ValidationError):
        GenerateRequest(**payload)


def test_label_aliases():
    assert GenerateRequest(label="Pneumonia").label == 1
    assert GenerateRequest(label="normal").label == 0
    assert GenerateRequest(label="1").label == 1


def test_response_is_marked_synthetic_and_reproducible(svc):
    r1 = svc.handle(GenerateRequest(label=1, n=3, seed=5, return_grid=True))
    r2 = svc.handle(GenerateRequest(label=1, n=3, seed=5))
    assert r1["synthetic"] is True and "SYNTHETIC" in r1["disclaimer"]
    assert r1["label_name"] == "pneumonia" and len(r1["images"]) == 3 and "grid" in r1
    assert r1["images"] == r2["images"]
    img = Image.open(io.BytesIO(base64.b64decode(r1["images"][0])))
    assert img.size == (64, 64) and img.info["Synthetic"] == "true"


def test_size_28_and_batching_consistency(svc):
    a = svc.generate_batch([(0, 2, 9, 28)])[0]
    b = svc.generate_batch([(1, 5, 3, 64), (0, 2, 9, 28)])[1]
    assert a.shape == (2, 28, 28)
    # identical up to float rounding: batch composition must not change a request's images
    assert abs(a.astype(int) - b.astype(int)).max() <= 1
