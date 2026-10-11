import importlib.util
from pathlib import Path
import sys
import pytest
import torch
from realign_vla.utils import atomic_save, load_config, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def test_cached_feature_training_and_resume_are_identical(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("train_entry", ROOT / "scripts/train_realign.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    config = load_config(ROOT / "configs/train_realign.yaml")
    config["model"].update(feature_dim=12, agent_tokens=8, hidden_dim=8, rank=3)
    config["optimization"].update(total_steps=3, validate_every=1)
    config_path = tmp_path / "config.json"
    write_json(config, config_path)  # JSON is also valid YAML.
    cache = tmp_path / "cache"; cache.mkdir()
    shards = []
    torch.manual_seed(123)
    for split in ("train", "val"):
        clean = torch.randn(3, 8, 12).half()
        source = torch.arange(3).repeat_interleave(3)
        corrupted = (clean[source].float() + .2 * torch.randn(9, 8, 12)).half()
        path = cache / (split + ".pt")
        atomic_save(dict(clean=clean, corrupted=corrupted, source=source), path)
        shards.append(dict(file=path.name, sha256=sha256(path), split=split))
    write_json(dict(shards=shards), cache / "index.json")

    def run(output, resume=None):
        argv = ["train_realign.py", "--config", str(config_path), "--features", str(cache), "--output", str(output), "--device", "cpu"]
        if resume is not None:
            argv += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", argv)
        training.main()
    complete = tmp_path / "complete"
    run(complete)
    save = training.atomic_save
    class Interruption(Exception):
        pass
    def stop_after_two_steps(checkpoint, path):
        save(checkpoint, path)
        if Path(path).name == "last.pt" and checkpoint["next_step"] == 2:
            raise Interruption()
    monkeypatch.setattr(training, "atomic_save", stop_after_two_steps)
    interrupted = tmp_path / "interrupted"
    with pytest.raises(Interruption):
        run(interrupted)
    monkeypatch.setattr(training, "atomic_save", save)
    run(interrupted, interrupted / "last.pt")
    a = torch.load(complete / "last.pt", weights_only=True)
    b = torch.load(interrupted / "last.pt", weights_only=True)
    assert a["curve"] == b["curve"]
    assert a["selected_step"] == b["selected_step"]
    assert all(torch.equal(value, b["state_dict"][key]) for key,value in a["state_dict"].items())
    # Cache corruption is refused before any optimization.
    (cache / "train.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        run(tmp_path / "invalid")


def test_sampling_schedule_and_native_chunk_horizons():
    import numpy as np
    from realign_vla.upstream import rollout
    class Env:
        def reset(self):
            self.actions = 0
        def set_init_state(self, state):
            return {}
        def step(self, action):
            self.actions += 1
            return {}, 0, False, {}
    class Policy:
        def query(self, obs, instruction):
            return np.zeros((8, 7))
    for horizon in (220, 280):
        env, sampled = Env(), []
        def sample(obs, step):
            if step >= 10 and (step-10) % 4 == 0:
                sampled.append(step)
        result = rollout(env, Policy(), None, "clean task", max_steps=horizon, settling_steps=10, on_step=sample)
        assert result["policy_steps"] == horizon and env.actions == horizon + 10
        assert sampled == list(range(10, 10+horizon+1, 4))
        assert result["success"] is False


def test_corruption_config_and_seeded_raw_image_bank():
    import numpy as np
    from realign_vla.corruptions import BANK, SPEC, random_corruption
    config = load_config(ROOT / "configs/train_realign.yaml")
    assert config["corruptions"] == SPEC
    assert set(BANK) == {"gaussian_blur", "motion_blur", "zoom_blur", "glass_blur", "gaussian_noise"}
    raw = np.random.default_rng(2).integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    a, b = np.random.default_rng(0), np.random.default_rng(0)
    for _ in range(20):
        out_a, label_a, params_a = random_corruption(raw, a)
        out_b, label_b, params_b = random_corruption(raw, b)
        assert np.array_equal(out_a, out_b) and label_a == label_b and params_a == params_b
        assert out_a.shape == raw.shape and out_a.dtype == np.uint8
