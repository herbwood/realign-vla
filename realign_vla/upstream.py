"""Thin integration with separately installed, pinned OFT and LIBERO code."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml

REVISIONS = {
    "oft": "e4287e94541f459edc4feabc4e181f537cd569a8",
    "libero": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "libero_plus": "4976dc30028e805ff8094b55501d532c48fec182",
}


def prepare_upstreams(config: dict) -> Path:
    oft, libero = Path(config["oft_root"]).resolve(), Path(config["libero_root"]).resolve()
    for name, root in (("oft", oft), (config["backend"], libero)):
        if not root.is_dir():
            raise FileNotFoundError(f"Missing upstream {name}; follow README installation instructions")
        try:
            revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        except subprocess.CalledProcessError as error:
            raise ValueError("Use the documented pinned upstream git checkouts") from error
        if revision != REVISIONS[name]:
            raise ValueError(f"Unexpected {name} revision: {revision}")
        changed = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
        if changed.strip():
            raise ValueError(f"Tracked upstream code changed: {name}")
    runtime = Path(config["runtime_dir"]).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    benchmark_root = libero / "libero/libero"
    paths = dict(benchmark_root=str(benchmark_root), bddl_files=str(benchmark_root / "bddl_files"),
                 init_states=str(benchmark_root / "init_files"), assets=str(benchmark_root / "assets"),
                 datasets=str(libero / "libero/datasets"))
    private_config = runtime / "libero_config"
    private_config.mkdir(exist_ok=True)
    (private_config / "config.yaml").write_text(yaml.safe_dump(paths))
    os.environ["LIBERO_CONFIG_PATH"] = str(private_config)
    os.environ["HF_MODULES_CACHE"] = str(runtime / "hf_modules")
    sys.path[:0] = [str(oft), str(libero)]
    return runtime


def base_task_name(name: str) -> str:
    return re.sub(r"_(light|table|tb)_\d+$", "", name.split("_view_")[0])


class OFTPolicy:
    """Loads one frozen policy using the upstream image, action and proprio helpers."""
    def __init__(self, upstream: dict, inference: dict, suite: str, device: str):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Real policy extraction/evaluation requires an available CUDA GPU; use smoke_test.py for CPU tests")
        runtime = prepare_upstreams(upstream)
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            from experiments.robot import openvla_utils as ov, robot_utils as robot
            from experiments.robot.libero import libero_utils as lib
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ImportError("Missing OFT/LIBERO dependencies; install requirements-oft.txt and the documented upstream packages") from error
        ov.DEVICE = robot.DEVICE = self.device  # Upstream helpers otherwise select their default device.
        path = Path(upstream["checkpoint"])
        if not path.is_dir():
            path = Path(snapshot_download(upstream["checkpoint"], revision=upstream["checkpoint_revision"],
                                          ignore_patterns=["lora_adapter/*"]))
        # Upstream loading edits config/model Python. Only this private disposable
        # view is writable; original checkpoint blobs and upstream trees stay intact.
        self._view = tempfile.TemporaryDirectory(prefix="checkpoint-", dir=runtime)
        view = Path(self._view.name)
        for source in path.iterdir():
            target = view / source.name
            if source.is_file() and source.suffix in (".json", ".py"):
                shutil.copy2(source, target)
            else:
                target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        self.cfg = SimpleNamespace(
            pretrained_checkpoint=str(view), model_family="openvla", task_suite_name=suite,
            use_l1_regression=True, use_diffusion=False, use_film=False,
            num_images_in_input=inference["num_images_in_input"], use_proprio=inference["use_proprio"],
            center_crop=inference["center_crop"], num_open_loop_steps=inference["action_chunk"],
            lora_rank=32, load_in_8bit=False, load_in_4bit=False, unnorm_key=suite,
            num_diffusion_steps_train=50, num_diffusion_steps_inference=50,
        )
        if self.cfg.num_images_in_input != 2 or self.cfg.num_open_loop_steps != 8 or not self.cfg.use_proprio:
            raise ValueError("This release implements the paper dual-image/proprio/eight-action OFT path")
        previous = Path.cwd()
        try:
            os.chdir(upstream["oft_root"])
            self.vla = ov.get_vla(self.cfg)
            self.proprio = ov.get_proprio_projector(self.cfg, self.vla.llm_dim, proprio_dim=8)
            self.action_head = ov.get_action_head(self.cfg, self.vla.llm_dim)
            self.processor = ov.get_processor(self.cfg)
        finally:
            os.chdir(previous)
        if suite not in self.vla.norm_stats and suite + "_no_noops" in self.vla.norm_stats:
            self.cfg.unnorm_key = suite + "_no_noops"
        if self.cfg.unnorm_key not in self.vla.norm_stats:
            raise ValueError("Checkpoint has no normalization statistics for the selected suite")
        for component in (self.vla, self.action_head, self.proprio):
            component.eval().requires_grad_(False)
        self.ov, self.robot, self.lib = ov, robot, lib

    @torch.inference_mode()
    def query(self, obs: dict, instruction: str) -> np.ndarray:
        state = np.concatenate([obs["robot0_eef_pos"], self.lib.quat2axisangle(obs["robot0_eef_quat"].copy()),
                                obs["robot0_gripper_qpos"]])
        prepared = dict(full_image=self.ov.resize_image_for_policy(self.lib.get_libero_image(obs), 224),
                        wrist_image=self.ov.resize_image_for_policy(self.lib.get_libero_wrist_image(obs), 224), state=state)
        actions = self.ov.get_vla_action(self.cfg, self.vla, self.processor, prepared, instruction,
                                         action_head=self.action_head, proprio_projector=self.proprio)
        actions = np.stack([self.robot.invert_gripper_action(self.robot.normalize_gripper_action(a, binarize=True)) for a in actions])
        if actions.shape != (8, 7) or not np.isfinite(actions).all():
            raise ValueError("Native OFT produced invalid actions")
        return actions

    @torch.inference_mode()
    def features(self, raw: np.ndarray) -> torch.Tensor:
        image = self.ov.prepare_images_for_vla([self.ov.resize_image_for_policy(raw[::-1, ::-1], 224)], self.cfg)[0]
        backbone = self.vla.vision_backbone
        backbone.set_num_images_in_input(1)
        try:
            pixels = self.processor("In: What action should the robot take to x?\nOut:", image)["pixel_values"]
            feature = backbone(pixels.to(self.device, torch.bfloat16))[0].half().cpu()
            if feature.shape != (256, 2176):
                raise ValueError(f"Unexpected fused visual features: {tuple(feature.shape)}")
            return feature
        finally:
            backbone.set_num_images_in_input(self.cfg.num_images_in_input)


def make_task(suite, suite_name: str, variant: dict):
    from libero.libero import get_libero_path
    if variant.get("task_idx") is not None:
        task = suite.get_task(variant["task_idx"])
        if task.name != variant["name"]:
            raise ValueError("Task index/name mismatch: use the pinned upstream and exact manifest")
        return task, suite.get_task_init_states(variant["task_idx"])
    task = SimpleNamespace(language=variant.get("plus_language", variant["instruction"]),
                           problem_folder=suite_name, bddl_file=variant["name"] + ".bddl")
    path = Path(get_libero_path("init_states")) / suite_name / (variant["base_name"] + ".pruned_init")
    # Upstream init-state files may contain NumPy arrays; use only trusted upstream assets.
    return task, torch.load(path, map_location="cpu", weights_only=False)


def rollout(env, policy: OFTPolicy, initial_state, instruction: str, *, max_steps: int, settling_steps: int, on_step=None) -> dict:
    env.reset()
    obs = env.set_init_state(initial_state)
    for step in range(settling_steps):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        if on_step is not None:
            on_step(obs, step + 1)
    steps = queries = 0
    with torch.inference_mode():
        while steps < max_steps:
            actions = policy.query(obs, instruction)
            queries += 1
            for action in actions:
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
                if on_step is not None:
                    on_step(obs, settling_steps + steps)
                if not isinstance(done, (bool, np.bool_)):
                    raise ValueError("Environment success must be boolean")
                if done or steps >= max_steps:
                    return dict(success=bool(done), policy_steps=steps, n_queries=queries)
    raise RuntimeError("Episode did not execute any policy action")
