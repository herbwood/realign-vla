"""Collect successful frozen clean Spatial trajectories and paired synthetic features."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from realign_vla.corruptions import SPEC, random_corruption
from realign_vla.upstream import OFTPolicy, make_task, rollout
from realign_vla.utils import atomic_save, load_config, seed_everywhere, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_realign.yaml")
    parser.add_argument("--output", default="data/feature_cache")
    parser.add_argument("--device", default="cuda", help="Available CUDA device chosen by the user")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["corruptions"] != SPEC:
        raise ValueError("Config differs from the fixed audited corruption bank; update code/config explicitly")
    data, inference = config["data"], config["inference"]
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Feature output must be empty; extraction never overwrites an existing cache")
    output.mkdir(parents=True, exist_ok=True)
    if set(data["train_init_states"]) & set(data["val_init_states"]):
        raise ValueError("Training/validation init states must be disjoint")
    if not data["successful_trajectories_only"]:
        raise ValueError("Final training uses successful clean trajectories only")
    seed_everywhere(inference["seed"])
    policy = OFTPolicy(config["upstream"], inference, data["suite"], args.device)
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[data["suite"]]()
    rng = np.random.default_rng(data["corruption_seed"])
    index = dict(format_version=1, shards=[], rollouts=[], config=config)
    for split, init_ids in (("train", data["train_init_states"]), ("val", data["val_init_states"])):
        for task_index, base in enumerate(data["task_names"]):
            name = base + ("_view_0_0_100_0_0_initstate_0" if config["upstream"]["backend"] == "libero_plus" else "")
            variant = dict(name=name, base_name=base, instruction=base.replace("_", " "), task_idx=None)
            task, states = make_task(suite, data["suite"], variant)
            for init in init_ids:
                seed_everywhere(inference["seed"])
                env, _ = policy.lib.get_libero_env(task, "openvla", resolution=inference["env_image_resolution"])
                frames = []
                def capture(obs, step):
                    wait = inference["settling_steps"]
                    if step >= wait and (step-wait) % data["sample_every_env_steps"] == 0:
                        frames.append(obs["agentview_image"].copy())
                try:
                    result = rollout(env, policy, states[init], variant["instruction"], max_steps=220,
                                     settling_steps=inference["settling_steps"], on_step=capture)
                finally:
                    env.close()
                row = dict(split=split, task=task_index, init_state=init, frames=len(frames), **result)
                index["rollouts"].append(row)
                print(json.dumps(row), flush=True)
                if not result["success"]:
                    continue
                clean, corrupted, source, metadata = [], [], [], []
                for frame_index, raw in enumerate(frames):
                    clean.append(policy.features(raw))
                    for _ in range(data["corruptions_per_clean_frame"]):
                        image, family, parameters = random_corruption(raw, rng)
                        corrupted.append(policy.features(image)); source.append(frame_index)
                        metadata.append(dict(frame=frame_index, family=family, parameters=parameters))
                shard = dict(clean=torch.stack(clean), corrupted=torch.stack(corrupted), source=torch.tensor(source, dtype=torch.long),
                             metadata=json.loads(json.dumps(metadata)))
                filename = f"{split}_task{task_index:02d}_init{init:02d}.pt"
                atomic_save(shard, output / filename)
                index["shards"].append(dict(file=filename, split=split, sha256=sha256(output / filename), clean_frames=len(clean)))
                write_json(index, output / "index.json")
    write_json(index, output / "index.json")
    if not index["shards"]:
        raise RuntimeError("No successful clean trajectories; inspect assets and native policy integration")


if __name__ == "__main__":
    main()
