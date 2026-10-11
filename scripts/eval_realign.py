"""LIBERO example evaluation with frozen OFT, optionally plus the frozen adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from realign_vla.intervention import attach_realign_hook, remove_realign_hook
from realign_vla.upstream import OFTPolicy, base_task_name, make_task, rollout
from realign_vla.utils import load_adapter, load_config, seed_everywhere, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_libero.yaml")
    parser.add_argument("--method", choices=["frozen", "realign"], default="realign")
    parser.add_argument("--adapter", help="Adapter-only checkpoint, required for --method realign")
    parser.add_argument("--manifest", help="Optional user-supplied JSON with variants; no paper manifests are bundled")
    parser.add_argument("--output", default="runs/evaluation")
    parser.add_argument("--device", default="cuda", help="Available CUDA device chosen by the user")
    args = parser.parse_args()
    if args.method == "realign" and not args.adapter:
        parser.error("--adapter is required for ReAlign evaluation")
    config = load_config(args.config)
    evaluation = config["evaluation"]
    suite_name = evaluation["suite"]
    if suite_name not in ("libero_spatial", "libero_object"):
        raise ValueError("This example supports Spatial and Object")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(config=config, method=args.method,
                      adapter_sha256=sha256(args.adapter) if args.adapter else None,
                      manifest_sha256=sha256(args.manifest) if args.manifest else None)
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    config_file = output / "run_config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != provenance:
        raise ValueError("Output contains a different evaluation; use a separate output directory")
    write_json(provenance, config_file)
    journal = output / "episodes.jsonl"
    done = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row["fingerprint"] != fingerprint:
                raise ValueError("Result provenance differs")
            if row["status"] == "valid":
                done.add((row["name"], row["init_state"]))
    seed_everywhere(evaluation["seed"])
    policy = OFTPolicy(config["upstream"], evaluation, suite_name, args.device)
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[suite_name]()
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        if manifest.get("suite", suite_name) != suite_name:
            raise ValueError("Manifest suite differs from config")
        selected = manifest["variants"]
    else:
        selected = []
        for task_index in evaluation["task_indices"]:
            task = suite.get_task(task_index)
            base = base_task_name(task.name)
            selected.extend(dict(task_idx=task_index, name=task.name, base_name=base,
                                 instruction=base.replace("_", " "), init_state=init)
                            for init in evaluation["init_states"])
    hook = attach_realign_hook(policy.vla, load_adapter(args.adapter, args.device)) if args.method == "realign" else None
    try:
        for variant in selected:
            init = variant.get("init_state", 0)
            if (variant["name"], init) in done:
                continue
            seed_everywhere(evaluation["seed"])
            task, states = make_task(suite, suite_name, variant)
            if not evaluation["use_clean_instruction"] and variant.get("task_idx") is None and "plus_language" not in variant:
                raise ValueError("Virtual task needs explicit plus_language when use_clean_instruction=false")
            instruction = variant["instruction"] if evaluation["use_clean_instruction"] else task.language
            env, _ = policy.lib.get_libero_env(task, "openvla", resolution=evaluation["env_image_resolution"])
            row = dict(name=variant["name"], init_state=init, instruction=instruction, fingerprint=fingerprint,
                       use_clean_instruction=evaluation["use_clean_instruction"])
            try:
                result = rollout(env, policy, states[init], instruction,
                                 max_steps=280 if suite_name == "libero_object" else 220,
                                 settling_steps=evaluation["settling_steps"])
                row.update(status="valid", **result)
            except Exception as error:
                row.update(status="infrastructure_error", success=None, error=f"{type(error).__name__}: {error}")
                raise
            finally:
                env.close()
                with journal.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
                print(json.dumps(row), flush=True)
            done.add((variant["name"], init))
    finally:
        if hook is not None:
            remove_realign_hook(hook)


if __name__ == "__main__":
    main()
