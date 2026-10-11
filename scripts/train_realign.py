"""Optimize only ReAlign parameters from an immutable frozen-feature cache."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from realign_vla.data import load_feature_cache
from realign_vla.losses import realign_loss, relative_error, token_cosine
from realign_vla.model import TokenwiseLowRankAdapter
from realign_vla.utils import atomic_save, load_config, model_kwargs, seed_everywhere, warmup_cosine, write_json


@torch.no_grad()
def validate(model, corrupted, clean, source, pair_indices, clean_indices):
    model.eval()
    cosine, reconstruction, identity = [], [], []
    for ids in pair_indices.split(32):
        prediction = model(corrupted[ids].to(torch.bfloat16)).float()
        target = clean[source[ids]].float()
        cosine.append(token_cosine(prediction, target))
        reconstruction.append(relative_error(prediction, target))
    for ids in clean_indices.split(32):
        target = clean[ids].float()
        identity.append(relative_error(model(target.to(torch.bfloat16)).float(), target))
    model.train()
    mean = lambda parts: float(np.mean(torch.cat(parts).cpu().tolist()))
    return dict(token_cosine=mean(cosine), relative_error=mean(reconstruction),
                clean_input_distortion=mean(identity))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_realign.yaml")
    parser.add_argument("--features", default="data/feature_cache")
    parser.add_argument("--output", default="runs/realign")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", help="Our last.pt checkpoint; config and cache must match")
    parser.add_argument("--steps", type=int, help="Override total steps for a small pipeline check")
    args = parser.parse_args()
    config = load_config(args.config)
    opt_cfg = config["optimization"]
    if args.steps is not None:
        opt_cfg["total_steps"] = args.steps
    if opt_cfg["total_steps"] < 1 or opt_cfg["warmup_steps"] < 1:
        raise ValueError("Total steps and warmup must be positive")
    output = Path(args.output)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Existing training run: pass --resume or choose another output")
    output.mkdir(parents=True, exist_ok=True)
    seed_everywhere(opt_cfg["seed"])
    cache = load_feature_cache(args.features)
    device = torch.device(args.device)
    clean, corrupted = cache["clean"].to(device), cache["corrupted"].to(device)
    source, is_train = cache["source"].to(device), cache["is_train"].to(device)
    train_pairs, val_pairs = torch.where(is_train[source])[0], torch.where(~is_train[source])[0]
    train_clean, val_clean = torch.where(is_train)[0], torch.where(~is_train)[0]
    if any(len(ids) == 0 for ids in (train_pairs, val_pairs, train_clean, val_clean)):
        raise ValueError("Need nonempty, disjoint training and validation trajectories")
    model = TokenwiseLowRankAdapter(**model_kwargs(config["model"])).to(device)
    if clean.shape[1:] != (model.num_tokens, model.dim):
        raise ValueError("Cache tensor dimensions differ from the adapter configuration")
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: warmup_cosine(
        step, warmup_steps=opt_cfg["warmup_steps"], total_steps=opt_cfg["total_steps"]))
    generator = torch.Generator(device=device).manual_seed(opt_cfg["seed"])
    start, best_score, best_state, selected_step, curve = 0, -1.0, None, None, []
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)
        if previous["training_config"] != config or previous["cache_sha256"] != cache["index_sha256"]:
            raise ValueError("Resume config or feature cache differs")
        if previous["device_type"] != device.type:
            raise ValueError("Resume on the same CPU/CUDA generator type")
        model.load_state_dict(previous["state_dict"])
        optimizer.load_state_dict(previous["optimizer"])
        schedule.load_state_dict(previous["schedule"])
        generator.set_state(previous["sampler_state"])
        start, best_score = previous["next_step"], previous["selected_val_tc"]
        selected_step, best_state, curve = previous["selected_step"], previous["best_state_dict"], previous["curve"]
    for step in range(start, opt_cfg["total_steps"]):
        ids = train_pairs[torch.randint(len(train_pairs), (opt_cfg["corrupted_batch_size"],), device=device, generator=generator)]
        ci = train_clean[torch.randint(len(train_clean), (opt_cfg["identity_batch_size"],), device=device, generator=generator)]
        features = torch.cat([corrupted[ids].float(), clean[ci].float()])
        target = torch.cat([clean[source[ids]].float(), clean[ci].float()])
        residual = model.residual(features)
        loss, terms = realign_loss(features, target, residual, **config["loss"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite adapter training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["grad_clip"])
        optimizer.step(); schedule.step()
        if step % opt_cfg["validate_every"] == 0 or step == opt_cfg["total_steps"] - 1:
            metrics = validate(model, corrupted, clean, source, val_pairs, val_clean)
            row = dict(step=step, loss=loss.item(), terms={k:v.item() for k,v in terms.items()}, validation=metrics)
            curve.append(row)
            if metrics["token_cosine"] > best_score:
                best_score, selected_step = metrics["token_cosine"], step
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                atomic_save(dict(model_type="tokenwise_lowrank", config=config["model"], state_dict=best_state,
                                 selected_step=selected_step, selected_val_tc=best_score, seed=opt_cfg["seed"],
                                 cache_sha256=cache["index_sha256"]), output / "best.pt")
            atomic_save(dict(model_type="tokenwise_lowrank", config=config["model"], state_dict=model.state_dict(),
                             optimizer=optimizer.state_dict(), schedule=schedule.state_dict(), sampler_state=generator.get_state(),
                             next_step=step+1, selected_step=selected_step, selected_val_tc=best_score,
                             best_state_dict=best_state, curve=curve, training_config=config,
                             device_type=device.type, cache_sha256=cache["index_sha256"]), output / "last.pt")
            write_json(dict(selected_step=selected_step, selected_val_tc=best_score, curve=curve,
                            cache_sha256=cache["index_sha256"]), output / "metrics.json")
            write_json(config, output / "config.json")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
