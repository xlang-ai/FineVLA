import dataclasses
import json
import logging
import os
import argparse
import importlib

@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5678
    resize_size = [224, 224]

    env_name: str = "robocasa/PickPlaceCounterToCabinet"
    n_episodes: int = 50
    n_envs: int = 1
    max_episode_steps: int = 600
    n_action_steps: int = 8

    video_out_path: str = "results/robocasa365/videos/debug"
    seed: int = 7

    pretrained_path: str = "results/Checkpoints/robocasa365_qwenoft/checkpoints/steps_10000_pytorch_model.pt"
    unnorm_key: str = "new_embodiment"


def _load_dependencies():
    run_eval_mod = importlib.import_module("examples.Robocasa_tabletop.eval_files.simulation_env")
    iface_mod = importlib.import_module("examples.Robocasa365.eval_files.model2robocasa365_interface")
    return run_eval_mod.run_evaluation, iface_mod.PolicyWarper365


def eval_robocasa365(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    if os.getenv("DEBUG", False):
        start_debugpy_once()

    run_evaluation, PolicyWarper365 = _load_dependencies()

    model = PolicyWarper365(
        policy_ckpt_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
    )

    run_evaluation(
        env_name=args.env_name,
        model=model,
        video_dir=args.video_out_path,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--args.host", dest="host", type=str, default=Args.host)
    parser.add_argument("--args.port", dest="port", type=int, default=Args.port)
    parser.add_argument("--args.env-name", dest="env_name", type=str, default=Args.env_name)
    parser.add_argument("--args.n-episodes", dest="n_episodes", type=int, default=Args.n_episodes)
    parser.add_argument("--args.n-envs", dest="n_envs", type=int, default=Args.n_envs)
    parser.add_argument(
        "--args.max-episode-steps", dest="max_episode_steps", type=int, default=Args.max_episode_steps
    )
    parser.add_argument("--args.n-action-steps", dest="n_action_steps", type=int, default=Args.n_action_steps)
    parser.add_argument("--args.video-out-path", dest="video_out_path", type=str, default=Args.video_out_path)
    parser.add_argument("--args.seed", dest="seed", type=int, default=Args.seed)
    parser.add_argument("--args.pretrained-path", dest="pretrained_path", type=str, default=Args.pretrained_path)
    parser.add_argument("--args.unnorm-key", dest="unnorm_key", type=str, default=Args.unnorm_key)
    parser.add_argument("--args.resize-size", dest="resize_size", nargs=2, type=int, default=Args.resize_size)

    cli_args = parser.parse_args()
    args = Args(
        host=cli_args.host,
        port=cli_args.port,
        resize_size=cli_args.resize_size,
        env_name=cli_args.env_name,
        n_episodes=cli_args.n_episodes,
        n_envs=cli_args.n_envs,
        max_episode_steps=cli_args.max_episode_steps,
        n_action_steps=cli_args.n_action_steps,
        video_out_path=cli_args.video_out_path,
        seed=cli_args.seed,
        pretrained_path=cli_args.pretrained_path,
        unnorm_key=cli_args.unnorm_key,
    )
    eval_robocasa365(args)
