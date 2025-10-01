import os

os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import tyro
from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.alivev0origin import AliveV0Origin
from dreamer.algorithms.alivev0space_attention import AliveV0SpaceAttention
from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.envs.envs import get_env_infos, make_atari_env, make_dmc_env
from dreamer.utils.utils import get_base_directory, load_config


@dataclass
class Args:
    config_file: str = "dmc-walker-walk.yml"
    algorithm: Literal[
        "dreamer-v1",
        "plan2explore",
        "alive-v0-origin",
        "alive-v0-pc",
        "alive-v0-ensemble",
        "alive-v0-space_attention",
    ] = "alive-v0-origin"
    disable_logger: bool = False
    run_name: str = ""


def main(args: Args):
    config = load_config(args.config_file)
    if config.environment.benchmark == "atari":
        env = make_atari_env(
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            height=config.environment.height,
            width=config.environment.width,
            skip_frame=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    elif config.environment.benchmark == "dmc":
        env = make_dmc_env(
            domain_name=config.environment.domain_name,
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            visualize_reward=config.environment.visualize_reward,
            from_pixels=config.environment.from_pixels,
            height=config.environment.height,
            width=config.environment.width,
            frame_skip=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    else:
        raise ValueError(f"Unknown benchmark: {config.environment.benchmark}")
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)

    config.algorithm = args.algorithm
    log_dir = (
        get_base_directory()
        + "/runs/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.algorithm
        + "_"
        + config.operation.log_dir
        + f"-{args.run_name}"
    )
    writer = None if args.disable_logger else SummaryWriter(log_dir)
    device = config.operation.device

    match config.algorithm:
        case "dreamer-v1":
            print("Training DreamerV1")
            agent = Dreamer(obs_shape, discrete_action_bool, action_size, writer, device, config)
        case "plan2explore":
            print("Training Plan2Explore")
            agent = Plan2Explore(
                obs_shape, discrete_action_bool, action_size, writer, device, config
            )
        case "alive-v0-origin":
            print("Training AliveV0Origin")
            agent = AliveV0Origin(
                obs_shape, discrete_action_bool, action_size, writer, device, config
            )
        case "alive-v0-space_attention":
            print("Training AliveV0SpaceAttention")
            agent = AliveV0SpaceAttention(
                obs_shape, discrete_action_bool, action_size, writer, device, config
            )
        case "alive-v0-pc":
            print("Training AliveV0PC")
            from dreamer.algorithms.alivev0pc import AliveV0Pc

            agent = AliveV0Pc(obs_shape, discrete_action_bool, action_size, writer, device, config)
        case "alive-v0-ensemble":
            ...
        case _:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

    agent.train(env)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
