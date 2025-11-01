import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import tyro
from torch.utils.tensorboard import SummaryWriter

from whetstone.envs.envs import get_env_infos, make_atari_env, make_dmc_env
from whetstone.utils.utils import get_base_directory, load_config


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
    headless: bool = True
    speed: float = 1.0


def main(args: Args):
    config = load_config(args.config_file)
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
        headless=args.headless,
    )

    timestep = env.unwrapped.control_timestep()

    target_step_duration = timestep / args.speed if args.speed > 0 else 0.0

    obs_shape, discrete_action_bool, action_size = get_env_infos(env)

    while True:
        obs = env.reset()
        done = False
        while not done:
            loop_start_time = time.monotonic()
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if not args.headless and target_step_duration > 0:
                loop_end_time = time.monotonic()
                elapsed_time = loop_end_time - loop_start_time
                delay = target_step_duration - elapsed_time
                if delay > 0:
                    time.sleep(delay)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
