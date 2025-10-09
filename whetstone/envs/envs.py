import gymnasium as gym

from whetstone.envs.wrappers import *

from .dmc_wrappers import DMControlWrapper


def make_dmc_env(
    domain_name,
    task_name,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
    channels_first=True,
    pixel_norm=True,
):
    env_id = f"dmc_{domain_name}_{task_name}_{seed}-v1"
    if from_pixels:
        assert not visualize_reward, "cannot use visualize reward when learning from pixels"
    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip
    if env_id not in gym.registry:
        task_kwargs = {}
        if seed is not None:
            task_kwargs["random"] = seed
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit
        gym.register(
            id=env_id,
            entry_point=DMControlWrapper,
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
            ),
            max_episode_steps=max_episode_steps,
        )
    env = gym.make(env_id)
    if pixel_norm:
        env = PixelNormalization(env)
    return env


def make_atari_env(task_name, skip_frame, width, height, seed, pixel_norm=True):
    env = gym.make(task_name)
    env = gym.wrappers.ResizeObservation(env, (height, width))
    env = ChannelFirstEnv(env)
    env = SkipFrame(env, skip_frame)
    if pixel_norm:
        env = PixelNormalization(env)
    env.seed(seed)
    return env


def get_env_infos(env):
    obs_shape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete_action_bool = True
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discrete_action_bool = False
        action_size = env.action_space.shape[0]
    else:
        raise Exception
    return obs_shape, discrete_action_bool, action_size
