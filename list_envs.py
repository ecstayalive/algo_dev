import cv2
import numpy as np
from dm_control import suite

max_len = max(len(d) for d, _ in suite.BENCHMARKING)
for domain, task in suite.BENCHMARKING:
    print(f"{domain:<{max_len}}  {task}")
random_state = np.random.RandomState(42)
env = suite.load("humanoid", "walk")
duration = 5
spec = env.action_spec()
time_step = env.reset()
while env.physics.data.time < duration:
    action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
    time_step = env.step(action)
    camera0 = env.physics.render(camera_id=0, height=200, width=200)
    camera1 = env.physics.render(camera_id=1, height=200, width=200)
    cv2.imshow("frame", cv2.cvtColor(camera0, cv2.COLOR_RGB2BGR))
    cv2.waitKey(int(env.control_timestep() * 1000))
