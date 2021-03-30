import os
import sys
from gym.envs.registration import register


register(
    id='detection-v0',
    entry_point='deepspace.environments.xray.detection:Detection',
    max_episode_steps=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)

path = os.path.dirname(os.path.abspath(__file__))

for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for cls in classes:
        setattr(sys.modules[__name__], cls.__name__, cls)
