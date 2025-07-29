import rl_env_cl as rl_cl
import run as r
import config as c 

if c.RL_ALG == 'ppo':
    r.train()
elif c.RL_ALG == 'random':
    r.create_random_baseline()

