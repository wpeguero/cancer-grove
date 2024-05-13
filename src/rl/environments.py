"""Environments Module.

Contains all of the custom environments for building machine learning
models.
"""

from collections import defaultdict
from typing import Optional

import torch
from torch import nn

import matplotlib.pyplot as plt

import numpy as np

from torchrl.envs import CatTensors, EnvBase, Transform, TransformedEnv, UnsqueezeTransform
from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

DEFAULT_X = np.pi
DEFAULT_Y = 1.0
bs = 32 #batch size

#TODO: Add transforms so that the observations can be picked up
def _main():
    """Test classes."""
    env = Pendulum()
    net = nn.Sequential(
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(1)
            )
    policy = TensorDictModule(
            net,
            in_keys=["observation"],
            out_keys=["action"]
            )

    opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
    steps = range(20_000 // bs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20_000)
    logs = defaultdict(list)

    for _ in steps:
        init_td = env.reset(env.gen_params(batch_size=[bs]))
        print("init_td\n{}".format(init_td))
        print("\npolicy:\n{}".format(policy))
        rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        logs['return'].append(traj_return.item())
        logs['last_reward'].append(rollout[..., -1]['next', 'reward'].mean().item())
        scheduler.step()
    plot_train_data(logs)


def plot_train_data(logs):
    with plt.ion():
        plt.figure(figsize(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(logs['return'])
        plt.title('returns')
        plt.xlabel('iteration')
        plt.subplot(1, 2, 2)
        plt.plot(logs['last_reward'])
        plt.title('last reward')
        plt.xlabel('iteration')
        plt.show()

def simple_rollout(env:EnvBase, steps:int=100):
    #Preallocate:
    data = TensorDict({}, [steps])
    #reset
    _data = env.reset()
    for i in range(steps):
        _data["action"] = env.action_spec.rand()
        _data = env.step(_data)
        data[i] = _data
        _data = step_mdp(_data, keep_other=True)
    return data

def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

def make_composite_from_td(td:TensorDict):
    """
    Custom function to convert a `tensordict` in a similar spec
    structure of unbounded values.
    """
    composite = CompositeSpec(
            {
                key:make_composite_from_td(tensor) if isinstance(tensor, TensorDictBase) else UnboundedContinuousTensorSpec(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape) for key, tensor in td.items()
                },
            shape=td.shape
            )
    return composite


class Pendulum(EnvBase):
    metadata = {
            'render_modes': ['human', 'rgb_array'],
            'render_fps':30
            }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device='cpu'):
        if td_params is None:
            td_params = self.gen_params()
        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    @staticmethod
    def _step(tdict:TensorDict):
        theta, theta_dot = tdict["theta"], tdict["theta_dot"]

        # Set parameters
        gravity = tdict['params', 'g']
        mass = tdict['params', 'm']
        length = tdict['params', 'l'] # Length of bar
        dt = tdict['params', 'dt'] # Small changes in time

        u = tdict['action'].squeeze(-1)
        u = u.clamp(-tdict['params', 'max_torque'], tdict['params', 'max_torque'])

        costs = angle_normalize(theta) ** 2 + 0.1 * theta_dot ** 2 + 0.001 * (u**2)

        new_theta_dot = (theta_dot + (3 * gravity / (2 * length) * theta.sin() + 3.0 / (mass * length ** 2) * u) * dt)
        new_theta_dot = new_theta_dot.clamp(-tdict['params', 'max_speed'], tdict['params', 'max_speed'])
        new_theta = theta + new_theta_dot * dt
        reward = -costs.view(*tdict.shape, 1)
        done = torch.zeros_like(reward, dtype=torch.bool)
        out = TensorDict(
                {
                    'theta':new_theta,
                    'theta_dot':new_theta_dot,
                    'params':tdict['params'],
                    'reward':reward,
                    'done':done
                    },
                tdict.shape,
                )
        return out

    def _reset(self, tdict:TensorDict):
        if tdict is None or tdict.is_empty():
            # Generate a single set of hyperparameters
            # otherwise, we assume that the input tdict
            # contains all the relevant parameters to get
            # started.
            tdict = self.gen_params(batch_size=self.batch_size)

        high_theta = torch.tensor(DEFAULT_X, device=self.device)
        high_theta_dot = torch.tensor(DEFAULT_Y, device=self.device)

        low_theta = -high_theta
        low_theta_dot = -high_theta_dot

        theta = (torch.rand(tdict.shape, generator=self.rng, device=self.device) * (high_theta - low_theta) + low_theta_dot)
        theta_dot = (torch.rand(tdict.shape, generator=self.rng, device=self.device) * (high_theta_dot - low_theta_dot) + low_theta)
        out = TensorDict(
                {
                    'theta':theta,
                    'theta_dot':theta_dot,
                    'params': tdict['params']
                    },
                batch_size=tdict.shape,
                )
        return out

    def _make_spec(self, td_params):
        """Populates the self.output_spec['observation']."""
        self.observation_spec = CompositeSpec(
                theta = BoundedTensorSpec(
                    low = -torch.pi,
                    high = torch.pi,
                    shape = (),
                    dtype = torch.float32,
                    ),
                theta_dot = BoundedTensorSpec(
                    low = -td_params['params', 'max_speed'],
                    high = td_params['params', 'max_speed'],
                    shape=(),
                    dtype=torch.float32
                    ),
                params = make_composite_from_td(td_params['params']),
                shape=()
                )

        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
                low=-td_params['params', 'max_torque'],
                high=td_params['params', 'max_torque'],
                shape=(1,),
                dtype=torch.float32
                )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))
        return self

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
        return self

    @staticmethod
    def gen_params(g=9.71, batch_size=None) -> TensorDictBase:
        """Returns a `tensordict` containing the physical parameters such as gravitational force and torch or speed limits."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
                {
                    'params':TensorDict(
                        {
                            'max_speed': 8.0,
                            'max_torque':2.0,
                            'dt':0.05,
                            'g':g,
                            'm':1.0,
                            'l':1.0,
                            },
                        []
                        )
                    },
                []
                )
        if batch_size is not None:
            td = td.expand(batch_size).contiguous()
            return td


class HangmanEnvironment(EnvBase):
    """Pytorch Implementation of the Hangman Environment.

    This is an implementation based on the TensorFlow Hangman
    Environment made for the job application.

    Arguments
    ---------
    words : list
        Contains a list of words for playing hangman.
    """

    def __init___(self, words:list):
        """Init the class."""
        super().__init__(device=device, batch_size=[])

    def _step(self, td:TensorDictBase):
        """Compute the Next Step for the actor.

        Parameters
        ----------
        td: TensorDictBase
            Contains the data required for calculating the next step.
        """
        pass


if __name__ == "__main__":
    _main()
