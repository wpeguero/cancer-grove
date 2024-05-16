"""Set of tests against reinforcement learning sub-library."""
import pytest
import src.rl.environments as envs
from torchrl.envs.utils import check_env_specs

def test_pendulum_environment():
    """Test whether the environment Outputs Correctly"""
    env = envs.Pendulum()
    check_env_specs(env)
