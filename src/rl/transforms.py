"""Set of custom transformations for Reinforcement Learning Using Pytorch."""
from torchrl.envs import CatTensors, EnvBase, Transform, TransformedEnv, UnsqueezeTransform
import torch
from tensordict import TensorDictBase, BoundedTensorSpec


class SineTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.sin()

    def _reset(self, tensordict:TensorDictBase, tensordict_reset:TensorDictBase) -> TensorDictBase:
        """The transform must also modify the data at reset time."""
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        """Decorator will execute the observation spec transform
        accross all in_keys/out_keys pairs and write the result in
        the observation_spec which is of type `Composite`.
        """
        return BoundedTensorSpec(
                low=-1,
                high=1,
                shape=observation_spec.shape,
                device=observation_spec.device,
                dtype=observation_spec.dtype,
                )


class CosineTransform(Transform):
    def _apply_transform(self, obs:torch.Tensor) -> None:
        return obs.cos()

    def _reset(self, tensordict:TensorDictBase, tensordict_reset:TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
                low=-1,
                high=1,
                shape=observation_spec.shape,
                device=observation_spec.device,
                dtype=observation_spec.dtype
                )
