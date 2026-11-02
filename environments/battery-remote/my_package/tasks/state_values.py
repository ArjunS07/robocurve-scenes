"""Conversions for Isaac Lab 3 state and recorder callbacks (no simulator imports)."""


def as_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def env_indices(env_ids, num_envs):
    """Recorder callbacks may supply lists, tensors, or None for all environments."""
    return list(range(num_envs)) if env_ids is None else [int(i) for i in as_list(env_ids)]


def quaternion_xyzw(value):
    """Isaac Lab 3 and Arena Pose both use xyzw; do not reorder the components."""
    values = as_list(value)
    if len(values) != 4:
        raise ValueError("Expected a quaternion with four components")
    return tuple(float(v) for v in values)
