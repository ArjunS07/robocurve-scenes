"""Conversions for Isaac Lab 3 state and recorder callbacks (no simulator imports)."""

# Component order of root_quat_w and write_root_pose_to_sim in the installed runtime.
# Isaac Lab 3 documents xyzw; Isaac Lab 2.x used wxyz. Not yet confirmed on the
# server. If the runtime probe shows wxyz, change only this value.
SIM_QUAT_ORDER = "xyzw"


def as_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def env_indices(env_ids, num_envs):
    """Recorder callbacks may supply lists, tensors, slices, or None for all environments."""
    if env_ids is None or isinstance(env_ids, slice):
        return list(range(num_envs))[env_ids or slice(None)]
    return [int(i) for i in as_list(env_ids)]


def quaternion_xyzw(value, order=None):
    """Read a simulator quaternion into the scorer's xyzw order."""
    values = [float(v) for v in as_list(value)]
    if len(values) != 4:
        raise ValueError("Expected a quaternion with four components")
    if (order or SIM_QUAT_ORDER) == "wxyz":
        return (values[1], values[2], values[3], values[0])
    return tuple(values)


def sim_quaternion(xyzw, order=None):
    """Convert an xyzw quaternion into the order the simulator writes."""
    x, y, z, w = (float(v) for v in xyzw)
    if (order or SIM_QUAT_ORDER) == "wxyz":
        return (w, x, y, z)
    return (x, y, z, w)
