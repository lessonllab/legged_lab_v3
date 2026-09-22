"""Display-only camera geometry in the ROS optical frame (+Z forward)."""
import numpy as np


def view_boundary(intrinsic, bounds, distance):
    """Wire boundary at the optical-depth clip plane; bounds are pixel edges.

    distance_to_image_plane clips camera Z, not Euclidean ray length.
    """
    u0, v0, u1, v1 = bounds
    corners = np.array([[u0,v0], [u1,v0], [u1,v1], [u0,v1]], dtype=float)
    outline = []
    for i in range(4):
        uv = np.linspace(corners[i], corners[(i+1)%4], 9)
        rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(intrinsic).T
        rays *= distance / rays[:, 2:3]
        outline.extend(np.stack((rays[:-1], rays[1:]), axis=1))
        outline.append(np.stack((np.zeros(3), rays[0])))
    return np.asarray(outline, dtype=np.float32)


def enlarged_depth(depth):
    """Nearest-neighbor display enlargement; never modify policy observations."""
    pixels = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    factor = max(1, int(np.ceil(640 / pixels.shape[1])))
    pixels = np.repeat(np.repeat(pixels, factor, axis=0), factor, axis=1)
    return np.repeat(pixels[..., None], 3, axis=-1)
