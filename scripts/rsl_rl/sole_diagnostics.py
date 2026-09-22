"""Playback explanations of sole ray samples, not physical contact labels."""
import numpy as np


def sole_diagnostic(heights, force_z):
    h = np.asarray(heights).reshape(5, 11)
    valid = np.isfinite(h)
    supported = valid & (np.abs(h + .035) <= .025)
    fraction = float(supported[1:-1, 1:-1].mean())

    def jump(values):
        finite = np.isfinite(values)
        with np.errstate(invalid='ignore'):
            dx = (np.abs(np.diff(values, axis=1)) > .05) & finite[:, 1:] & finite[:, :-1]
            dy = (np.abs(np.diff(values, axis=0)) > .05) & finite[1:, :] & finite[:-1, :]
        return bool(dx.any() or dy.any())

    near = jump(h)
    inside = jump(h[1:-1, 1:-1])
    missing = int((~valid).sum())
    if not np.isfinite(force_z):
        status = '⚪ 接触力无效，无法判断'
    elif force_z <= 20.:
        status = '⚪ 离地或轻触，不判踩边'
    elif not valid[1:-1, 1:-1].all():
        status = '⚪ 脚底采样缺失，无法确认踩边'
    elif inside and fraction < .9:
        status = '🔴 疑似跨边承重（脚底内部有高度跳变）'
    elif inside:
        status = '🟠 脚底内部检测到高度跳变，支撑采样充分'
    elif near:
        status = '🟡 脚外周圈临边，不等于踩边'
    elif missing:
        status = '⚪ 外圈采样缺失，未确认边缘'
    elif fraction < .9:
        status = '🟡 支撑采样不足，未检出跨边'
    else:
        status = '🟢 未检出跨边'
    return status, fraction, missing
