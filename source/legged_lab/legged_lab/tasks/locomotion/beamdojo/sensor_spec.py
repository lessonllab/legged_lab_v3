"""Verified MID-360 references; these constants do not create a LiDAR sensor.

Paper: https://arxiv.org/html/2502.10363v3 (IV-D, IV-E)
Specifications: https://www.livoxtech.com/cn/mid-360/specs
Mount source: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof.urdf
Local mount was read via pxr from g1_29dof_rev_1_0.usd on 2026-10-02.
"""
MID360_SPEC = {
    'horizontal_fov_deg': 360., 'vertical_fov_deg': (-7., 52.),
    'typical_frame_rate_hz': 10., 'points_per_second': 200_000,
    'blind_zone_m': .1, 'range_m_at_10_percent_reflectivity': 40.,
    'range_m_at_80_percent_reflectivity': 70.,
    'mount_parent': 'torso_link', 'mount_frame': 'mid360_link',
    'local_usd_mount_xyz_m': (.0002835, .00003, .41618),
    'official_urdf_mount_xyz_m': (.0002835, .00003, .40618),
    'mount_rpy_rad': (0., .04014257279586953, 0.),
    # Mount frame is NOT established to equal the driver scan frame.
    # Do not silently assume identity or a pi roll; require calibration/validation.
    'mount_to_scan_transform': None,
    'scan_pattern': 'non-repetitive; exact simulation pattern not verified',
}
BEAMDOJO_MAP_SPEC = {'size': 15, 'resolution_m': .1, 'publication_rate_hz': 10.}
