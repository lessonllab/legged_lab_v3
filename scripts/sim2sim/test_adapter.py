"""CPU tests for the cross-simulator observation contract."""
import ast
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from run_mujoco import Histories, ROOT, configure_geometry_groups, orientation_features, preprocess_depth, rotation, target_command
from interactive_viewer import depth_rgb, walkable_surface


def test_training_stairs_have_seventeen_12cm_rises_and_30cm_treads():
    scene = ET.parse(ROOT / 'scripts/sim2sim/scenes/training_stairs_12cm.xml').getroot()
    # Check actual raycast surfaces, including the central spawn platform and
    # the final transition to the outer border, independently of the exporter.
    root = ET.Element('mujoco')
    root.append(scene.find('worldbody'))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for level in range(18):
            radius = 0. if level == 0 else 1.2 + (level - .5) * .3
            gid = np.zeros(1, dtype=np.int32)
            distance = mujoco.mj_ray(model, data, np.array([radius * direction[0], radius * direction[1], 3.]),
                                     np.array([0., 0., -1.]), None, 1, -1, gid)
            np.testing.assert_allclose(3. - distance, level * .12, atol=1e-10)


def test_depth_matches_training_implementation():
    source = ROOT / 'source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/instinct_depth.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_instinct_depth')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    generator = torch.Generator().manual_seed(42)
    raw = torch.rand((36, 64), generator=generator) * 4
    raw[20, 20:25] = torch.tensor([float('nan'), float('inf'), -float('inf'), 0., .05])
    expected = namespace['preprocess_instinct_depth'](raw[None, :, :, None])[0]
    torch.testing.assert_close(preprocess_depth(raw), expected, atol=0, rtol=0)


def test_term_major_history_and_sparse_depth_delay():
    for delay in (0, 1):
        histories = Histories(delay)
        for tick in range(45):
            obs = histories.update([np.full(size, 100 * term + tick) for term, size in enumerate([3, 6, 3, 29, 29, 29])], torch.full((18, 32), float(tick)))
            expected_depth = np.maximum(0, tick - np.arange(35, -1, -5) - delay)
            np.testing.assert_array_equal(obs['depth'][0, :, 0, 0].numpy(), expected_depth)
            expected_terms = np.concatenate([np.repeat((100 * term + np.maximum(0, np.arange(tick - 4, tick + 1)))[:, None], size, axis=1).ravel() for term, size in enumerate([3, 6, 3, 29, 29, 29])])
            np.testing.assert_array_equal(obs['policy'][0].numpy(), expected_terms)


def test_orientation_removes_heading_retains_tilt():
    pitch = .3
    tilt = rotation([np.cos(pitch/2), 0, np.sin(pitch/2), 0])
    yaw = rotation([np.cos(.7), 0, 0, np.sin(.7)])
    np.testing.assert_allclose(orientation_features(yaw @ tilt), orientation_features(tilt), atol=1e-14)
    np.testing.assert_allclose(orientation_features(np.eye(3)), [1, 0, 0, 0, 0, 1])


def test_target_turn_stop_and_translation_invariance():
    cfg = {'target_dis_threshold': .4, 'velocity_control_stiffness': 2.,
           'heading_control_stiffness': 2., 'yaw_limit': 1., 'heading_slowdown': True}
    zero = np.zeros(3)
    np.testing.assert_allclose(target_command(zero, np.eye(3), [2, 0], .6, cfg), [.6, 0, 0])
    np.testing.assert_allclose(target_command(zero, np.eye(3), [0, 2], .6, cfg), [0, 0, 1])
    np.testing.assert_allclose(target_command(zero, np.eye(3), [.2, 0], .6, cfg), [0, 0, 0])
    result = target_command(np.array([100., -80., 3.]), np.eye(3), [102, -80], .6, cfg)
    np.testing.assert_allclose(result, [.6, 0, 0])
    yaw90 = rotation([2**-.5, 0, 0, 2**-.5])
    np.testing.assert_allclose(target_command(zero, yaw90, [0, 2], .6, cfg), [.6, 0, 0], atol=1e-14)


def test_head_is_visible_but_not_in_policy_depth_mask():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <asset><mesh name="head_link" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>
      <worldbody><body><geom name="head" type="mesh" mesh="head_link" group="1" contype="0" conaffinity="0"/>
      <geom name="visual" type="sphere" size=".1" group="1" contype="0" conaffinity="0"/>
      <geom name="collision" type="sphere" size=".1"/></body></worldbody></mujoco>''')
    configure_geometry_groups(model)
    assert model.geom('head').group == 3
    assert model.geom('visual').group == 1
    assert model.geom('collision').group == 2
    depth_mask = [1, 1, 0, 0, 0, 0]
    visible_mask = [1, 1, 0, 1, 0, 0]
    assert visible_mask[int(model.geom('head').group[0])]
    assert not depth_mask[int(model.geom('head').group[0])]


def test_depth_preview_handles_invalid_returns_without_mutating_input():
    raw = np.array([[0., 1.], [np.inf, np.nan]])
    original = raw.copy()
    image = depth_rgb(raw, 4, 4)
    assert image.shape == (4, 4, 3) and image.dtype == np.uint8
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(image[0, 2], image[2, 0])
    np.testing.assert_array_equal(image[0, 2], image[2, 2])
    assert not np.array_equal(image[0, 0], image[0, 2])


def test_target_picking_accepts_stair_top_rejects_wall_face():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="10 10 .1"/>
      <geom name="step" type="box" pos="2 0 .2" size=".5 .5 .2"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert walkable_surface(model, data, model.geom('floor').id, np.zeros(3))
    assert walkable_surface(model, data, model.geom('step').id, np.array([2., 0., .4]))
    assert not walkable_surface(model, data, model.geom('step').id, np.array([1.5, 0., .2]))
