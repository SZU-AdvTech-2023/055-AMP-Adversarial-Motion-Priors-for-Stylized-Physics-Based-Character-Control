# @author: zrk
# @desc:   utils for motion upsampling, calc pivot rotation along y-axis , foot contact, etc.

import torch
import torch.nn.functional as F
import fmbvh.motion_tensor as mot


def sample_frames(motion: torch.Tensor, scale_factor=None, target_frame=None, sampler='nearest'):
    """
    upsample a given rotation to a certain frame_time, scale_factor, ... ...
    :param motion: JxCxF   (J: num_joints; C: channel, 3(euler, position) or 4(quaternion); F: num_frames)
    :param scale_factor: source_frame_time / target_frame_time
    :param target_frame: how many frames are needed
                         NOTE: if scale_factor is not none, another interpolate will be applied
                               to obtain `target_frame` frames
    :param sampler: 'nearest', 'linear', etc.
    :return:
    """
    assert len(motion.shape) == 3, 'input rotation should be Jx(3/4)xF'

    if scale_factor is not None and abs(scale_factor - 1.0) > 1e-3:
        # noinspection PyArgumentList
        motion = F.interpolate(motion, size=None, recompute_scale_factor=False, scale_factor=scale_factor, mode=sampler)

    if target_frame is not None and motion.shape[-1] != target_frame:  # frames not aligned yet
        # noinspection PyArgumentList
        motion = F.interpolate(motion, size=target_frame, recompute_scale_factor=False, scale_factor=None, mode=sampler)

    return motion


def align_root_rot(pos: torch.Tensor, root_rot, hip: tuple, sho: tuple, to_axis='Z', up_axis='Y'):
    """
    align root rotation to a certain direction (x/y/z-axis)
    :param pos:  [(B), J, 3, F], positions
    :param root_rot: [(B), 4, F], rotations
    :param to_axis: axis to align (to face towards)
    :param up_axis: up axis
    :param hip: (L-hip, R-hip)
    :param sho:  (L-shoulder, R-shoulder)
    :return: [(B), 4, F] for new root rotation (towards `to_axis` axis), [(B), 1, F] for rotation radius along `up_axis`
    """
    batch = True if len(pos.shape) == 4 else False
    if not batch:
        pos = pos[None, ...]
        root_rot = root_rot[None, ...]

    if up_axis != 'Y':
        raise NotImplementedError

    h = pos[:, hip[0], ...] - pos[:, hip[1], ...]  # L<-R
    s = pos[:, sho[0], ...] - pos[:, sho[1], ...]  # L<-R
    ve = (h + s) * 0.5
    up = torch.zeros_like(ve, device=pos.device)  # [B, 3, F]
    up[:, 'XYZ'.index(up_axis), :] = 1.0

    forward = torch.cross(ve, up, dim=1)
    forward = torch.nn.functional.normalize(forward, p=2.0, dim=1)

    to_dir = torch.zeros_like(forward, device=pos.device)
    to_dir[..., 'XYZ'.index(to_axis), :] = 1.0

    to_dir_qua = mot.rotations.quaternion_from_two_vectors(forward, to_dir)
    new_root = mot.rotations.mul_two_quaternions(to_dir_qua, root_rot)
    neg_w = (new_root[:, [0], :] < 0).expand(-1, 4, -1)
    new_root[neg_w] = -new_root[neg_w]
    root_eul_y = mot.rotations.quaternion_to_euler(to_dir_qua[:, None, :, :], 'XZY', intrinsic=False)[:, 0, [2], :]

    if not batch:
        new_root = new_root[0]
        root_eul_y = root_eul_y[0]

    return new_root, root_eul_y


def calc_joint_offset(trs: torch.Tensor):
    """
    get offset from position (root translation)

    offset of frame i     ===> O_{i} = F_{i} - F_{i-1}
    offset of first frame ===> O_{0} = 0.5 * (O_{1} + O_{2})
    :param trs: root translation (1x3xF)
    :return: 1x3xF
    """
    _, _, f = trs.shape
    off = torch.empty_like(trs, device=trs.device)
    off[:, :, 1:f] = trs[:, :, 1:f] - trs[:, :, 0:f-1]
    off[:, :, 0] = 0.5 * (off[:, :, 1] + off[:, :, 2])   # approximate the first frame
    return off


def sum_joint_offset(off: torch.Tensor, world=None):
    """
    sum up all the offset vectors to obtain position

    NOTE:
        world = frame_0_absolute_position - frame_0_offset_vector
    that means:
        world = frame_{-1}_absolute_position   (-1 <==> last frame)
    since:
        frame_0_absolution_position = frame_{-1}_absolution_position + frame_0_offset_vector

    :param off:    root translation (1x3xF)
    :param world:  world position (is a tensor: 1x3x1, not a tensor: len==3)
    :return: 1x3xF
    """
    trs = torch.cumsum(off, dim=2)
    if world is not None:
        world = torch.tensor(world, device=trs.device)
        world = world.view(1, -1, 1)
    if isinstance(world, torch.Tensor):
        trs += world
    return trs


def pad_motion(motion: torch.Tensor, l_pad=0, r_pad=0, mode='reflect'):
    return F.pad(motion, (l_pad, r_pad), mode, value=0)


def get_motion_masked(motion: torch.Tensor, mask: list) -> torch.Tensor:
    """
    get a subset of motion
    e.g.
        Hand:1 --- Index:2
               |
               +-- Thumb:3

        mask = [1, 2] will cut off the `Thumb` node and returns [Hand, Index]

    :param motion: JxNxF or BxJxNxF
    :param mask: a list of `int`
    :return: JxMxF or BxJxMxF,  where M = N - len(mask)
    """
    return motion[..., mask, :, :]


# def demo():
#     import bvh
#     a = torch.zeros((31, 3, 360))
#     b = torch.zeros((4, 31, 4, 360))
#
#     # test get motion masked
#     print(get_motion_masked(a, [0, 1, 2, 5, 7, 9]).shape)
#     print(get_motion_masked(b, [0, 1, 2, 6, 7, 8]).shape)
#
#     # test get selected joints
#     bvh_obj = bvh.parser.BVH('../data/assets/test.bvh')
#
#     cmu_mask = [
#         'Hips',             # 0
#         'LeftUpLeg',        # 2
#         'LeftLeg',          # 3
#         'LeftFoot',         # 4
#         'LeftToeBase',      # 5
#         'RightUpLeg',       # 7
#         'RightLeg',         # 8
#         'RightFoot',        # 9
#         'RightToeBase',     # 10
#         'Spine',            # 12
#         'Spine1',           # 13
#         'Neck1',            # 15
#         'Head',             # 16
#         'LeftArm',          # 18
#         'LeftForeArm',      # 19
#         'LeftHand',         # 20
#         'LeftHandIndex1',   # 22
#         'RightArm',         # 25
#         'RightForeArm',     # 26
#         'RightHand',        # 27
#         'RightHandIndex1',  # 29
#     ]
#     mask = bvh_obj.get_index_of_selected_joints(cmu_mask)
#     print(mask)
#
#     _, qua = mot.bvh_casting.get_quaternion_from_bvh(bvh_obj)
#     print(get_motion_masked(qua, mask).shape)
#
#     # test motion sampling
#     a = torch.zeros((31, 3, 360))
#     b = torch.zeros((31, 4, 360))
#
#     print(sample_frames(a, 0.25).shape)
#     print(sample_frames(b, 0.25).shape)
#
#     print(sample_frames(a, 0.25, 90).shape)
#     print(sample_frames(b, 0.25, 90).shape)
#
#     print(sample_frames(a, 0.25, sampler='linear').shape)
#     print(sample_frames(b, 0.25, sampler='linear').shape)
#
#     trs = torch.arange(300).view(1, 3, 100)
#     off = calc_joint_offset(trs)
#     trs_local = sum_joint_offset(off)
#
#     p0 = (0, 100, 200)
#     o0 = (1, 1, 1)
#     w0 = tuple([a - b for a, b in zip(p0, o0)])
#     trs_world = sum_joint_offset(off, w0)
#
#     a = trs.numpy()
#     b = off.numpy()
#     c = trs_local.numpy()
#     d = trs_world.numpy()
#     print(a)
#     print(b)
#     print(c)
#     print(d)
#
#     print((trs - trs_local).numpy())
#     print((trs - trs_world).numpy())
#
#     mo = torch.arange(12).view(2, 2, 3).float()
#     print(pad_motion(mo, 1, 2, 'constant'))
#     print(pad_motion(mo, 2, 1, 'constant'))
#     print(pad_motion(mo, 2, 1, 'reflect'))
#
#
# if __name__ == '__main__':
#     demo()
