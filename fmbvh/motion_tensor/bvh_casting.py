import torch
import fmbvh.bvh as bvh
import fmbvh.motion_tensor as mot
from typing import Tuple


def write_euler_to_bvh(trs: torch.Tensor, eul: torch.Tensor, t_bvh: bvh.parser.BVH,
                       order='ZYX', to_deg=180.0/3.1415926535, frame_time=None):
    """
    write euler(rad) to bvh object(NOTE: this will overwrite `t_bvh`) (degree)
    :param trs:  1x3xF
    :param eul:  Jx3xF
    :param order: ZYX or XYZ
    :param t_bvh: template bvh object
    :param to_deg:
    :param frame_time: new frame time, None for default
    :return: a bvh object (t_bvh)
    """
    assert len(trs.shape) == 3, '`trs` should be: 1x3xF'
    assert len(eul.shape) == 3, '`eul` should be: Jx3xF'
    assert eul.shape[1] == 3
    assert trs.shape[2] == eul.shape[2]

    for i, (name, depth) in enumerate(t_bvh.dfs()):
        old_order = t_bvh.offset_data[name].order
        if len(old_order) == 6: old_order = f'XYZ{order}'
        else: old_order = order
        t_bvh.offset_data[name].order = old_order
        deg_eul = eul[i] * to_deg
        if depth == 0:
            root = torch.cat([trs[0], deg_eul], dim=0)
            t_bvh.motion_data[name].data = root.permute((1, 0)).tolist()
        else:
            t_bvh.motion_data[name].data = deg_eul.permute((1, 0)).tolist()

    t_bvh.frames = trs.shape[2]
    if frame_time is not None: t_bvh.frame_time = frame_time
    return t_bvh


def write_quaternion_to_bvh(trs: torch.Tensor, qua: torch.Tensor,
                            t_bvh: bvh.parser.BVH, to_deg=180.0/3.1415926535,
                            frame_time=None):
    """
    write quaternion to bvh object(NOTE: this will overwrite `t_bvh`) (degree),
    the quaternions will be normalized automatically.
    :param trs:   1x3xF
    :param qua:   Jx3xF
    :param t_bvh:  template bvh object
    :param to_deg:
    :param frame_time: new frame time, None for default
    :return: a bvh object (t_bvh)
    """
    qua = mot.rotations.normalize_quaternion(qua[None, ...])
    eul = mot.rotations.quaternion_to_euler(qua, order='ZYX', intrinsic=False)[0]
    # eul = mot.rotations.quaternion_to_euler_2(qua, order='ZYX', intrinsic=False)  # slower
    return write_euler_to_bvh(trs, eul, t_bvh, 'ZYX', to_deg, frame_time)


def write_offsets_to_bvh(offsets: torch.Tensor, bvh_obj: bvh.parser.BVH):
    """
    :param bvh_obj:
    :param offsets: Jx3x1
    :return:
    """
    for i, (name, _) in enumerate(bvh_obj.dfs()):
        bvh_obj.offset_data[name].offset = [e.item() for e in offsets[i]]
    return bvh_obj


def get_euler_from_bvh(bvh_obj: bvh.parser.BVH, to_rad=3.1415926535/180.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    get root_translation, euler(rad) from an bvh object
    :param bvh_obj: bvh object
    :param to_rad: degree to rad ratio
    :return: root translation(1x3xF), joint rotation(Jx3xF) (rad)
    """
    # assert root channel == 6
    # assert joint channel == 3

    data = [torch.tensor(e.data) for e in bvh_obj.motion_data.values()]  # [3+Jx3, Jx3, ... ]
    data = torch.cat(data, dim=1)  # Fx(3+J3)
    trs = data[:, :3]  # Fx3
    eul = data[:, 3:]  # FxJ3
    eul = eul * to_rad

    trs = trs.permute((1, 0))[None, ...]  # 1x3xF
    eul = eul.reshape(eul.shape[0], -1, 3)  # FxJx3
    eul = eul.permute(1, 2, 0)  # Jx3xF

    return trs, eul


def get_quaternion_from_bvh(bvh_obj: bvh.parser.BVH) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    get root_translation, quaternion from an bvh object
    :param bvh_obj: bvh object
    :return: root translation(1x3xF), joint rotation(Jx4xF) (quaternion)
    """
    trs, eul = get_euler_from_bvh(bvh_obj)
    order = bvh_obj.offset_data[bvh_obj.root_name].order
    if len(order) == 6: order = order[3:]
    qua = mot.rotations.euler_to_quaternion(eul[None, ...], to_rad=1.0, order=order)
    return trs, qua[0]


def get_offsets_from_bvh(bvh_obj: bvh.parser.BVH) -> torch.Tensor:
    """
    get joint offsets from bvh object
    :param bvh_obj:
    :return: offsets Jx3x1 (joint_num, `xyz`, `1 frame`)
    """
    offsets = [e.offset for e in bvh_obj.offset_data.values()]
    return torch.tensor(offsets, dtype=torch.float32)[..., None]


def get_positions_from_bvh(bvh_obj: bvh.parser.BVH, locomotion=True) -> torch.Tensor:
    """
    get joint positions from bvh object
    :param bvh_obj:
    :param locomotion: set root to `zero` if False
    :return: offsets Jx3xF (joint_num, `xyz`, `frames`)
    """
    p_index = bvh_obj.dfs_parent()
    off = get_offsets_from_bvh(bvh_obj)
    trs, qua = get_quaternion_from_bvh(bvh_obj)
    trs = trs if locomotion else None
    mat = mot.rotations.quaternion_to_matrix(qua)
    pos = mot.kinematics.forward_kinematics(p_index, mat, trs, off)
    return pos


def get_t_pose_from_bvh(bvh_obj: bvh.parser.BVH) -> torch.Tensor:
    """
    get t-pose from bvh object
    :param bvh_obj:
    :return: offsets Jx3x1 (joint_num, `xyz`, `1 frame`)
    """
    positions = get_offsets_from_bvh(bvh_obj)
    for i, p in enumerate(bvh_obj.dfs_parent()):
        if p == -1:
            continue
        positions[i, ...] += positions[p, ...]
    return positions
