import matplotlib
# matplotlib.use("agg")
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt

from bvh.parser import BVH
from motion_tensor.bvh_casting import *
from motion_tensor.motion_process import sample_frames
from motion_tensor.rotations import *
from PIL import Image


def _get_uniform_pos_from_bvh(obj: BVH, fps=2.0, start=0, end=5, rearrange_xz=True, facing_z=True, scale=1.0):
    """
    (J, 3, T)
    """
    """ 1. get rotations from bvh obj """
    p_index = obj.dfs_parent()
    off = get_offsets_from_bvh(obj)
    trs, qua = get_quaternion_from_bvh(obj)

    """ 2. sample frames to reduce frame count """
    trs = sample_frames(trs, scale_factor=obj.frame_time * fps)
    qua = sample_frames(qua, scale_factor=obj.frame_time * fps)

    """ 3. select the most desired part """
    end = min(end, trs.shape[-1], qua.shape[-1])
    if start >= end:
        start = max(0, end-5)
    trs = trs[:, :, start:end]
    qua = qua[:, :, start:end]

    """ 4. rectify the facing direction """
    if facing_z:
        facing = qua[0, :, 0][None, :, None]
        y_rot = quaternion_to_euler(facing, 'XZY', intrinsic=False)[0, 2, 0].item()
        neg_rot = torch.tensor([0.0, 1.57-y_rot, 0.0])[None, :, None]
        rectify = euler_to_quaternion(neg_rot, to_rad=1.0, order='XYZ', intrinsic=True)
        qua[[0], :, :] = mul_two_quaternions(rectify, qua[[0], :, :])

    """ 5. apply FK """
    mat = mot.rotations.quaternion_to_matrix(qua)
    pos = mot.kinematics.forward_kinematics(p_index, mat, trs, off)

    """ 6. normalize the poses to uniform unit """
    t_pos = get_t_pose_from_bvh(obj)
    h = (torch.max(t_pos[:, 1]) - torch.min(t_pos[:, 1])).item()
    pos = pos / h * scale

    """ 7. move every poses to separate them """
    if rearrange_xz:
        pos[:, [0, 2], :] -= pos[[0], [0, 2], :]  # move every frames' root xz to origin
        n = pos.shape[-1]
        for i in range(n):
            pos[:, 0, i] += ((i/(n-1)) - 0.5) * 2.0 * scale

    return pos


def _frame_to_lines(p_index, frame):
    """
    (J, 3)
    """
    lines = []
    for c, p in enumerate(p_index):
        if p == -1: continue
        x1 = frame[c, 0].item()
        x2 = frame[p, 0].item()
        y1 = frame[c, 1].item()
        y2 = frame[p, 1].item()
        z1 = frame[c, 2].item()
        z2 = frame[p, 2].item()
        lines.append(([x1, x2], [y1, y2], [z1, z2]))
    return lines


def _render_pos(p_index, pos, color, elev=2.0, azim=-90, hind_axis=False, save_to=None, scale=1.0):

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    for i in range(0, pos.shape[-1], 1):
        frame = pos[..., i]
        lines = _frame_to_lines(p_index, frame)
        for x, y, z in lines:
            ax.plot(x, z, y, linewidth=4, c=color)

    ax.set_xlim3d(-scale, +scale)
    ax.set_ylim3d(-scale, +scale)
    ax.set_zlim3d(-0, +2 * scale)
    ax.margins(0.0)
    ax.view_init(elev=elev, azim=azim)

    if hind_axis: plt.axis('off')

    if save_to is None:
        plt.show()
    else:
        if save_to[-3:] == 'png':
            plt.savefig(save_to, transparent=True)
        else:
            plt.savefig(save_to)
        plt.close()


def _post_processing(filepath, overwrite=True, weights=None):
    """
    :param filepath:
    :param overwrite: overwrite the input image file or not
    :param weights: crop ratio: (left, upper, right, bottom)
    :return:
    """
    img = Image.open(filepath)
    w, h = img.size

    if weights is None:
        weights = (0.2, 0.45, 0.15, 0.25)
    l, u, r, b = weights
    u, b = int(h * u), int(h * b)
    l, r = int(w * l), int(w * r)
    img = img.crop((l, u, w-r, h-b))
    img.save(filepath if overwrite else filepath[:-4]+'.post_processed'+filepath[-4:])


def render_bvh(obj: BVH, color='midnightblue', elev=2.0, azim=-90, hind_axis=True, save_to=None,
               fps=2.0, start=0, end=5, rearrange_xz=True, facing_z=True, scale=1.0):
    """
    :param obj: BVH object
    :param color: the color for rendering
                  NOTE: Using different color for different body part is meaningless
                        since matplotlib doesn't have something like depth buffer.
                        One possible solution is to use other rendering libs like
                        visvis or mayavi, please refer to:
                        https://stackoverflow.com/questions/12193709/is-there-something-like-a-depth-buffer-in-matplotlib
    :param hind_axis: hind plot axis or not
    :param save_to: file path to save the image, save as `*.png` to preserve alpha channel,
                    `None` for not saving but just displaying it
    :param elev: ax.view_init(...)
    :param azim: ax.view_init(...)
    :param fps: fps of the motion to be rendered, large fps may cause a very dense result
    :param start: start frame (after sampled to desired fps)
    :param end: end frame (after sampled to desired fps)
    :param rearrange_xz: put root positions of rendered poses in a line on xz-plane
    :param facing_z: rotate rendered poses toward z-axis
    :param scale: scale of rendered poses
    :return:
    """
    pos = _get_uniform_pos_from_bvh(obj, fps, start, end, rearrange_xz, facing_z, scale)
    _render_pos(obj.dfs_parent(), pos, color, elev, azim, hind_axis, save_to, scale)
    if save_to is not None:
        _post_processing(save_to)


if __name__ == '__main__':

    def render_bvh_folder(folder, color, params):
        import glob
        ls = []
        for file in glob.glob(os.path.join(folder, "*.bvh")):
            ls.append(file)

        for file in ls:
            st = 0
            fps = 2
            for k, v in params.items():
                if k in file:
                    st, fps = v
            if not os.path.exists(file[:-4]+'.png'):
                print('[processing]' + file)
                obj = BVH(file)
                render_bvh(obj, save_to=file[:-4]+'.png', color=color, start=st, end=st+5, fps=fps)
            else:
                print('[skip]' + file)

    def foo():
        start = {
            'box': 0,
            'dance': 10,
            'swim': 2,
            'yoga': 0,
            'basketball': 4,

            # from dataset
            'run':  0,
            'walk': 0,
            'jump': 0,
            'turn': 0,

            # MP different results
            'diff': 0,

            # bfa
            'bfa_old': 0,
            'bfa_str': 0,
            'bfa_dep': 0,

            '_SJ': 0,
            '_SW': 0,
        }
        fps = {
            'box': 2,
            'dance': 2,
            'swim': 2,
            'yoga': 2,
            'basketball': 8,

            # from dataset
            'run':  12,
            'walk': 8,
            'jump': 8,
            'turn': 8,

            # MP different results
            'diff': 1.5,

            # bfa
            'bfa_old': 4,
            'bfa_str': 4,
            'bfa_dep': 1,

            '_SJ': 16,
            '_SW': 16,
        }
        params = {}
        for k, v in start.items():
            params[k] = (v, fps[k])
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\org", 'black', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\rm\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\rm\str", 'maroon', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\rm\dep", 'midnightblue', params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\dme\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\dme\str", 'maroon', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\dme\dep", 'midnightblue', params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\rm_pos\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\rm_pos\str", 'maroon', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\rm_pos\dep", 'midnightblue', params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\dme\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\dme\str", 'maroon', params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\dataset\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\dataset\str", 'maroon', params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\mp_diff\old", 'darkgoldenrod',  params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\mp_diff\org", 'black',  params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\bfa\old", 'darkgoldenrod',  params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\bfa\str", 'maroon',  params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\bfa\dep", 'midnightblue',  params)
        #
        # for k, v in fps.items():
        #     fps[k] = v // 2
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\png\MP_OUT\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\png\MP_OUT\str", 'maroon', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\png\MP_OUT\dep", 'midnightblue', params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\diff-style\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\diff-style\str", 'maroon', params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\mp-diff\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\mp-diff\str", 'maroon', params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\sig20-diff\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\sig20-diff\str", 'maroon', params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\ours-diff\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\ours-diff\str", 'maroon', params)
        #
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\sig20-sup\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\sig20-sup\str", 'maroon', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\sig20-sup\dep", 'midnightblue', params)

        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\ours-lim\old", 'darkgoldenrod', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\ours-lim\org", 'black', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\ours-lim\zom", 'crimson', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\ours-lim\ang", 'mediumvioletred', params)
        # render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-4\ours-lim\str", 'maroon', params)

        render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-15\fem", '#FF1493', params)
        render_bvh_folder(r"D:\_Projects\_Thesis\Rendering\CHEAT1\4-15\old", 'darkgoldenrod', params)

    foo()
