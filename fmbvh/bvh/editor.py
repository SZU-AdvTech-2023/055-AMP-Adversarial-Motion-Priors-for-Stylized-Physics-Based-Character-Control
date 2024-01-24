from fmbvh.bvh.parser import BVH, JointOffset, JointMotion
import itertools
from copy import deepcopy
import torch
import fmbvh.motion_tensor.rotations as mor
import fmbvh.motion_tensor.bvh_casting as bvc


def reorder_bvh(obj: BVH):
    """
    this function is to reorder the offset_data and motion_data,
    since both of them use `OrderedDict` to store data and thus
    their keys need to be reordered after inserting or deleting
    some joints
    :param obj:
    :return:
    """
    dfs = [name for name, _ in obj.dfs()]
    for name in dfs:
        obj.offset_data.move_to_end(name)
        obj.motion_data.move_to_end(name)
    return obj


def reorder_joints(obj: BVH, parent_name: str, children_names_reordered: list):
    pj = obj.offset_data[parent_name]
    if len([None for name in pj.children_names if name not in children_names_reordered]):
        raise ValueError(f"should contain all the children joints: {pj.children_names}")

    pj.children_names = deepcopy(children_names_reordered)

    return reorder_bvh(obj)


def get_remaining_joint_names(obj: BVH, joint_names: list):
    names = [name for name, _ in obj.dfs()]
    return [name for name in names if name not in joint_names]


def rectify_joint(obj: BVH, parent: str, target: str, direction: list):
    """
    to correct a joint, e.g. A-pose to T-pose
    :param obj:
    :param parent:
    :param target:
    :param direction: e.g. direction [0, -1, 0] for `UpperLeg` joint forces it to face to the ground (A-pose to T-pose)
    :return:
    """
    QUA = mor.pad_position_to_quaternion
    MUL = mor.mul_two_quaternions
    INV = mor.inverse_quaternion
    V2Q = mor.quaternion_from_two_vectors
    POS = lambda x: x[:, 1:, :]
    NORM = mor.normalize_quaternion
    RECT = lambda x: mor.rectify_w_of_quaternion(x, True)
    ROTATE = lambda q_, p_: MUL(q_, MUL(p_, INV(q_)))

    q_ls = [parent]

    def __gather_children(l_, n_):
        j_ = obj.offset_data[n_]
        l_ += j_.children_names
        for c_ in j_.children_names:
            __gather_children(l_, c_)

    __gather_children(q_ls, q_ls[0])

    pis = obj.get_index_of_selected_joints(q_ls)
    tis = pis[1:]

    t, q = bvc.get_quaternion_from_bvh(obj)
    o = bvc.get_offsets_from_bvh(obj)

    src = o[[pis[1]]]
    dst = torch.empty_like(src, dtype=src.dtype, device=src.device)
    dst[:, 0, :], dst[:, 1, :], dst[:, 2, :] = direction[0], direction[1], direction[2]
    Q = V2Q(src, dst)
    Q_ = INV(Q)

    Qx = q[pis]
    Qx = MUL(Qx, Q_)
    Qx[1:] = MUL(Q, Qx[1:])
    Qx = RECT(NORM(Qx))
    q[pis] = Qx[:]
    obj = bvc.write_quaternion_to_bvh(t, q, obj)

    Lx = QUA(o[tis])
    Lx = MUL(Q, MUL(Lx, Q_))
    o[tis] = POS(Lx)
    obj = bvc.write_offsets_to_bvh(o, obj)

    return obj


def cut_off_joints(obj: BVH, remove_names: list):

    def try_remove(name_):
        off: JointOffset = obj.offset_data[name_]
        if name_ == obj.root_name or len(off.children_names) != 0:
            raise ValueError(f"'{name_}' is not a leaf node.")
        par_ = off.parent_name
        p_off: JointOffset = obj.offset_data[par_]
        p_off.children_names.remove(name_)
        del obj.offset_data[name_]
        del obj.motion_data[name_]

    reversed_names = [name for name, _ in obj.dfs()][::-1]
    for fit_name, _ in filter(lambda x: x[0] == x[1], itertools.product(reversed_names, remove_names)):
        try_remove(fit_name)

    return reorder_bvh(obj)


def shift_joint(obj: BVH, target_name: str, offset: list):
    """
    change target joint offset to a new one
    :param obj:
    :param target_name:
    :param offset:
    :return:
    """
    QUA = mor.pad_position_to_quaternion
    MUL = mor.mul_two_quaternions
    INV = mor.inverse_quaternion
    V2Q = mor.quaternion_from_two_vectors
    POS = lambda x: x[:, 1:, :]
    NORM = mor.normalize_quaternion
    RECT = lambda x: mor.rectify_w_of_quaternion(x, True)
    ROTATE = lambda q_, p_: MUL(q_, MUL(p_, INV(q_)))

    tj = obj.offset_data[target_name]
    pn = tj.parent_name
    tn = target_name
    cn = tj.children_names

    pi = obj.get_index_of_selected_joints([pn])  # parent index
    ti = obj.get_index_of_selected_joints([tn])  # target index
    ci = obj.get_index_of_selected_joints(cn)  # indices of children

    if len(ci) != 0:
        t, q = bvc.get_quaternion_from_bvh(obj)
        o = bvc.get_offsets_from_bvh(obj)

        Qp_, Qt_, Qc_ = q[pi], q[ti], q[ci]
        Lp_, Lt_, Lc_ = o[pi], o[ti], o[ci]
        Lp_, Lt_, Lc_ = QUA(Lp_), QUA(Lt_), QUA(Lc_)

        Ltp = torch.empty_like(Lt_, dtype=Lt_.dtype, device=Lt_.device)
        Ltp[:, 0, :], Ltp[:, 1, :], Ltp[:, 2, :], Ltp[:, 3, :] = 0.0, offset[0], offset[1], offset[2]
        Lcp = Lt_ + Lc_ - Ltp

        vec = ROTATE(Qp_, Lt_) + ROTATE(MUL(Qp_, Qt_), Lc_) - ROTATE(Qp_, Ltp)
        vec = MUL(INV(Qp_), MUL(vec, Qp_))
        tqp = V2Q(POS(Lcp), POS(vec))  # what the target rotation should be for every child
        tqp = tqp.mean(dim=0, keepdim=True)  # get the average result if there are multiple children
        tqp = RECT(NORM(tqp))
        q[ti] = tqp
        obj = bvc.write_quaternion_to_bvh(t, q, obj)
    else:
        pass  # just move the end-effector so there is no need to change

    # set the new target offset
    org = tj.offset
    tj.offset = [e for e in offset]
    for c in cn:
        cj = obj.offset_data[c]
        cj.offset = [a + b - c for a, b, c in zip(org, cj.offset, tj.offset)]
    return reorder_bvh(obj)


def remove_joint(obj: BVH, remove_names: [list, str], inherent='mul'):
    """
    remove a list of given joints,
    this is useful for removing joints that are at the same position
    :param obj:
    :param remove_names:
    :param inherent: inherent joint rotation: none, mul, recompute
    :return: edited bvh object
    """
    if isinstance(remove_names, str):
        remove_names = [remove_names]
    for rm_name in remove_names:
        if rm_name == obj.root_name:
            raise ValueError("cannot remove root joint!")

        tj = obj.offset_data[rm_name]  # target joint
        pn = tj.parent_name            # parent name
        tn = rm_name

        if inherent == 'none':
            ic = obj.get_index_of_selected_joints(obj.offset_data[rm_name].children_names)  # indices of children
            if len(ic) != 0:
                # add target offset to children
                for cn in tj.children_names:
                    cj = obj.offset_data[cn]
                    cj.offset = [a + b for a, b in zip(cj.offset, tj.offset)]
            else:
                pass
        elif inherent == 'mul':
            # add target motion to children
            MUL = mor.mul_two_quaternions
            NORM = mor.normalize_quaternion
            RECT = lambda x: mor.rectify_w_of_quaternion(x, True)

            t, q = bvc.get_quaternion_from_bvh(obj)

            it = obj.get_index_of_selected_joints([tn])  # target index
            ic = obj.get_index_of_selected_joints(obj.offset_data[rm_name].children_names)  # indices of children

            if len(ic) != 0:
                Qt, Qc = q[it], q[ic]
                q[ic] = RECT(NORM(MUL(Qt, Qc)))  # directly combine the two rotations
                obj = bvc.write_quaternion_to_bvh(t, q, obj)

                # add target offset to children
                for cn in tj.children_names:
                    cj = obj.offset_data[cn]
                    cj.offset = [a+b for a, b in zip(cj.offset, tj.offset)]
            else:
                pass
        elif inherent == 'recompute':
            # add target motion to children
            QUA = mor.pad_position_to_quaternion
            MUL = mor.mul_two_quaternions
            INV = mor.inverse_quaternion
            V2Q = mor.quaternion_from_two_vectors
            POS = lambda x: x[:, 1:, :]
            NORM = mor.normalize_quaternion
            RECT = lambda x: mor.rectify_w_of_quaternion(x, True)
            ROTATE = lambda q_, p_: MUL(q_, MUL(p_, INV(q_)))

            t, q = bvc.get_quaternion_from_bvh(obj)
            o = bvc.get_offsets_from_bvh(obj)

            cns = obj.offset_data[rm_name].children_names  # children of target joint
            ip = obj.get_index_of_selected_joints([pn])  # parent index
            it = obj.get_index_of_selected_joints([tn])  # target index
            ics = obj.get_index_of_selected_joints(cns)  # indices of children

            if len(ics) != 0:
                for ic, cn in zip(ics, cns):
                    gns = obj.offset_data[cn].children_names  # grandchildren names
                    if len(gns) == 0:
                        continue

                    Qp, Qt, Qc = q[ip], q[it], q[ic]
                    Lp, Lt, Lc = o[ip], o[it], o[ic]
                    Lp, Lt, Lc = QUA(Lp), QUA(Lt), QUA(Lc)

                    igs = obj.get_index_of_selected_joints(gns)
                    Lg = o[igs].mean(dim=0, keepdims=True)  # get the average results as the succeeded joint's offset
                    Lg = QUA(Lg)

                    a = ROTATE(Qp, Lt) + ROTATE(MUL(Qp, Qt), Lc) + ROTATE(MUL(Qp, MUL(Qt, Qc)), Lg)
                    b = ROTATE(Qp, Lt + Lc)
                    vec = MUL(INV(Qp), MUL(a - b, Qp))
                    Qc_ = V2Q(POS(Lg), POS(vec))

                    q[ic] = RECT(NORM(Qc_))

                obj = bvc.write_quaternion_to_bvh(t, q, obj)
            else:
                pass  # use parent rotation

            # add target offset to children
            for cn in tj.children_names:
                cj = obj.offset_data[cn]
                cj.offset = [a+b for a, b in zip(cj.offset, tj.offset)]
        else:
            raise ValueError(f'inherent_rotation should be `none`, `mul` or `recompute`, not {inherent}')

        # ---- for parent ---- #
        pj = obj.offset_data[pn]           # parent joint
        pj.children_names.remove(rm_name)  # delete target joint
        pj.children_names += tj.children_names  # take over all the children of target joint

        # ---- for children ---- #
        # assign a new parent (i.e. target joint's parent)
        for cn in tj.children_names:  # cn: child name
            obj.offset_data[cn].parent_name = pn

        # ---- for target ---- #
        # delete target joint
        del obj.offset_data[rm_name]
        del obj.motion_data[rm_name]

        obj = reorder_bvh(obj)

    return obj


def insert_joint_between(obj: BVH, j1: str, j2: str, new_name: str, new_offset: [None, list]=None, divide_ratio=0.5):
    """
    :param obj: bvh object
    :param j1: name of joint 1
    :param j2: name of joint 2
    :param new_name: name of inserted joint
    :param new_offset: [Optional] new offset of the new joint
    :param divide_ratio: split between two (j1, j2), this parameter is ignored if new_offset is not None
    :return: edited bvh object (inplace!)
    """
    if new_name in obj.offset_data:
        raise ValueError(f"Name {new_name} already exists!")
    ja: JointOffset = obj.offset_data[j1]
    jb: JointOffset = obj.offset_data[j2]

    if j1 in jb.children_names:  # j1, ja: parent, j2, jb: child
        j1, j2 = j2, j1
        ja, jb = jb, ja
    elif j2 not in ja.children_names:
        raise ValueError(f"{j1} and {j2} should be father and child.")

    # input:   ja +---> jb ----> [...]
    #             |
    #             +---> jc
    #             |
    #             +---> [...]
    #
    # output:  ja +---> jn ----> jb ----> [...]
    #             |
    #             +---> jc
    #             |
    #             +---> [...]
    #
    # ---- offset data ---- #
    off_jn = [b*divide_ratio for b in jb.offset] if new_offset is None else new_offset
    off_jb = [b - a for a, b in zip(off_jn, jb.offset)]

    jn = deepcopy(jb)  # joint new
    jn.name = new_name
    jn.offset = off_jn
    jn.children_names = [j2]
    jn.parent_name = j1

    ja.children_names[ja.children_names.index(jb.name)] = jn.name
    jb.offset = off_jb
    jb.parent_name = jn.name
    obj.offset_data[new_name] = jn

    # ---- motion data ---- #
    mb: JointMotion = obj.motion_data[j2]
    mn = deepcopy(mb)
    mn.name = new_name
    mn.data = [[0 for _ in range(len(mb.data[0]))] for _ in range(len(mb.data))]  # no motion!
    obj.motion_data[new_name] = mn

    # ---- reorder ---- #
    return reorder_bvh(obj)


def zero_motion(obj: BVH, name: str):
    """
    :param obj:
    :param name:
    :return:
    """
    data = obj.motion_data[name].data
    for i in range(len(data)):
        data[i] = [0, 0, 0]

    return obj


def append_joint(obj: BVH, parent_name: str, new_name: str, offset):
    """
    :param obj:
    :param parent_name:
    :param new_name:
    :param offset:
    :return:
    """
    pn = parent_name
    pj: JointOffset = obj.offset_data[pn]
    pm: JointMotion = obj.motion_data[pn]
    pj.children_names.append(new_name)

    # offset
    nj = deepcopy(pj)
    nj.parent_name = pn
    nj.children_names = []
    nj.offset = deepcopy(offset)
    nj.name = new_name
    obj.offset_data[new_name] = nj

    # motion
    nm = deepcopy(pm)
    nm.name = new_name
    nm.data = [[0 for _ in range(len(pm.data[0]))] for _ in range(len(pm.data))]  # no motion!
    obj.motion_data[new_name] = nm

    # ---- reorder ---- #
    return reorder_bvh(obj)


def rename_joints(obj: BVH, src_names: list, dst_names: list):
    def __find_and_replace(name_):
        index_ = src_names.index(name_)
        new_ = dst_names[index_]

        # parent
        if name_ != obj.root_name:
            p_name = obj.offset_data[name_].parent_name
            obj.offset_data[p_name].children_names.remove(name_)
            obj.offset_data[p_name].children_names.append(new_)

        # children
        for cn in obj.offset_data[name_].children_names:
            obj.offset_data[cn].parent_name = new_

        # self
        obj.offset_data[name_].name = new_
        obj.motion_data[name_].name = new_

        jo = obj.offset_data[name_]
        jm = obj.motion_data[name_]

        del obj.offset_data[name_]
        del obj.motion_data[name_]

        obj.offset_data[new_] = jo
        obj.motion_data[new_] = jm

    nm_list = [name for name, _ in obj.dfs()][::-1]

    for name in nm_list:
        if name in src_names:
            __find_and_replace(name)

    return reorder_bvh(obj)


# def demo_test():
#     show_t_pose = False
#     is_cmu = True
#     edge_repr = True
#
#     if not is_cmu:
#         bvh_obj = BVH('D:/_dataset/Motion Retargeting/Mixamo/std_bvhs/Ortiz_m.bvh')
#         # bvh_obj = BVH('D:/_dataset/Motion Retargeting/Mixamo/std_bvhs/Jasper_m.bvh')
#         # bvh_obj = BVH('D:/_dataset/Motion Retargeting/Mixamo/std_bvhs/Abe_m.bvh')
#         # bvh_obj = BVH('D:/_dataset/Motion Retargeting/Mixamo/std_bvhs/Malcolm_m.bvh')
#         # bvh_obj = BVH('D:/_dataset/Motion Retargeting/Mixamo/std_bvhs/Knight_m.bvh')
#
#         remain_names = [
#             'Hips',
#             'Spine', 'Spine1', 'Spine1_split', 'Spine2', 'Neck', 'Head', 'HeadTop_End',
#             'LeftShoulder', 'LeftShoulder_split', 'LeftArm', 'LeftForeArm', 'LeftHand',
#             'RightShoulder', 'RightShoulder_split', 'RightArm', 'RightForeArm', 'RightHand',
#             'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End',
#             'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End'
#         ]
#         remove_names = get_remaining_joint_names(bvh_obj, remain_names)
#         cut_off_joints(bvh_obj, remove_names)
#     else:
#         bvh_obj = BVH(r'D:\_dataset\Motion Style Transfer\mocap_xia\0_angry\angry_01_000.bvh')
#         insert_joint_between(bvh_obj, 'LowerBack', 'Spine', 'MySpine')
#         delete_joints(bvh_obj, 'LowerBack')
#         insert_joint_between(bvh_obj, 'Spine', 'Spine1', 'MySpine2')
#
#         rectify_joint(bvh_obj, 'LeftUpLeg', 'LeftLeg', [0, -1, 0])
#         rectify_joint(bvh_obj, 'RightUpLeg', 'RightLeg', [0, -1, 0])
#
#     from visualization.visualize_motion import MoVisualizer
#     import motion_tensor as mot
#     import torch
#
#     t_pos = mot.bvh_casting.get_positions_from_bvh(bvh_obj)[..., 0]  # [J, 3]
#     max_y = torch.max(t_pos[:, 1])
#     min_y = torch.min(t_pos[:, 1])
#     height = (max_y - min_y).item()
#
#     trs, qua = mot.bvh_casting.get_quaternion_from_bvh(bvh_obj)  # [1, 3, F], [J, 4, F]
#     trs /= height
#
#     if show_t_pose:
#         trs[:, :, :] = 0.0
#         qua[:, :1, :] = 1.0
#         qua[:, 1:, :] = 0.0
#
#     mat = mot.rotations.quaternion_to_matrix(qua[None, ...])  # [B, J, 3, 3, F]
#     J, _, F = qua.shape
#
#     offsets = mot.bvh_casting.get_offsets_from_bvh(bvh_obj)[None, ...]  # [B, J, 3, 1]
#     offsets = torch.broadcast_to(offsets, (1, J, 3, F))  # [B, J, 3, F]
#
#     if not edge_repr:
#         fk_pos = mot.kinematics.forward_kinematics(bvh_obj.dfs_parent(), mat,
#                                                    trs[None, ...], offsets, is_edge=False)    # [B, J, 3, F]
#     else:
#         ide = torch.eye(3)[None, None, ..., None]
#         ide = torch.broadcast_to(ide, (1, 1, 3, 3, F))
#         edge_i = [e for e in bvh_obj.dfs_parent() if e != -1]
#         mat = mat[:, edge_i, :, :, :]
#         mat = torch.concat([ide, mat], dim=1)  # append root rotation
#         fk_pos = mot.kinematics.forward_kinematics(bvh_obj.dfs_parent(), mat,
#                                                    trs[None, ...], offsets, is_edge=True)    # [B, J, 3, F]
#
#     fk_pos /= height
#
#     def _next():
#         f = 0
#         while True:
#             rpt = trs[:, :, f]  # [1, 3]
#             cps = fk_pos[0, :, :, f]  # [J, 3]
#             pos_ls = cps + rpt  # [J, 3]
#             yield pos_ls.numpy().tolist()
#             f = (f + 1) % F
#
#     p_index = bvh_obj.dfs_parent()
#     mvz = MoVisualizer(p_index, _next(), max_fps=30, add_coordinate=True)
#     # mvz.add_grids(10, 10, height*0.2)
#     mvz.run()
#
#
# if __name__ == '__main__':
#     demo_test()
