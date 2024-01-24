import torch


def forward_kinematics(parent_index: list, mat3x3: torch.Tensor,
                       root_pos: [torch.Tensor, None], offset: torch.Tensor,
                       world=True, is_edge=False):
    """
    implement forward kinematics in a batched manner
    :param parent_index: index of parents (-1 for no parent)
    :param mat3x3: rotation matrix, [(B), J, 3, 3, F] (batch_size x joint_num x 3 x 3 x frame_num)
    :param root_pos: root position [(B), 1, 3, F], None for Zero
    :param offset: joint offsets [(B), J, 3, F]
    :param world: world position or local position ?
    :param is_edge:
            True:  mat3x3[i] represents a rotation matrix of the parent of joint i,
                   i.e. edge rotation
            False: mat3x3[i] represents a rotation matrix of joint i,
                   i.e. joint rotation
    :return: tensor of positions in the shape of [(B), J, 3, F]
    """
    assert parent_index[0] == -1, f"the first parent index should be -1 (root), not {parent_index[0]}."

    batch = len(mat3x3.shape) == 5
    if not batch:
        mat3x3 = mat3x3[None, ...]
        if root_pos is not None:
            root_pos = root_pos[None, ...]
        offset = offset[None, ...]

    assert len(mat3x3.shape) == 5
    assert len(offset.shape) == 4
    assert root_pos is None or len(root_pos.shape) == 4

    B, J, _, _, F = mat3x3.shape

    mat3x3 = mat3x3.permute(0, 4, 1, 2, 3)                  # mat:    [B, F, J, 3, 3]
    offset = offset.permute(0, 3, 1, 2)[..., None]          # offset: [B, F, J, 3, 1]
    if root_pos is not None:
        root_pos = root_pos.permute(0, 3, 1, 2)[..., None]  # root:   [B, F, 1, 3, 1]

    mat_mix = torch.empty_like(mat3x3, dtype=mat3x3.dtype, device=mat3x3.device)  # avoid in-place operation

    position = torch.empty((B, F, J, 3, 1), device=offset.device)  # [B, F, J, 3, 1]

    if root_pos is not None:
        position[..., 0, :, :] = root_pos[..., 0, :, :]
    else:
        position[..., 0, :, :].zero_()

    mat_mix[..., 0, :, :] = mat3x3[..., 0, :, :]
    for ci, pi in enumerate(parent_index[1:], 1):
        off_i = offset[..., ci, :, :]

        if not is_edge:
            mat_p = mat_mix[..., pi, :, :]
            trs_i = torch.matmul(mat_p, off_i)
            position[..., ci, :, :] = trs_i
            mat_mix[..., ci, :, :] = torch.matmul(mat_p, mat3x3[..., ci, :, :])
        else:
            combo = torch.matmul(mat_mix[..., pi, :, :], mat3x3[..., ci, :, :])
            trs_i = torch.matmul(combo, off_i)
            position[..., ci, :, :] = trs_i
            mat_mix[..., ci, :, :] = combo

        if world:
            position[..., ci, :, :] += position[..., pi, :, :]

    position = position[..., 0].permute(0, 2, 3, 1)

    if not batch:
        position = position[0]

    return position
