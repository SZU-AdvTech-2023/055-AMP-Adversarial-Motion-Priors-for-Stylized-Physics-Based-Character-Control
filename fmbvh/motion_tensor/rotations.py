import warnings
import torch


# NOTE:
#   the code below will be removed in future versions
#
class _SymbolWrapper:
    """
    Avoid allocating `zero` tensors in memory, which is replaced by `None` object
    """
    def __init__(self, obj):
        self.obj = obj

    def __mul__(self, other):
        if self.obj is None: return _SymbolWrapper(None)
        if other.obj is None: return _SymbolWrapper(None)
        return _SymbolWrapper(self.obj * other.obj)

    def __truediv__(self, other):
        if self.obj is None: return _SymbolWrapper(None)
        if other.obj is None: raise ZeroDivisionError('Error: divided by Zero')
        return _SymbolWrapper(self.obj / other.obj)

    def __floordiv__(self, other):
        if self.obj is None: return _SymbolWrapper(None)
        if other.obj is None: raise ZeroDivisionError('Error: divided by Zero')
        return _SymbolWrapper(self.obj // other.obj)

    def __add__(self, other):
        if self.obj is None: return _SymbolWrapper(other.obj)
        if other.obj is None: return _SymbolWrapper(self.obj)
        return _SymbolWrapper(self.obj + other.obj)

    def __sub__(self, other):
        if self.obj is None: return _SymbolWrapper(None if other.obj is None else -other.obj)
        if other.obj is None: return _SymbolWrapper(self.obj)
        return _SymbolWrapper(self.obj - other.obj)


def _warped_mul_two_quaternions(qa: tuple, qb: tuple) -> tuple:
    """
    perform quaternion multiplication qa * qb
    e.g.
        (w, 0, y, 0) * (w, 0, 0, z)
        ==> _mul_quaternion((w, None, y, None), (w, None, None, z))
        where `None` stands for `zero` in the quaternion
    :param qa: quaternion a
    :param qb: quaternion b
    :return: qa * qb
    """
    if len(qa) != len(qb) or len(qa) != 4:
        raise ValueError(f"Length should be the same and equals to 4, but got qa={len(qa)} while qb={len(qb)}.")

    w1, x1, y1, z1 = tuple(list(_SymbolWrapper(e) for e in qa))
    w2, x2, y2, z2 = tuple(list(_SymbolWrapper(e) for e in qb))
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = x1*w2 + w1*x2 - z1*y2 + y1*z2
    y = y1*w2 + z1*x2 + w1*y2 - x1*z2
    z = z1*w2 - y1*x2 + x1*y2 + w1*z2
    return w.obj, x.obj, y.obj, z.obj


def euler_to_quaternion(eul: torch.Tensor, to_rad, order="ZYX", intrinsic=True) -> torch.Tensor:
    """
    euler rotation -> quaternion
    :param order: rotation order, default is "ZYX"
    :param to_rad: degree to radius (3.14159265 / 180.0)
    :param eul: [(B), J, 3, T] (rad)
    :param intrinsic: intrinsic or extrinsic rotation
    :return: [(B), J, 4, T]
    """
    batch, eul = (True, eul) if len(eul.shape) == 4 else (False, eul[None, ...])

    if len(eul.shape) != 4 or eul.shape[2] != 3:
        raise ValueError(f'Input tensor should be in the shape of BxJx3xF, but got {eul.shape}')

    if to_rad != 1.0:
        eul = eul * to_rad
    half_eul = eul * 0.5
    s = [torch.sin(half_eul[..., 0:1, :]),
         torch.sin(half_eul[..., 1:2, :]),
         torch.sin(half_eul[..., 2:3, :])]

    c = [torch.cos(half_eul[..., 0:1, :]),
         torch.cos(half_eul[..., 1:2, :]),
         torch.cos(half_eul[..., 2:3, :])]
    r = []
    for i, od in enumerate(order):
        if od == "X": r.append((c[i], s[i], None, None))
        if od == "Y": r.append((c[i], None, s[i], None))
        if od == "Z": r.append((c[i], None, None, s[i]))
    if len(r) != 3:
        raise ValueError(f'Error: Unknown order {order}')

    if intrinsic:
        w, x, y, z = _warped_mul_two_quaternions(r[0], _warped_mul_two_quaternions(r[1], r[2]))
    else:
        w, x, y, z = _warped_mul_two_quaternions(r[2], _warped_mul_two_quaternions(r[1], r[0]))

    ret = torch.cat((w, x, y, z), dim=2)
    return ret if batch else ret[0]


# def matrix_to_euler(mtx: torch.Tensor, fix_grad=False) -> torch.Tensor:
#     """
#     matrix -> euler
#     :param mtx: [(B), J, 3, 3, T]
#     :param fix_grad: fix gradient when using
#     :return: euler, [(B), J, 3, T]  (order=ZYX)
#     """
#     batch, mtx = (True, mtx) if len(mtx.shape) == 5 else (False, mtx[None, ...])
#
#     if len(mtx.shape) != 5 or mtx.shape[2] != 3 or mtx.shape[3] != 3:
#         raise ValueError(f'Input tensor should be in the shape of BxJx3x3xF, but got {mtx.shape}')
#
#     # reference: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
#
#     r11 = mtx[..., 0:1, 0, :]
#     r21 = mtx[..., 1:2, 0, :]
#     r31 = mtx[..., 2:3, 0, :]
#     r32 = mtx[..., 2:3, 1, :]
#     r33 = mtx[..., 2:3, 2, :]
#
#     if not mtx.requires_grad:
#         the1 = -torch.asin(torch.clip(r31, min=-1.0, max=1.0))
#     elif not fix_grad:
#         warnings.warn("Convert a matrix with grad to euler may produce INF gradient, please set `fix_grad=True`.")
#         the1 = -torch.asin(torch.clip(r31, min=-1.0, max=1.0))
#     else:
#         the1 = -torch.asin(torch.clip(r31, min=-0.9999, max=0.9999))
#
#     cos1 = torch.cos(the1)
#     # pai1 = torch.atan2((r32 / cos1), (r33 / cos1))
#     # phi1 = torch.atan2((r21 / cos1), (r11 / cos1))
#     # -- avoid division by zero
#     pai1 = torch.atan2((r32 * cos1), (r33 * cos1))
#     phi1 = torch.atan2((r21 * cos1), (r11 * cos1))
#
#     ret = torch.cat((phi1, the1, pai1), dim=2)
#     return ret if batch else ret[0]


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")

    frame_last = False
    if matrix.shape[-1] != 3 and (matrix.shape[-2] == 3 == matrix.shape[-3]):
        frame_last = True
        matrix = torch.einsum('...ijk->...kij', matrix)
    else:
        assert matrix.shape[-2] == 3, f"must be (..., 3, 3, T) or (..., 3, 3), not {matrix.shape}"

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    ret = torch.stack(o, -1)
    if frame_last:
        ret = torch.einsum('...ij->...ji', ret)
    return ret


def quaternion_to_euler(qua: torch.Tensor, order='XYZ', intrinsic=True) -> torch.Tensor:
    """
    quaternion to euler
    :param qua:  [(B), J, 4, T]
    :param order: rotation order of euler angles
    :param intrinsic: intrinsic rotation or extrinsic rotation
    :return: [(B), J, 3, T]
    """

    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        roll = atan2(2xw + 2yz, 1 - 2xx - 2yy)
        pitch = asin(2yw - 2xz)
        yaw = atan2(2zw + 2xy, 1 - 2yy - 2zz)
    """
    batch, qua = (True, qua) if len(qua.shape) == 4 else (False, qua[None, ...])

    if len(qua.shape) != 4 or qua.shape[2] != 4:
        raise ValueError('Input tensor must be in the shape of BxJx4xF.')

    # extrinsic to intrinsic
    if not intrinsic:
        # intrinsic = True
        order = order[::-1]

    w = qua[..., 0:1, :]
    x = qua[..., 1:2, :]
    y = qua[..., 2:3, :]
    z = qua[..., 3:4, :]

    if order == "XYZ":
        xx = 2 * x*x
        yy = 2 * y*y
        zz = 2 * z*z
        xy = 2 * x*y
        xz = 2 * x*z
        xw = 2 * x*w
        yz = 2 * y*z
        yw = 2 * y*w
        zw = 2 * z*w

        roll  = torch.atan2(xw + yz, 1 - xx - yy)
        pitch = torch.arcsin(torch.clip(yw - xz, min=-0.9999, max=0.9999))
        yaw   = torch.atan2(zw + xy, 1 - yy - zz)

        # first roll, then pitch, then yaw
        # yaw * pitch * roll * V
        ret = torch.cat((yaw, pitch, roll), dim=2)

    elif order == "YZX":
        xx, yy, zz, ww = x * x, y * y, z * z, w * w
        ex = torch.atan2(2 * (x * w - y * z), -xx + yy - zz + ww)
        ey = torch.atan2(2 * (y * w - x * z), xx - yy - zz + ww)
        ez = torch.asin(torch.clamp(2 * (x * y + z * w), min=-0.9999, max=0.9999))
        ret = torch.cat((ex, ez, ey), dim=2)
    else:
        raise NotImplementedError

    return ret if batch else ret[0]


def quaternion_to_euler_2(qua: torch.Tensor, order) -> torch.Tensor:
    """
    quaternion to euler
    :param qua:  [(B), J, 4, T]
    :param order: rotation order of euler angles
    :return: [(B), J, 3, T]
    """
    if len(qua.shape) != 4 or qua.shape[2] != 4:
        raise ValueError('Input tensor must be in the shape of BxJx4xF.')

    mtx = quaternion_to_matrix(qua)
    eul = matrix_to_euler(mtx, order)
    return eul


def normalize_quaternion(qua: torch.Tensor) -> torch.Tensor:
    """
    euler rotation -> quaternion
    :param qua: [(B), J, 4, T]
    :return: [(B), J, 4, T]
    """
    batch, qua = (True, qua) if len(qua.shape) == 4 else (False, qua[None, ...])

    if len(qua.shape) != 4 or qua.shape[2] != 4:
        raise ValueError('Input tensor should be in the shape of BxJx4xF.')

    ret = torch.nn.functional.normalize(qua, p=2.0, dim=2)
    return ret if batch else ret[0]

    # s = torch.norm(qua, dim=2, keepdim=True)
    # # s = torch.sqrt(torch.sum(qua**2, dim=2, keepdim=True))
    # s = torch.broadcast_to(s, qua.shape)
    # return torch.div(qua, s)


def quaternion_to_matrix(qua: torch.Tensor) -> torch.Tensor:
    """
    quaternion -> matrix
    :param qua: [(B), J, 4, T]
    :return: [(B), J, 3, 3, T]
    """
    batch, qua = (True, qua) if len(qua.shape) == 4 else (False, qua[None, ...])

    if len(qua.shape) != 4 or qua.shape[2] != 4:
        raise ValueError('Input tensor should be in the shape of BxJx4xF.')

    w = qua[..., 0:1, :][..., None, :]
    x = qua[..., 1:2, :][..., None, :]
    y = qua[..., 2:3, :][..., None, :]
    z = qua[..., 3:4, :][..., None, :]
    xx = 2 * x*x
    yy = 2 * y*y
    zz = 2 * z*z
    xy = 2 * x*y
    xz = 2 * x*z
    xw = 2 * x*w
    yz = 2 * y*z
    yw = 2 * y*w
    zw = 2 * z*w

    r11, r12, r13 = 1 - yy - zz,      xy - zw,      xz + yw
    r21, r22, r23 =     xy + zw,  1 - xx - zz,      yz - xw
    r31, r32, r33 =     xz - yw,      yz + xw,  1 - xx - yy

    r1 = torch.cat((r11, r12, r13), dim=3)
    r2 = torch.cat((r21, r22, r23), dim=3)
    r3 = torch.cat((r31, r32, r33), dim=3)
    ret = torch.cat((r1, r2, r3), dim=2)
    return ret if batch else ret[0]


def quaternion_from_two_vectors(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    :param v0: [..., 3, F], start
    :param v1: [..., 3, F], end
    :return:
    """
    # Quaternion q;
    # vector a = crossproduct(v1, v2);
    # q.xyz = a;
    # q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
    a = torch.cross(v0, v1, dim=-2)
    l0 = torch.norm(v0, dim=-2, keepdim=True)
    l1 = torch.norm(v1, dim=-2, keepdim=True)
    dot = torch.sum(v0 * v1, dim=-2, keepdim=True)
    w = l0*l1 + dot  # fix bug: since torch.norm <==> torch.sqrt(v ** 2) there is no need to use torch.sqrt any more
    qua = torch.cat([w, a], dim=-2)
    qua = torch.nn.functional.normalize(qua, p=2.0, dim=-2)
    return qua


def conjugate_quaternion(q) -> torch.Tensor:
    """
    :param q: [..., 4, F]
    :return: [..., 4, F]
    """
    qc = q.clone()
    qc[..., 1:, :] = -q[..., 1:, :]
    return qc


def norm_of_quaternion(q) -> torch.Tensor:
    """
    :param q: [..., 4, F]
    :return: [..., 1, F]
    """
    qn = torch.norm(q, dim=-2, keepdim=True)
    return qn


def inverse_quaternion(q) -> torch.Tensor:
    """
    :param q: [..., 4, F]
    :return: [..., 4, F]
    """
    return conjugate_quaternion(q) / norm_of_quaternion(q)


def mul_two_quaternions(q0, q1) -> torch.Tensor:
    """
    perform quaternion multiplication qa * qb
    :param q0: [..., 4, F], quaternion 0
    :param q1: [..., 4, F], quaternion 0
    :return: q0 * q1
    """
    qa = (q0[..., 0:1, :], q0[..., 1:2, :], q0[..., 2:3, :], q0[..., 3:4, :])
    qb = (q1[..., 0:1, :], q1[..., 1:2, :], q1[..., 2:3, :], q1[..., 3:4, :])
    return torch.cat(_warped_mul_two_quaternions(qa, qb), dim=-2)


def rectify_w_of_quaternion(qua: torch.Tensor, inplace=False) -> torch.Tensor:
    """
    quaternion[w < 0] --> quaternion[w < 0]
    :param qua: [(B), J, 4, T]
    :param inplace: inplace operator or not
    :return: [(B), J, 4, T]
    """
    batch, qua = (True, qua) if len(qua.shape) == 4 else (False, qua[None, ...])

    if len(qua.shape) != 4 or qua.shape[2] != 4:
        raise ValueError('Input tensor should be in the shape of [(B), J, 4, T].')

    w_lt = (qua[:, :, [0], :] < 0.0).expand(-1, -1, 4, -1)  # w less than 0.0
    w_ge = torch.logical_not(w_lt)  # w greater equal than 0.0

    if inplace:
        qua[w_lt] *= -1
    else:
        new = torch.empty_like(qua, dtype=qua.dtype, device=qua.device)
        new[w_ge] = qua[w_ge]
        new[w_lt] = qua[w_lt] * (-1)
        qua = new

    return qua if batch else qua[0]


def pad_position_to_quaternion(_xyz: torch.Tensor) -> torch.Tensor:
    """
    :param _xyz: [..., 3, (F)]
    :return:
    """
    no_frame = len(_xyz.shape) == 1
    if no_frame: _xyz = _xyz[:, None]
    assert _xyz.shape[-2] == 3, "input tensor shape should be [..., 3, (F)]"

    zero = torch.zeros_like(_xyz, dtype=_xyz.dtype, device=_xyz.device)[..., [0], :]
    wxyz = torch.cat([zero, _xyz], dim=-2)

    if no_frame: wxyz = wxyz[..., 0]

    return wxyz


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    :param d6: [..., 6, T]
    :return: [..., 3, 3, T]
    """
    a1, a2 = d6[..., :3, :], d6[..., 3:, :]

    # gram-schmidt
    b1 = torch.nn.functional.normalize(a1, dim=-2)
    b2 = a2 - (b1 * a2).sum(dim=-2, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-2)
    b3 = torch.cross(b1, b2, dim=-2)
    return torch.stack((b1, b2, b3), dim=-3)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    :param matrix: [..., 3, 3, T]
    :return: [..., 6, T]
    """
    return matrix[..., :2, :, :].clone().reshape(*matrix.size()[:-3], 6, -1)


# def matrix_to_quaternion(matrix):
#     """
#     Convert rotations given as rotation matrices to quaternions.
#
#     Args:
#         matrix: Rotation matrices as tensor of shape (..., 3, 3).
#
#     Returns:
#         quaternions with real part first, as tensor of shape (..., 4).
#     """
#
#     def _copy_sign(a, b):
#         """
#         Return a tensor where each element has the absolute value taken from the,
#         corresponding element of a, with sign taken from the corresponding
#         element of b. This is like the standard copysign floating-point operation,
#         but is not careful about negative 0 and NaN.
#
#         Args:
#             a: source tensor.
#             b: tensor whose signs will be used, of the same shape as a.
#
#         Returns:
#             Tensor of the same shape as a with the signs of b.
#         """
#         signs_differ = (a < 0) != (b < 0)
#         return torch.where(signs_differ, -a, a)
#
#     def _sqrt_positive_part(x):
#         """
#         Returns torch.sqrt(torch.max(0, x))
#         but with a zero sub-gradient where x is 0.
#         """
#         ret = torch.zeros_like(x)
#         positive_mask = x > 0
#         ret[positive_mask] = torch.sqrt(x[positive_mask])
#         return ret
#
#     if matrix.size(-1) != 3 or matrix.size(-2) != 3:
#         raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
#     m00 = matrix[..., 0, 0]
#     m11 = matrix[..., 1, 1]
#     m22 = matrix[..., 2, 2]
#     o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
#     x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
#     y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
#     z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
#     o1 = _copy_sign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
#     o2 = _copy_sign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
#     o3 = _copy_sign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
#     return torch.stack((o0, o1, o2, o3), -1)


# def simple_test():
#     v0 = torch.zeros((2, 3, 1))
#     v1 = torch.zeros((2, 3, 1))
#     v0[..., 0, :] = 1.0
#     v1[..., 1, :] = 1.0
#     print(quaternion_from_two_vectors(v0, v1))
#
#     v0 = torch.zeros((1, 2, 3, 1))
#     v1 = torch.zeros((1, 2, 3, 1))
#     v0[..., 1, :] = 1.0
#     v1[..., 0, :] = 1.0
#     print(quaternion_from_two_vectors(v0, v1))
#
#
# if __name__ == '__main__':
#     simple_test()
