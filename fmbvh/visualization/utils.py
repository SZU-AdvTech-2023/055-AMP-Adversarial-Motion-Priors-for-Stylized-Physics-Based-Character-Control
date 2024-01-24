from motion_tensor.rotations import quaternion_to_matrix as q2m
from motion_tensor.kinematics import forward_kinematics as fk
from visualization.visualize_motion import MoVisualizer


def quick_visualize_fk(p_index, off, qua, scale=200.0):
    if len(qua.shape) == 4:
        off = off[0]
        qua = qua[0]

    if len(qua.shape) == 3:
        assert len(off.shape) == 3
        if qua.shape[0] == len(p_index) and qua.shape[1] == 4:  # [J, 4, T]
            pass
        else:
            qua = qua[0]
            qua = qua.view(qua.shape[0], 4, qua.shape[-1])
    else:
        raise ValueError(f"incorrect shape: {qua.shape}")

    off = off.detach().cpu()
    qua = qua.detach().cpu()
    mat = q2m(qua)
    pos = fk(p_index, mat, None, off, True, False)

    def _next():
        f = 0
        while True:
            yield pos[..., f].tolist()
            f = (f + 1) % pos.shape[-1]

    mvz = MoVisualizer(p_index, _next(), scale=scale)
    mvz.run()


def quick_visualize(p_index, pos, scale=200.0, callback_fn=None):
    """
    pos: [J, 3, F]
    """
    if len(pos.shape) == 4:
        pos = pos[0]

    assert pos.shape[1] == 3
    pos = pos.detach().cpu()

    def _next():
        f = 0
        while True:
            yield pos[..., f].tolist()
            f = (f + 1) % pos.shape[-1]
            if callback_fn is not None:
                callback_fn(f)

    mvz = MoVisualizer(p_index, _next(), scale=scale)
    mvz.run()
