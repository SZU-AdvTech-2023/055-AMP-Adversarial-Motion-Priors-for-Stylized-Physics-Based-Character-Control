import bvh
import visualization as vis
from typing import Iterator
import time


class MoVisualizer:
    def __init__(self, p_index: list, mo_source: Iterator[list], max_fps=60.0, add_coordinate=True, scale=1.0):
        """
        :param p_index:  a list of parent index
        :param mo_source: a iterator that yields position for next frame, end with `None`
                          yields a [N, 3] list
        :param max_fps:   maximum fps
        """
        self.rnd = vis.renderer.O3DRenderer()
        self.p_index = p_index
        self.mo_source = mo_source
        self.max_fps = max_fps

        # add coordinate
        if add_coordinate:
            coord_p = [[-1, 0, 0], [+1, 0, 0], [0, 0, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]]
            coord_l = [[0, 1], [2, 3], [4, 5]]
            coord_p = [[e*scale for e in v] for v in coord_p]
            self.rnd.add_lines('__default__$coord', coord_p, coord_l)

        # add skeleton
        self.rnd.add_lines('motion',
                           [[0, 0, 0] for _ in range(len(p_index))],
                           [[i, p] if p != -1 else [i, i] for i, p in enumerate(p_index)])
        self.rnd.set_animation_callback(self.__update)

    def add_grids(self, m: int, n: int, scale=1.0) -> None:
        """
        add mxn grids (y = 0)
        :param m: m grids along z-axis
        :param n: n grids along x-axis
        :param scale: scale of object(s)
        :return: None
        """
        grids_p = []
        grids_l = []
        for z in range(m+1):
            for x in range(n+1):
                grids_p.append([x - (n / 2), 0, z - (m / 2)])
        for z in range(m+1):
            for x in range(0, n):
                grids_l.append([x+z*(n+1), x+1+z*(n+1)])
            if z == m:
                break
            for x in range(0, n+1):
                grids_l.append([x+z*(n+1), x+(z+1)*(n+1)])

        grids_p = [[e*scale for e in v] for v in grids_p]
        self.rnd.add_lines('__default__$grids', grids_p, grids_l)

    def run(self):
        self.rnd.show_window()

    def __update(self, _):
        st = time.time()
        points = next(self.mo_source, None)
        if points is not None:
            self.rnd.set_lines('motion', points)
        delta = time.time() - st
        rest = (1.0 / self.max_fps) - delta
        if rest > 0:
            time.sleep(rest)


def demo():
    bvh_obj = bvh.parser.BVH('../data/assets/test.bvh')

    import motion_tensor as mot
    import torch

    trs, qua = mot.bvh_casting.get_quaternion_from_bvh(bvh_obj)  # [1, 3, F], [J, 3, F]
    mat = mot.rotations.quaternion_to_matrix(qua[None, ...])  # [B, J, 3, 3, F]
    J, _, F = qua.shape

    offsets = mot.bvh_casting.get_offsets_from_bvh(bvh_obj)[None, ...]  # [B, J, 3, 1]
    offsets = torch.broadcast_to(offsets, (1, J, 3, F))  # [B, J, 3, F]

    fk_pos = mot.kinematics.forward_kinematics(bvh_obj.dfs_parent(), mat,
                                               trs[None, ...], offsets)    # [B, J, 3, F]

    t_pos = mot.bvh_casting.get_positions_from_bvh(bvh_obj)[..., 0]  # [J, 3]
    max_y = torch.max(t_pos[:, 1])
    min_y = torch.min(t_pos[:, 1])
    height = (max_y - min_y).item()

    def _next():
        f = 0
        while True:
            rpt = trs[:, :, f]  # [1, 3]
            cps = fk_pos[0, :, :, f]  # [J, 3]
            pos_ls = cps + rpt  # [J, 3]
            yield pos_ls.numpy().tolist()
            f = (f + 1) % F

    p_index = bvh_obj.dfs_parent()
    mvz = MoVisualizer(p_index, _next(), scale=height*2, max_fps=60)
    mvz.add_grids(10, 10, height*0.2)
    mvz.run()


if __name__ == '__main__':
    demo()

