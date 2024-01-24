import open3d as o3d


class _CallbackContainer:
    def __init__(self, fun, renderer):
        self.fun = fun
        self.rnd = renderer

    def __call__(self, *args, **kwargs):
        self.fun(self.rnd)


class O3DRenderer:
    def __init__(self, left=400, top=300, width=800, height=600):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height, left=left, top=top)
        self.geometries_set = {}

    def create_window(self, left=400, top=300, width=800, height=600):
        self.vis.destroy_window()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height, left=left, top=top)
        self.geometries_set = {}

    def show_window(self) -> None:
        """
        run until window closed
        :return: None
        """
        self.vis.run()
        self.vis.destroy_window()

    def set_keyboard_callback(self, key: str, fun: callable) -> None:
        """
        callback function
        :param key: 'a' ~ 'z' or 'A' ~ 'Z'
        :param fun: fun = def your_callback_function(obj: O3DRenderer)
        :return: None
        """
        self.vis.register_key_callback(ord(key.upper()[0]), _CallbackContainer(fun, self))

    def set_animation_callback(self, fun: callable) -> None:
        """
        set animation callback function
        :param fun: fun = def your_callback_function(obj: O3DRenderer)
        :return: None
        """
        self.vis.register_animation_callback(_CallbackContainer(fun, self))

    def is_name_exists(self, name):
        return name in self.geometries_set

    def add_lines(self, name, points, lines, colors=None) -> None:
        """
        add new lines with a unique name
        :param name: geometry name
        :param points: Nx3 list for N points
        :param lines:  Kx2 list for K lines, index starts from 0
        :param colors: Kx3 list for K lines, range from 0 ~ 1
        :return: None
        """
        if name in self.geometries_set:
            raise KeyError(f'Key "{name}" already exists!')

        if colors is None:
            colors = [[0, 0, 0] for _ in range(len(lines))]

        # range check
        assert len(lines) == len(colors), f"size not match for lines: {len(lines)} and colors: {len(colors)}"
        N = len(points)
        for p in lines:
            for e in p:
                assert e < N, f"index out of range: {e} > {N - 1}"

        point_set = o3d.geometry.PointCloud()
        point_set.points = o3d.utility.Vector3dVector(points)
        point_set.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in range(N)])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(line_set)
        self.vis.add_geometry(point_set)

        geo_dic = {
            'lines': line_set,
            'points': point_set,
            'N': len(points),
            'K': len(lines)}
        self.geometries_set[name] = geo_dic

    def set_lines(self, name, points=None, lines=None, colors=None) -> None:
        """
        set lines with the given name
        :param name: geometry name
        :param points: points
        :param lines:
        :param colors:
        :return:
        """
        geo_dic = self.geometries_set[name]
        line_set = geo_dic['lines']
        point_set = geo_dic['points']

        if points is not None:
            line_set.points = o3d.utility.Vector3dVector(points)
            point_set.points = o3d.utility.Vector3dVector(points)
            geo_dic['N'] = len(points)

        if lines is not None:
            N = geo_dic['N']
            for p in lines:
                for e in p:
                    assert e < N, f"index out of range: {e} > {N-1}"

            line_set.lines = o3d.utility.Vector2iVector(lines)
            geo_dic['K'] = len(lines)

        if colors is not None:
            K = geo_dic['K']
            assert K == len(colors), f"size not match for lines: {K} and colors: {len(colors)}"
            line_set.colors = o3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(line_set)
        self.vis.update_geometry(point_set)
        self.vis.poll_events()
        self.vis.update_renderer()

        self.geometries_set[name] = geo_dic


def demo():
    def fun_a(rnd: O3DRenderer):
        rnd.set_lines('lines', [[1, 2, 3],
                                [1, 4, 3],
                                [2, 1, 3],
                                [1, 1, 0]],
                      [[0, 1], [1, 2], [2, 3]])

    def fun_b(rnd: O3DRenderer):
        # ERROR: out of range: 4
        rnd.set_lines('lines', [[1, 2, 3],
                                [1, 4, 3],
                                [2, 1, 3],
                                [1, 1, 0]],
                      [[1, 2], [2, 3], [3, 4]])

    def fun_ani(rnd: O3DRenderer):
        # ERROR: press `A` can cause a size not match error
        import time
        time.sleep(1 / 60.0)  # 60FPS
        colors = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0.5, 0.5, 0]]
        import random
        cl = [random.choice(colors) for _ in range(4)]
        rnd.set_lines('lines', colors=cl)

    renderer = O3DRenderer()
    renderer.add_lines('lines', [[1, 2, 3],
                                 [1, 4, 3],
                                 [2, 1, 3],
                                 [1, 1, 0]],
                      [[0, 1], [1, 2], [2, 3], [3, 0]])
    renderer.set_keyboard_callback('A', fun_a)
    renderer.set_keyboard_callback('B', fun_b)
    renderer.set_animation_callback(fun_ani)
    renderer.show_window()


if __name__ == '__main__':
    demo()
