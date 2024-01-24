"""
@author: zrk
@desc: this script provides tools for creating and loading motion dataset
"""

import fmbvh.bvh as bvh
import os
import json
import torch

from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Type

import fmbvh.motion_tensor.motion_process as mop
import fmbvh.motion_tensor.bvh_casting as casting


class BVHDataExtractor:
    """
        This class decides what kinds of data to store on disk.
        `extract` function receive a `BVH` object and return user-defined features (static and dynamic).
        These features will be processed again latter.
    """
    def __init__(self, desired_frame_time=1/60.0):
        """
        :param desired_frame_time: e.g. 0.03333 - 30FPS, 0.08333 - 120FPS
        """
        self.desired_frame_time = desired_frame_time

    def scale(self, motion: torch.Tensor, frame_time) -> torch.Tensor:
        scale_factor = frame_time / self.desired_frame_time
        sampler = 'nearest'
        motion = mop.sample_frames(motion, scale_factor=scale_factor, sampler=sampler)
        return motion

    # noinspection PyMethodMayBeStatic
    def extract(self, bvh_obj: bvh.parser.BVH) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        :param bvh_obj:
        :return: (static features), (dynamic features), frame_time
                 where for each s in (static features),  s is a [J, C, 1] tensor
                       for each d in (dynamic features), d is a [J, C, T] tensor
                 NOTE: call .scale(dynamic features[i]) before returning.
        """
        trs_, qua_ = casting.get_quaternion_from_bvh(bvh_obj)
        off_ = casting.get_offsets_from_bvh(bvh_obj)
        tps_ = casting.get_t_pose_from_bvh(bvh_obj)

        trs_ = self.scale(trs_, bvh_obj.frame_time)
        qua_ = self.scale(qua_, bvh_obj.frame_time)
        return (off_, tps_), (trs_, qua_)


class MotionDataDivider:
    def __init__(self, window, window_step, skip):
        """
        :param window:   motion clip length (T)
        :param window_step:  motion clip step size (overlapping = window - window_step)
        :param skip:     skip some heading frames
        """
        self.window = window
        self.window_step = window_step
        self.skip = skip

    def divide(self, motion: torch.Tensor) -> List[torch.Tensor]:
        K, S, W = self.skip, self.window_step, self.window
        motion = motion[..., K:]
        clip_list = []
        total = motion.shape[-1]

        for j in range(0, total, S):
            mo_clip = motion[..., j: j + W].clone()
            if mo_clip.shape[-1] < W:
                break
            clip_list.append(mo_clip)

        return clip_list


def make_mo_clip_dataset(bvh_file_folder, cache_file_folder,
                         data_divider: MotionDataDivider=None,
                         bvh_extractor: BVHDataExtractor=None):
    """
    split every *.bvh files into motion clips, and save them to:
        *.mo_clip           // [pickle] a list of N tuples of two tensors  (note: one mo_clip file per bvh file)
        mo_clip_v2.meta    // [json] a list of dicts for recording (unique id, class id, file id, number of clips)

    :param bvh_file_folder:     root/class_1/*.bvh, root/class_2/*.bvh, ...
    :param cache_file_folder:  out/(unique id)_(class id)_(file id).mo_clip.pkl; out/mo_clip_v2.meta
    :param data_divider: a divider that divides a long motion to short clips
    :param bvh_extractor: a class factory that used to extract features from a motion clip
    :return: None
    """
    if os.path.isfile(os.path.join(cache_file_folder, 'mo_clip_v2.meta')):
        print(f"[INFO] motion meta file already exists: {os.path.join(cache_file_folder, 'mo_clip_v2.meta')}")
        return

    if bvh_extractor is None:
        bvh_extractor = BVHDataExtractor()
    else:
        assert isinstance(bvh_extractor, BVHDataExtractor)

    dataset = bvh.folder.BVHFolder(bvh_file_folder)
    os.makedirs(cache_file_folder, exist_ok=True)

    meta = []
    file_id = 0
    last_class_id = 0

    for class_id, obj in dataset:
        if obj.frames == 0:
            print(f'[WARNING] frame count of bvh file {obj.filepath} is 0!')
            continue
        else:
            static, dynamic = bvh_extractor.extract(obj)

        # ---- file ids ---- #
        if last_class_id != class_id:  # check if we are stepping into a new class
            last_class_id = class_id
            file_id = 0

        # ---- static feature ---- #
        sta_list = list(static)

        # sta_list:
        #   [feature 1, feature 2, feature 3, ...]

        # ---- dynamic feature ---- #
        dyn_list = []
        for dyn in dynamic:
            if data_divider is None:
                dyn_list.append([dyn])
            else:
                mo_list = data_divider.divide(dyn)
                dyn_list.append(mo_list)

        # dyn_list:
        #   [[f1, f1, ...], [f2, f2, ...], ...]
        # ==>
        #   [(f1, f2, ...), (f1, f2, ...), ...]
        #
        dyn_list = [e for e in zip(*dyn_list)]
        meta.append({
            'class_id': class_id,
            'file_id': file_id,
            'num_clip': len(dyn_list)
        })

        filepath = os.path.join(cache_file_folder, f'{class_id:03d}_{file_id:03d}.mo_clip.pkl')
        torch.save({
            "static": sta_list,
            "dynamic": dyn_list
        }, filepath)

        file_id += 1

    with open(os.path.join(cache_file_folder, 'mo_clip_v2.meta'), 'w') as meta_f:
        json.dump(meta, meta_f, indent=4, sort_keys=True)


class MoClipProcessor(ABC):
    """
    the processed results will be stored in memory to speed up data loading
    """
    @abstractmethod
    def f_process_static(self, class_id, *args) -> tuple:
        """
        please refer to f_process_dynamic
        :param args:
        :param class_id:
        :return: by default it returns [offset, position], refer to BVHDataExtractor.extract
        """
        raise NotImplementedError

    @abstractmethod
    def f_process_dynamic(self, class_id, *args) -> tuple:
        """
        this function is used to extract useful features of a motion clip
        e.g.
            a simple process is the identity mapping:
                f_process = lambda x: x

        :param args:  dynamic features
        :param class_id:     please refer to make_mo_clip_dataset
        :return: by default it returns [root translation, joint rotations(quaternion)]
                 refer to BVHDataExtractor.extract
        """
        raise NotImplementedError


class MoClipDataset:
    """
    This class processes the extracted features that can be
    directly fed into a neural network for training.

    self._cached_data:
        a list of tuples/Tensors that caches all motion clips in memory:
            tuple(f_process_static(...)) + tuple(f_process_dynamic(...)) + (class_id, )
            ...

    self._class_ids:
        a dict of (int, list) that records all clip ids of the same class

    """
    def __init__(self, cache_file_folder, meta, processor: MoClipProcessor=None, enable_lazy_loading=True):
        self._meta = meta
        self._path = cache_file_folder
        self._cached_data = []
        self._class_ids = {}
        self._item_to_file = []
        self._total_clips = 0
        self._loaded_clips = 0
        self._processor = processor

        iter_ = range(len(self._meta))
        for file_item in iter_:
            file_id   = self._meta[file_item]['file_id']
            class_id  = self._meta[file_item]['class_id']
            num_clip  = self._meta[file_item]['num_clip']
            path = os.path.join(self._path, f'{class_id:03d}_{file_id:03d}.mo_clip.pkl')

            if class_id not in self._class_ids:
                self._class_ids[class_id] = []

            for _ in range(num_clip):
                cur_clip_id = len(self._cached_data)
                # self._cached_raw.append(sta + dyn + (class_id,))
                self._cached_data.append((class_id,))  # load the class_id only
                self._class_ids[class_id].append(cur_clip_id)
                self._item_to_file.append(path)
                self._total_clips += 1

        if not enable_lazy_loading:
            for i in range(self._total_clips):
                self._lazy_load(i)  # force load into memory

    # noinspection PyMethodMayBeStatic
    def _before_load_to_cache_memory(self, static: tuple, dynamic: tuple, class_id: int) -> tuple:
        """
        determines what data should be returned.
        this is useful for applying normalization on data or not
        :param static:
        :param dynamic:
        :param class_id:
        :return:
        """
        return static + dynamic + (class_id, )

    def _lazy_load(self, item: int):
        """
        if cache missed then load and add the CURRENT data and ALL NEARBY data (same pickle file) to cache memory.
        :param item: item index to fetch
        :return:
        """
        if self._loaded_clips == self._total_clips:
            return self._cached_data[item]

        path = self._item_to_file[item]
        if path is None:
            return self._cached_data[item]
        else:
            mo_data = torch.load(path)
            static, dynamic = mo_data['static'], mo_data['dynamic']

            class_id, = self._cached_data[item]
            sta: tuple = self._processor.f_process_static(class_id, *static)
            if not isinstance(sta, tuple) and not isinstance(sta, list):
                sta = (sta,)

            # each clip
            ret_ls = []
            for dyn in dynamic:
                dyn: tuple = self._processor.f_process_dynamic(class_id, *dyn)
                if not isinstance(dyn, tuple) and not isinstance(dyn, list):
                    dyn = (dyn,)

                ret = (sta, dyn, class_id, )  # (static features, ) + (dynamic features, ) + (class_id, )
                ret_ls.append(ret)

            up_i = item - 1
            dw_i = item + 1
            while up_i >= 0 and self._item_to_file[up_i] == path:
                up_i -= 1
            while dw_i < self._total_clips and self._item_to_file[dw_i] == path:
                dw_i += 1

            for i in range(up_i+1, dw_i):
                # noinspection PyTypeChecker
                self._item_to_file[i] = None
                self._cached_data[i] = self._before_load_to_cache_memory(*ret_ls[i - up_i - 1])
                self._loaded_clips += 1

            return self._cached_data[item]

    def __len__(self):
        return len(self._cached_data)

    def __getitem__(self, item):
        """
        :param item:
        :return: self._before_save_to_cache_memory(static features, dynamic features, class_id)
        """
        return self._lazy_load(item)

    @property
    def total_clips(self):
        return self._total_clips


def load_mo_clip_dataset(cache_file_folder, processor: MoClipProcessor,
                         cls: Type[MoClipDataset] = MoClipDataset, enable_lazy_loading=True) -> Any:
    """
    load motion clip dataset
    :param cache_file_folder: see make_mo_clip_dataset (..., output_data_folder, ...)
    :param cls:
    :param processor:
    :param enable_lazy_loading:
    :return: a torch dataset
    """
    with open(os.path.join(cache_file_folder, f'mo_clip_v2.meta'), 'r') as meta_f:
        meta = json.load(meta_f)
        cls = cls(cache_file_folder, meta, processor, enable_lazy_loading)
    return cls


class MoStatisticCollector(ABC):
    """
    This collector decides what statistics to use, e.g. mean and var.
    """
    @abstractmethod
    def get_stat(self, class_id: int, feature: List[tuple]) -> Any:
        """
        compute statistics from a list of feature, e.g. mean and var.
        :param class_id:
        :param feature: list of features return from f_process_static & f_process_dynamic
        :return:
        """
        pass

    @abstractmethod
    def get_stat_all(self, feature: List[tuple]) -> Any:
        pass


def gather_statistic(bvh_file_folder, cache_file_folder,
                     extractor: BVHDataExtractor,
                     processor: MoClipProcessor,
                     collector: MoStatisticCollector) -> dict:
    """
    gather a statistic array of [(static features, dynamic features), ...] and send it to collector
    :param bvh_file_folder:     root/class_1/*.bvh, root/class_2/*.bvh, ...
    :param cache_file_folder:   where to store the precomputed results
    :param extractor: a class factory that used to extract features from a motion clip
    :param processor:
    :param collector:
    :return: dict {
                "per": {
                        cid: get_stat(cid, feature),
                        cid: get_stat(cid, feature),
                        ...
                },
                "all": get_stat_all(feature)
             }
    """

    if extractor is None:
        extractor = BVHDataExtractor()
    else:
        assert isinstance(extractor, BVHDataExtractor)

    dataset = bvh.folder.BVHFolder(bvh_file_folder)

    def _gather_per_class():
        file_id = 0
        last_class_id = 0
        stat_list = []

        for class_id, obj in dataset:
            if last_class_id != class_id:  # check if we are stepping into a new class
                yield last_class_id, stat_list

                stat_list = []
                last_class_id = class_id
                file_id = 0

            if obj.frames == 0:
                print(f'[WARNING] frame count of bvh file {obj.filepath} is 0!')
                continue
            else:
                static, dynamic = extractor.extract(obj)
            static = processor.f_process_static(class_id, *static)
            dynamic = processor.f_process_dynamic(class_id, *dynamic)
            stat_list.append(static + dynamic)

            file_id += 1

        yield last_class_id, stat_list  # yield last class

    os.makedirs(cache_file_folder, exist_ok=True)
    cache_file = os.path.join(cache_file_folder, 'stat.pkl')

    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as f:
            dic: dict = torch.load(f)
        print(f"[INFO] statistic file already exists: {cache_file}")
        return dic

    dic = {}
    dic_per = {}
    stat_all = []
    for cid, stat in _gather_per_class():
        dic_per[cid] = collector.get_stat(cid, stat)
        stat_all += stat
    lis_all = collector.get_stat_all(stat_all)
    dic['per'] = dic_per
    dic['all'] = lis_all
    with open(cache_file, 'wb') as f:
        torch.save(dic, f)
    return dic


def test_make_and_load():

    class _MyExtractor(BVHDataExtractor):
        def __init__(self):
            super(_MyExtractor, self).__init__(desired_frame_time=1/30.0)

        def extract(self, bvh_obj: bvh.parser.BVH) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
            (off_, tps_), (trs_, qua_) = super(_MyExtractor, self).extract(bvh_obj)
            p_index = bvh_obj.dfs_parent()
            pos_ = casting.get_positions_from_bvh(bvh_obj)
            pos_ = self.scale(pos_, frame_time=bvh_obj.frame_time)
            return (off_, tps_, p_index), (trs_, qua_, pos_)

    divider = MotionDataDivider(64, 64, 0)
    extractor = _MyExtractor()
    make_mo_clip_dataset(r"D:\_dataset\bvh_test", r"D:\_dataset\bvh_test_pickle", divider, extractor)

    class _MyProcessor(MoClipProcessor):

        def f_process_static(self, class_id, *args) -> tuple:
            (off_, tps_, p_index) = args
            return off_, tps_, p_index

        def f_process_dynamic(self, class_id, *args) -> tuple:
            (trs_, qua_, pos_) = args
            return trs_, qua_, pos_

    processor = _MyProcessor()
    dataset = load_mo_clip_dataset(r"D:\_dataset\bvh_test_pickle", processor)

    def run_vis():
        from visualization.visualize_motion import MoVisualizer
        for (off_, tps_, p_index, trs_, qua_, pos_, cid) in dataset:
            def _next():
                f = 0
                while True:
                    yield pos_[..., f].tolist()
                    f = (f + 1) % pos_.shape[-1]
            mvz = MoVisualizer(p_index, _next(), scale=200.0)
            mvz.run()

    run_vis()


def test_mean_and_var():
    class _MyExtractor(BVHDataExtractor):
        def __init__(self):
            super(_MyExtractor, self).__init__(desired_frame_time=1 / 30.0)

        def extract(self, bvh_obj: bvh.parser.BVH) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
            (off_, tps_), (trs_, qua_) = super(_MyExtractor, self).extract(bvh_obj)
            p_index = bvh_obj.dfs_parent()
            pos_ = casting.get_positions_from_bvh(bvh_obj)
            pos_ = self.scale(pos_, frame_time=bvh_obj.frame_time)
            return (off_, tps_, p_index), (trs_, qua_, pos_)

    extractor = _MyExtractor()

    class _MyProcessor(MoClipProcessor):

        def f_process_static(self, class_id, *args) -> tuple:
            (off_, tps_, p_index) = args
            return off_, tps_, p_index

        def f_process_dynamic(self, class_id, *args) -> tuple:
            (trs_, qua_, pos_) = args
            return trs_, qua_, pos_

    processor = _MyProcessor()

    class _MyCollector(MoStatisticCollector):

        def get_stat(self, class_id: int, feature: List[tuple]) -> Any:
            # for off_, tps_, p_index, trs_, qua_, pos_ in feature:
            p_index = feature[0][2]
            pos_ls = [e[5] for e in feature]
            m = torch.mean(torch.concat(pos_ls, dim=-1), dim=(-1))
            v = torch.var(torch.concat(pos_ls, dim=-1), dim=(-1))

            return m, v, p_index

    collector = _MyCollector()
    stat_dic = gather_statistic(r"D:\_dataset\bvh_test", r"D:\_dataset\bvh_test_pickle",
                                extractor, processor, collector)

    def run_vis():
        from visualization.visualize_motion import MoVisualizer
        for m, v, p_index in stat_dic.values():
            pos = torch.zeros(m.shape[0], m.shape[1], 100)
            for i in range(100):
                pos[..., i] = v * (i/99 - 0.5) + m

            def _next():
                f = 0
                while True:
                    yield pos[..., f].tolist()
                    f = (f + 1) % pos.shape[-1]

            mvz = MoVisualizer(p_index, _next(), scale=200.0)
            mvz.run()

    run_vis()


if __name__ == '__main__':
    test_make_and_load()
    test_mean_and_var()
