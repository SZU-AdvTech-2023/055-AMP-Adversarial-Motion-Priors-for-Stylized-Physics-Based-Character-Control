"""
gather all the *.bvh files and load them into memory as BVH objects
"""

import os
import fmbvh.bvh as bvh
import glob


class BVHSubFolder:
    def __init__(self, bvh_file_folder: str=""):
        self.file_list = []
        self.bvh_cache = {}

        if bvh_file_folder != "":
            self.create_dataset(bvh_file_folder)

    def create_dataset(self, bvh_file_folder: str) -> None:
        self.file_list = []
        import os.path as path
        for file in glob.iglob(path.join(bvh_file_folder, "*.bvh"), recursive=False):
            self.file_list.append(file)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item) -> bvh.parser.BVH:
        filename = self.file_list[item]  # may raise an IndexError
        if filename not in self.bvh_cache:
            bvh_obj = bvh.parser.BVH(self.file_list[item])
            self.bvh_cache[filename] = bvh_obj
        return self.bvh_cache[filename]


# Load BVH files from multiple folders with index, sorted by filename
class BVHFolder:
    def __init__(self, bvh_file_folder=""):
        self.dataset_list = []

        if bvh_file_folder != "":
            self.create_dataset(bvh_file_folder)

    def __len__(self):
        return sum([len(e) for e in self.dataset_list])

    def __getitem__(self, item: int) -> [bvh.parser.BVH, int]:
        cls = 0
        for ds in self.dataset_list:
            if item >= len(ds):
                item -= len(ds)
                cls += 1
            else:
                obj = ds[item]
                return cls, obj
        raise IndexError

    def __iter__(self):
        for cls, sub in enumerate(self.dataset_list):
            for obj in sub:
                yield cls, obj

    def create_dataset(self, bvh_file_folder: str, name_ascending=True) -> None:
        for file in sorted(os.listdir(bvh_file_folder), reverse=not name_ascending):
            subdir = os.path.join(bvh_file_folder, file)
            if os.path.isdir(subdir):
                inst_dataset = BVHSubFolder(subdir)
                self.dataset_list.append(inst_dataset)


def test():
    folder = BVHFolder(r"D:\_dataset\bvh_test")
    print(len(folder))
    for cls, obj in folder:
        print(cls, obj.filepath)


if __name__ == '__main__':
    test()
