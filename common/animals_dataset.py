import numpy as np
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset

# AP-10K based quadruped animal skeleton
quadruped_skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15],
    joints_left=[5, 6, 7, 8, 13, 14, 15, 16],  # All left side joints
    joints_right=[1, 2, 3, 4, 9, 10, 11, 12]  # All right side joints
)

class AnimalsDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=False):
        super().__init__(fps=30, skeleton=quadruped_skeleton)

        # 加载数据
        data = np.load(path, allow_pickle=True)
        if 'positions_3d' in data:
            data = data['positions_3d'].item()
        else:
            data = data.item() if hasattr(data, 'item') else data

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions.astype('float32')
                }


    def supports_semi_supervised(self):
        return True

    def __getitem__(self, key):
        return self._data[key]

    def subjects(self):
        return list(self._data.keys())

    def fps(self):
        return self._fps

    def skeleton(self):
        return self._skeleton
