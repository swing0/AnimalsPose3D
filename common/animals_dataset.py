import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates

# AP-10K based quadruped animal skeleton
quadruped_skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15],
    joints_left=[5, 6, 7, 8, 13, 14, 15, 16],  # All left side joints
    joints_right=[1, 2, 3, 4, 9, 10, 11, 12]  # All right side joints
)

# 虚拟相机参数 - 为四足动物数据集创建
animal_cameras_intrinsic_params = [
    {
        'id': 'camera_01',
        'center': [640.0, 360.0],  # 假设1280x720分辨率
        'focal_length': [1000.0, 1000.0],
        'radial_distortion': [0.0, 0.0, 0.0],
        'tangential_distortion': [0.0, 0.0],
        'res_w': 1280,
        'res_h': 720,
        'azimuth': 45,
    },
    {
        'id': 'camera_02',
        'center': [640.0, 360.0],
        'focal_length': [1000.0, 1000.0],
        'radial_distortion': [0.0, 0.0, 0.0],
        'tangential_distortion': [0.0, 0.0],
        'res_w': 1280,
        'res_h': 720,
        'azimuth': -45,
    },
    {
        'id': 'camera_03',
        'center': [640.0, 360.0],
        'focal_length': [1000.0, 1000.0],
        'radial_distortion': [0.0, 0.0, 0.0],
        'tangential_distortion': [0.0, 0.0],
        'res_w': 1280,
        'res_h': 720,
        'azimuth': 135,
    },
    {
        'id': 'camera_04',
        'center': [640.0, 360.0],
        'focal_length': [1000.0, 1000.0],
        'radial_distortion': [0.0, 0.0, 0.0],
        'tangential_distortion': [0.0, 0.0],
        'res_w': 1280,
        'res_h': 720,
        'azimuth': -135,
    },
]

# 虚拟外参参数 - 为所有动物创建统一的相机设置
animal_cameras_extrinsic_params = {
    'default': [
        {
            'orientation': [0.707, 0.0, 0.0, 0.707],  # 四元数表示
            'translation': [0.0, 0.0, 5.0],  # 相机位置
        },
        {
            'orientation': [0.5, 0.5, 0.5, 0.5],
            'translation': [5.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.707, 0.707, 0.0],
            'translation': [0.0, 0.0, -5.0],
        },
        {
            'orientation': [0.5, -0.5, -0.5, 0.5],
            'translation': [-5.0, 0.0, 0.0],
        },
    ]
}


class AnimalsDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=False):
        super().__init__(fps=30, skeleton=quadruped_skeleton)  # 假设30fps

        # 设置相机参数
        self._cameras = copy.deepcopy(animal_cameras_extrinsic_params)
        for subject, cameras in self._cameras.items():
            for i, cam in enumerate(cameras):
                cam.update(animal_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # 归一化相机坐标系
                cam['center'] = normalize_screen_coordinates(
                    cam['center'], w=cam['res_w'], h=cam['res_h']
                ).astype('float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2

                # 添加内参向量
                cam['intrinsic'] = np.concatenate((
                    cam['focal_length'],
                    cam['center'],
                    cam['radial_distortion'],
                    cam['tangential_distortion']
                ))

        # 加载序列化数据集
        data = np.load(path, allow_pickle=True)
        if 'positions_3d' in data:
            data = data['positions_3d'].item()
        else:
            # 如果直接是positions_3d字典
            data = data.item() if hasattr(data, 'item') else data

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                # 为所有动物使用相同的相机设置
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras['default'],  # 使用默认相机
                }

        # 如果需要移除静态关节（当前四足动物骨架已经是17关节，不需要进一步简化）
        if remove_static_joints:
            # 这里可以根据需要移除静态关节
            # 当前四足动物骨架已经是最简形式
            pass

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