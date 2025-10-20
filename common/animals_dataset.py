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

# 基于数据分析优化的虚拟相机参数
# 数据最大跨度: 12.438，建议相机距离: 18.657
animal_cameras_intrinsic_params = [
    {
        'id': 'camera_01_front_right',
        'center': [960.0, 540.0],  # 1920x1080分辨率
        'focal_length': [800.0, 800.0],  # 较短焦距适应大尺度数据
        'radial_distortion': [-0.15, 0.20, 0.001],
        'tangential_distortion': [0.001, -0.001],
        'res_w': 1920,
        'res_h': 1080,
        'azimuth': 45,  # 前右视角
    },
    {
        'id': 'camera_02_front_left',
        'center': [960.0, 540.0],
        'focal_length': [800.0, 800.0],
        'radial_distortion': [-0.12, 0.18, 0.001],
        'tangential_distortion': [-0.001, 0.001],
        'res_w': 1920,
        'res_h': 1080,
        'azimuth': -45,  # 前左视角
    },
    {
        'id': 'camera_03_rear_right',
        'center': [960.0, 540.0],
        'focal_length': [850.0, 850.0],
        'radial_distortion': [-0.18, 0.22, 0.001],
        'tangential_distortion': [0.001, 0.001],
        'res_w': 1920,
        'res_h': 1080,
        'azimuth': 135,  # 后右视角
    },
    {
        'id': 'camera_04_rear_left',
        'center': [960.0, 540.0],
        'focal_length': [850.0, 850.0],
        'radial_distortion': [-0.16, 0.19, 0.001],
        'tangential_distortion': [-0.001, -0.001],
        'res_w': 1920,
        'res_h': 1080,
        'azimuth': -135,  # 后左视角
    },
]

# 优化后的虚拟外参参数 - 基于实际数据尺度
animal_cameras_extrinsic_params = {
    'default': [
        # 相机1: 前右视角 - 看向数据中心 [0.341, -0.008, 0.758]
        {
            'orientation': [0.653, 0.271, 0.271, 0.653],  # 四元数：看向前右方
            'translation': [15.0, -12.0, 8.0],  # 前右上方，距离约20单位
        },
        # 相机2: 前左视角
        {
            'orientation': [0.653, -0.271, -0.271, 0.653],  # 看向前左方
            'translation': [-15.0, -12.0, 8.0],  # 前左上方
        },
        # 相机3: 后右视角
        {
            'orientation': [0.271, 0.653, 0.653, 0.271],  # 看向后右方
            'translation': [12.0, -10.0, -15.0],  # 后右上方
        },
        # 相机4: 后左视角
        {
            'orientation': [0.271, -0.653, -0.653, 0.271],  # 看向后左方
            'translation': [-12.0, -10.0, -15.0],  # 后左上方
        },
    ]
}


class AnimalsDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=False):
        super().__init__(fps=30, skeleton=quadruped_skeleton)

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
            data = data.item() if hasattr(data, 'item') else data

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras['default'],
                }

        if remove_static_joints:
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

    def get_data_statistics(self):
        """
        获取数据统计信息
        """
        all_positions = []
        for subject in self.subjects():
            for action in self[subject].keys():
                all_positions.append(self[subject][action]['positions'])

        if all_positions:
            combined = np.concatenate(all_positions, axis=0)
            stats = {
                'mean': combined.mean(axis=(0, 1)),
                'std': combined.std(axis=(0, 1)),
                'range': [
                    [combined[..., 0].min(), combined[..., 0].max()],
                    [combined[..., 1].min(), combined[..., 1].max()],
                    [combined[..., 2].min(), combined[..., 2].max()]
                ],
                'span': combined.ptp(axis=(0, 1))
            }
            return stats
        return None