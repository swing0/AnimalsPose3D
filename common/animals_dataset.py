import numpy as np
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates

# AP-10K based quadruped animal skeleton
quadruped_skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15],
    joints_left=[5, 6, 7, 8, 13, 14, 15, 16],  # All left side joints
    joints_right=[1, 2, 3, 4, 9, 10, 11, 12]  # All right side joints
)


class QuadrupedAnimalDataset(MocapDataset):
    def __init__(self, npz_path, remove_static_joints=True):
        super().__init__(fps=50, skeleton=quadruped_skeleton)

        # Load NPZ data - 适配新的数据格式
        data = np.load(npz_path, allow_pickle=True)

        keypoints_3d = data['keypoints']

        print(f"Loaded keypoints with shape: {keypoints_3d.shape}")

        # Setup virtual cameras
        self._setup_virtual_cameras()

        self._data = self._reorganize_data(keypoints_3d)

        if remove_static_joints:
            # Remove joints if needed
            pass

    def _setup_virtual_cameras(self):
        """Setup virtual camera parameters"""
        self._cameras = {}

        # Base camera intrinsics
        base_intrinsics = {
            'res_w': 1000,
            'res_h': 1000,
            'focal_length': np.array([1145.0, 1145.0], dtype='float32'),
            'center': np.array([500.0, 500.0], dtype='float32'),
            'radial_distortion': np.array([-0.2, 0.25, 0.0], dtype='float32'),
            'tangential_distortion': np.array([0.0, 0.0], dtype='float32')
        }

        # 4 virtual camera positions
        camera_configs = [
            {'azimuth': 70, 'distance': 1.0, 'elevation': 0.1},
            {'azimuth': -70, 'distance': 1.0, 'elevation': 0.1},
            {'azimuth': 110, 'distance': 1.0, 'elevation': 0.1},
            {'azimuth': -110, 'distance': 1.0, 'elevation': 0.1},
        ]

        subject = 'Animal'
        self._cameras[subject] = []

        for i, config in enumerate(camera_configs):
            camera = base_intrinsics.copy()

            orientation = self._compute_orientation(config['azimuth'], config['elevation'])
            translation = self._compute_translation(config['azimuth'], config['distance'], config['elevation'])

            camera.update({
                'id': f'virtual_camera_{i}',
                'azimuth': config['azimuth'],
                'orientation': orientation,
                'translation': translation
            })

            # Normalize camera parameters
            camera['center'] = normalize_screen_coordinates(
                camera['center'].copy(),
                w=camera['res_w'],
                h=camera['res_h']
            ).astype('float32')

            camera['focal_length'] = camera['focal_length'] / camera['res_w'] * 2

            # Combine intrinsic parameters
            camera['intrinsic'] = np.concatenate((
                camera['focal_length'],
                camera['center'],
                camera['radial_distortion'],
                camera['tangential_distortion']
            ))

            self._cameras[subject].append(camera)

    def _compute_orientation(self, azimuth, elevation):
        """Compute camera orientation (quaternion)"""
        # 简化的方向计算 - 返回单位四元数
        return np.array([0.0, 0.0, 0.0, 1.0], dtype='float32')

    def _compute_translation(self, azimuth, distance, elevation):
        """Compute camera translation"""
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)

        x = distance * np.cos(el_rad) * np.sin(az_rad)
        y = distance * np.sin(el_rad)
        z = distance * np.cos(el_rad) * np.cos(az_rad)

        return np.array([x, y, z], dtype='float32')

    def _reorganize_data(self, keypoints_3d):
        data = {}
        subject = 'Animal'
        data[subject] = {}

        # 使用完整序列作为一个action，不进行分割
        action_name = 'complete_sequence'
        data[subject][action_name] = {
            'positions': keypoints_3d,
            'cameras': self._cameras[subject]
        }

        print(f"Created complete sequence with {len(keypoints_3d)} frames")
        return data

    def supports_semi_supervised(self):
        return True

    def __getitem__(self, key):
        return self._data[key]

    def subjects(self):
        return list(self._data.keys())

    def cameras(self):
        return self._cameras