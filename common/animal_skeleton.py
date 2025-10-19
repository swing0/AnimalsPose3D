from . import math3d
from . import bvh_helper

import numpy as np


class AnimalSkeleton(object):

    def __init__(self):
        self.root = 'Hip'
        self.keypoint2index = {
            'Hip': 0,
            'RightHip': 14,
            'RightKnee': 15,
            'RightAnkle': 16,
            'LeftHip': 11,
            'LeftKnee': 12,
            'LeftAnkle': 13,
            'Spine': 17,  # 使用虚拟索引，不在原始数据中
            'Thorax': 18,  # 使用虚拟索引，不在原始数据中
            'Neck': 4,
            'Nose': 3,
            'LeftShoulder': 5,
            'LeftElbow': 6,
            'LeftWrist': 7,
            'RightShoulder': 8,
            'RightElbow': 9,
            'RightWrist': 10
        }
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)

        self.children = {
            'Hip': ['RightHip', 'LeftHip', 'Spine'],
            'RightHip': ['RightKnee'],
            'RightKnee': ['RightAnkle'],
            'RightAnkle': [],  # 移除了EndSite子节点
            'LeftHip': ['LeftKnee'],
            'LeftKnee': ['LeftAnkle'],
            'LeftAnkle': [],  # 移除了EndSite子节点
            'Spine': ['Thorax'],
            'Thorax': ['Neck', 'LeftShoulder', 'RightShoulder'],
            'Neck': ['Nose'],
            'Nose': [],  # Head is an end site
            'LeftShoulder': ['LeftElbow'],
            'LeftElbow': ['LeftWrist'],
            'LeftWrist': [],  # 移除了EndSite子节点
            'RightShoulder': ['RightElbow'],
            'RightElbow': ['RightWrist'],
            'RightWrist': []  # 移除了EndSite子节点
        }
        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent

        self.left_joints = [
            joint for joint in self.keypoint2index
            if 'Left' in joint
        ]
        self.right_joints = [
            joint for joint in self.keypoint2index
            if 'Right' in joint
        ]

        # Animal-pose
        self.initial_directions = {
            'Hip': [0, 0, 0],
            'RightHip': [-1, 0, 0],
            'RightKnee': [0, 0, -1],
            'RightAnkle': [0, 0, -1],
            'LeftHip': [1, 0, 0],
            'LeftKnee': [0, 0, -1],
            'LeftAnkle': [0, 0, -1],
            'Spine': [0, 0, 1],
            'Thorax': [0, 0, 1],
            'Neck': [0, 0, 1],
            'Nose': [0, 0, 1],
            'LeftShoulder': [1, 0, 0],
            'LeftElbow': [1, 0, 0],
            'LeftWrist': [1, 0, 0],
            'RightShoulder': [-1, 0, 0],
            'RightElbow': [-1, 0, 0],
            'RightWrist': [-1, 0, 0]
        }

    def get_estimated_position(self, pose, joint_name):
        """根据Hip和Neck的位置估计Spine或Thorax的位置"""
        hip_pos = pose[self.keypoint2index['Hip']]
        neck_pos = pose[self.keypoint2index['Neck']]

        if joint_name == 'Spine':
            # Spine在Hip到Neck的1/3处
            return hip_pos + 0.33 * (neck_pos - hip_pos)
        elif joint_name == 'Thorax':
            # Thorax在Hip到Neck的2/3处
            return hip_pos + 0.67 * (neck_pos - hip_pos)
        else:
            # 其他关节直接返回原始位置
            return pose[self.keypoint2index[joint_name]]

    def get_initial_offset(self, poses_3d):
        # 使用第一帧来估算骨骼长度
        sample_pose = poses_3d[0]

        bone_lens = {self.root: [0]}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            p_pos = self.get_estimated_position(sample_pose, parent)

            for child in self.children[parent]:
                # 跳过EndSite节点的长度计算
                if 'EndSite' in child:
                    continue

                stack.append(child)
                c_pos = self.get_estimated_position(sample_pose, child)

                # 计算骨骼长度
                bone_length = np.linalg.norm(p_pos - c_pos)
                bone_lens[child] = [bone_length]

        bone_len = {}
        for joint in self.keypoint2index:
            if joint in bone_lens:
                bone_len[joint] = np.mean(bone_lens[joint])
            else:
                # 如果没有计算过长度，使用默认值
                bone_len[joint] = 0.5  # 默认骨骼长度

        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]

        return initial_offset

    def get_bvh_header(self, poses_3d):
        initial_offset = self.get_initial_offset(poses_3d)

        nodes = {}
        for joint in self.keypoint2index:
            is_root = joint == self.root
            is_end_site = 'EndSite' in joint
            # 不为EndSite关节创建BVH节点
            if is_end_site:
                continue

            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',
                is_root=is_root,
                is_end_site=is_end_site,
            )

        # 重建父子关系（跳过EndSite关节）
        for joint, children in self.children.items():
            if joint not in nodes:
                continue
            valid_children = [child for child in children if child in nodes]
            nodes[joint].children = [nodes[child] for child in valid_children]
            for child in valid_children:
                nodes[child].parent = nodes[joint]

        header = bvh_helper.BvhHeader(root=nodes[self.root], nodes=nodes)
        return header

    def pose2euler(self, pose, header):
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]

        while stack:
            node = stack.pop()
            joint = node.name
            joint_pos = self.get_estimated_position(pose, joint)

            if node.is_root:
                channel.extend(joint_pos)

            order = None
            if joint == 'Hip':
                left_hip_pos = self.get_estimated_position(pose, 'LeftHip')
                right_hip_pos = self.get_estimated_position(pose, 'RightHip')
                spine_pos = self.get_estimated_position(pose, 'Spine')

                x_dir = left_hip_pos - right_hip_pos
                y_dir = None
                z_dir = spine_pos - joint_pos
                order = 'zyx'
            elif joint in ['RightHip', 'RightKnee']:
                hip_pos = self.get_estimated_position(pose, 'Hip')
                right_hip_pos = self.get_estimated_position(pose, 'RightHip')
                child_pos = self.get_estimated_position(pose, node.children[0].name) if node.children else None

                x_dir = hip_pos - right_hip_pos
                y_dir = None
                z_dir = joint_pos - child_pos if child_pos is not None else [0, 0, 1]
                order = 'zyx'
            elif joint in ['LeftHip', 'LeftKnee']:
                left_hip_pos = self.get_estimated_position(pose, 'LeftHip')
                hip_pos = self.get_estimated_position(pose, 'Hip')
                child_pos = self.get_estimated_position(pose, node.children[0].name) if node.children else None

                x_dir = left_hip_pos - hip_pos
                y_dir = None
                z_dir = joint_pos - child_pos if child_pos is not None else [0, 0, 1]
                order = 'zyx'
            elif joint == 'Spine':
                left_hip_pos = self.get_estimated_position(pose, 'LeftHip')
                right_hip_pos = self.get_estimated_position(pose, 'RightHip')
                thorax_pos = self.get_estimated_position(pose, 'Thorax')

                x_dir = left_hip_pos - right_hip_pos
                y_dir = None
                z_dir = thorax_pos - joint_pos
                order = 'zyx'
            elif joint == 'Thorax':
                left_shoulder_pos = self.get_estimated_position(pose, 'LeftShoulder')
                right_shoulder_pos = self.get_estimated_position(pose, 'RightShoulder')
                spine_pos = self.get_estimated_position(pose, 'Spine')

                x_dir = left_shoulder_pos - right_shoulder_pos
                y_dir = None
                z_dir = joint_pos - spine_pos
                order = 'zyx'
            elif joint == 'Neck':
                thorax_pos = self.get_estimated_position(pose, 'Thorax')
                head_pos = self.get_estimated_position(pose, 'Nose')

                x_dir = None
                y_dir = thorax_pos - joint_pos
                z_dir = head_pos - thorax_pos
                order = 'zxy'
            elif joint == 'LeftShoulder':
                left_elbow_pos = self.get_estimated_position(pose, 'LeftElbow')
                left_wrist_pos = self.get_estimated_position(pose, 'LeftWrist')

                x_dir = left_elbow_pos - joint_pos
                y_dir = left_elbow_pos - left_wrist_pos
                z_dir = None
                order = 'xzy'
            elif joint == 'LeftElbow':
                left_wrist_pos = self.get_estimated_position(pose, 'LeftWrist')
                left_shoulder_pos = self.get_estimated_position(pose, 'LeftShoulder')

                x_dir = left_wrist_pos - joint_pos
                y_dir = joint_pos - left_shoulder_pos
                z_dir = None
                order = 'xzy'
            elif joint == 'RightShoulder':
                right_elbow_pos = self.get_estimated_position(pose, 'RightElbow')
                right_wrist_pos = self.get_estimated_position(pose, 'RightWrist')

                x_dir = joint_pos - right_elbow_pos
                y_dir = right_elbow_pos - right_wrist_pos
                z_dir = None
                order = 'xzy'
            elif joint == 'RightElbow':
                right_wrist_pos = self.get_estimated_position(pose, 'RightWrist')
                right_shoulder_pos = self.get_estimated_position(pose, 'RightShoulder')

                x_dir = joint_pos - right_wrist_pos
                y_dir = joint_pos - right_shoulder_pos
                z_dir = None
                order = 'xzy'

            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()

            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )

            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel

    def poses2bvh(self, poses_3d, header=None, output_file=None):
        if not header:
            header = self.get_bvh_header(poses_3d)

        channels = []
        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))

        if output_file:
            bvh_helper.write_bvh(output_file, header, channels)

        return channels, header