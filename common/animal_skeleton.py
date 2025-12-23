import numpy as np
from . import math3d
from . import bvh_helper


class AnimalSkeleton(object):
    def __init__(self):
        # 根节点定义
        self.root = 'Hip'

        # 关节名称到原始数据索引的映射
        self.keypoint2index = {
            'Hip': 0,
            'RightHip': 14,
            'RightKnee': 15,
            'RightAnkle': 16,
            'LeftHip': 11,
            'LeftKnee': 12,
            'LeftAnkle': 13,
            'Spine': 17,  # 虚拟索引
            'Thorax': 18,  # 虚拟索引
            'RightHipTop': 19,  # 虚拟索引
            'LeftHipTop': 20,  # 虚拟索引
            'LeftShoulderTop': 21,  # 虚拟索引
            'RightShoulderTop': 22,  # 虚拟索引
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

        # 骨骼层级关系
        self.children = {
            'Hip': ['RightHipTop', 'LeftHipTop', 'Spine'],
            'RightHipTop': ['RightHip'],
            'RightHip': ['RightKnee'],
            'RightKnee': ['RightAnkle'],
            'RightAnkle': [],
            'LeftHipTop': ['LeftHip'],
            'LeftHip': ['LeftKnee'],
            'LeftKnee': ['LeftAnkle'],
            'LeftAnkle': [],
            'Spine': ['Thorax'],
            'Thorax': ['Neck', 'LeftShoulderTop', 'RightShoulderTop'],
            'Neck': ['Nose'],
            'Nose': [],
            'LeftShoulderTop': ['LeftShoulder'],
            'LeftShoulder': ['LeftElbow'],
            'LeftElbow': ['LeftWrist'],
            'LeftWrist': [],
            'RightShoulderTop': ['RightShoulder'],
            'RightShoulder': ['RightElbow'],
            'RightElbow': ['RightWrist'],
            'RightWrist': []
        }

        # 自动生成父节点字典
        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent

        # 这里的方向决定了 BVH 的静态姿态
        self.initial_directions = {
           'Hip': [0, 0, 0],
           'Spine': [0, 0, 1],  # 脊椎向正前方延伸
           'Thorax': [0, 0, 1],
           'Neck': [0, 0.5, 0.5],  # 脖子斜向上
           'Nose': [0, 0, 1],
           'RightHipTop': [1, 0, 0],  # 向右偏置
           'RightHip': [0, -1, 0],  # 腿向下
           'RightKnee': [0, -1, 0],
           'RightAnkle': [0, -1, 0],
           'LeftHipTop': [-1, 0, 0],
           'LeftHip': [0, -1, 0],
           'LeftKnee': [0, -1, 0],
           'LeftAnkle': [0, -1, 0],
           'RightShoulderTop': [1, 0, 0],
           'RightShoulder': [0, -1, 0],
           'RightElbow': [0, -1, 0],
           'RightWrist': [0, -1, 0],
           'LeftShoulderTop': [-1, 0, 0],
           'LeftShoulder': [0, -1, 0],
           'LeftElbow': [0, -1, 0],
           'LeftWrist': [0, -1, 0],
        }


    def get_estimated_position(self, pose, joint_name):
        """根据基础关键点计算虚拟/辅助关节点位置"""
        hip_pos = pose[self.keypoint2index['Hip']]
        neck_pos = pose[self.keypoint2index['Neck']]

        if joint_name == 'Spine':
            return hip_pos + 0.2 * (neck_pos - hip_pos)
        elif joint_name == 'Thorax':
            return hip_pos + 0.67 * (neck_pos - hip_pos)
        elif joint_name == 'RightHipTop':
            right_hip_pos = pose[self.keypoint2index['RightHip']]
            return np.array([right_hip_pos[0], hip_pos[1], right_hip_pos[2]])
        elif joint_name == 'LeftHipTop':
            left_hip_pos = pose[self.keypoint2index['LeftHip']]
            return np.array([left_hip_pos[0], hip_pos[1], left_hip_pos[2]])
        elif joint_name == 'LeftShoulderTop':
            left_shoulder_pos = pose[self.keypoint2index['LeftShoulder']]
            thorax_pos = self.get_estimated_position(pose, 'Thorax')
            return np.array([left_shoulder_pos[0], thorax_pos[1], left_shoulder_pos[2]])
        elif joint_name == 'RightShoulderTop':
            right_shoulder_pos = pose[self.keypoint2index['RightShoulder']]
            thorax_pos = self.get_estimated_position(pose, 'Thorax')
            return np.array([right_shoulder_pos[0], thorax_pos[1], right_shoulder_pos[2]])
        else:
            return pose[self.keypoint2index[joint_name]]

    def get_initial_offset(self, poses_3d):
        # 计算每一帧中各骨骼的长度
        bone_lens = {joint: [] for joint in self.keypoint2index}

        for pose in poses_3d:
            for parent, children in self.children.items():
                p_pos = self.get_estimated_position(pose, parent)
                for child in children:
                    c_pos = self.get_estimated_position(pose, child)
                    dist = np.linalg.norm(p_pos - c_pos)
                    bone_lens[child].append(dist)

        # 计算平均长度并结合初始方向
        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            avg_len = np.mean(bone_lens[joint]) if bone_lens[joint] else 0
            dir_norm = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = dir_norm * avg_len

        return initial_offset

    def get_bvh_header(self, poses_3d):
        initial_offset = self.get_initial_offset(poses_3d)
        nodes = {}

        for joint in self.keypoint2index:
            is_root = (joint == self.root)
            is_end_site = (len(self.children[joint]) == 0)

            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',
                is_root=is_root,
                is_end_site=is_end_site,
            )

        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        return bvh_helper.BvhHeader(root=nodes[self.root], nodes=nodes)

    def pose2euler(self, pose, header):
        channel = []
        quats = {}
        stack = [header.root]

        # 预计算所有关节点的 3D 坐标（包括虚拟点）
        positions = {joint: self.get_estimated_position(pose, joint) for joint in self.keypoint2index}

        while stack:
            node = stack.pop()
            joint = node.name

            if node.is_root:
                channel.extend(positions[joint])  # Root 包含全局位移

            # 初始化变量
            order = None
            x_dir, y_dir, z_dir = None, None, None

            # --- 确定当前关节的局部轴向 (DCM) ---

            # 1. 躯干/脊椎部分 (Hip, Spine, Thorax)
            if joint in ['Hip', 'Spine', 'Thorax']:
                # Z轴：指向脊椎链的下一个节点 (前进方向)
                # 根据 self.children 结构，Spine 是 Hip 的第 3 个孩子 (index 2)
                if joint == 'Hip':
                    z_dir = positions['Spine'] - positions['Hip']
                elif joint == 'Spine':
                    z_dir = positions['Thorax'] - positions['Spine']
                elif joint == 'Thorax':
                    z_dir = positions['Neck'] - positions['Thorax']

                # X轴：使用左右胯部/肩膀的连线作为水平参考
                x_dir = positions['LeftHipTop'] - positions['RightHipTop']
                order = 'zxy'

            # 2. 颈部 (Neck)
            elif joint == 'Neck':
                # Y轴：指向鼻尖 (向上/向前)
                y_dir = positions['Nose'] - positions['Neck']
                # X轴：参考肩膀宽度
                x_dir = positions['LeftShoulderTop'] - positions['RightShoulderTop']
                order = 'yxz'

            # 3. 四肢部分 (肩膀、跨部、膝盖、肘部等)
            elif any(k in joint for k in ['Shoulder', 'Hip', 'Knee', 'Elbow', 'Leg', 'Arm']):
                if len(node.children) > 0:
                    child_name = node.children[0].name
                    # Y轴：主轴，指向下一个关节 (骨骼延伸方向)
                    y_dir = positions[joint] - positions[child_name]
                    # Z轴：参考轴，使用身体的前进方向 (Hip -> Thorax) 保持四肢朝向稳定
                    z_dir = positions['Thorax'] - positions['Hip']
                    order = 'yzx'

            # --- 计算旋转矩阵与四元数 ---
            if order:
                # 调用你修改后的 math3d.dcm_from_axis，它现在支持 None 传参了
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                # 末端节点或未定义节点：继承父节点旋转
                quats[joint] = quats[self.parent[joint]].copy() if self.parent[joint] else np.array([1, 0, 0, 0])

            # --- 计算局部旋转并转为欧拉角 ---
            local_quat = quats[joint].copy()
            if node.parent:
                # 相对旋转：Q_local = Q_parent^-1 * Q_global
                local_quat = math3d.quat_divide(quats[joint], quats[node.parent.name])

            # 转换为 BVH 指定的欧拉角顺序
            euler = math3d.quat2euler(local_quat, order=node.rotation_order)
            # BVH 使用角度制
            channel.extend(np.rad2deg(euler))

            # 遍历子节点
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