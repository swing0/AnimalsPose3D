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
            'RightWrist': 10,
            'RightAnkle_End': -1,
            'LeftAnkle_End': -1,
            'RightWrist_End': -1,
            'LeftWrist_End': -1,
            'Nose_End': -1
        }
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)

        # 骨骼层级关系
        self.children = {
            'Hip': ['RightHipTop', 'LeftHipTop', 'Spine'],
            'RightHipTop': ['RightHip'],
            'RightHip': ['RightKnee'],
            'RightKnee': ['RightAnkle'],
            'RightAnkle': ['RightAnkle_End'],  # 变为 JOINT
            'RightAnkle_End': [],  # 真正的 End Site
            'LeftHipTop': ['LeftHip'],
            'LeftHip': ['LeftKnee'],
            'LeftKnee': ['LeftAnkle'],
            'LeftAnkle': ['LeftAnkle_End'],
            'LeftAnkle_End': [],
            'Spine': ['Thorax'],
            'Thorax': ['Neck', 'LeftShoulderTop', 'RightShoulderTop'],
            'Neck': ['Nose'],
            'Nose': ['Nose_End'],
            'Nose_End': [],
            'LeftShoulderTop': ['LeftShoulder'],
            'LeftShoulder': ['LeftElbow'],
            'LeftElbow': ['LeftWrist'],
            'LeftWrist': ['LeftWrist_End'],
            'LeftWrist_End': [],
            'RightShoulderTop': ['RightShoulder'],
            'RightShoulder': ['RightElbow'],
            'RightElbow': ['RightWrist'],
            'RightWrist': ['RightWrist_End'],
            'RightWrist_End': []
        }

        # 自动生成父节点字典
        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent

        # 这里的方向决定了 BVH 的静态姿态
        self.initial_directions = {
            'Hip': [0, 0, 0],
            'Spine': [0, 0, 1], 'Thorax': [0, 0, 1],
            'Neck': [0, 0, 1], 'Nose': [0, 0, 1], 'Nose_End': [0, 0, 1],
            'RightHipTop': [1, 0, 0], 'LeftHipTop': [-1, 0, 0],
            'RightShoulderTop': [1, 0, 0], 'LeftShoulderTop': [-1, 0, 0],
            'RightHip': [0, -1, 0], 'RightKnee': [0, -1, 0], 'RightAnkle': [0, -1, 0], 'RightAnkle_End': [0, -1, 0],
            'LeftHip': [0, -1, 0], 'LeftKnee': [0, -1, 0], 'LeftAnkle': [0, -1, 0], 'LeftAnkle_End': [0, -1, 0],
            'RightShoulder': [0, -1, 0], 'RightElbow': [0, -1, 0], 'RightWrist': [0, -1, 0],
            'RightWrist_End': [0, -1, 0],
            'LeftShoulder': [0, -1, 0], 'LeftElbow': [0, -1, 0], 'LeftWrist': [0, -1, 0], 'LeftWrist_End': [0, -1, 0],
        }

    def get_estimated_position(self, pose, joint_name):
        """根据基础关键点计算虚拟/辅助关节点位置"""
        # 处理新增的虚拟末端点 (_End)
        if '_End' in joint_name:
            parent_name = self.parent[joint_name]
            parent_pos = self.get_estimated_position(pose, parent_name)
            # 给末端一个微小的偏移（例如沿父级方向延伸一点），防止骨骼长度为0
            # 这里简单返回父节点位置，但在 get_initial_offset 中我们会赋予其固定长度
            return parent_pos

        # 基础点索引
        idx = self.keypoint2index

        # 1. 躯干虚拟点计算
        hip_pos = pose[idx['Hip']]
        neck_pos = pose[idx['Neck']]

        if joint_name == 'Spine':
            return hip_pos + 0.2 * (neck_pos - hip_pos)
        elif joint_name == 'Thorax':
            return hip_pos + 0.67 * (neck_pos - hip_pos)

        # 2. 髋部与肩部辅助点（用于确定 X 轴平面）
        elif joint_name == 'RightHipTop':
            right_hip_pos = pose[idx['RightHip']]
            return np.array([right_hip_pos[0], hip_pos[1], right_hip_pos[2]])
        elif joint_name == 'LeftHipTop':
            left_hip_pos = pose[idx['LeftHip']]
            return np.array([left_hip_pos[0], hip_pos[1], left_hip_pos[2]])
        elif joint_name == 'LeftShoulderTop':
            left_shoulder_pos = pose[idx['LeftShoulder']]
            thorax_pos = self.get_estimated_position(pose, 'Thorax')
            return np.array([left_shoulder_pos[0], thorax_pos[1], left_shoulder_pos[2]])
        elif joint_name == 'RightShoulderTop':
            right_shoulder_pos = pose[idx['RightShoulder']]
            thorax_pos = self.get_estimated_position(pose, 'Thorax')
            return np.array([right_shoulder_pos[0], thorax_pos[1], right_shoulder_pos[2]])

        # 3. 原始关键点
        else:
            return pose[idx[joint_name]]

    def get_initial_offset(self, poses_3d):
        """根据输入的 3D 序列动态计算每根骨骼的平均长度"""
        bone_lens = {joint: [] for joint in self.keypoint2index}

        # 1. 遍历所有帧，计算父子节点间的动态距离
        for pose in poses_3d:
            # 预计算当前帧所有点（含虚拟点）的位置
            positions = {j: self.get_estimated_position(pose, j) for j in self.keypoint2index}

            for parent, children in self.children.items():
                p_pos = positions[parent]
                for child in children:
                    c_pos = positions[child]
                    dist = np.linalg.norm(p_pos - c_pos)
                    bone_lens[child].append(dist)

        # 2. 计算平均长度
        avg_bone_len = {}
        for joint, lens in bone_lens.items():
            if lens:
                # 方案：使用中位数或平均值以减少噪点影响
                avg_bone_len[joint] = np.mean(lens)
            else:
                avg_bone_len[joint] = 0.0

        # 3. 对称性平滑 (可选，让左右腿长度强制一致)
        side_pairs = [
            ('LeftHip', 'RightHip'), ('LeftKnee', 'RightKnee'), ('LeftAnkle', 'RightAnkle'),
            ('LeftShoulder', 'RightShoulder'), ('LeftElbow', 'RightElbow'), ('LeftWrist', 'RightWrist'),
            ('LeftHipTop', 'RightHipTop'), ('LeftShoulderTop', 'RightShoulderTop')
        ]
        for left, right in side_pairs:
            if left in avg_bone_len and right in avg_bone_len:
                mean_val = (avg_bone_len[left] + avg_bone_len[right]) / 2
                avg_bone_len[left] = avg_bone_len[right] = mean_val

        # 4. 将长度应用到初始方向模板上
        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            if joint == self.root:
                initial_offset[joint] = np.array([0.0, 0.0, 0.0])
                continue

            # 归一化模板方向向量
            dir_vec = np.array(direction)
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-12:
                dir_vec = dir_vec / norm

            # 最终偏移量 = 模板方向 * 动态计算的平均长度
            initial_offset[joint] = dir_vec * avg_bone_len[joint]

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
        """将单帧 3D 坐标转换为 BVH 欧拉角通道数据"""
        channel = []
        quats = {}
        stack = [header.root]

        # 预计算当前帧所有点（包含虚拟点）的位置
        positions = {joint: self.get_estimated_position(pose, joint) for joint in self.keypoint2index}

        while stack:
            node = stack.pop()
            joint = node.name

            # 1. 处理根节点位移 (Root Translation)
            if node.is_root:
                channel.extend(positions[joint])

            # 2. 如果是 End Site，跳过旋转计算（End Site 在 BVH 中没有旋转通道）
            if node.is_end_site:
                # 遍历子节点（虽然 End Site 通常没有子节点，但为了逻辑严谨保留）
                for child in node.children[::-1]:
                    stack.append(child)
                continue

            # 3. 确定当前关节的局部坐标轴 (DCM)
            order = None
            x_dir, y_dir, z_dir = None, None, None

            # 情况 A: 躯干部分 (Hip, Spine, Thorax) - 以 Z 为主轴（前进方向）
            if joint in ['Hip', 'Spine', 'Thorax', 'Neck', 'Nose']:
                if len(node.children) > 0:
                    # Z轴：始终指向下一个节点（前进方向）
                    z_dir = positions[node.children[0].name] - positions[joint]
                    # X轴：始终参考肩部/胯部宽度，保持头不左右乱晃
                    x_dir = positions['LeftShoulderTop'] - positions['RightShoulderTop']
                    order = 'zxy'

            # 情况 B: 四肢部分 (含原本是末端的 Ankle, Wrist) - 以 Y 为主轴（骨骼指向）
            elif any(k in joint for k in ['Shoulder', 'Hip', 'Knee', 'Elbow', 'Ankle', 'Wrist', 'Leg', 'Arm']):
                if len(node.children) > 0:
                    # Y 轴：指向子节点
                    y_dir = positions[joint] - positions[node.children[0].name]
                    # Z 轴：参考躯干的前进方向
                    z_dir = positions['Thorax'] - positions['Hip']
                    order = 'yzx'

            # # 情况 C: 头部与颈部
            # elif joint == 'Neck' or joint == 'Nose':
            #     if len(node.children) > 0:
            #         y_dir = positions[node.children[0].name] - positions[joint]
            #         x_dir = positions['LeftShoulderTop'] - positions['RightShoulderTop']
            #         order = 'yzx'

            # 4. 计算全局旋转四元数
            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                # 兜底方案：继承父节点旋转
                quats[joint] = quats[self.parent[joint]].copy() if self.parent[joint] else np.array([1, 0, 0, 0])

            # 5. 计算局部旋转 (Local Rotation)
            local_quat = quats[joint].copy()
            if node.parent:
                # 相对旋转公式: q_local = q_parent_inverse * q_global
                local_quat = math3d.quat_divide(quats[joint], quats[node.parent.name])

            # 6. 转换为欧拉角并加入通道数据
            # 注意：node.rotation_order 在 get_bvh_header 中定义（通常为 'zxy'）
            euler = math3d.quat2euler(local_quat, order=node.rotation_order)
            channel.extend(np.rad2deg(euler))

            # 7. 深度优先遍历：将子节点压入栈
            # 注意：此处不再过滤 is_end_site，确保所有节点（包括末端）都被访问
            for child in node.children[::-1]:
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