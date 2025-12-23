"""
! left handed coordinate, z-up, y-forward
! left to right rotation matrix multiply: v'=vR
! non-standard quaternion multiply
左手坐标系，Z向上，Y向前
从左到右旋转矩阵旋转相乘：v'=vR
非标准四元数乘法
"""

import numpy as np


def normalize(x):
    return x / max(np.linalg.norm(x), 1e-12)


# 从坐标轴向量计算方向余弦矩阵(direction cosine matrix)
def dcm_from_axis(x_dir, y_dir, z_dir, order):
    """
    从给定的轴向量计算方向余弦矩阵。
    支持传入其中两个轴，自动计算第三个轴。
    """
    assert order in ['yzx', 'yxz', 'xyz', 'xzy', 'zxy', 'zyx']

    # 将轴存入字典，方便根据 order 索引
    axis_dict = {'x': x_dir, 'y': y_dir, 'z': z_dir}

    # 获取顺序索引
    name = ['x', 'y', 'z']
    o0, o1, o2 = order[0], order[1], order[2]

    # 1. 规范化第一个主轴（必须提供）
    if axis_dict[o0] is None:
        raise ValueError(f"Order starts with {o0}, but {o0}_dir is None.")
    axis_dict[o0] = normalize(axis_dict[o0])

    # 2. 计算/规范化第二个轴
    # 如果第二个轴缺失，尝试通过 3 x 1 叉乘得到
    if axis_dict[o1] is None:
        # 寻找第三个轴来做叉乘，如果第三个也没有，就报错
        if axis_dict[o2] is None:
            raise ValueError(f"Both {o1}_dir and {o2}_dir are None. Need at least two axes.")
        # 叉乘逻辑满足右手/左手定则循环
        axis_dict[o1] = normalize(np.cross(axis_dict[o2], axis_dict[o0]))
    else:
        axis_dict[o1] = normalize(axis_dict[o1])

    # 3. 计算第三个轴（确保三个轴正交）
    # 始终重新计算第三个轴以保证矩阵的正交性
    axis_dict[o2] = normalize(np.cross(axis_dict[o0], axis_dict[o1]))

    # 构建 DCM (方向余弦矩阵)
    dcm = np.stack([axis_dict['x'], axis_dict['y'], axis_dict['z']])

    return dcm

def dcm2quat(dcm):
    q = np.zeros([4])
    tr = np.trace(dcm)

    if tr > 0:
        sqtrp1 = np.sqrt(tr + 1.0)
        q[0] = 0.5 * sqtrp1
        q[1] = (dcm[1, 2] - dcm[2, 1]) / (2.0 * sqtrp1)
        q[2] = (dcm[2, 0] - dcm[0, 2]) / (2.0 * sqtrp1)
        q[3] = (dcm[0, 1] - dcm[1, 0]) / (2.0 * sqtrp1)
    else:
        d = np.diag(dcm)
        if d[1] > d[0] and d[1] > d[2]:
            sqdip1 = np.sqrt(d[1] - d[0] - d[2] + 1.0)
            q[2] = 0.5 * sqdip1

            if sqdip1 != 0:
                sqdip1 = 0.5 / sqdip1

            q[0] = (dcm[2, 0] - dcm[0, 2]) * sqdip1
            q[1] = (dcm[0, 1] + dcm[1, 0]) * sqdip1
            q[3] = (dcm[1, 2] + dcm[2, 1]) * sqdip1

        elif d[2] > d[0]:
            sqdip1 = np.sqrt(d[2] - d[0] - d[1] + 1.0)
            q[3] = 0.5 * sqdip1

            if sqdip1 != 0:
                sqdip1 = 0.5 / sqdip1

            q[0] = (dcm[0, 1] - dcm[1, 0]) * sqdip1
            q[1] = (dcm[2, 0] + dcm[0, 2]) * sqdip1
            q[2] = (dcm[1, 2] + dcm[2, 1]) * sqdip1

        else:
            sqdip1 = np.sqrt(d[0] - d[1] - d[2] + 1.0)
            q[1] = 0.5 * sqdip1

            if sqdip1 != 0:
                sqdip1 = 0.5 / sqdip1

            q[0] = (dcm[1, 2] - dcm[2, 1]) * sqdip1
            q[2] = (dcm[0, 1] + dcm[1, 0]) * sqdip1
            q[3] = (dcm[2, 0] + dcm[0, 2]) * sqdip1

    return q


def quat_dot(q0, q1):
    original_shape = q0.shape
    q0 = np.reshape(q0, [-1, 4])
    q1 = np.reshape(q1, [-1, 4])

    w0, x0, y0, z0 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    q_product = w0 * w1 + x1 * x1 + y0 * y1 + z0 * z1
    q_product = np.expand_dims(q_product, axis=1)
    q_product = np.tile(q_product, [1, 4])

    return np.reshape(q_product, original_shape)


def quat_inverse(q):
    original_shape = q.shape
    q = np.reshape(q, [-1, 4])

    q_conj = [q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]]
    q_conj = np.stack(q_conj, axis=1)
    q_inv = np.divide(q_conj, quat_dot(q_conj, q_conj))

    return np.reshape(q_inv, original_shape)


def quat_mul(q0, q1):
    original_shape = q0.shape
    q1 = np.reshape(q1, [-1, 4, 1])
    q0 = np.reshape(q0, [-1, 1, 4])
    terms = np.matmul(q1, q0)
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    q_product = np.stack([w, x, y, z], axis=1)
    return np.reshape(q_product, original_shape)


def quat_divide(q, r):
    return quat_mul(quat_inverse(r), q)


def quat2euler(q, order='zxy', eps=1e-8):
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = np.reshape(q, [-1, 4])

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    if order == 'zxy':
        # arcsin(2(wx + yz))
        res_x = np.arcsin(np.clip(2 * (w * x + y * z), -1 + eps, 1 - eps))
        res_y = np.arctan2(2 * (w * y - z * x), 1 - 2 * (x * x + y * y))
        res_z = np.arctan2(2 * (w * z - x * y), 1 - 2 * (x * x + z * z))
        euler = np.stack([res_z, res_x, res_y], axis=1)  # 返回顺序 Z, X, Y

    elif order == 'yzx':
        res_z = np.arcsin(np.clip(2 * (w * z + x * y), -1 + eps, 1 - eps))
        res_y = np.arctan2(2 * (w * y - x * z), 1 - 2 * (y * y + z * z))
        res_x = np.arctan2(2 * (w * x - y * z), 1 - 2 * (x * x + z * z))
        euler = np.stack([res_y, res_z, res_x], axis=1)  # 返回顺序 Y, Z, X

    elif order == 'yxz':
        res_x = np.arcsin(np.clip(-2 * (w * x - y * z), -1 + eps, 1 - eps))
        res_y = np.arctan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y))
        res_z = np.arctan2(2 * (w * z + x * y), 1 - 2 * (x * x + z * z))
        euler = np.stack([res_y, res_x, res_z], axis=1)  # 返回顺序 Y, X, Z

    else:
        # 如果遇到未实现的顺序，默认返回 0
        euler = np.zeros((q.shape[0], 3))

    return np.reshape(euler, original_shape)


def quat2SMPLEuler(q, order='xyz', eps=1e-8):
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = np.reshape(q, [-1, 4])

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = np.arcsin(np.clip(2 * (q0 * q1 + q2 * q3), -1 + eps, 1 - eps))
        y = np.arctan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        euler = np.stack([x, y, z], axis=1)
    else:
        raise ValueError('Not implemented')

    return np.reshape(euler, original_shape)
