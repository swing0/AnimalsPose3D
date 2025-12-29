import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_random_camera(dist_range=(5.0, 8.0), elev_range=(-10, 45), azim_range=(0, 360)):
    """在球面上随机采样一个相机位置并计算 LookAt 矩阵"""
    dist = np.random.uniform(*dist_range)
    elev = np.deg2rad(np.random.uniform(*elev_range))
    azim = np.deg2rad(np.random.uniform(*azim_range))

    # 相机在世界坐标系的位置 (Y-up)
    cam_x = dist * np.cos(elev) * np.sin(azim)
    cam_y = dist * np.sin(elev)
    cam_z = dist * np.cos(elev) * np.cos(azim)
    cam_pos = np.array([cam_x, cam_y, cam_z])

    target = np.array([0, 0, 0])  # 始终注视原点
    up = np.array([0, 1, 0])  # Y方向向上

    # 计算 LookAt 旋转矩阵
    z_axis = cam_pos - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3, 3) 外参旋转
    return R, cam_pos

def analyze_camera_positions(num_samples=100):
    """分析摄像机位置分布"""
    print("=== 摄像机位置分析 ===")
    print(f"采样 {num_samples} 个摄像机位置")
    
    # 采样多个摄像机位置
    camera_positions = []
    elevations = []
    azimuths = []
    distances = []
    
    for i in range(num_samples):
        R, cam_pos = get_random_camera()
        camera_positions.append(cam_pos)
        
        # 计算球坐标
        dist = np.linalg.norm(cam_pos)
        elev = np.arcsin(cam_pos[1] / dist)  # 仰角
        azim = np.arctan2(cam_pos[0], cam_pos[2])  # 方位角
        
        distances.append(dist)
        elevations.append(np.degrees(elev))
        azimuths.append(np.degrees(azim))
    
    camera_positions = np.array(camera_positions)
    
    # 统计信息
    print(f"\n=== 统计信息 ===")
    print(f"距离范围: [{min(distances):.2f}, {max(distances):.2f}] 米")
    print(f"仰角范围: [{min(elevations):.1f}, {max(elevations):.1f}] 度")
    print(f"方位角范围: [{min(azimuths):.1f}, {max(azimuths):.1f}] 度")
    
    # 检查不合理的位置
    print(f"\n=== 不合理位置检查 ===")
    
    # 检查过低的位置（低于动物高度）
    low_elevations = [e for e in elevations if e < -5]
    if low_elevations:
        print(f"❌ 发现 {len(low_elevations)} 个过低位置（仰角 < -5°）")
        print(f"   这些位置可能导致从下方拍摄，视角不合理")
    else:
        print("✅ 没有过低位置")
    
    # 检查过高的位置（俯视角度过大）
    high_elevations = [e for e in elevations if e > 60]
    if high_elevations:
        print(f"❌ 发现 {len(high_elevations)} 个过高位置（仰角 > 60°）")
        print(f"   这些位置可能导致过于俯视，不利于姿态估计")
    else:
        print("✅ 没有过高位置")
    
    # 检查距离过近的位置
    close_distances = [d for d in distances if d < 3]
    if close_distances:
        print(f"❌ 发现 {len(close_distances)} 个过近位置（距离 < 3米）")
        print(f"   这些位置可能导致透视畸变过大")
    else:
        print("✅ 没有过近位置")
    
    # 检查距离过远的位置
    far_distances = [d for d in distances if d > 12]
    if far_distances:
        print(f"❌ 发现 {len(far_distances)} 个过远位置（距离 > 12米）")
        print(f"   这些位置可能导致动物太小，细节丢失")
    else:
        print("✅ 没有过远位置")
    
    # 可视化摄像机位置分布
    print(f"\n=== 创建可视化图 ===")
    fig = plt.figure(figsize=(15, 5))
    
    # 3D分布图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(camera_positions[:, 0], camera_positions[:, 2], camera_positions[:, 1], 
               c=elevations, cmap='viridis', s=30)
    ax1.scatter([0], [0], [0], c='red', s=100, marker='*', label='动物位置')
    ax1.set_xlabel('X (左右)')
    ax1.set_ylabel('Z (前后)')
    ax1.set_zlabel('Y (高度)')
    ax1.set_title('摄像机位置3D分布')
    ax1.legend()
    
    # 仰角分布
    ax2 = fig.add_subplot(132)
    ax2.hist(elevations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(-5, color='red', linestyle='--', label='过低阈值 (-5°)')
    ax2.axvline(60, color='orange', linestyle='--', label='过高阈值 (60°)')
    ax2.set_xlabel('仰角 (度)')
    ax2.set_ylabel('频次')
    ax2.set_title('仰角分布')
    ax2.legend()
    
    # 距离分布
    ax3 = fig.add_subplot(133)
    ax3.hist(distances, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(3, color='red', linestyle='--', label='过近阈值 (3m)')
    ax3.axvline(12, color='orange', linestyle='--', label='过远阈值 (12m)')
    ax3.set_xlabel('距离 (米)')
    ax3.set_ylabel('频次')
    ax3.set_title('距离分布')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('camera_position_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ 摄像机位置分析图已保存: camera_position_analysis.png")
    
    # 建议改进
    print(f"\n=== 改进建议 ===")
    
    issues_found = len(low_elevations) + len(high_elevations) + len(close_distances) + len(far_distances)
    if issues_found > 0:
        print(f"发现 {issues_found} 个潜在问题位置")
        print("建议调整参数范围：")
        print("  - 仰角范围: (-10, 45) → (0, 45) 避免过低位置")
        print("  - 距离范围: (5.0, 8.0) → (4.0, 10.0) 增加多样性但避免极端")
    else:
        print("✅ 当前参数设置合理")

if __name__ == "__main__":
    analyze_camera_positions(100)