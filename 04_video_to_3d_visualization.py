# 04b_video_to_3d_visualization.py
# 视频到3D关键点的可视化工具 - 适配最佳合成模型 (Best Synth Model)

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from tqdm import tqdm
import sys

# 添加路径
sys.path.append('./common')

try:
    from common.ap10k_detector import AP10KAnimalPoseDetector
    from common.keypoint_mapper import KeypointMapper
    from common.transformer_model import AnimalPoseTransformer
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

# 骨架分组定义
SKELETON_GROUPS = {
    'trunk': {'edges': [(0, 4), (4, 3), (3, 1), (3, 2)], 'color': 'black', 'label': 'Head & Neck'},
    'front_left': {'edges': [(4, 5), (5, 6), (6, 7)], 'color': 'red', 'label': 'Front Left'},
    'front_right': {'edges': [(4, 8), (8, 9), (9, 10)], 'color': 'orange', 'label': 'Front Right'},
    'back_left': {'edges': [(0, 11), (11, 12), (12, 13)], 'color': 'blue', 'label': 'Back Left'},
    'back_right': {'edges': [(0, 14), (14, 15), (15, 16)], 'color': 'cyan', 'label': 'Back Right'}
}

SKELETON_COLORS_2D = {
    'trunk': (0, 0, 0), 'front_left': (0, 0, 255), 'front_right': (0, 165, 255),
    'back_left': (255, 0, 0), 'back_right': (255, 255, 0)
}

class VideoTo3DVisualizer:
    def __init__(self, model_checkpoint, onnx_model_path, target_species_id=0):
        print("🎯 初始化视频到3D可视化器...")
        self.detector = AP10KAnimalPoseDetector(onnx_model_path)
        self.mapper = KeypointMapper()
        self.target_species_id = target_species_id
        
        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_3d = self.load_3d_model(model_checkpoint)
        
        self.video_frames = []
        self.keypoints_2d_sequence = []
        self.keypoints_3d_sequence = []
        
        print("✅ 可视化器初始化完成")
    
    def load_3d_model(self, checkpoint_path):
        print(f"📥 加载3D模型: {checkpoint_path}")
        # 参数必须与 03c_train_enhanced.py 一致
        model = AnimalPoseTransformer(
            num_joints=17, 
            embed_dim=256, 
            depth=4, 
            num_heads=8, 
            seq_len=27, 
            num_species=20
        ).to(self.device)
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ 模型文件不存在: {checkpoint_path}")
            return None
            
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"✅ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
            
        return model
    
    def extract_video_frames(self, video_path, max_frames=300):
        print(f"🎥 处理视频: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames: total_frames = min(total_frames, max_frames)
        
        self.video_frames = []
        self.keypoints_2d_sequence = []
        
        pbar = tqdm(total=total_frames, desc="提取视频和关键点")
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            self.video_frames.append(frame.copy())
            
            # Detect
            # 为了速度，可以每隔几帧检测一次然后插值，这里演示逐帧
            temp_path = f"temp_{i}.jpg"
            cv2.imwrite(temp_path, frame)
            try:
                result = self.detector.predict(temp_path)
                kps = result['keypoints']
                if np.sum(kps[:, 2] > 0.3) >= 8:
                    kps_train = self.mapper.map_ap10k_to_training(kps)
                    self.keypoints_2d_sequence.append(kps_train[:, :2])
                else:
                    self.keypoints_2d_sequence.append(np.full((17, 2), np.nan))
            except:
                self.keypoints_2d_sequence.append(np.full((17, 2), np.nan))
                
            if os.path.exists(temp_path): os.remove(temp_path)
            pbar.update(1)
        pbar.close()
        cap.release()
        return len(self.video_frames)

    def normalize_root_relative(self, kps_seq):
        """
        归一化逻辑 (与 SyntheticAnimalDataset 训练时一致):
        1. Root Relative (减去根节点)
        2. Scale Normalization (除以最大绝对值)
        """
        normalized_seq = []
        stored_scales = [] # 用于反归一化 (Visualization purpose)
        
        for kps in kps_seq:
            if np.any(np.isnan(kps)):
                normalized_seq.append(kps) # Keep NaN
                stored_scales.append(1.0)
                continue
                
            # 1. Root Relative
            # 假设第0个关节是根节点 (Hip/Pelvis)
            root = kps[0:1, :] 
            kps_centered = kps - root
            
            # 2. Scale
            max_val = np.max(np.abs(kps_centered))
            if max_val < 1e-5: max_val = 1.0
            
            kps_norm = kps_centered / max_val
            
            # Flip Y to match Model's Up-positive convention (Image is Down-positive)
            kps_norm[:, 1] *= -1
            
            normalized_seq.append(kps_norm)
            stored_scales.append(max_val)
            
        return np.array(normalized_seq), stored_scales

    def convert_2d_to_3d(self):
        print("🔄 转换 2D -> 3D...")
        if not self.keypoints_2d_sequence: return False
        
        kps_2d_arr = np.array(self.keypoints_2d_sequence)
        kps_norm_arr, scales = self.normalize_root_relative(kps_2d_arr)
        
        self.keypoints_3d_sequence = []
        seq_len = 27
        
        # 滑动窗口处理
        # 为简单起见，我们逐帧滑动，或者分块
        # 这里使用简单的 padding 策略
        
        num_frames = len(kps_norm_arr)
        
        # Pad beginning
        pad_len = seq_len // 2
        padded_input = np.pad(kps_norm_arr, ((pad_len, pad_len), (0,0), (0,0)), mode='edge')
        
        # Inference Loop
        bs = 32
        
        # 准备 batch
        input_batches = []
        indices = []
        
        current_batch = []
        
        for i in range(num_frames):
            # 取窗口 [i : i+seq_len]
            # 对应的输出应该是中间帧，或者最后一帧？
            # 训练是 seq -> all joints 3d.
            # 为了平滑，我们通常取 centered window.
            # padding 之后，第 i 帧 对应 window [i : i+seq_len] (centered at i + pad_len)
            
            window = padded_input[i : i+seq_len]
            if np.any(np.isnan(window)):
                # 如果窗口内有 NaN，这一帧可能不准，但还是得预测
                # fill nan with 0
                window = np.nan_to_num(window, 0.0)
                
            current_batch.append(window)
            if len(current_batch) == bs or i == num_frames - 1:
                input_batches.append(np.array(current_batch))
                current_batch = []

        # Run Model
        all_preds = []
        with torch.no_grad():
            for batch_np in tqdm(input_batches, desc="3D 推理"):
                batch_tensor = torch.tensor(batch_np, dtype=torch.float32).to(self.device)
                
                # Species ID
                species = torch.full((batch_tensor.shape[0],), self.target_species_id, dtype=torch.long).to(self.device)
                
                # Forward
                pred_norm = self.model_3d(batch_tensor, species) # (B, 27, 17, 3)
                
                # 取中间帧还是最后一帧？
                # 训练时可以取序列。
                # Transformer 输出也是序列 (B, T, J, 3)
                # 我们取中间帧 pad_len
                pred_frame = pred_norm[:, seq_len // 2, :, :] # (B, 17, 3)
                
                all_preds.append(pred_frame.cpu().numpy())
                
        all_preds = np.concatenate(all_preds, axis=0) # (N, 17, 3)
        
        # 反归一化 (仅 Scale，因为是 Root Relative 的 3D)
        # 3D 预测也是 Root Relative 的
        # 我们给它乘以一个系数，让它看起来大一点
        # 或者乘以原始 2D 的 scale
        
        for i, pred_3d_norm in enumerate(all_preds):
            scale = scales[i] if i < len(scales) else 100.0
            if scale == 1.0: scale = 100.0 # Default if unknown
            
            # 乘以 Scale 还原物理大小 (Approx)
            # 这里的 Scale 是 2D pixel scale。
            # 直接用它会让 3D 看起来和 2D 像素空间对应。
            pred_3d = pred_3d_norm * scale * 2.0 # 2.0 是经验系数 (Z-depth usually larger)
            
            self.keypoints_3d_sequence.append(pred_3d)
            
        print(f"✅ 3D序列生成完毕: {len(self.keypoints_3d_sequence)} 帧")
        return True

    def visualize_combined(self, video_path):
        if not self.extract_video_frames(video_path): return
        if not self.convert_2d_to_3d(): return
        
        print("🎬 启动可视化界面...")
        import matplotlib
        matplotlib.use('TkAgg')
        
        fig = plt.figure(figsize=(18, 9))
        ax_2d = fig.add_subplot(121)
        ax_3d = fig.add_subplot(122, projection='3d')
        
        ax_2d.axis('off')
        ax_2d.set_title("Input Video")
        ax_3d.set_title("3D Reconstruction")
        
        # Plot objects
        img_plot = None
        scatter_3d = None
        lines_3d = []
        
        # Axis limits
        valid_3d = [p for p in self.keypoints_3d_sequence if not np.any(np.isnan(p))]
        if valid_3d:
            valid_3d = np.array(valid_3d)
            limit = np.max(np.abs(valid_3d)) * 1.2
            ax_3d.set_xlim(-limit, limit)
            ax_3d.set_ylim(-limit, limit)
            ax_3d.set_zlim(-limit, limit)
        
        ax_3d.view_init(elev=20, azim=45)
        
        def update(frame_idx):
            nonlocal img_plot, scatter_3d, lines_3d
            
            # 2D
            if frame_idx < len(self.video_frames):
                frame = self.video_frames[frame_idx].copy()
                kps = self.keypoints_2d_sequence[frame_idx]
                
                # Draw Skeleton
                if not np.any(np.isnan(kps)):
                    for s, e in list(SKELETON_GROUPS['trunk']['edges']) + list(SKELETON_GROUPS['front_left']['edges']) + \
                                list(SKELETON_GROUPS['front_right']['edges']) + list(SKELETON_GROUPS['back_left']['edges']) + \
                                list(SKELETON_GROUPS['back_right']['edges']):
                        if s < len(kps) and e < len(kps):
                            pt1 = (int(kps[s][0]), int(kps[s][1]))
                            pt2 = (int(kps[e][0]), int(kps[e][1]))
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                            
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if img_plot is None:
                    img_plot = ax_2d.imshow(frame)
                else:
                    img_plot.set_data(frame)
            
            # 3D
            if frame_idx < len(self.keypoints_3d_sequence):
                pose = self.keypoints_3d_sequence[frame_idx]
                if scatter_3d is not None: scatter_3d.remove()
                for l in lines_3d: l.remove()
                lines_3d = []
                
                if not np.any(np.isnan(pose)):
                    # Rotate for better view (Optional)
                    # pose = pose @ R_x(np.pi/2) ...
                    
                    scatter_3d = ax_3d.scatter(pose[:,0], pose[:,1], pose[:,2], c='r', s=20)
                    
                    for group_name, info in SKELETON_GROUPS.items():
                        for s, e in info['edges']:
                             lines_3d.append(ax_3d.plot(
                                 [pose[s,0], pose[e,0]],
                                 [pose[s,1], pose[e,1]],
                                 [pose[s,2], pose[e,2]],
                                 color=info['color']
                             )[0])
                             
            ax_2d.set_title(f"Frame {frame_idx}")

        total_frames = len(self.video_frames)
        ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50)
        
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='video/luotuo.mp4', help='Path to video')
    parser.add_argument('--species_id', type=int, default=0, help='Species ID (0-19)')
    args = parser.parse_args()
    
    checkpoint = 'checkpoints_enhanced/best_synth_model.pt'
    onnx_path = 'model/ap10k/end2end.onnx'
    
    if not os.path.exists(args.video):
        print(f"Please provide valid video path. {args.video} not found.")
        # Try finding a video
        if os.path.exists("video"):
            vids = os.listdir("video")
            if vids:
                args.video = os.path.join("video", vids[0])
                print(f"Using found video: {args.video}")
    
    viz = VideoTo3DVisualizer(checkpoint, onnx_path, target_species_id=args.species_id)
    if viz.model_3d:
        viz.visualize_combined(args.video)

if __name__ == '__main__':
    main()