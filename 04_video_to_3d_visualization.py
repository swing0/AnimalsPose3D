# 17_video_to_3d_visualization_animal_poseformer.py
# 视频到3D关键点的可视化工具 - 适配 AnimalPoseFormer 模型

import os
import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from tqdm import tqdm
import sys

# 添加路径
sys.path.append('./common')

try:
    from common.apt36k_video_detector import APT36KVideoPoseDetector, OneEuroFilter
    from common.keypoint_mapper import KeypointMapper
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

SKELETON_GROUPS = {
    'trunk': {'edges': [(0, 4), (4, 3), (3, 1), (3, 2)], 'color': 'black', 'label': 'Head & Neck'},
    'front_left': {'edges': [(4, 5), (5, 6), (6, 7)], 'color': 'red', 'label': 'Front Left'},
    'front_right': {'edges': [(4, 8), (8, 9), (9, 10)], 'color': 'orange', 'label': 'Front Right'},
    'back_left': {'edges': [(0, 11), (11, 12), (12, 13)], 'color': 'blue', 'label': 'Back Left'},
    'back_right': {'edges': [(0, 14), (14, 15), (15, 16)], 'color': 'cyan', 'label': 'Back Right'}
}

BONE_CHAIN = [
    (0, 4), (4, 3), (3, 1), (3, 2),
    (4, 5), (5, 6), (6, 7),
    (4, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16),
]

SYMMETRY_PAIRS = {
    (3, 1): (3, 2),
    (4, 5): (4, 8), (5, 6): (8, 9), (6, 7): (9, 10),
    (0, 11): (0, 14), (11, 12): (14, 15), (12, 13): (15, 16),
}


def enforce_per_frame_consistency(kps_3d, edges, symmetry_pairs):
    kps = kps_3d.copy()

    target = {}
    for p, c in edges:
        current = np.linalg.norm(kps[c] - kps[p])
        if (p, c) in symmetry_pairs:
            mirror = symmetry_pairs[(p, c)]
            mirror_len = np.linalg.norm(kps[mirror[1]] - kps[mirror[0]])
            target[(p, c)] = (current + mirror_len) / 2.0
        else:
            target[(p, c)] = current

    for (parent, child), target_len in target.items():
        vec = kps[child] - kps[parent]
        current_len = np.linalg.norm(vec)
        if current_len < 1e-6:
            continue
        kps[child] = kps[parent] + vec * (target_len / current_len)

    return kps

MODEL_META = {
    'animalposeformer': {'seq_len': 27, 'ckpt': 'checkpoints/animal_poseformer_best_model.pt'},
    'poseformer':        {'seq_len': 27, 'ckpt': 'checkpoints/compare_poseformer_best.pt'},
    'poseformerv2':      {'seq_len': 27, 'ckpt': 'checkpoints/compare_poseformerv2_best.pt'},
    'videopose3d':       {'seq_len': 27, 'ckpt': 'checkpoints/compare_videopose3d_best.pt'},
    'dstformer':         {'seq_len': 27, 'ckpt': 'checkpoints/compare_dstformer_best.pt'},
    'dstformer_full':    {'seq_len': 243, 'ckpt': 'checkpoints/compare_dstformer_full_best.pt'},
    'motionagformer':   {'seq_len': 27, 'ckpt': 'checkpoints/compare_motionagformer_best.pt'},
    'stcformer':        {'seq_len': 27, 'ckpt': 'checkpoints/compare_stcformer_best.pt'},
    'mixste':           {'seq_len': 27, 'ckpt': 'checkpoints/compare_mixste_best.pt'},
}


def _build_3d_model(model_name, seq_len, device):
    if model_name == 'animalposeformer':
        from common.animal_poseformer import AnimalPoseFormer
        return AnimalPoseFormer(
            num_frame=seq_len, num_joints=17, in_chans=2,
            embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2.,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.2, use_lme=True,
            num_frame_kept=seq_len, num_coeff_kept=seq_len
        ).to(device)
    elif model_name == 'poseformer':
        from common.poseformer.model_poseformer import PoseTransformer
        return PoseTransformer(
            num_frame=seq_len, num_joints=17, in_chans=2,
            embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2.,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.2
        ).to(device)
    elif model_name == 'poseformerv2':
        from common.poseformerv2.model_poseformerV2 import PoseTransformerV2
        import argparse as _ap
        args_ns = _ap.Namespace(
            embed_dim_ratio=32, depth=4,
            number_of_kept_frames=seq_len, number_of_kept_coeffs=seq_len
        )
        return PoseTransformerV2(
            num_frame=seq_len, num_joints=17, in_chans=2,
            num_heads=8, mlp_ratio=2., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, args=args_ns
        ).to(device)
    elif model_name == 'videopose3d':
        from common.videopose3D.model import TemporalModel
        return TemporalModel(
            num_joints_in=17, in_features=2, num_joints_out=17,
            filter_widths=[3, 3, 3], dropout=0.25, channels=1024
        ).to(device)
    elif model_name == 'dstformer':
        from common.MotionBERT.DSTformer import DSTformer
        return DSTformer(
            dim_in=2, dim_out=3, dim_feat=448, depth=4, num_heads=8,
            mlp_ratio=2, num_joints=17, maxlen=seq_len
        ).to(device)
    elif model_name == 'dstformer_full':
        from common.MotionBERT.DSTformer import DSTformer
        return DSTformer(
            dim_in=2, dim_out=3, dim_feat=512, dim_rep=512,
            depth=5, num_heads=8, mlp_ratio=4,
            num_joints=17, maxlen=seq_len
        ).to(device)
    elif model_name == 'motionagformer':
        from common.MotionAGFormer.MotionAGFormer import MotionAGFormer
        return MotionAGFormer(
            n_layers=8, dim_in=2, dim_feat=256, dim_rep=512, dim_out=3,
            mlp_ratio=4, num_heads=8, num_joints=17, n_frames=seq_len
        ).to(device)
    elif model_name == 'stcformer':
        import argparse as _ap
        from common.STCFormer.stcformer import Model
        stc_args = _ap.Namespace(
            layers=8, d_hid=320, frames=seq_len,
            n_joints=17, out_joints=17
        )
        return Model(stc_args).to(device)
    elif model_name == 'mixste':
        import argparse as _ap
        from common.MixSTE.model_cross import MixSTE2
        return MixSTE2(
            num_frame=seq_len, num_joints=17, in_chans=2,
            embed_dim_ratio=512, depth=4, num_heads=8, mlp_ratio=2.,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.2
        ).to(device)
    raise ValueError(f"Unknown model: {model_name}")


class VideoTo3DVisualizer:
    def __init__(self, model_checkpoint, onnx_model_path,
                 smooth_2d: bool = True, model_name: str = 'animalposeformer'):
        print("🎯 初始化视频到3D可视化器...")
        self.detector = APT36KVideoPoseDetector(onnx_model_path)
        self.mapper = KeypointMapper()
        self.smooth_2d = smooth_2d
        self.model_name = model_name
        self.meta = MODEL_META[model_name]
        self.seq_len = self.meta['seq_len']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_3d = self.load_3d_model(model_checkpoint)
        
        self.video_frames = []
        self.keypoints_2d_sequence = []
        self.video_fps = 30.0
        self.keypoints_3d_sequence = []
        
        print("✅ 可视化器初始化完成")
    
    def load_3d_model(self, checkpoint_path):
        print(f"📥 加载3D模型 ({self.model_name}): {checkpoint_path}")
        model = _build_3d_model(self.model_name, self.seq_len, self.device)

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
    
    def _get_cache_path(self, video_path: str, total_frames: int) -> str:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cache_dir = os.path.join("video", "temp2D")
        prefix = f"{video_name}_{total_frames}frames"
        suffix = "_smooth" if self.smooth_2d else ""
        return os.path.join(cache_dir, f"{prefix}{suffix}.npz")

    def extract_video_frames(self, video_path, max_frames=300):
        print(f"🎥 处理视频: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError(f"无法打开视频: {video_path}")
        
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0

        if reported_frames <= 0:
            total_frames = max_frames
        elif max_frames:
            total_frames = min(reported_frames, max_frames)
        else:
            total_frames = reported_frames

        self.video_fps = video_fps

        duration = total_frames / video_fps if video_fps > 0 else 0
        print(f"  视频信息: {total_frames} 帧, {video_fps:.1f} fps, {duration:.1f} 秒")

        cache_path = self._get_cache_path(video_path, total_frames)

        if os.path.exists(cache_path):
            print(f"  💾 命中缓存: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self.keypoints_2d_sequence = list(data['keypoints_2d'])
            cached_fps = float(data['video_fps'])
            if abs(cached_fps - video_fps) > 0.1:
                self.video_fps = cached_fps

            self.video_frames = []
            for _ in tqdm(range(total_frames), desc="读取视频帧"):
                ret, frame = cap.read()
                if not ret: break
                self.video_frames.append(frame.copy())
            cap.release()

            print(f"  ✅ 从缓存加载: {len(self.keypoints_2d_sequence)} 个关键点序列, "
                  f"{len(self.video_frames)} 帧")
            return len(self.video_frames)

        self.video_frames = []
        self.keypoints_2d_sequence = []
        
        if self.smooth_2d:
            filters_2d = [OneEuroFilter(
                freq=video_fps, min_cutoff=0.8, beta=0.01, d_cutoff=1.0
            ) for _ in range(17)]

        t_start = time.time()
        pbar = tqdm(total=total_frames, desc="提取视频和关键点")
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            self.video_frames.append(frame.copy())
            
            try:
                result = self.detector.predict_frame(frame)
                kps = result['keypoints']
                if np.sum(kps[:, 2] > 0.3) >= 8:
                    kps_train = self.mapper.map_ap10k_to_training(kps)
                else:
                    kps_train = np.full((17, 2), np.nan)

                if self.smooth_2d and not np.any(np.isnan(kps_train)):
                    for k in range(17):
                        kps_train[k, :] = filters_2d[k].filter(kps_train[k, :])

                self.keypoints_2d_sequence.append(kps_train[:, :2])
            except:
                self.keypoints_2d_sequence.append(np.full((17, 2), np.nan))
                
            pbar.update(1)
        pbar.close()
        cap.release()
        
        elapsed = time.time() - t_start
        actual_fps = len(self.video_frames) / elapsed if elapsed > 0 else 0
        print(f"  2D检测完成: {len(self.video_frames)} 帧, "
              f"耗时 {elapsed:.1f}s ({actual_fps:.2f} fps)")

        if len(self.keypoints_2d_sequence) > 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path,
                     keypoints_2d=np.array(self.keypoints_2d_sequence),
                     video_fps=self.video_fps)
            print(f"  💾 缓存已保存: {cache_path}")

        return len(self.video_frames)

    def normalize_root_relative(self, kps_seq):
        """
        归一化逻辑:
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
        print(f"🔄 转换 2D -> 3D ({self.model_name})...")
        if not self.keypoints_2d_sequence: return False

        kps_2d_arr = np.array(self.keypoints_2d_sequence)
        kps_norm_arr, scales = self.normalize_root_relative(kps_2d_arr)

        self.keypoints_3d_sequence = []
        num_frames = len(kps_norm_arr)

        pad_len = self.seq_len // 2
        padded_input = np.pad(kps_norm_arr, ((pad_len, pad_len), (0, 0), (0, 0)), mode='edge')

        bs = 32
        input_batches = []
        current_batch = []

        for i in range(num_frames):
            window = padded_input[i:i + self.seq_len]
            if np.any(np.isnan(window)):
                window = np.nan_to_num(window, 0.0)
            current_batch.append(window)
            if len(current_batch) == bs or i == num_frames - 1:
                input_batches.append(np.array(current_batch))
                current_batch = []

        all_preds = []
        with torch.no_grad():
            for batch_np in tqdm(input_batches, desc="3D 推理"):
                batch_tensor = torch.tensor(batch_np, dtype=torch.float32).to(self.device).contiguous()
                pred = self.model_3d(batch_tensor)

                if pred.shape[1] == 1:
                    pred_frame = pred[:, 0, :, :]
                else:
                    pred_frame = pred[:, self.seq_len // 2, :, :]

                all_preds.append(pred_frame.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)

        for i, pred_3d_norm in enumerate(all_preds):
            scale = scales[i] if i < len(scales) else 100.0
            if scale == 1.0:
                scale = 100.0
            pred_3d = pred_3d_norm * scale * 2.0
            self.keypoints_3d_sequence.append(pred_3d)

        for i in range(len(self.keypoints_3d_sequence)):
            if not np.any(np.isnan(self.keypoints_3d_sequence[i])):
                self.keypoints_3d_sequence[i] = enforce_per_frame_consistency(
                    self.keypoints_3d_sequence[i], BONE_CHAIN, SYMMETRY_PAIRS
                )

        print(f"✅ 3D序列生成完毕: {len(self.keypoints_3d_sequence)} 帧 "
              f"(逐帧骨骼自洽)")
        return True

    def visualize_combined(self, video_path):
        if not self.extract_video_frames(video_path, max_frames=10000): return
        if not self.convert_2d_to_3d(): return

        total_frames = len(self.video_frames)
        base_interval = 1000.0 / self.video_fps

        print("🎬 启动可视化界面...")
        import matplotlib
        matplotlib.use('TkAgg')

        fig = plt.figure(figsize=(18, 9))
        ax_2d = fig.add_subplot(121)
        ax_3d = fig.add_subplot(122, projection='3d')

        ax_2d.axis('off')
        ax_2d.set_title("Input Video")
        ax_3d.set_title("3D Reconstruction (AnimalPoseFormer)")

        self.paused = False
        self.speed = 1.0

        vis_frames_2d = []
        for i in range(total_frames):
            frame = self.video_frames[i].copy()
            kps = self.keypoints_2d_sequence[i]
            if not np.any(np.isnan(kps)):
                for group_name, info in SKELETON_GROUPS.items():
                    for s, e in info['edges']:
                        if s < len(kps) and e < len(kps):
                            pt1 = (int(kps[s][0]), int(kps[s][1]))
                            pt2 = (int(kps[e][0]), int(kps[e][1]))
                            cv2.line(frame, pt1, pt2,
                                     (0, 255, 0) if group_name == 'trunk' else
                                     (0, 0, 255) if group_name == 'front_left' else
                                     (0, 165, 255) if group_name == 'front_right' else
                                     (255, 0, 0) if group_name == 'back_left' else
                                     (255, 255, 0), 2)
            vis_frames_2d.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        img_plot = ax_2d.imshow(vis_frames_2d[0])

        valid_3d = [p for p in self.keypoints_3d_sequence if not np.any(np.isnan(p))]
        if valid_3d:
            valid_3d = np.array(valid_3d)
            limit = np.max(np.abs(valid_3d)) * 1.2
            ax_3d.set_xlim(-limit, limit)
            ax_3d.set_ylim(-limit, limit)
            ax_3d.set_zlim(-limit, limit)

        ax_3d.view_init(elev=20, azim=-90)

        scatter_3d = ax_3d.scatter([], [], [], c='r', s=20)
        lines_3d = {}
        for group_name, info in SKELETON_GROUPS.items():
            for s, e in info['edges']:
                key = (group_name, s, e)
                lines_3d[key] = ax_3d.plot([], [], [], color=info['color'],
                                           linewidth=2)[0]

        ax_pause = plt.axes([0.25, 0.02, 0.1, 0.04])
        btn_pause = Button(ax_pause, 'pause', color='lightblue', hovercolor='0.975')

        ax_speed = plt.axes([0.38, 0.02, 0.2, 0.04])
        slider_speed = Slider(ax_speed, 'speed', 0.1, 3.0, valinit=1.0, valstep=0.1)

        ax_frame = plt.axes([0.62, 0.02, 0.3, 0.04])
        slider_frame = Slider(ax_frame, 'frame', 0, total_frames - 1,
                              valinit=0, valfmt='%d', valstep=1)

        def on_speed(val):
            self.speed = float(val)
            ani.event_source.interval = base_interval / self.speed

        slider_speed.on_changed(on_speed)

        frame_idx = [0]
        slider_programmatic = [False]

        def on_frame_slider(val):
            if not slider_programmatic[0]:
                self.paused = True
                btn_pause.label.set_text('continue')
                btn_pause.color = 'lightcoral'

        slider_frame.on_changed(on_frame_slider)

        def toggle_pause(event):
            self.paused = not self.paused
            btn_pause.label.set_text('continue' if self.paused else 'pause')
            btn_pause.color = 'lightcoral' if self.paused else 'lightblue'
            if not self.paused:
                frame_idx[0] = int(slider_frame.val)
            plt.draw()

        btn_pause.on_clicked(toggle_pause)

        def update(anim_frame):
            if self.paused:
                idx = int(slider_frame.val)
            else:
                idx = frame_idx[0]
                slider_programmatic[0] = True
                slider_frame.set_val(idx)
                slider_programmatic[0] = False
                frame_idx[0] = (idx + 1) % total_frames

            if idx < total_frames:
                img_plot.set_data(vis_frames_2d[idx])
                pose3d = self.keypoints_3d_sequence[idx] if idx < len(self.keypoints_3d_sequence) else None
                if pose3d is not None and not np.any(np.isnan(pose3d)):
                    scatter_3d._offsets3d = (pose3d[:, 0], pose3d[:, 1], pose3d[:, 2])
                    for group_name, info in SKELETON_GROUPS.items():
                        for s, e in info['edges']:
                            key = (group_name, s, e)
                            lines_3d[key].set_data([pose3d[s, 0], pose3d[e, 0]],
                                                   [pose3d[s, 1], pose3d[e, 1]])
                            lines_3d[key].set_3d_properties([pose3d[s, 2], pose3d[e, 2]])
                else:
                    scatter_3d._offsets3d = ([], [], [])
                    for key, line in lines_3d.items():
                        line.set_data([], [])
                        line.set_3d_properties([])

            ax_2d.set_title(f"Frame {idx}/{total_frames}"
                            f"{' (pause)' if self.paused else ''}"
                            f" x{self.speed:.1f}")

            return [img_plot, scatter_3d] + list(lines_3d.values())

        ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                       interval=base_interval, blit=False,
                                       repeat=True, cache_frame_data=False)

        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='video/animals.mp4', help='Path to video')
    parser.add_argument('--model', type=str, default='animalposeformer',
                        choices=list(MODEL_META.keys()), help='3D model name')
    args = parser.parse_args()

    meta = MODEL_META[args.model]
    checkpoint = meta['ckpt']
    onnx_path = 'model/apt36k/vitpose-b-apt36k.onnx'

    if not os.path.exists(args.video):
        print(f"Please provide valid video path. {args.video} not found.")
        if os.path.exists("video"):
            vids = [v for v in os.listdir("video") if v.endswith(".mp4")]
            if vids:
                args.video = os.path.join("video", vids[0])
                print(f"Using found video: {args.video}")

    viz = VideoTo3DVisualizer(checkpoint, onnx_path, model_name=args.model)
    if viz.model_3d:
        viz.visualize_combined(args.video)

if __name__ == '__main__':
    main()
