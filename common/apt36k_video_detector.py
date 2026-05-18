import onnxruntime as ort
import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Any


class OneEuroFilter:
    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.0, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.first_time = True

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray) -> np.ndarray:
        if self.first_time:
            self.first_time = False
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()

        dx = (x - self.x_prev) * self.freq

        alpha_d = self._alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat.copy()

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._alpha(cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat.copy()

        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.first_time = True


class APT36KVideoPoseDetector:
    def __init__(self, onnx_path: str,
                 temporal_smooth: bool = True,
                 smooth_freq: float = 30.0,
                 smooth_min_cutoff: float = 0.8,
                 smooth_beta: float = 0.01):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape

        self.input_size = (input_shape[3], input_shape[2])
        print(f"模型输入尺寸: {self.input_size}")

        self.output_names = [output.name for output in self.session.get_outputs()]
        print(f"模型输出: {self.output_names}")

        output_shape = self.session.get_outputs()[0].shape
        self.num_keypoints = output_shape[1] if len(output_shape) == 4 else output_shape[2]
        self.is_heatmap = len(output_shape) == 4
        print(f"关键点数量: {self.num_keypoints}, 输出类型: {'heatmap' if self.is_heatmap else 'SimCC'}")

        self.temporal_smooth = temporal_smooth
        self.smooth_freq = smooth_freq
        self.smooth_min_cutoff = smooth_min_cutoff
        self.smooth_beta = smooth_beta

        self.keypoint_info = {
            0: dict(name='L_Eye', id=0, color=[0, 255, 0], type='upper', swap='R_Eye'),
            1: dict(name='R_Eye', id=1, color=[255, 128, 0], type='upper', swap='L_Eye'),
            2: dict(name='Nose', id=2, color=[51, 153, 255], type='upper', swap=''),
            3: dict(name='Neck', id=3, color=[51, 153, 255], type='upper', swap=''),
            4: dict(name='Root of tail', id=4, color=[51, 153, 255], type='lower', swap=''),
            5: dict(name='L_Shoulder', id=5, color=[51, 153, 255], type='upper', swap='R_Shoulder'),
            6: dict(name='L_Elbow', id=6, color=[51, 153, 255], type='upper', swap='R_Elbow'),
            7: dict(name='L_F_Paw', id=7, color=[0, 255, 0], type='upper', swap='R_F_Paw'),
            8: dict(name='R_Shoulder', id=8, color=[0, 255, 0], type='upper', swap='L_Shoulder'),
            9: dict(name='R_Elbow', id=9, color=[255, 128, 0], type='upper', swap='L_Elbow'),
            10: dict(name='R_F_Paw', id=10, color=[0, 255, 0], type='lower', swap='L_F_Paw'),
            11: dict(name='L_Hip', id=11, color=[255, 128, 0], type='lower', swap='R_Hip'),
            12: dict(name='L_Knee', id=12, color=[255, 128, 0], type='lower', swap='R_Knee'),
            13: dict(name='L_B_Paw', id=13, color=[0, 255, 0], type='lower', swap='R_B_Paw'),
            14: dict(name='R_Hip', id=14, color=[0, 255, 0], type='lower', swap='L_Hip'),
            15: dict(name='R_Knee', id=15, color=[0, 255, 0], type='lower', swap='L_Knee'),
            16: dict(name='R_B_Paw', id=16, color=[0, 255, 0], type='lower', swap='L_B_Paw'),
        }

        self.skeleton_info = {
            0: dict(link=('L_Eye', 'R_Eye'), id=0, color=[0, 0, 255]),
            1: dict(link=('L_Eye', 'Nose'), id=1, color=[0, 0, 255]),
            2: dict(link=('R_Eye', 'Nose'), id=2, color=[0, 0, 255]),
            3: dict(link=('Nose', 'Neck'), id=3, color=[0, 255, 0]),
            4: dict(link=('Neck', 'Root of tail'), id=4, color=[0, 255, 0]),
            5: dict(link=('Neck', 'L_Shoulder'), id=5, color=[0, 255, 255]),
            6: dict(link=('L_Shoulder', 'L_Elbow'), id=6, color=[0, 255, 255]),
            7: dict(link=('L_Elbow', 'L_F_Paw'), id=7, color=[0, 255, 255]),
            8: dict(link=('Neck', 'R_Shoulder'), id=8, color=[6, 156, 250]),
            9: dict(link=('R_Shoulder', 'R_Elbow'), id=9, color=[6, 156, 250]),
            10: dict(link=('R_Elbow', 'R_F_Paw'), id=10, color=[6, 156, 250]),
            11: dict(link=('Root of tail', 'L_Hip'), id=11, color=[0, 255, 255]),
            12: dict(link=('L_Hip', 'L_Knee'), id=12, color=[0, 255, 255]),
            13: dict(link=('L_Knee', 'L_B_Paw'), id=13, color=[0, 255, 255]),
            14: dict(link=('Root of tail', 'R_Hip'), id=14, color=[6, 156, 250]),
            15: dict(link=('R_Hip', 'R_Knee'), id=15, color=[6, 156, 250]),
            16: dict(link=('R_Knee', 'R_B_Paw'), id=16, color=[6, 156, 250]),
        }

        self.name_to_id = {info['name']: kid for kid, info in self.keypoint_info.items()}

    def get_affine_transform(self, center: Tuple[float, float], scale: float, rot: float,
                             output_size: Tuple[int, int]) -> np.ndarray:
        src_w = scale
        dst_w = output_size[0]
        dst_h = output_size[1]

        src_dir = self.get_dir([0, src_w * -0.5], rot)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(src, dst)
        return trans

    def get_dir(self, src_point: np.ndarray, rot_rad: float) -> np.ndarray:
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return np.array(src_result, dtype=np.float32)

    def get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def preprocess(self, image: np.ndarray, center: Tuple[float, float], scale: float) -> np.ndarray:
        trans = self.get_affine_transform(center, scale, 0, self.input_size)

        input_image = cv2.warpAffine(
            image, trans,
            (int(self.input_size[0]), int(self.input_size[1])),
            flags=cv2.INTER_LINEAR
        )

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        input_image = (input_image - mean) / std

        input_image = np.transpose(input_image, (2, 0, 1)).astype(np.float32)
        input_image = np.expand_dims(input_image, axis=0)

        return input_image

    def _taylor_refine(self, heatmap: np.ndarray, x: int, y: int) -> Tuple[float, float]:
        h, w = heatmap.shape

        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            return float(x), float(y)

        dx = 0.5 * (heatmap[y, x + 1] - heatmap[y, x - 1])
        dy = 0.5 * (heatmap[y + 1, x] - heatmap[y - 1, x])

        dxx = heatmap[y, x + 1] - 2.0 * heatmap[y, x] + heatmap[y, x - 1]
        dyy = heatmap[y + 1, x] - 2.0 * heatmap[y, x] + heatmap[y - 1, x]
        dxy = 0.25 * (heatmap[y + 1, x + 1] - heatmap[y + 1, x - 1]
                      - heatmap[y - 1, x + 1] + heatmap[y - 1, x - 1])

        det = dxx * dyy - dxy * dxy
        if abs(det) < 1e-6:
            return float(x), float(y)

        H_inv_00 = dyy / det
        H_inv_01 = -dxy / det
        H_inv_10 = -dxy / det
        H_inv_11 = dxx / det

        offset_x = -(H_inv_00 * dx + H_inv_01 * dy)
        offset_y = -(H_inv_10 * dx + H_inv_11 * dy)

        offset_x = np.clip(offset_x, -1.0, 1.0)
        offset_y = np.clip(offset_y, -1.0, 1.0)

        return float(x + offset_x), float(y + offset_y)

    def heatmap_decode(self, heatmap: np.ndarray) -> np.ndarray:
        batch_size, num_kpts, h, w = heatmap.shape
        keypoints = []

        for i in range(num_kpts):
            kp_heatmap = heatmap[0, i]
            max_idx = np.argmax(kp_heatmap)
            max_y = max_idx // w
            max_x = max_idx % w
            score = float(kp_heatmap[max_y, max_x])

            refined_x, refined_y = self._taylor_refine(kp_heatmap, max_x, max_y)

            x_norm = refined_x / (w - 1)
            y_norm = refined_y / (h - 1)

            keypoints.append([x_norm, y_norm, score])

        return np.array(keypoints)

    def predict_frame(self, frame: np.ndarray, bbox: List[float] = None) -> Dict[str, Any]:
        original_h, original_w = frame.shape[:2]

        if bbox is None:
            bbox = [0, 0, original_w, original_h]

        x1, y1, x2, y2 = bbox
        bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        bbox_size = np.array([x2 - x1, y2 - y1])

        scale = max(bbox_size) * 1.25

        input_tensor = self.preprocess(frame, bbox_center, scale)

        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        if self.is_heatmap:
            keypoints_normalized = self.heatmap_decode(outputs[0])
        else:
            simcc_x, simcc_y = outputs
            keypoints_normalized = self.simcc_decode(simcc_x, simcc_y)

        keypoints_original = self.transform_keypoints_to_original(
            keypoints_normalized, bbox_center, scale, (original_w, original_h)
        )

        return {
            'keypoints': keypoints_original,
            'bbox': bbox,
            'image_shape': (original_h, original_w),
            'keypoints_normalized': keypoints_normalized
        }

    def simcc_decode(self, simcc_x: np.ndarray, simcc_y: np.ndarray) -> np.ndarray:
        keypoints = []

        batch_size, num_kpts, simcc_dim_x = simcc_x.shape
        _, _, simcc_dim_y = simcc_y.shape

        for i in range(num_kpts):
            max_idx_x = np.argmax(simcc_x[0, i])
            max_idx_y = np.argmax(simcc_y[0, i])

            score_x = simcc_x[0, i, max_idx_x]
            score_y = simcc_y[0, i, max_idx_y]
            score = (score_x + score_y) / 2

            x = max_idx_x / (simcc_dim_x - 1)
            y = max_idx_y / (simcc_dim_y - 1)

            keypoints.append([x, y, score])

        return np.array(keypoints)

    def transform_keypoints_to_original(self, keypoints: np.ndarray, center: np.ndarray,
                                        scale: float, image_size: Tuple[int, int]) -> np.ndarray:
        output_size = self.input_size
        trans = self.get_affine_transform(center, scale, 0, output_size)
        trans_inv = cv2.invertAffineTransform(trans)

        keypoints_original = []
        for kp in keypoints:
            x_norm, y_norm, score = kp
            x_input = x_norm * (output_size[0] - 1)
            y_input = y_norm * (output_size[1] - 1)

            point = np.array([x_input, y_input, 1.0])
            x_orig, y_orig = np.dot(trans_inv, point)[:2]

            x_orig = np.clip(x_orig, 0, image_size[0] - 1)
            y_orig = np.clip(y_orig, 0, image_size[1] - 1)

            keypoints_original.append([x_orig, y_orig, score])

        return np.array(keypoints_original)

    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray,
                       confidence_threshold: float = 0.3) -> np.ndarray:
        vis_frame = frame.copy()

        for skeleton_id, skeleton in self.skeleton_info.items():
            start_name, end_name = skeleton['link']
            color = skeleton['color']

            start_id = self.name_to_id[start_name]
            end_id = self.name_to_id[end_name]

            start_kp = keypoints[start_id]
            end_kp = keypoints[end_id]

            if start_kp[2] > confidence_threshold and end_kp[2] > confidence_threshold:
                start_point = (int(start_kp[0]), int(start_kp[1]))
                end_point = (int(end_kp[0]), int(end_kp[1]))
                cv2.line(vis_frame, start_point, end_point,
                         (color[0], color[1], color[2]), 2, cv2.LINE_AA)

        for kp_id, kp in enumerate(keypoints):
            x, y, score = kp
            if score > confidence_threshold:
                color = self.keypoint_info[kp_id]['color']
                cv2.circle(vis_frame, (int(x), int(y)), 5,
                           (color[0], color[1], color[2]), -1, cv2.LINE_AA)
                cv2.putText(vis_frame, self.keypoint_info[kp_id]['name'],
                            (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (color[0], color[1], color[2]), 1, cv2.LINE_AA)

        return vis_frame

    def process_video(self, video_path: str, output_path: str = None,
                      confidence_threshold: float = 0.3,
                      max_frames: int = -1,
                      show_preview: bool = False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if video_fps <= 0:
            video_fps = 30.0

        if reported_frames > 0 and max_frames < 0:
            max_frames = reported_frames
        elif max_frames < 0:
            max_frames = 999999

        duration = reported_frames / video_fps if video_fps > 0 and reported_frames > 0 else 0
        print(f"视频信息: {width}x{height}, {video_fps:.1f} fps, "
              f"共 {max_frames} 帧 ({duration:.1f} 秒)")

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        else:
            out = None

        if self.temporal_smooth:
            self._filters = [
                OneEuroFilter(
                    freq=video_fps,
                    min_cutoff=self.smooth_min_cutoff,
                    beta=self.smooth_beta,
                    d_cutoff=1.0
                ) for _ in range(self.num_keypoints)
            ]
        else:
            self._filters = None

        frame_count = 0
        results = []
        t_start = time.time()
        t_last_report = t_start

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.predict_frame(frame)
            keypoints_raw = result['keypoints']
            results.append(result)

            if self._filters is not None:
                smoothed = keypoints_raw.copy()
                for k in range(self.num_keypoints):
                    if keypoints_raw[k, 2] > confidence_threshold:
                        smoothed_xy = self._filters[k].filter(keypoints_raw[k, :2])
                        smoothed[k, :2] = smoothed_xy
                keypoints = smoothed
            else:
                keypoints = keypoints_raw

            frame_count += 1

            n_visible = int(np.sum(keypoints[:, 2] > confidence_threshold))
            t_now = time.time()
            if t_now - t_last_report >= 1.0:
                elapsed = t_now - t_start
                fps_actual = frame_count / elapsed
                eta = (max_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                print(f"\r帧 {frame_count}/{max_frames} | {n_visible}/{self.num_keypoints} 关键点 | "
                      f"{fps_actual:.1f} fps | 已耗时 {elapsed:.0f}s | 剩余约 {eta:.0f}s",
                      end='', flush=True)
                t_last_report = t_now

            vis_frame = self.draw_keypoints(frame, keypoints, confidence_threshold)

            if show_preview:
                cv2.imshow('APT36K Pose Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if out:
                out.write(vis_frame)

        elapsed = time.time() - t_start
        fps_actual = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n处理完成: {frame_count} 帧, 总耗时 {elapsed:.1f}s ({fps_actual:.2f} fps)")

        cap.release()
        if out:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()

        if output_path:
            print(f"输出视频: {output_path}")

        return results


def main():
    detector = APT36KVideoPoseDetector("model/apt36k/vitpose-b-apt36k.onnx")

    video_path = "video/liebao.mp4"
    output_path = "video/liebao_apt36k_result.mp4"

    try:
        detector.process_video(
            video_path,
            output_path=output_path,
            confidence_threshold=0.3,
            max_frames=-1,
            show_preview=False
        )
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
