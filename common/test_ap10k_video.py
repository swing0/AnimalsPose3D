# test_ap10k_video.py
import os

import cv2
import numpy as np

from common.ap10k_detector import AP10KAnimalPoseDetector


def debug_ap10k_detection(video_path, output_path="../video/test_ap10k.mp4", max_frames=500):
    """单独调试AP10K检测器"""
    cap = cv2.VideoCapture(video_path)
    detector = AP10KAnimalPoseDetector("../model/ap10k/end2end.onnx")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30,
                          (int(cap.get(3)), int(cap.get(4))))

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        temp_path = f"temp_debug_{i}.jpg"
        cv2.imwrite(temp_path, frame)

        # 检测关键点
        result = detector.predict(temp_path)
        keypoints = result['keypoints']

        # 绘制检测结果
        debug_frame = frame.copy()
        for j, kp in enumerate(keypoints):
            if kp[2] > 0.3:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(debug_frame, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(debug_frame, f"{j}", (x + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 打印检测信息
        print(f"帧 {i}: 检测到 {np.sum(keypoints[:, 2] > 0.3)}/{len(keypoints)} 个关键点")
        # print(f"  坐标范围: X[{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
        # print(f"           Y[{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")

        out.write(debug_frame)
        os.remove(temp_path)

    cap.release()
    out.release()
    print(f"AP10K调试视频保存到: {output_path}")



if __name__ == '__main__':
    # test_ap10k_detector()
    debug_ap10k_detection("../video/test_video.mp4")