#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段3端到端: USB 摄像头实时帧 -> YOLO Pose -> NPU 推理统计
默认 1280x720@30, 跑 30 秒。输出 FPS/延迟/NPU 负载。
默认不保存视频; 仅在 --save 时保存 1 张带骨架标注的当前帧用于调试。
"""
import os, sys, time, json
import cv2, numpy as np

sys.path.insert(0, os.path.expanduser('~/posture_monitor'))
from pose.engine import YoloPoseRKNN

DEV = 17
NPU_LOAD = '/sys/kernel/debug/rknpu/load'

def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 30
    save = '--save' in sys.argv
    eng = YoloPoseRKNN()
    cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    assert cap.isOpened(), 'camera open failed'
    # 目标帧率设为推理能力的上限, 避免无意义抽帧
    infer_lat = []
    n_frames = 0
    n_person = 0
    t0 = time.time()
    last_sample = 0
    samples = []
    snap = None
    while time.time() - t0 < dur:
        ret, frame = cap.read()
        if not ret:
            continue
        boxes = eng.infer(frame)
        infer_lat.append(eng.last_infer_ms)
        n_frames += 1
        if boxes:
            n_person += 1
        if time.time() - last_sample > 5:
            samples.append(open(NPU_LOAD).read().strip())
            last_sample = time.time()
        if save and snap is None and boxes:
            snap = eng.draw(frame, boxes)
    el = time.time() - t0
    cap.release()
    lat = np.array(infer_lat)
    e2e_fps = n_frames / el
    print(f"运行 {el:.1f}s: 帧数={n_frames}  端到端FPS={e2e_fps:.1f}  "
          f"推理延迟avg={lat.mean():.1f}ms p95={np.percentile(lat,95):.1f}ms "
          f"有帧比例={n_person}/{n_frames}")
    print('NPU 负载采样:', samples)
    rep = {'duration_s': round(el,1), 'frames': n_frames, 'e2e_fps': round(e2e_fps,2),
           'infer_ms_avg': round(float(lat.mean()),2),
           'infer_ms_p95': round(float(np.percentile(lat,95)),2),
           'frames_with_person': n_person, 'npu_load_samples': samples}
    with open(os.path.expanduser('~/posture_monitor/logs/stage3_live.json'), 'w') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    if snap is not None:
        out = os.path.expanduser('~/posture_monitor/logs/stage3_live_frame.jpg')
        cv2.imwrite(out, snap)
        print('已保存 1 张调试帧:', out)
    eng.release()

if __name__ == '__main__':
    main()
