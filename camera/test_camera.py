#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段2：USB 摄像头 OpenCV 测试
用法: python3 test_camera.py [WxH] [时长秒]
默认 1920x1080, 测试 10 秒。只实时读帧统计, 不保存视频。
"""
import sys, time, datetime
import cv2
import numpy as np

DEVICE = 17  # /dev/video17 UGREEN Camera 4K (uvcvideo)

def parse_wh(arg):
    w, h = arg.lower().split('x')
    return int(w), int(h)

def run(width, height, duration_s):
    print(f"[{now()}] 打开 /dev/video{DEVICE} 目标 {width}x{height} MJPG ...")
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        print("!! 无法打开设备"); return 1

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_req = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace')
    print(f"实际分辨率={aw}x{ah} fourcc={fourcc} 驱动报告FPS={fps_req}")
    if (aw, ah) != (width, height):
        print(f"!! 分辨率不符: 请求 {width}x{height}, 实际 {aw}x{ah}")

    # 预热 1 秒
    t_end = time.time() + 1
    while time.time() < t_end:
        cap.read()

    frames = 0
    fails = 0
    times = []
    lum_mean = []
    lum_std = []
    last = time.time()
    t_start = time.time()
    while time.time() - t_start < duration_s:
        ret, frame = cap.read()
        now_t = time.time()
        if not ret or frame is None:
            fails += 1
            continue
        dt = now_t - last
        last = now_t
        if 0 < dt < 0.5:
            times.append(dt)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lum_mean.append(float(g.mean()))
        lum_std.append(float(g.std()))
        frames += 1

    elapsed = time.time() - t_start
    cap.release()

    print(f"\n[{now()}] 测试结果 ({elapsed:.1f}s):")
    print(f"  成功帧数={frames} 失败={fails}")
    if frames > 10 and times:
        arr = np.array(times)
        inst_fps = 1.0 / arr
        p5, p50, p95 = np.percentile(inst_fps, [5, 50, 95])
        print(f"  平均FPS={frames/elapsed:.1f}  中位瞬时FPS={p50:.1f}  P5={p5:.1f}  P95={p95:.1f}")
    if lum_mean:
        lm = np.array(lum_mean); ls = np.array(lum_std)
        print(f"  画面亮度均值={lm.mean():.1f} (范围 {lm.min():.1f}~{lm.max():.1f}) 对比度std均值={ls.mean():.1f}")
        verdict = "全黑/遮挡?" if lm.mean() < 10 else ("正常" if lm.mean() > 10 else "")
        print(f"  亮度判定: {verdict}")
    return 0

def now():
    return datetime.datetime.now().strftime('%H:%M:%S')

if __name__ == '__main__':
    w, h = parse_wh(sys.argv[1]) if len(sys.argv) > 1 else (1920, 1080)
    d = float(sys.argv[2]) if len(sys.argv) > 2 else 10
    sys.exit(run(w, h, d))
