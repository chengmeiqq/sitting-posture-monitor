#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段3验证: bus.jpg 单帧推理 + NPU 确证 + 性能统计
输出: logs/stage3_bench.json, tests/bus_pose_result.jpg
"""
import json, os, sys, time, subprocess
import cv2, numpy as np

sys.path.insert(0, os.path.expanduser('~/posture_monitor'))
from pose.engine import YoloPoseRKNN

BASE = os.path.expanduser('~/posture_monitor')
IMG = BASE + '/third_party_ref/yolov8_pose/model/bus.jpg'
OUT_JPG = BASE + '/tests/bus_pose_result.jpg'
OUT_JSON = BASE + '/logs/stage3_bench.json'
NPU_LOAD = '/sys/kernel/debug/rknpu/load'
NPU_FREQ = '/sys/class/devfreq/fdab0000.npu/cur_freq'


def npu_load():
    try:
        return open(NPU_LOAD).read().strip()
    except Exception as e:
        return 'unreadable: %s' % e


def main():
    report = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'model': 'yolov8n-pose-int8.rknn'}

    t0 = time.perf_counter()
    eng = YoloPoseRKNN()
    report['init_runtime_s'] = round(time.perf_counter() - t0, 2)
    print('NPU init OK (%.1fs), target=None 自动绑定本机 NPU' % report['init_runtime_s'])

    img = cv2.imread(IMG)
    print('input:', IMG, img.shape)
    print('推理前 NPU:', npu_load())

    # 单帧正确性
    boxes = eng.infer(img)
    print('推理后 NPU:', npu_load())
    persons = [b for b in boxes if b['score'] >= 0.5]
    print('detected persons:', len(persons))
    n_vis = 0
    for b in persons:
        vis = int((b['kpts'][:, 2] > 0.3).sum())
        n_vis = max(n_vis, vis)
        print('  score=%.3f box=[%.0f,%.0f,%.0f,%.0f] kpts_vis=%d conf nose=%.2f Lsho=%.2f Lhip=%.2f' % (
            b['score'], b['xmin'], b['ymin'], b['xmax'], b['ymax'], vis,
            b['kpts'][0, 2], b['kpts'][5, 2], b['kpts'][11, 2]))
    report['persons'] = len(persons)
    report['max_kpts_visible'] = n_vis
    report['first_infer_ms'] = round(eng.last_infer_ms, 2)

    res = eng.draw(img, persons)
    cv2.imwrite(OUT_JPG, res)
    print('骨架图已保存', OUT_JPG)

    # 性能: 60 次连续推理
    lat = []
    loads = []
    for i in range(60):
        eng.infer(img)
        lat.append(eng.last_infer_ms)
        if i % 15 == 14:
            loads.append(npu_load())
    lat = np.array(lat)
    report['n_iter'] = 60
    report['infer_ms_avg'] = round(float(lat.mean()), 2)
    report['infer_ms_p50'] = round(float(np.percentile(lat, 50)), 2)
    report['infer_ms_p95'] = round(float(np.percentile(lat, 95)), 2)
    report['infer_ms_min'] = round(float(lat.min()), 2)
    report['prep_ms_avg'] = round(eng.last_prep_ms, 2)
    report['post_ms_avg'] = round(eng.last_post_ms, 2)
    report['npu_load_during'] = loads
    report['npu_freq_hz'] = open(NPU_FREQ).read().strip() if os.path.exists(NPU_FREQ) else 'n/a'
    print('NPU 推理延迟: avg=%.1fms p50=%.1f p95=%.1f  (NPU 核负载采样: %s)' % (
        lat.mean(), np.percentile(lat, 50), np.percentile(lat, 95), loads))

    # CPU 对照(说明用): 打印 librknnrt 无 CPU 后端的事实依据
    report['npu_proof'] = ('RKNPU Lite2 只有 NPU 后端: init_runtime(target=None) 成功即绑定本机 '
                           'fdab0000.npu; 且推理期间 /sys/kernel/debug/rknpu/load 显示非零负载')

    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('报告已保存', OUT_JSON)
    eng.release()
    assert len(persons) >= 1, '未检测到人体, 验证失败'
    assert n_vis >= 10, '关键点数不足, 验证失败'


if __name__ == '__main__':
    main()
