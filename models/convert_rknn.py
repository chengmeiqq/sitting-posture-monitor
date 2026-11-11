#!/usr/bin/env python3
# 基于 airockchip/rknn_model_zoo examples/yolov8_pose/python/convert.py
# rk3588 INT8 混合量化转换: onnx -> rknn
import sys, os, time
from rknn.api import RKNN

ONNX = os.path.expanduser('~/posture_monitor/models/yolov8n-pose.onnx')
DATASET = os.path.expanduser('~/posture_monitor/models/coco_subset_20.txt')
OUT = os.path.expanduser('~/posture_monitor/models/yolov8n-pose-int8.rknn')

t0 = time.time()
rknn = RKNN(verbose=False)

print('--> Config model')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
            target_platform='rk3588')

# 官方 yolov8 系列用混合量化: step1 生成 .model/.data/.quantization.cfg
name = os.path.basename(ONNX).replace('.onnx', '')
print('--> Loading model')
ret = rknn.load_onnx(model=ONNX)
assert ret == 0, 'Load model failed: %d' % ret
print('load done %.0fs' % (time.time() - t0))
os.chdir(os.path.dirname(ONNX))
print('--> Hybrid quantization step1')
rknn.hybrid_quantization_step1(
    dataset=DATASET,
    proposal=False,
    custom_hybrid=[['/model.22/cv4.0/cv4.0.0/act/Mul_output_0', '/model.22/Concat_6_output_0'],
                   ['/model.22/cv4.1/cv4.1.0/act/Mul_output_0', '/model.22/Concat_6_output_0'],
                   ['/model.22/cv4.2/cv4.2.0/act/Mul_output_0', '/model.22/Concat_6_output_0']])
print('step1 done %.0fs' % (time.time() - t0))

print('--> Hybrid quantization step2')
rknn.hybrid_quantization_step2(
    model_input=name + '.model',
    data_input=name + '.data',
    model_quantization_cfg=name + '.quantization.cfg')
print('step2 done %.0fs' % (time.time() - t0))

print('--> Export rknn model')
ret = rknn.export_rknn(OUT)
assert ret == 0, 'Export failed: %d' % ret
print('done:', OUT, '%.0f KB' % (os.path.getsize(OUT) / 1024))
rknn.release()
print('TOTAL %.0fs' % (time.time() - t0))
