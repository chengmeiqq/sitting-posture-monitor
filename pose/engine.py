# -*- coding: utf-8 -*-
"""YOLOv8n-Pose RKNN 推理引擎（板端 rknnlite2 + NPU）

后处理语义与官方 airockchip/rknn_model_zoo
examples/yolov8_pose/python/yolov8_pose.py 一致，但向量化实现。
关键点顺序为 COCO 17 点。
"""
import os
import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite

MODEL_PATH = '/root/posture_monitor/models/yolov8n-pose-int8.rknn'
INPUT_SIZE = 640
BG_COLOR = 56
OBJ_THRESH = 0.5
NMS_THRESH = 0.45
KPT_CONF_THRESH = 0.3

KP_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
KP = {n: i for i, n in enumerate(KP_NAMES)}

# 0-based 骨架连线 (与官方 skeleton 表一致)
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)]
_PALETTE = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
                     [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
                     [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
                     [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
                     [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)
KPT_COLOR = _PALETTE[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
LIMB_COLOR = _PALETTE[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


def letterbox_resize(image, size=INPUT_SIZE, bg_color=BG_COLOR):
    th = tw = size if isinstance(size, int) else size[0]
    ih, iw = image.shape[:2]
    r = min(tw / iw, th / ih)
    nw, nh = int(iw * r), int(ih * r)
    img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.ones((th, tw, 3), dtype=np.uint8) * bg_color
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = img
    return out, r, ox, oy


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _nms(boxes, thresh=NMS_THRESH):
    boxes.sort(key=lambda b: -b['score'])
    keep = []
    for b in boxes:
        ok = True
        for k in keep:
            ix = min(b['xmax'], k['xmax']) - max(b['xmin'], k['xmin'])
            iy = min(b['ymax'], k['ymax']) - max(b['ymin'], k['ymin'])
            if ix <= 0 or iy <= 0:
                continue
            inter = ix * iy
            a1 = (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin'])
            a2 = (k['xmax'] - k['xmin']) * (k['ymax'] - k['ymin'])
            if inter / (a1 + a2 - inter) > thresh:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


class YoloPoseRKNN:
    def __init__(self, model_path=MODEL_PATH, core_mask=RKNNLite.NPU_CORE_AUTO):
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(model_path)
        assert ret == 0, 'load_rknn failed: %d' % ret
        # target=None: 由 librknnrt 自动绑定本机 NPU（该运行时没有 CPU 后端，
        # init 成功即说明跑在 NPU 上）
        ret = self.rknn.init_runtime(core_mask=core_mask)
        assert ret == 0, 'init_runtime failed: %d (NPU 不可用?)' % ret
        self.last_prep_ms = 0.0
        self.last_infer_ms = 0.0
        self.last_post_ms = 0.0

    def _decode(self, outs):
        # 官方 layout: det=(1,65,H,W) [c,h,w]; kpt=(1,17,3,8400)，列号=全局网格序号
        k = np.asarray(outs[3])
        if k.ndim == 4 and k.shape[1] == 17:        # (1,17,3,8400) NCHW
            kpt_all = k[0].reshape(51, -1)          # row=k*3+d, col=全局网格序
        elif k.ndim == 4 and k.shape[3] == 51:      # NHWC 兼容
            kpt_all = k[0].reshape(-1, 51).T
        else:
            raise ValueError('unexpected kpt layout %s' % (k.shape,))
        GRID_BASE = {80: 0, 40: 6400, 20: 8000}     # 官方 index 偏移 (同 ultralytics)
        boxes = []
        for x in outs[:3]:
            f = np.asarray(x)
            if f.ndim == 4 and f.shape[1] != 65:    # NHWC -> NCHW
                f = f.transpose(0, 3, 1, 2)
            f = f.reshape(65, -1)
            A = f.shape[1]
            W = int(round(A ** 0.5))
            stride = INPUT_SIZE // W
            conf = _sigmoid(f[64])
            j = np.where(conf > OBJ_THRESH)[0]
            if len(j) == 0:
                continue
            gi, gj = j // W, j % W
            d = f[:64, j].reshape(4, 16, len(j))
            d = _softmax(d, 1)
            bins = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
            dist = (d * bins).sum(1)  # row0..3 = dist[0..3]
            xmin = (gj + 0.5 - dist[0]) * stride
            ymin = (gi + 0.5 - dist[1]) * stride
            xmax = (gj + 0.5 + dist[2]) * stride
            ymax = (gi + 0.5 + dist[3]) * stride
            gidx = j + GRID_BASE[W]                 # 全局网格序号 (官方 index 偏移)
            kp = kpt_all[:, gidx].T.reshape(len(j), 17, 3).astype(np.float32)
            kp[..., 0:2] = np.floor(kp[..., 0:2])   # 与官方一致
            for i in range(len(j)):
                boxes.append(dict(score=float(conf[j[i]]),
                                  xmin=float(xmin[i]), ymin=float(ymin[i]),
                                  xmax=float(xmax[i]), ymax=float(ymax[i]),
                                  kpts=kp[i]))
        return _nms(boxes)

    def infer(self, bgr_img):
        t0 = time.perf_counter()
        lb, ratio, ox, oy = letterbox_resize(bgr_img)
        rgb = lb[:, :, ::-1]
        self.last_prep_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        outs = self.rknn.inference(inputs=[rgb[np.newaxis, ...]])  # 4D NHWC for Lite2
        self.last_infer_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        boxes = self._decode(outs)
        for b in boxes:
            b['xmin'] = (b['xmin'] - ox) / ratio
            b['ymin'] = (b['ymin'] - oy) / ratio
            b['xmax'] = (b['xmax'] - ox) / ratio
            b['ymax'] = (b['ymax'] - oy) / ratio
            k = b['kpts'].copy()
            k[:, 0] = (k[:, 0] - ox) / ratio
            k[:, 1] = (k[:, 1] - oy) / ratio
            k[:, 2] = _sigmoid(k[:, 2])
            b['kpts'] = k
        self.last_post_ms = (time.perf_counter() - t2) * 1000
        return boxes

    @staticmethod
    def draw(bgr_img, boxes):
        img = bgr_img.copy()
        for b in boxes:
            cv2.rectangle(img, (int(b['xmin']), int(b['ymin'])),
                          (int(b['xmax']), int(b['ymax'])), (0, 255, 0), 2)
            cv2.putText(img, 'person %.2f' % b['score'],
                        (int(b['xmin']), max(0, int(b['ymin']) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            k = b['kpts']
            for i, (x, y, c) in enumerate(k):
                if c > KPT_CONF_THRESH and x > 0 and y > 0:
                    cv2.circle(img, (int(x), int(y)), 4,
                               [int(v) for v in KPT_COLOR[i]], -1, cv2.LINE_AA)
            for s, (i, j) in enumerate(SKELETON):
                x1, y1, c1 = k[i]
                x2, y2, c2 = k[j]
                if (c1 > KPT_CONF_THRESH and c2 > KPT_CONF_THRESH
                        and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0):
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                             [int(v) for v in LIMB_COLOR[s]], 2, cv2.LINE_AA)
        return img

    def release(self):
        self.rknn.release()
