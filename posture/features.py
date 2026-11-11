# -*- coding: utf-8 -*-
"""坐姿特征计算 (阶段4)
基于 YOLOv8-Pose COCO-17 关键点计算姿态几何特征。
所有特征做了人体尺度归一化(肩宽/躯干长), 便于跨体型/距离比较。

注意: 单目 2D 关键点无法直接测量脊柱曲度, "驼背"只是趋势特征, 非医学诊断。
相机假设: 书桌侧前方 30-45°, 画面 x≈前后方向(朝桌=通常向画面深处), y≈竖直。
"""
import numpy as np
from pose.engine import KP

# 躯干/头部分析所需关键点的最小置信度
MIN_CONF = 0.3
CORE_KPTS = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
# 侧向相机时左右侧关键点遮挡常见, 用耳/鼻增强头部姿态鲁棒性
HEAD_KPTS = ['nose', 'left_ear', 'right_ear']


def _mid(a, b):
    return (a[:2] + b[:2]) / 2.0


def _pick_visible(k, names, min_conf=MIN_CONF):
    """返回第一个置信度足够的关键点索引, 找不到返回 None"""
    for n in names:
        p = k[KP[n]]
        if p[2] >= min_conf:
            return p
    return None


def compute_features(kpts):
    """kpts: (17,3) [x,y,conf] 原图像素坐标.
    返回 dict; features 为 None 表示无法计算(调用方应给 UNKNOWN)。
    """
    f = {'kconf': {}, 'note': []}
    core_ok = all(kpts[KP[n]][2] >= MIN_CONF for n in CORE_KPTS)
    for n in CORE_KPTS + HEAD_KPTS + ['left_knee', 'right_knee']:
        f['kconf'][n] = round(float(kpts[KP[n]][2]), 3)
    if not core_ok:
        f['valid'] = False
        f['reason'] = 'core_keypoints_low_conf'
        return f

    ls, rs = kpts[KP['left_shoulder']], kpts[KP['right_shoulder']]
    lh, rh = kpts[KP['left_hip']], kpts[KP['right_hip']]
    nose = kpts[KP['nose']]

    sho_c = _mid(ls, rs)
    hip_c = _mid(lh, rh)

    shoulder_w = float(np.linalg.norm(ls[:2] - rs[:2]))
    torso_vec = hip_c - sho_c                       # 图像坐标 y 向下
    torso_len = float(np.linalg.norm(torso_vec))
    if torso_len < 1e-3 or shoulder_w < 1e-3:
        f['valid'] = False
        f['reason'] = 'degenerate_geometry'
        return f

    # 躯干倾角: 向量 sho->hip 与 (0,+1)(图像竖直向下) 的夹角, 逆时针为正。
    # 肩相对髋朝画面右倒 -> torso_tilt>0; 朝左倒 -> <0。
    # 前倾方向(朝桌)是画面左还是右, 由阶段5校准确定符号。
    torso_tilt = float(np.degrees(np.arctan2(torso_vec[0], torso_vec[1])))

    # 肩部连线角度: 无向直线, 归一到 (-90, 90], 正=画面右端更低
    dy = rs[1] - ls[1]
    dx = rs[0] - ls[0]
    shoulder_line_ang = float(np.degrees(np.arctan2(dy, dx if abs(dx) > 1e-3 else 1e-3)))
    while shoulder_line_ang > 90:
        shoulder_line_ang -= 180
    while shoulder_line_ang <= -90:
        shoulder_line_ang += 180
    # 肩线接近竖直时(纯侧视)该值无意义, 标记
    if abs(dx) < 0.25 * shoulder_w:
        f['note'].append('shoulder_line_unreliable_near_side_view')

    # 头部前移量(颈部前伸): 鼻相对肩中心水平偏移 / 躯干长 (带符号按 facing 在 judge 里乘)
    head_fwd = float((nose[0] - sho_c[0]) / torso_len)
    # 头-肩垂直距离 / 躯干长 (低头/驼背时变小)
    head_down = float((sho_c[1] - nose[1]) / torso_len)
    # 耳-肩垂直特征(头部姿态趋势, 侧视更可靠)
    ear = _pick_visible(kpts, ['left_ear', 'right_ear'])
    if ear is not None:
        f['ear_sho_dy_norm'] = float((sho_c[1] - ear[1]) / torso_len)
        f['ear_fwd_norm'] = float((ear[0] - sho_c[0]) / torso_len)
    else:
        f['ear_sho_dy_norm'] = None
        f['ear_fwd_norm'] = None
        f['note'].append('ears_low_conf')

    # 肩中心相对髋中心水平偏移 / 躯干长 (整体前倾/侧歪共用带符号量)
    trunk_off = float((sho_c[0] - hip_c[0]) / torso_len)

    # 参考 BNDSer/Sitting-posture-detection-system: 眼(或鼻)-髋距 / 肩宽。
    # 儿童颈部前伸/低头时, 侧前方机位下该比值明显减小; 对一年级小孩
    # (头身比大)比纯像素位移更稳。eye 不可见退用 nose。
    eye_mid = None
    le, re_ = kpts[KP['left_eye']], kpts[KP['right_eye']]
    if le[2] >= MIN_CONF and re_[2] >= MIN_CONF:
        eye_mid = _mid(le, re_)
    elif le[2] >= MIN_CONF:
        eye_mid = le[:2]
    elif re_[2] >= MIN_CONF:
        eye_mid = re_[:2]
    elif nose[2] >= MIN_CONF:
        eye_mid = nose[:2]
    if eye_mid is not None:
        f['eye_hip_ratio'] = round(float(np.linalg.norm(eye_mid - hip_c) / shoulder_w), 3)
    else:
        f['eye_hip_ratio'] = None
        f['note'].append('eyes_nose_low_conf')

    f.update(dict(
        valid=True,
        sho_c=sho_c.tolist(), hip_c=hip_c.tolist(),
        shoulder_w=round(shoulder_w, 1), torso_len=round(torso_len, 1),
        torso_tilt_deg=round(torso_tilt, 1),
        shoulder_line_deg=round(shoulder_line_ang, 1),
        head_fwd_norm=round(head_fwd, 3),
        head_down_norm=round(head_down, 3),
        trunk_off_norm=round(trunk_off, 3),
        nose_conf=float(nose[2]),
        eye_ref='eye' if (le[2] >= MIN_CONF or re_[2] >= MIN_CONF) else 'nose',
    ))
    return f
