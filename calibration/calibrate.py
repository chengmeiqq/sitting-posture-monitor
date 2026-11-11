# -*- coding: utf-8 -*-
"""个人坐姿校准 (阶段5)
采集中: 连续采集 N 秒正常学习姿态的有效帧特征, 取中位数作基准, MAD 作离散度。
保存: config/calibration.json (重新校准=覆盖)
facing(脸朝画面哪侧)由校准数据的 trunk 水平漂移符号+人工确认: 自动取肩-髋偏移主方向。
"""
import json, os, time
import numpy as np

CFG_PATH = os.path.expanduser('~/posture_monitor/config/calibration.json')
KEYS = ['torso_tilt_deg', 'shoulder_line_deg', 'head_fwd_norm', 'head_down_norm',
        'trunk_off_norm', 'ear_sho_dy_norm', 'ear_fwd_norm', 'eye_hip_ratio']


class Calibrator:
    def __init__(self, min_frames=60, duration_s=15.0):
        self.min_frames = min_frames
        self.duration_s = duration_s
        self.samples = []
        self.rejected = 0
        self.t_start = None

    def start(self):
        self.samples = []
        self.rejected = 0
        self.t_start = time.time()

    def add(self, feats):
        if not feats.get('valid'):
            return False
        # 姿态合理性门槛: 只有结构上像"坐人的"帧才能进基准
        # (防止手/杂物引起的关键点幻觉污染校准)
        if not self._plausible(feats):
            self.rejected += 1
            return False
        s = {k: (np.nan if feats.get(k) is None else feats[k]) for k in KEYS}
        s['torso_len'] = feats['torso_len']
        self.samples.append(s)
        return True

    @staticmethod
    def _plausible(f):
        hd = f.get('head_down_norm')
        # 鼻必须在肩上方 0.1~0.9 个躯干长之间
        if hd is None or not (0.10 <= hd <= 0.90):
            return False
        # 躯干不能倒置/横倒: 肩中心在髋中心上方
        if f['sho_c'][1] >= f['hip_c'][1]:
            return False
        # 躯干不能过度倾斜(>45°, 坐姿正常时肩-髋连线接近竖直)
        if abs(f['torso_tilt_deg']) > 45:
            return False
        # 关键核心点置信度都要高
        if min(f['kconf'][n] for n in ('nose', 'left_shoulder', 'right_shoulder',
                                       'left_hip', 'right_hip')) < 0.5:
            return False
        # 人体尺度: 躯干长至少 80px(离镜头太远/太小不可信)
        if f['torso_len'] < 80:
            return False
        return True

    def progress(self):
        if self.t_start is None:
            return 0.0
        return min(1.0, (time.time() - self.t_start) / self.duration_s)

    def ready(self):
        return (self.t_start and len(self.samples) >= self.min_frames
                and time.time() - self.t_start >= self.duration_s)

    def finish(self):
        if not self.ready():
            return None, 'insufficient valid samples: %d (need %d in %.0fs; %d rejected as implausible) 请确认孩子正面坐在书桌前且坐姿端正' % (
                len(self.samples), self.min_frames, self.duration_s, self.rejected)
        arr = {k: np.array([s[k] for s in self.samples], dtype=float) for k in KEYS}
        ref, spread, notes = {}, {}, []
        for k in KEYS:
            v = arr[k][~np.isnan(arr[k])]
            if len(v) < max(10, self.min_frames // 3):
                notes.append('%s insufficient' % k)
                ref[k] = float('nan'); spread[k] = float('nan')
                continue
            ref[k] = float(np.median(v))
            spread[k] = float(np.median(np.abs(v - ref[k])))
        # facing: 正常坐时鼻相对肩朝桌子侧偏移; 用 head_fwd 符号(校准中孩子朝桌坐,
        # 默认基准帧 head_fwd 应接近 0; 用 ear_fwd 与 head_fwd 联合判断)
        hf = ref.get('head_fwd_norm', np.nan)
        ef = ref.get('ear_fwd_norm', np.nan)
        val = hf if not np.isnan(hf) else ef
        if abs(val) < 0.02:
            facing = 'right'
            notes.append('facing_ambiguous_default_right')
        else:
            facing = 'right' if val > 0 else 'left'
        cal = dict(ref=ref, spread=spread, facing=facing,
                   n_frames=len(self.samples), ts=time.strftime('%Y-%m-%d %H:%M:%S'),
                   notes=notes)
        os.makedirs(os.path.dirname(CFG_PATH), exist_ok=True)
        with open(CFG_PATH, 'w') as f:
            json.dump(cal, f, ensure_ascii=False, indent=2)
        return cal, 'saved %s' % CFG_PATH


def load_calibration():
    if os.path.exists(CFG_PATH):
        try:
            return json.load(open(CFG_PATH))
        except Exception:
            return None
    return None
