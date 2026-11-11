# -*- coding: utf-8 -*-
"""坐姿状态判定 (阶段4)
基于 features.compute_features 的几何量 + 个人校准基准(阶段5) 判定姿态。
瞬时状态: NORMAL / HEAD_FWD / BODY_FWD / SLUMP / HEAD_NEAR_DESK /
          TILT_L / TILT_R / SIDE_LEAN / AWAY / UNKNOWN
规则使用"相对个人基准的偏移", 不用统一固定角度。
未校准时使用保守默认基准 + 更宽阈值, 并在 note 标注。
"""
import numpy as np

GREEN, YELLOW, RED = 'green', 'yellow', 'red'

# 默认基准(正常直立端坐): torso_tilt≈±小角, head_fwd≈0(相机在侧前方, 人在画面中朝向桌子)
# 注: eye_hip_ratio 默认值按一年级小孩(头身比大, 肩窄)经验取 3.2; 校准会用实测中位数覆盖。
DEFAULT_REF = dict(
    torso_tilt=0.0,        # 校准会覆盖
    trunk_off=0.0,
    head_fwd=0.0,
    head_down=0.35,        # 鼻到肩垂距/躯干长 典型值, 校准覆盖
    ear_sho_dy=0.18,
    shoulder_line=0.0,
    eye_hip_ratio=3.2,     # 眼-髋距/肩宽 (借鉴 BNDSer 项目特征, 用个人基准而非固定系数)
)
# 默认偏移阈值(未校准, 保守): 超过=提醒, 远超=异常
DEFAULT_TH = dict(
    head_fwd_dy=0.18, head_fwd_red=0.30,     # 头部持续前倾(相对基准的前移)
    body_fwd_dy=0.14, body_fwd_red=0.24,     # 身体前倾 trunk_off
    slump_dy=0.10, slump_red=0.16,           # 头颈下垂 head_down 减小量
    near_desk=0.30,                          # head_down 绝对下限(近似; 桌线未知时按基准漂移)
    tilt_line_dy=12.0, tilt_line_red=20.0,   # 肩线角度偏移(仅前视分量可靠)
    side_lean_dy=14.0, side_lean_red=24.0,   # 躯干倾角偏移
    eye_hip_dy=0.35, eye_hip_red=0.7,        # 眼-髋距/肩宽 相对基准下降量(颈部前伸证据)
)


def _wrap90(a):
    while a > 90:
        a -= 180
    while a <= -90:
        a += 180
    return a


def _wrap180(a):
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a


class PostureJudge:
    def __init__(self, ref=None, th=None, calibrated=False, facing=None):
        """facing: 'left'|'right' 孩子脸朝向画面哪侧(桌子在哪侧); 校准确定。
        未校准: 调用 set_instant_ref(feats) 用首个有效帧作临时基准(仍标记未校准)。"""
        self.ref = dict(DEFAULT_REF)
        self.th = dict(DEFAULT_TH)
        self.calibrated = calibrated
        self.facing = facing or 'right'
        if ref:
            # 容错: 校准文件用特征名(torso_tilt_deg 等), 内部基准用短名; 统一映射
            _kmap = {'torso_tilt_deg': 'torso_tilt', 'shoulder_line_deg': 'shoulder_line',
                     'head_fwd_norm': 'head_fwd', 'head_down_norm': 'head_down',
                     'trunk_off_norm': 'trunk_off', 'ear_sho_dy_norm': 'ear_sho_dy',
                     'ear_fwd_norm': 'ear_fwd'}
            norm = {}
            for k, v in ref.items():
                nk = _kmap.get(k, k)
                if isinstance(v, float) and v != v:   # NaN: 保持默认值
                    continue
                norm[nk] = v
            self.ref.update(norm)
        if th:
            self.th.update(th)

    def set_instant_ref(self, feats):
        """无校准时以当前帧为临时基准(阈值仍保守)"""
        self.ref.update(
            torso_tilt=feats['torso_tilt_deg'],
            shoulder_line=feats['shoulder_line_deg'],
            head_fwd=feats['head_fwd_norm'],
            head_down=feats['head_down_norm'],
            trunk_off=feats['trunk_off_norm'],
            ear_sho_dy=feats.get('ear_sho_dy_norm') or DEFAULT_REF['ear_sho_dy'])
        if feats.get('eye_hip_ratio') is not None:
            self.ref['eye_hip_ratio'] = feats['eye_hip_ratio']
        if self.facing is None:
            self.facing = 'right'

    def _sgn(self):
        """朝向符号: 前倾量在特征里的方向"""
        return 1.0 if self.facing == 'right' else -1.0

    def judge(self, feats):
        """feats: compute_features 输出 -> (state, level, detail)"""
        det = {}
        if not feats.get('valid'):
            return 'UNKNOWN', YELLOW, {'reason': feats.get('reason', 'invalid')}

        s = self._sgn()
        trunk_off = s * feats['trunk_off_norm']          # >0 = 朝桌前倾
        head_fwd = s * feats['head_fwd_norm']
        body_fwd_dev = trunk_off - s * self.ref['trunk_off'] if self.calibrated else trunk_off - self.ref['trunk_off']
        head_fwd_dev = head_fwd - (s * self.ref['head_fwd'] if self.calibrated else self.ref['head_fwd'])
        head_down_dev = (feats['head_down_norm'] - self.ref['head_down'])  # <0 = 头更低
        slump_dev = -head_down_dev                        # >0 = 头颈下垂趋势
        tilt_line_dev = _wrap90(feats['shoulder_line_deg'] - self.ref['shoulder_line'])
        side_lean_dev = abs(_wrap180(feats['torso_tilt_deg'] - self.ref['torso_tilt']))

        # 眼-髋距/肩宽 下降 = 颈部前伸/探头 的第二证据(侧前方机位下对小孩更稳)
        ehrs = feats.get('eye_hip_ratio')
        eye_hip_dev = None
        if ehrs is not None and self.ref.get('eye_hip_ratio') is not None \
                and not (isinstance(self.ref['eye_hip_ratio'], float) and self.ref['eye_hip_ratio'] != self.ref['eye_hip_ratio']):
            eye_hip_dev = self.ref['eye_hip_ratio'] - ehrs   # >0 = 头相对基线更靠近髋线(前伸/低下)

        tilt_line_dev = abs(tilt_line_dev)
        th = self.th
        states = []  # (state, level)

        # 头部前倾
        if head_fwd_dev >= th['head_fwd_red']:
            states.append(('HEAD_FWD', RED))
        elif head_fwd_dev >= th['head_fwd_dy']:
            states.append(('HEAD_FWD', YELLOW))
        # 颈部前伸双证据: 眼-髋比明显下降 + 头颈显著下沉, 才报 HEAD_FWD
        # (单靠比值会误报: 小孩拿笔/翻书时手臂使 hip 拖动; 阈值偏保守)
        if eye_hip_dev is not None and eye_hip_dev >= th['eye_hip_dy'] and slump_dev >= th['slump_dy']:
            lvl = RED if eye_hip_dev >= th['eye_hip_red'] else YELLOW
            if head_fwd_dev < th['head_fwd_dy']:   # 避免与主判定重复叠加
                states.append(('HEAD_FWD', lvl))
        # 身体前倾
        if body_fwd_dev >= th['body_fwd_red']:
            states.append(('BODY_FWD', RED))
        elif body_fwd_dev >= th['body_fwd_dy']:
            states.append(('BODY_FWD', YELLOW))
        # 驼背趋势 (头部相对躯干下垂 + 前倾)
        if slump_dev >= th['slump_red'] and head_fwd_dev > 0:
            states.append(('SLUMP', RED))
        elif slump_dev >= th['slump_dy'] and head_fwd_dev > 0:
            states.append(('SLUMP', YELLOW))
        # 头过度靠近桌面(头显著低于基准)
        if head_down_dev <= -th['near_desk']:
            states.append(('HEAD_NEAR_DESK', RED))
        # 左右倾斜: 肩线(侧视分量) 或 躯干倾角
        if tilt_line_dev >= th['tilt_line_red'] or side_lean_dev >= th['side_lean_red']:
            st = 'TILT_R' if feats['shoulder_line_deg'] > 0 else 'TILT_L'
            if side_lean_dev >= th['side_lean_red'] and tilt_line_dev < th['tilt_line_dy']:
                st = 'SIDE_LEAN'
            states.append((st, RED))
        elif tilt_line_dev >= th['tilt_line_dy'] or side_lean_dev >= th['side_lean_dy']:
            st = 'TILT_R' if feats['shoulder_line_deg'] > 0 else 'TILT_L'
            if side_lean_dev >= th['side_lean_dy'] and tilt_line_dev < th['tilt_line_dy']:
                st = 'SIDE_LEAN'
            states.append((st, YELLOW))

        det.update(body_fwd_dev=round(float(body_fwd_dev), 3),
                   head_fwd_dev=round(float(head_fwd_dev), 3),
                   slump_dev=round(float(slump_dev), 3),
                   tilt_line_dev=round(float(tilt_line_dev), 1),
                   side_lean_dev=round(float(side_lean_dev), 1))
        if eye_hip_dev is not None:
            det['eye_hip_dev'] = round(float(eye_hip_dev), 3)

        if not states:
            return 'NORMAL', GREEN, det
        # 取最严重(RED优先), 同级取第一个
        level = RED if any(l == RED for _, l in states) else YELLOW
        state = next(st for st, l in states if l == level)
        det['all_flags'] = states
        return state, level, det
