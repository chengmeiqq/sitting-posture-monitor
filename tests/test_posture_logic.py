#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段4/6 坐姿逻辑单元测试: 合成关键点驱动 features/judge/temporal"""
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser('~/posture_monitor'))
from posture.features import compute_features
from posture.judge import PostureJudge, GREEN, YELLOW, RED
from posture.temporal import StateMachine
from pose.engine import KP


def synth(torso_len=200, tilt=0.0, head_fwd=0.0, head_dn=0.35, sh_line=0.0,
          facing='right', conf=0.9, drop_core=False):
    """head_fwd = 鼻相对肩中心水平偏移/躯干长; head_dn = 鼻高于肩的垂距/躯干长"""
    k = np.zeros((17, 3), np.float32); k[:, 2] = conf
    if drop_core:
        k[KP['left_hip'], 2] = 0.05
    sgn = 1 if facing == 'right' else -1
    hip = np.array([300., 600.])
    sho = hip + np.array([sgn * np.sin(np.radians(tilt)) * torso_len,
                          -np.cos(np.radians(tilt)) * torso_len])
    nose = sho + np.array([sgn * head_fwd * torso_len, -head_dn * torso_len])
    k[KP['nose']] = [nose[0], nose[1], conf]
    k[KP['left_ear']] = [nose[0] - 5, nose[1] + 8, conf]
    k[KP['right_ear']] = [nose[0] + 5, nose[1] + 8, conf * 0.4]
    k[KP['left_shoulder']] = [sho[0], sho[1], conf]
    k[KP['right_shoulder']] = [sho[0] + 40 * np.cos(np.radians(sh_line)),
                               sho[1] + 40 * np.sin(np.radians(sh_line)), conf]
    k[KP['left_hip']] = [hip[0], hip[1], conf if not drop_core else 0.05]
    k[KP['right_hip']] = [hip[0] + 40, hip[1], conf]
    return k


def mk_ref(J, f):
    J.ref.update(torso_tilt=f['torso_tilt_deg'], shoulder_line=f['shoulder_line_deg'],
                 head_fwd=f['head_fwd_norm'], head_down=f['head_down_norm'],
                 trunk_off=f['trunk_off_norm'], ear_sho_dy=f['ear_sho_dy_norm'])


def main():
    f0 = compute_features(synth(tilt=8, head_fwd=0.10, head_dn=0.35, sh_line=2))
    assert f0['valid']
    J = PostureJudge(calibrated=True, facing='right'); mk_ref(J, f0)
    assert J.judge(f0)[0] == 'NORMAL'
    Ji = PostureJudge(calibrated=False, facing='right'); Ji.set_instant_ref(f0)
    assert Ji.judge(f0)[0] == 'NORMAL'

    cases = [
        ('head_forward', synth(tilt=8, head_fwd=0.45, head_dn=0.32, sh_line=2), ('HEAD_FWD',), RED),
        ('body_forward', synth(tilt=34, head_fwd=0.10, head_dn=0.35, sh_line=2), ('BODY_FWD',), RED),
        ('slump',        synth(tilt=16, head_fwd=0.35, head_dn=0.14, sh_line=2), ('SLUMP', 'HEAD_NEAR_DESK', 'HEAD_FWD'), None),
        ('near_desk',    synth(tilt=12, head_fwd=0.15, head_dn=0.03, sh_line=2), ('HEAD_NEAR_DESK', 'SLUMP'), None),
        ('shoulder_tilt', synth(tilt=8, head_fwd=0.10, head_dn=0.35, sh_line=20), ('TILT_R', 'TILT_L'), None),
        ('side_lean',    synth(tilt=-16, head_fwd=0.0, head_dn=0.33, sh_line=2), ('SIDE_LEAN', 'TILT_L', 'TILT_R'), None),
    ]
    for name, k, want, wl in cases:
        st, lv, d = J.judge(compute_features(k))
        print('%-14s -> %-14s %-6s' % (name, st, lv))
        assert st in want, (name, st, d)
        if wl:
            assert lv == wl, (name, lv)
    assert J.judge(compute_features(synth(drop_core=True)))[0] == 'UNKNOWN'

    JL = PostureJudge(calibrated=True, facing='left')
    fL = compute_features(synth(tilt=8, head_fwd=0.10, head_dn=0.35, sh_line=2, facing='left'))
    mk_ref(JL, fL); assert JL.judge(fL)[0] == 'NORMAL'
    st, lv, _ = JL.judge(compute_features(synth(tilt=8, head_fwd=0.45, head_dn=0.32, sh_line=2, facing='left')))
    assert st in ('HEAD_FWD', 'SLUMP')

    sm = StateMachine()
    for i in np.arange(0, 3., .2): sm.update('BODY_FWD', RED, i)
    for i in np.arange(3.2, 10., .2): sm.update('NORMAL', GREEN, i)
    assert (sm.shown_state, sm.shown_level) == ('NORMAL', GREEN), 'flap-suppression'

    s2 = StateMachine()
    for i in np.arange(0, 4.2, .2): s2.update('HEAD_FWD', YELLOW, 4 + i)
    assert s2.shown_state == 'HEAD_FWD'
    for i in np.arange(0, 2.1, .2): s2.update('NORMAL', GREEN, 9 + i)
    assert s2.shown_state == 'HEAD_FWD', 'recover-hold'
    for i in np.arange(2.2, 7., .2): s2.update('NORMAL', GREEN, 9 + i)
    assert (s2.shown_state, s2.shown_level) == ('NORMAL', GREEN)
    s2.update('UNKNOWN', YELLOW, 20.); assert s2.shown_state == 'UNKNOWN'

    s3 = StateMachine()
    for i in np.arange(0, 4.2, .2): s3.update('SLUMP', YELLOW, i)
    assert s3.shown_level == YELLOW
    for i in np.arange(5., 9.5, .2): s3.update('SLUMP', RED, i)
    assert s3.shown_level == RED, 'escalate-instant'
    for i in np.arange(10., 11.5, .2): s3.update('SLUMP', YELLOW, i)
    assert s3.shown_level == RED, 'degrade-hold'
    for i in np.arange(12., 19., .2): s3.update('NORMAL', GREEN, i)
    assert s3.shown_state == 'NORMAL'
    print('ALL_POSTURE_TESTS_OK')


if __name__ == '__main__':
    main()
