"""Second lightweight confounder probe: find a lateral-offset range where
   matched_teacher_action > mild_straight_action > opposite_action
while mild straight stays safe (positive progress, no miss/fall).

No training, no env changes. Reuses probe_lateral_y's reference-state helpers,
the conedir env, and the trained full-state policy.

Run:  python scripts/probe_lateral_y2.py            # single seed, CPU
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_lateral_y as P            # reuse get_state/set_state/set_object_y/...

from crl.config import Config
from crl import envs as envs_mod
from crl import report_push as R

OUTDIR = r'D:/Users/trhua/Research/contrastive_rl/artifacts/lateral_y_probe'
DS = [0.04, 0.05, 0.06, 0.07]
MARGIN = 0.005                          # "clearly worse" threshold (m)
MILD = np.array([1.0, 0.0, 0.0, 0.0], np.float32)   # medium-forward, lateral 0


def run_action(env, u, restore, dy, action, goal):
    flat, od0, meas = restore(dy)
    obj0 = od0['achieved_goal'].copy(); q0 = P.obj_quat(u)
    d0 = float(np.linalg.norm(obj0 - goal))
    n_contact = 0; fell = False; minz = float(obj0[2])
    for _ in range(P.N_STEPS):
        g = np.asarray(u._get_obs()['observation'][:3])
        ob = np.asarray(u._get_obs()['achieved_goal'])
        if np.linalg.norm(g - ob) < P.CONTACT:
            n_contact += 1
        u.step(np.asarray(action, np.float32))
        oz = float(u._get_obs()['achieved_goal'][2]); minz = min(minz, oz)
        if oz < P.TABLE_Z:
            fell = True
    obj1 = np.asarray(u._get_obs()['achieved_goal']).copy(); q1 = P.obj_quat(u)
    d1 = float(np.linalg.norm(obj1 - goal))
    move = float(np.linalg.norm(obj1 - obj0))
    return dict(
        forward_progress=float(obj1[0] - obj0[0]),
        goal_improvement=float(d0 - d1),
        lateral_disp=float(obj1[1] - obj0[1]),
        rot_change_deg=P.quat_angle_deg(q0, q1),
        contact_steps=int(n_contact),
        missed=bool(move < P.MISS_MOVE), fell=bool(fell))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=P.CKPT_DEFAULT)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    cfg = Config(env_name=P.ENV)
    env = envs_mod.make_env(P.ENV, cfg, seed=args.seed)
    u = env._env.unwrapped
    greedy, step = R._load_greedy(P.ENV, args.ckpt, cfg)
    print(f'ckpt step {step} | dt={u.dt:.3f}s | seed {args.seed}')

    # ---- reference state (same construction as probe 1) ----
    o = env.reset()
    g_y = float(o[1]); obj_x, obj_z = 1.40, 0.425
    goal = np.array([obj_x + P.PUSH, g_y, obj_z], np.float32)
    u.goal = goal.astype(float)
    P.set_object_y(u, obj_x, g_y, obj_z)
    u.data.qvel[:] = 0.0; u._mujoco.mj_forward(u.model, u.data)
    ref = P.get_state(u)

    def restore(dy):
        P.set_state(u, ref)
        P.set_object_y(u, obj_x, g_y + dy, obj_z)
        od = u._get_obs()
        flat = env._flatten(od)
        return flat, od, float(flat[4] - flat[1])

    # ---- action-sign check: does +action_y raise grip_pos_y? ----
    P.set_state(u, ref)
    gy0 = float(u._get_obs()['observation'][1])
    for _ in range(2):
        u.step(np.array([0.0, 0.5, 0.0, 0.0], np.float32))
    gy1 = float(u._get_obs()['observation'][1])
    sign_dy = gy1 - gy0
    sign_ok = abs(sign_dy) > 1e-3
    print(f'\n[action-sign check] +action_y (0.5, 2 steps): grip_y {gy0:.4f} -> '
          f'{gy1:.4f}  (delta {sign_dy:+.4f}) => +action_y moves grip_y '
          f'{"UP/+y" if sign_dy > 0 else "DOWN/-y"}')

    # ---- sweep ----
    rows = []
    impl_issue = not sign_ok
    for d in DS:
        for sgn in (-1.0, 1.0):
            dy = sgn * d
            flat, _, meas = restore(dy)
            if abs(meas - dy) > 1e-3:
                impl_issue = True
            matched = np.asarray(greedy(flat), np.float32)
            opposite = matched.copy(); opposite[1] = -matched[1]
            acts = {'matched_teacher': matched, 'opposite': opposite,
                    'mild_straight': MILD}
            for name, av in acts.items():
                res = run_action(env, u, restore, dy, av, goal)
                rows.append(dict(d=d, delta_y=round(dy, 3), action_type=name,
                                 ax=float(av[0]), ay=float(av[1]), az=float(av[2]),
                                 **res))

    # ---- table ----
    print(f'\n{"d":>5} {"dy":>6} {"action":>15} {"ax":>5} {"ay":>6} {"az":>6} '
          f'{"fwd":>6} {"goalimp":>8} {"lat":>7} {"drot":>6} {"cts":>3} '
          f'{"miss":>5} {"fall":>5}')
    for r in rows:
        print(f'{r["d"]:>5.2f} {r["delta_y"]:>6.2f} {r["action_type"]:>15} '
              f'{r["ax"]:>5.2f} {r["ay"]:>6.2f} {r["az"]:>6.2f} '
              f'{r["forward_progress"]:>6.3f} {r["goal_improvement"]:>8.3f} '
              f'{r["lateral_disp"]:>7.3f} {r["rot_change_deg"]:>6.1f} '
              f'{r["contact_steps"]:>3} {str(r["missed"]):>5} {str(r["fell"]):>5}')

    # ---- pattern per d ----
    def get(d, dy, name, k):
        return next(r[k] for r in rows if r['d'] == d and r['delta_y'] == round(dy, 3)
                    and r['action_type'] == name)

    per_d = {}
    for d in DS:
        ok_signs = []
        for sgn in (-1.0, 1.0):
            dy = sgn * d
            gi_m = get(d, dy, 'matched_teacher', 'goal_improvement')
            gi_s = get(d, dy, 'mild_straight', 'goal_improvement')
            gi_o = get(d, dy, 'opposite', 'goal_improvement')
            mild_miss = get(d, dy, 'mild_straight', 'missed')
            mild_fell = get(d, dy, 'mild_straight', 'fell')
            matched_best = gi_m >= gi_s and gi_m >= gi_o
            mild_safe = (gi_s > 0) and (not mild_miss) and (not mild_fell)
            opp_worse = (gi_m - gi_o) > MARGIN
            ok_signs.append(matched_best and mild_safe and opp_worse)
        per_d[d] = all(ok_signs)

    # mild unsolvable at some d (missed or fell for BOTH signs)
    unsolvable_d = []
    for d in DS:
        both = all(get(d, sgn * d, 'mild_straight', 'missed') or
                   get(d, sgn * d, 'mild_straight', 'fell') for sgn in (-1.0, 1.0))
        if both:
            unsolvable_d.append(d)

    # straight >= matched across all d/signs (too weak)
    straight_competitive = all(
        get(d, sgn * d, 'mild_straight', 'goal_improvement') >=
        get(d, sgn * d, 'matched_teacher', 'goal_improvement') - MARGIN
        for d in DS for sgn in (-1.0, 1.0))
    all_miss = all(r['missed'] for r in rows)

    sweet = sorted([d for d in DS if per_d[d]])
    if impl_issue or all_miss:
        verdict = 'IMPLEMENTATION_ISSUE'
    elif sweet:
        verdict = 'Y_SWEET_SPOT_FOUND'
    elif unsolvable_d:
        verdict = 'Y_BECOMES_UNSOLVABLE'
    elif straight_competitive:
        verdict = 'Y_STILL_TOO_WEAK'
    else:
        verdict = 'Y_STILL_TOO_WEAK'

    # ---- outputs ----
    csv_path = os.path.join(OUTDIR, 'results2.csv')
    cols = ['d', 'delta_y', 'action_type', 'ax', 'ay', 'az', 'forward_progress',
            'goal_improvement', 'lateral_disp', 'rot_change_deg', 'contact_steps',
            'missed', 'fell']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')

    summary = dict(
        env=P.ENV, ckpt=args.ckpt, step=int(step), seed=args.seed,
        ds=DS, action_sign_grip_dy=sign_dy,
        pos_action_y_moves_grip=('+y' if sign_dy > 0 else '-y'),
        pattern_per_d={str(d): bool(per_d[d]) for d in DS},
        smallest_sweet_d=(sweet[0] if sweet else None),
        unsolvable_d=unsolvable_d, straight_competitive=bool(straight_competitive),
        verdict=verdict)
    json.dump(summary, open(os.path.join(OUTDIR, 'summary2.json'), 'w'), indent=2)

    print('\n' + '=' * 60)
    print('VERDICT:', verdict)
    print('=' * 60)
    print('pattern holds per d:', {d: per_d[d] for d in DS})
    if sweet:
        print(f'smallest valid d = {sweet[0]}')
    print(f'unsolvable_d={unsolvable_d}  straight_competitive={straight_competitive}')
    print(f'\nsaved: {csv_path}')
    print(f'saved: {os.path.join(OUTDIR, "summary2.json")}')


if __name__ == '__main__':
    main()
