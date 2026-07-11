"""Lightweight confounder probe: is U = delta_y = y_object - y_gripper a suitable
hidden variable in the conedir FetchPush setup?

No training, no env changes. Reuses the conedir env + trained full-state policy.
Builds ONE clean clonable reference state (fixed obj x/z, fixed gripper, fixed
straight-ahead goal, zero velocities, fixed orientation) and only varies object y.

  Part A: does the policy's lateral action (action_y) respond to delta_y?
  Part B: under fixed actions, does delta_y change the object transition?
  Part C: is there a mild action that stays safe (>=0 progress) across all delta_y?

Run:  python scripts/probe_lateral_y.py            # single seed, CPU
      python scripts/probe_lateral_y.py --ckpt <best.pkl> --seed 0
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crl.config import Config
from crl import envs as envs_mod
from crl import report_push as R

ENV = 'fetch_push_easy_conedir'
CKPT_DEFAULT = r'D:/Users/trhua/Research/contrastive_rl/easypush_colab/fetch_push_easy_conedir_s1/best.pkl'
OUTDIR = r'D:/Users/trhua/Research/contrastive_rl/artifacts/lateral_y_probe'
DELTA_YS = [-0.04, -0.02, 0.00, 0.02, 0.04]
PUSH = 0.075                 # fixed straight-ahead goal distance (m)
CONTACT = 0.06               # gripper-object proximity counted as contact
N_STEPS = 3                  # control steps per fixed-action test
TABLE_Z = 0.40
MISS_MOVE = 0.003            # object moved < 3mm over the test => contact missed

ACTIONS = {                  # [dx, dy, dz, gripper], components in [-1,1]
    'small_forward':  np.array([0.5,  0.0, 0.0, 0.0], np.float32),
    'medium_forward': np.array([1.0,  0.0, 0.0, 0.0], np.float32),
    'forward_neg_y':  np.array([0.5, -0.5, 0.0, 0.0], np.float32),
    'forward_pos_y':  np.array([0.5,  0.5, 0.0, 0.0], np.float32),
}

OBJ = slice(3, 6)


def get_state(u):
    d = u.data
    return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(),
                mocap_pos=np.array(d.mocap_pos).copy(),
                mocap_quat=np.array(d.mocap_quat).copy(), time=float(d.time))


def set_state(u, s):
    d = u.data
    d.qpos[:] = s['qpos']; d.qvel[:] = s['qvel']
    d.mocap_pos[:] = s['mocap_pos']; d.mocap_quat[:] = s['mocap_quat']
    d.time = s['time']
    u._mujoco.mj_forward(u.model, u.data)


def set_object_y(u, x, y, z):
    q = np.array(u._utils.get_joint_qpos(u.model, u.data, 'object0:joint'), float)
    q[0:3] = [x, y, z]; q[3:7] = [1.0, 0.0, 0.0, 0.0]     # fixed orientation
    u._utils.set_joint_qpos(u.model, u.data, 'object0:joint', q)
    # zero the object free-joint velocities too
    u._utils.set_joint_qvel(u.model, u.data, 'object0:joint', np.zeros(6))
    u._mujoco.mj_forward(u.model, u.data)


def quat_angle_deg(q0, q1):
    d = min(1.0, abs(float(np.dot(q0, q1))))
    return float(2.0 * np.degrees(np.arccos(d)))


def obj_quat(u):
    return np.array(u._utils.get_joint_qpos(u.model, u.data, 'object0:joint'), float)[3:7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT_DEFAULT)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    cfg = Config(env_name=ENV)
    env = envs_mod.make_env(ENV, cfg, seed=args.seed)
    u = env._env.unwrapped
    greedy, step = R._load_greedy(ENV, args.ckpt, cfg)
    print(f'ckpt step {step} | env dt={u.dt:.3f}s | seed {args.seed}\n')

    # ---- build reference state ----
    o = env.reset()
    grip = o[0:3].copy()
    obj_x, obj_z = 1.40, 0.425
    g_y = float(grip[1])
    goal = np.array([obj_x + PUSH, g_y, obj_z], np.float32)   # fixed, straight +x
    u.goal = goal.astype(float)
    set_object_y(u, obj_x, g_y, obj_z)                        # delta_y = 0 baseline
    u.data.qvel[:] = 0.0
    u._mujoco.mj_forward(u.model, u.data)
    ref = get_state(u)
    print(f'reference: gripper={np.round(grip,4)}  object=({obj_x},{g_y:.4f},{obj_z})  '
          f'goal={np.round(goal,4)}')

    def restore_dy(dy):
        set_state(u, ref)
        set_object_y(u, obj_x, g_y + dy, obj_z)
        od = u._get_obs()
        flat = env._flatten(od)
        meas_dy = float(flat[4] - flat[1])
        return flat, od, meas_dy

    # sanity: verify delta_y is realizable / correct sign
    impl_issue = False
    for dy in DELTA_YS:
        _, _, meas = restore_dy(dy)
        if abs(meas - dy) > 1e-3:
            print(f'  [WARN] delta_y target {dy:+.3f} measured {meas:+.4f}')
            impl_issue = True

    # =================== Part A: policy vs delta_y ===================
    print('\n=== Part A: policy action vs delta_y ===')
    print(f'{"delta_y":>8} {"action_x":>9} {"action_y":>9} {"action_z":>9}')
    partA = []
    for dy in DELTA_YS:
        flat, _, meas = restore_dy(dy)
        a = np.asarray(greedy(flat), float)
        partA.append(dict(delta_y=dy, meas_delta_y=meas,
                          action_x=float(a[0]), action_y=float(a[1]),
                          action_z=float(a[2])))
        print(f'{dy:>8.2f} {a[0]:>9.3f} {a[1]:>9.3f} {a[2]:>9.3f}')
    ay = np.array([r['action_y'] for r in partA])
    ay_range = float(ay.max() - ay.min())
    opposite = (np.sign(ay[0]) != np.sign(ay[-1])) and min(abs(ay[0]), abs(ay[-1])) > 0.02
    policy_uses_y = ay_range > 0.05
    print(f'action_y range={ay_range:.3f}  opposite-sign(+/-)={opposite}  '
          f'policy_uses_y={policy_uses_y}')

    # plot
    plt.figure(figsize=(5, 4))
    plt.plot([r['delta_y'] for r in partA], ay, '-o')
    plt.axhline(0, color='k', lw=.7); plt.axvline(0, color='k', lw=.7)
    plt.xlabel('delta_y = obj_y - grip_y (m)'); plt.ylabel('policy action_y')
    plt.title('Part A: lateral action vs delta_y')
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'delta_y_vs_action_y.png'), dpi=100)

    # =================== Part B: transition vs delta_y ===================
    print('\n=== Part B: transition under fixed actions (3 steps) ===')
    rows = []
    for dy in DELTA_YS:
        for an, av in ACTIONS.items():
            flat, od0, meas = restore_dy(dy)
            obj0 = od0['achieved_goal'].copy(); q0 = obj_quat(u)
            d0 = float(np.linalg.norm(obj0 - goal))
            n_contact = 0; fell = False; minz = obj0[2]
            for _ in range(N_STEPS):
                grip_now = np.asarray(u._get_obs()['observation'][:3])
                obj_now = np.asarray(u._get_obs()['achieved_goal'])
                if np.linalg.norm(grip_now - obj_now) < CONTACT:
                    n_contact += 1
                u.step(av)
                oz = float(u._get_obs()['achieved_goal'][2]); minz = min(minz, oz)
                if oz < TABLE_Z:
                    fell = True
            obj1 = np.asarray(u._get_obs()['achieved_goal']).copy(); q1 = obj_quat(u)
            d1 = float(np.linalg.norm(obj1 - goal))
            move = float(np.linalg.norm(obj1 - obj0))
            rows.append(dict(
                delta_y=dy, action=an,
                action_x=float(av[0]), action_y=float(av[1]), action_z=float(av[2]),
                obj_dx=float(obj1[0] - obj0[0]), obj_dy=float(obj1[1] - obj0[1]),
                obj_rot_change_deg=quat_angle_deg(q0, q1),
                contact_steps=int(n_contact),
                goal_progress=float(d0 - d1),          # + = moved closer
                missed=bool(move < MISS_MOVE), fell=bool(fell)))

    # print compact table
    print(f'{"delta_y":>7} {"action":>14} {"dx":>7} {"dy":>7} {"drot":>6} '
          f'{"cts":>3} {"prog":>7} {"miss":>4} {"fall":>4}')
    for r in rows:
        print(f'{r["delta_y"]:>7.2f} {r["action"]:>14} {r["obj_dx"]:>7.3f} '
              f'{r["obj_dy"]:>7.3f} {r["obj_rot_change_deg"]:>6.1f} '
              f'{r["contact_steps"]:>3} {r["goal_progress"]:>7.3f} '
              f'{str(r["missed"]):>4} {str(r["fell"]):>4}')

    # write CSV
    csv_path = os.path.join(OUTDIR, 'results.csv')
    cols = ['delta_y', 'action', 'action_x', 'action_y', 'action_z', 'obj_dx',
            'obj_dy', 'obj_rot_change_deg', 'contact_steps', 'goal_progress',
            'missed', 'fell']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')

    # =================== Part C: robust action ===================
    print('\n=== Part C: robustness across delta_y ===')
    print(f'{"action":>14} {"mean_prog":>10} {"worst_prog":>11} {"miss":>5} {"fall":>5}')
    partC = {}
    for an in ACTIONS:
        pr = [r['goal_progress'] for r in rows if r['action'] == an]
        ms = sum(r['missed'] for r in rows if r['action'] == an)
        fl = sum(r['fell'] for r in rows if r['action'] == an)
        partC[an] = dict(mean_progress=float(np.mean(pr)),
                         worst_progress=float(np.min(pr)),
                         miss=int(ms), fall=int(fl))
        print(f'{an:>14} {np.mean(pr):>10.3f} {np.min(pr):>11.3f} {ms:>5} {fl:>5}')

    # dynamics dependence: spread of medium_forward progress across delta_y
    mf = [r['goal_progress'] for r in rows if r['action'] == 'medium_forward']
    dyn_spread = float(np.max(mf) - np.min(mf))
    dynamics_depend_on_y = dyn_spread > 0.01
    # direction-specific trade-off: neg_y best at dy<0, pos_y best at dy>0?
    def prog(an, dy):
        return next(r['goal_progress'] for r in rows if r['action'] == an and r['delta_y'] == dy)
    tradeoff = (prog('forward_neg_y', -0.04) > prog('forward_neg_y', 0.04) and
                prog('forward_pos_y', 0.04) > prog('forward_pos_y', -0.04))
    # robust mild action: non-negative worst-case, no falls, prefer small/medium fwd
    robust = None
    for an in ['small_forward', 'medium_forward', 'forward_neg_y', 'forward_pos_y']:
        c = partC[an]
        if c['worst_progress'] >= -0.002 and c['fall'] == 0 and c['miss'] <= 1:
            robust = an; break
    all_miss = all(r['missed'] for r in rows)

    # =================== verdict ===================
    if impl_issue or all_miss:
        verdict = 'IMPLEMENTATION_ISSUE'
    elif not dynamics_depend_on_y:
        verdict = 'Y_TOO_WEAK'
    elif not policy_uses_y:
        verdict = 'Y_POLICY_DOES_NOT_USE_Y'
    elif robust is not None:
        verdict = 'Y_LOOKS_SUITABLE'
    else:
        verdict = 'Y_AFFECTS_DYNAMICS_BUT_NO_ROBUST_ACTION'

    summary = dict(
        env=ENV, ckpt=args.ckpt, step=int(step), seed=args.seed,
        delta_ys=DELTA_YS, goal=goal.tolist(), gripper_y=g_y,
        partA=partA, action_y_range=ay_range, opposite_sign=bool(opposite),
        policy_uses_y=bool(policy_uses_y),
        dynamics_spread_medium_forward=dyn_spread,
        dynamics_depend_on_y=bool(dynamics_depend_on_y),
        direction_tradeoff=bool(tradeoff),
        partC=partC, robust_action=robust, verdict=verdict)
    json.dump(summary, open(os.path.join(OUTDIR, 'summary.json'), 'w'), indent=2)

    print('\n' + '=' * 60)
    print('VERDICT:', verdict)
    print('=' * 60)
    print(f'  policy_uses_y={policy_uses_y} (action_y range {ay_range:.3f}, '
          f'opposite-sign {opposite})')
    print(f'  dynamics_depend_on_y={dynamics_depend_on_y} '
          f'(medium_forward progress spread {dyn_spread:.3f} m)')
    print(f'  direction_tradeoff={tradeoff}   robust_mild_action={robust}')
    print(f'\nsaved: {csv_path}')
    print(f'saved: {os.path.join(OUTDIR, "summary.json")}')
    print(f'saved: {os.path.join(OUTDIR, "delta_y_vs_action_y.png")}')


if __name__ == '__main__':
    main()
