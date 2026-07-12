"""Copy a training run's latest.pkl to gate_<step>.pkl at gate boundaries.

Polls <run_dir>/latest.pkl (written atomically by crl.checkpoint at each eval)
and snapshots it the first time its recorded step reaches each gate. Exits
when the last gate is captured or after --timeout_min.
"""
import argparse
import os
import pickle
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root: unpickling the
import crl.losses  # noqa: F401,E402  checkpoint needs TrainingState importable


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_dir', required=True)
  ap.add_argument('--gates', default='10000,20000,30000,40000,50000')
  ap.add_argument('--poll_s', type=float, default=15)
  ap.add_argument('--timeout_min', type=float, default=180)
  args = ap.parse_args()
  gates = sorted(int(g) for g in args.gates.split(','))
  latest = os.path.join(args.run_dir, 'latest.pkl')
  t0 = time.time()
  done = set()
  while len(done) < len(gates) and (time.time() - t0) < args.timeout_min * 60:
    try:
      with open(latest, 'rb') as f:
        step = pickle.load(f)['step']
      for g in gates:
        if g not in done and step >= g:
          dst = os.path.join(args.run_dir, f'gate_{g}.pkl')
          shutil.copy2(latest, dst)
          # record the actual step inside the snapshot (may exceed the gate)
          print(f'GATE {g} captured at step {step} -> {dst}', flush=True)
          done.add(g)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
      pass
    time.sleep(args.poll_s)
  print(f'watcher done: {sorted(done)}', flush=True)


if __name__ == '__main__':
  main()
