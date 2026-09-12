"""Evaluation-only teacher/simulator pairing with explicit exogenous coupling.

Hidden simulator fields stay in restoration records, never in model arguments.
Pairs for one underlying context are not samples from an exact conditional ETT.
"""
import copy
import numpy as np

from crl.envs import TwoRouteSwampWindyF4Env
from scripts.collect_swamp_windy import make_windy_teacher
from ett.action_response import select_contexts


def snapshot(env):
  return {'state':env.state.copy(),'goal':env.goal.copy(),'frames':np.array(env._frames),
      'bits':env._bits.copy(),'absorbing':bool(env._dead),
      'rng_state':copy.deepcopy(env._rng.bit_generator.state),
      'action_noise':env._action_noise,'active_prob':env.active_prob,
      'auto_resample':env._auto_resample,'max_episode_steps':env.max_episode_steps}


def restore(record):
  env=TwoRouteSwampWindyF4Env(action_noise=record['action_noise'],active_prob=record['active_prob'],
                            max_episode_steps=record['max_episode_steps'])
  env.state=np.array(record['state'],dtype=float);env.goal=np.array(record['goal'],dtype=float)
  env._frames=[np.array(v,dtype=float) for v in record['frames']]
  env._bits=np.array(record['bits'],dtype=bool);env._dead=bool(record['absorbing'])
  env._auto_resample=bool(record['auto_resample'])
  env._rng.bit_generator.state=copy.deepcopy(record['rng_state'])
  return env


def natural_action(policy,env,rng,memo,noise=.15):
  action=np.asarray(policy(env.state.copy(),env.goal.copy(),memo),np.float32)
  noisy=action.copy()
  if noise>0 and np.any(action!=0):noisy=action+rng.normal(0,noise,2)
  clipped=np.clip(noisy,-1,1).astype(np.float32)
  return clipped,bool(np.any(noisy!=np.clip(noisy,-1,1)))


def teacher_contexts(episodes,count,selection_seed=940001):
  env=TwoRouteSwampWindyF4Env(seed=930000)
  rng=np.random.default_rng(930001);policy=make_windy_teacher(env,rng,.05)
  candidates=[];mode_counts={};clipped=0
  for episode in range(episodes):
    env.reset();memo={}
    for t in range(50):
      before=snapshot(env);obs=env._get_obs().copy()
      behavior_before=copy.deepcopy(rng.bit_generator.state);memo_before=copy.deepcopy(memo)
      xp,was_clipped=natural_action(policy,env,rng,memo);clipped+=was_clipped
      next_obs,reward,done,_=env.step(xp)
      candidates.append({'state':obs[:8],'goal':obs[8:],'observational_action':xp,
          'episode':episode,'timestep':t,'restoration':before,'behavior_rng_before':behavior_before,
          'teacher_memo_before':memo_before,'teacher_action_clipped':was_clipped,
          'observed_next_state':next_obs[:8],'observed_reward':reward,'done':done,
          'environment_rng_after':copy.deepcopy(env._rng.bit_generator.state)})
    mode=memo['teacher_mode'];mode_counts[mode]=mode_counts.get(mode,0)+1
  ids,groups,counts=select_contexts(np.array([r['state'] for r in candidates]),
      np.array([r['episode'] for r in candidates]),np.array([r['timestep'] for r in candidates]),count,selection_seed)
  selected=[candidates[i] for i in ids]
  # Replay both teacher RNG/memo and the untouched environment RNG.
  for row in selected:
    clone=restore(row['restoration']);behavior=np.random.default_rng()
    behavior.bit_generator.state=copy.deepcopy(row['behavior_rng_before'])
    teacher=make_windy_teacher(clone,behavior,.05)
    xp,_=natural_action(teacher,clone,behavior,copy.deepcopy(row['teacher_memo_before']))
    if not np.array_equal(xp,row['observational_action']):raise RuntimeError('teacher action replay mismatch')
    obs,reward,done,_=clone.step(xp)
    if not np.array_equal(obs[:8],row['observed_next_state']) or reward!=row['observed_reward'] or done!=row['done']:
      raise RuntimeError('native simulator replay mismatch')
    if clone._rng.bit_generator.state!=row['environment_rng_after']:raise RuntimeError('RNG restoration mismatch')
  return selected,groups,{'eligible':counts,'episodes':episodes,'natural_steps':episodes*50,
      'mode_counts':mode_counts,'teacher_noise_clipped_steps':clipped,
      'environment_seed':930000,'behavior_seed':930001,'selection_seed':selection_seed,
      'native_teacher_action_observation_reward_rng_replay_exact':True,
      'population':'fresh episodes from the exact nominal-source teacher protocol: force-safe .05, nonzero-action Gaussian noise .15, no uniform coverage and no blind demonstrator',
      'scope':'one underlying context per selected row; no posterior matching or conditional ETT ground truth'}


class StepNoise:
  """Two independent per-time exogenous slots, regardless of branch call order.

  The native dead branch skips action noise. Copying one generator at t=0
  would therefore misalign later bit draws after branches disagree on death.
  Separate slots preserve the actual IID marginal noise law and a declared
  synchronous coupling; this coupling is not identified by observed data.
  """
  def __init__(self,normal,uniform):
    self.normal_value=np.asarray(normal);self.uniform_value=np.asarray(uniform)
  def normal(self,loc,scale,size):
    if tuple(np.atleast_1d(size))!=(2,):raise ValueError('unexpected normal draw')
    return loc+scale*self.normal_value.copy()
  def random(self,size):
    if size!=3:raise ValueError('unexpected uniform draw')
    return self.uniform_value.copy()


def continuation_indices(rows,horizon=5):
  if not 1<=horizon<=50:raise ValueError('invalid continuation horizon')
  return np.array([i for i,row in enumerate(rows) if row['timestep']+horizon<=50],dtype=int)


def prescribed_reference(rows,actions,horizon=5):
  if len(continuation_indices(rows,horizon))!=len(rows):
    raise ValueError('reference would exceed the original 50-step task horizon')
  states=[];rewards=[];audit=[]
  for c,row in enumerate(rows):
    context_states=[];context_rewards=[];context_audit=[]
    for action in actions[c]:
      env=restore(row['restoration']);ss=[env._get_obs()[:8]];rr=[];bits=[];absorbing=[]
      for t in range(horizon):
        if t>0:
          tape=np.random.default_rng(970000+c*100+t)
          env._rng=StepNoise(tape.normal(size=2),tape.random(3))
        # Hidden audit arrays never travel with model input arrays.
        bits.append(env._bits.copy());absorbing.append(env._dead)
        obs,reward,done,_=env.step(action)
        if done:raise RuntimeError('fixed-horizon environment unexpectedly terminated')
        ss.append(obs[:8]);rr.append(reward)
      context_states.append(ss);context_rewards.append(rr)
      context_audit.append({'bits_before':bits,'absorbing_before':absorbing})
    states.append(context_states);rewards.append(context_rewards);audit.append(context_audit)
  return {'states':np.asarray(states),'reward':np.asarray(rewards),'execution_action':actions},audit
