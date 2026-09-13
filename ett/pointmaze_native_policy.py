"""Frozen observational/pessimistic models, averaged region critics, actor updates."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess

import jax
import jax.numpy as j
import numpy as np
import optax
import torch

from crl import checkpoint, networks
from ett import pointmaze_update_reference as update
from ett import pointmaze_future_average as average
from ett import pointmaze_phase_sampling as phase
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett.policy_improvement import tree_sha
from ett.rollout_return import START, GOAL
from ett.finite_crl import write, sha

PREVIOUS=Path('artifacts/pointmaze_region_pilot/update_reference_s01_v1')
CONFIG=dict(base='817e4295c9dbb347fbf74f0704730767598a6dfa', seeds=[0,1],
    rounds=4, actor_steps=25, actor_batch=256, actor_lr=1e-5, gradient_clip=1.,
    critic_initial_steps=1500, critic_refresh_steps=400, critic_lr=.003,
    initial_kl_cap=.02, step_kl_cap=.002, action_rms_cap=.05, action_max_cap=.15,
    paths_per_root=8, public_start_roots=12, native_episodes=256, model_episodes=128,
    critic_seed=130000000, train_seed=131000000, native_reset_seed=140000000,
    native_action_seed=141000000, model_evaluation_seed=142000000, bootstrap_seed=143000000,
    model_cap=500000, native_cap=64000, gamma=.95, horizon=50,
    actor_attempts=400, critic_steps=10800, ett_updates=0, nominal_updates=0)


class Ledger(old.Ledger):
    def __init__(self,out):
        super().__init__(out); self.data['cap']=CONFIG['model_cap']; self.scope='shared'
    def add(self,n,purpose): super().add(n,self.scope+'/'+purpose)


def actor_setup():
    _, saved=checkpoint.load_checkpoint(old.inputs()['actor'])
    net=networks.make_networks(8,8,2)
    return net,saved.policy_params


def export_critic(critic):
    return {k:j.asarray(v.detach().numpy(),dtype=j.float32) for k,v in critic.state_dict().items()}


def decoded_q(parameters,state,action,h,mean,std):
    """Differentiable action path; same unclipped region decoding as Torch."""
    x=j.concatenate([(state-mean)/std,action,h[...,None]/50.],axis=-1)
    for layer in [0,2,4]:
        x=x@parameters[f'phi.{layer}.weight'].T+parameters[f'phi.{layer}.bias']
        if layer!=4: x=j.tanh(x)
    f1=x@parameters['psi.weight'][1]
    return j.where(h==0,0.,(1-.95**h)*15.5*j.exp(f1))


def policy_distribution(net,params,state):
    return net.policy_network.apply(params,j.concatenate([state,j.broadcast_to(j.asarray(GOAL),state.shape)],axis=-1))


def sample_actor(net,params,state,key):
    dist=policy_distribution(net,params,state)
    return j.tanh(dist.loc+dist.scale*jax.random.normal(key,dist.loc.shape))


def gaussian_kl(old_dist,new_dist):
    return j.sum(j.log(new_dist.scale/old_dist.scale)+
        (old_dist.scale**2+(old_dist.loc-new_dist.loc)**2)/(2*new_dist.scale**2)-.5,axis=-1)


def policy_change(net,reference,candidate,states):
    a=policy_distribution(net,reference,states); b=policy_distribution(net,candidate,states)
    delta=j.linalg.norm(j.tanh(a.loc)-j.tanh(b.loc),axis=-1)
    return j.stack([j.mean(gaussian_kl(a,b)),j.sqrt(j.mean(delta**2)),j.max(delta),j.mean(b.scale-a.scale)])


def policy_admissible(bank,current,step_kl):
    for change in [bank,current]:
        if not (np.isfinite(change).all() and change[0]<=.02 and change[1]<=.05 and change[2]<=.15): return False
    return bool(np.isfinite(step_kl) and step_kl<=.002)


class FrozenRollouts:
    def __init__(self,engine,net):
        self.engine=engine; self.net=net; self.compiled={}
    def rollout(self,params,theta,observational,states,key,horizon):
        cache_key=(observational,horizon)
        if cache_key not in self.compiled:
            def generate(params,theta,states,key):
                def step(s,key):
                    ak,bk,tk=jax.random.split(key,3)
                    a=sample_actor(self.net,params,s,ak)
                    xp=(a if observational else self.engine.nominal.sample(s,bk,1,goal=j.broadcast_to(j.asarray(GOAL),s.shape)))
                    y,d=self.engine._sample(theta,s,a,xp,tk,1)
                    ns=y[:,0]
                    return ns,dict(next=ns,action=a,x_prime=xp,reward=old.task_reward(ns,j.broadcast_to(j.asarray(GOAL),s.shape)),valid=d['box_valid'][:,0])
                _,r=jax.lax.scan(step,states,jax.random.split(key,horizon))
                r=jax.tree.map(lambda x:j.swapaxes(x,0,1),r)
                r['states']=j.concatenate([states[:,None],r.pop('next')],axis=1)
                return r
            self.compiled[cache_key]=jax.jit(generate)
        r=jax.tree.map(np.asarray,self.compiled[cache_key](params,theta,states,key))
        assert r['valid'].all() and np.isfinite(r['states']).all()
        np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
        assert np.abs(r['action']).max()<=1 and np.abs(r['x_prime']).max()<=1
        if observational: np.testing.assert_array_equal(r['action'],r['x_prime'])
        return r


def collect(generator,params,theta,observational,roots,times,seed,ledger,repeats=8):
    records=[]
    for time in np.unique(times):
        ids=np.flatnonzero(times==time); h=50-int(time); s=np.repeat(roots[ids],repeats,axis=0)
        ledger.add(len(s)*h,'current_actor_rollouts')
        r=generator.rollout(params,theta,observational,s,jax.random.PRNGKey(seed+h),h)
        r['root']=np.repeat(ids,repeats); records.append(r)
    return records


def actor_loss(params,critic,states,h,weight,epsilon,mean,std,net):
    dist=policy_distribution(net,params,states)
    action=j.tanh(dist.loc[:,None]+dist.scale[:,None]*epsilon)
    q=decoded_q(critic,j.repeat(states,4,axis=0),action.reshape(-1,2),j.repeat(h,4),mean,std).reshape(-1,4)
    return -j.mean(weight[:,None]*q)


def save_actor(path,params,optimizer=None):
    with path.open('wb') as f: pickle.dump(jax.tree.map(np.asarray,dict(params=params,optimizer=optimizer)),f)


def load_actor(path):
    with path.open('rb') as f: return jax.tree.map(j.asarray,pickle.load(f)['params'])


def models():
    kernels=dict(np.load(PREVIOUS/'final_kernels.npz')); result={}
    for seed in [0,1]:
        result[f'obs_s{seed}']=(kernels[f'diagonal_s{seed}'],True)
        result[f'pess_s{seed}']=(kernels[f'critic_s{seed}'],False)
    return result


def train(out,engine,net,initial,context,ledger):
    generator=FrozenRollouts(engine,net)
    roots=np.concatenate([context['train_roots'],np.tile(START,(12,1))])
    times=np.r_[context['train_indices'][:,1],np.zeros(12,int)]
    bank=np.concatenate([context['train_state'],context['train_roots'],np.tile(START,(64,1))])
    np.savez_compressed(out/'training_contexts.npz',roots=roots,times=times,policy_guard_bank=bank)
    transformation=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-5))
    loss_grad=jax.jit(jax.value_and_grad(lambda params,c,s,h,w,eps:actor_loss(params,c,s,h,w,eps,
        j.asarray(context['mean'],j.float32),j.asarray(context['std'],j.float32),net)))
    change=jax.jit(lambda ref,can,s:policy_change(net,ref,can,s))
    actors={'A':initial}; histories={}; initial_critic_hash={}
    save_actor(out/'checkpoints'/'A.pkl',initial)
    for seed in [0,1]:
        for arm,model_name in [('B',f'obs_s{seed}'),('C',f'pess_s{seed}')]:
            name=f'{arm}_s{seed}'; ledger.scope=name; params=initial; optimizer=transformation.init(params)
            critic=early.Critic(CONFIG['critic_seed']+seed); critic_opt=torch.optim.Adam(critic.parameters(),lr=.003)
            critic_initial=old.parameter_sha(critic)
            if seed in initial_critic_hash: assert critic_initial==initial_critic_hash[seed]
            else: initial_critic_hash[seed]=critic_initial
            theta,observational=models()[model_name]; history=[]
            for round_id in range(4):
                folder=out/'rounds'/f'{name}_r{round_id}'; folder.mkdir(parents=True)
                key=CONFIG['train_seed']+100000*seed+1000*round_id
                records=collect(generator,params,theta,observational,roots,times,key,ledger)
                phase.save_records(folder/'paths.npz',records)
                before_actor=tree_sha(params)
                fit=update.refresh(critic,critic_opt,records,context,key+100,1500 if round_id==0 else 400)
                assert tree_sha(params)==before_actor
                frozen=old.parameter_sha(critic); c=export_critic(critic)
                torch.save(dict(model=critic.state_dict(),optimizer=critic_opt.state_dict()),out/'checkpoints'/f'{name}_r{round_id}_critic.pt')
                save_actor(out/'checkpoints'/f'{name}_r{round_id}_before.pkl',params,optimizer)
                steps=[]
                for step in range(25):
                    query=update.visitation(records,key+200+step,256)
                    noise=np.random.default_rng(key+300+step).normal(size=(256,2,2)).astype(np.float32)
                    epsilon=np.concatenate([noise,-noise],axis=1)
                    args=(c,query['state'],query['h'],query['weight'].astype(np.float32),epsilon)
                    value,gradient=loss_grad(params,*args)
                    delta,new_optimizer=transformation.update(gradient,optimizer,params)
                    proposed=optax.apply_updates(params,delta)
                    on_bank=np.asarray(change(initial,proposed,bank)); on_current=np.asarray(change(initial,proposed,query['state']))
                    step_kl=float(change(params,proposed,query['state'])[0])
                    accepted=policy_admissible(on_bank,on_current,step_kl)
                    assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient))
                    candidate_value=float(actor_loss(proposed,c,query['state'],query['h'],query['weight'].astype(np.float32),epsilon,
                        j.asarray(context['mean'],j.float32),j.asarray(context['std'],j.float32),net))
                    if accepted: params,optimizer=proposed,new_optimizer
                    steps.append(dict(step=step,accepted=accepted,q_before=-float(value),q_proposed=-candidate_value,
                        gradient_norm=float(optax.global_norm(gradient)),bank_change=on_bank,current_change=on_current,step_kl=step_kl))
                assert old.parameter_sha(critic)==frozen
                save_actor(out/'checkpoints'/f'{name}_r{round_id}_after.pkl',params,optimizer)
                row=dict(round=round_id,critic=fit,critic_sha256=frozen,actor_before_sha256=before_actor,
                    actor_after_sha256=tree_sha(params),steps=steps)
                write(folder/'training.json',row); history.append(row)
                print(f'{name}: round {round_id+1}/4 complete; 25 guarded actor proposals.',flush=True)
            actors[name]=params; histories[name]=dict(model=model_name,initial_actor_sha256=tree_sha(initial),
                initial_critic_sha256=critic_initial,rounds=history,final_actor_sha256=tree_sha(params),
                final_bank_change=np.asarray(change(initial,params,bank)),
                parameter_change_norm=float(optax.global_norm(jax.tree.map(lambda a,b:a-b,params,initial))))
            save_actor(out/'checkpoints'/f'{name}_final.pkl',params,optimizer)
    write(out/'training.json',dict(utc=average.timestamp(),actors=histories,actor_attempts=400,critic_steps=10800,
        actor_A_sha256=tree_sha(initial),ett_updates=0,nominal_updates=0,
        final_checkpoints={name:sha(out/'checkpoints'/('A.pkl' if name=='A' else name+'_final.pkl')) for name in actors}))
    return actors,generator


def prepare(out):
    if out.exists(): raise ValueError('Fresh output directory required.')
    update.verify(PREVIOUS)
    sources=['ett/pointmaze_native_policy.py','ett/eval_pointmaze_native_policy.py',
        'ett/check_pointmaze_native_policy.py','scripts/test_pointmaze_native_policy.py','notes/pointmaze_native_policy.md',
        'crl/networks.py','ett/fixed_actor_continuation.py','crl/envs.py','crl/checkpoint.py']
    inherited=json.loads((PREVIOUS/'preregistration.json').read_text())['sources']
    inputs=[PREVIOUS/'final_kernels.npz',PREVIOUS/'results.json',PREVIOUS/'training.json',phase.PREVIOUS/'contexts.npz']
    hashes={**inherited,**{str(f):sha(f) for f in sources+inputs}}
    out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'rounds').mkdir()
    write(out/'config.json',CONFIG); (out/'PROTOCOL.md').write_bytes(Path(sources[4]).read_bytes())
    write(out/'preregistration.json',dict(utc=average.timestamp(),sources=hashes,
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=sha(out/'config.json'),protocol_sha256=sha(out/'PROTOCOL.md')))
    print('Native policy experiment protocol, budgets and hashes sealed.',flush=True)


def verify(out):
    d=json.loads((out/'preregistration.json').read_text()); assert json.loads((out/'config.json').read_text())==CONFIG
    assert sha(out/'config.json')==d['config_sha256'] and sha(out/'PROTOCOL.md')==d['protocol_sha256']
    for path,digest in d['sources'].items(): assert sha(path)==digest,path


def run(out):
    from ett import eval_pointmaze_native_policy as evaluation
    verify(out)
    if (out/'started.json').exists(): raise ValueError('Existing attempt protected.')
    write(out/'started.json',dict(utc=average.timestamp()))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    engine=old.Kernel();net,initial=actor_setup();context=dict(np.load(phase.PREVIOUS/'contexts.npz'))
    frozen=tree_sha((engine.base.params,engine.nominal.params,models()));ledger=Ledger(out)
    actors,generator=train(out,engine,net,initial,context,ledger)
    evaluation.evaluate(out,net,initial,actors,generator,ledger,context)
    assert tree_sha((engine.base.params,engine.nominal.params,models()))==frozen
    assert tree_sha(actors['A'])==tree_sha(initial)
    verify(out)
    write(out/'completion.json',dict(utc=average.timestamp(),frozen_models_nominal_A_verified=True,
        model_transitions=ledger.data['charged'],native_steps=64000,actor_attempts=400,critic_steps=10800))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=['prepare','run'])
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    (prepare if args.phase=='prepare' else run)(args.out)
