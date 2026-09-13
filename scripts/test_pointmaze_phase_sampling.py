"""Sampling-only regression and evaluation integrity checks."""
import unittest
import numpy as np
import torch
from ett import pointmaze_phase_sampling as p
from ett import eval_pointmaze_phase_sampling as e
from ett import pointmaze_region_pilot as old
from ett import pointmaze_early_pilot as early


def fixture():
    records=[]
    rng=np.random.default_rng(15)
    for root,h in enumerate([40,44,47,49]):
        records.append(dict(states=rng.normal(size=(1,h+1,8)).astype(np.float32),
            action=rng.uniform(-1,1,(1,h,2)).astype(np.float32),reward=rng.integers(2,size=(1,h)).astype(np.float32),root=np.array([root])))
    return records,p.training_rows(records,np.array([0,1,2,0]),np.zeros(8),np.ones(8))


class PhaseSamplingTests(unittest.TestCase):
    def test_phase_distribution_and_label_independence(self):
        _,data=fixture();n=200000
        uniform=p.sample_rows(np.random.default_rng(25),data,'uniform',n)
        balanced=p.sample_rows(np.random.default_rng(25),data,'balanced',n)
        expected=np.bincount(data['phase'],minlength=3)/len(data['phase'])
        np.testing.assert_allclose(np.bincount(data['phase'][uniform],minlength=3)/n,expected,atol=.004)
        np.testing.assert_allclose(np.bincount(data['phase'][balanced],minlength=3)/n,np.full(3,1/3),atol=.004)
        np.testing.assert_allclose(np.bincount(data['path'][balanced],minlength=4)/n,np.full(4,.25),atol=.004)
        changed=dict(data);changed['label']=1-data['label'];changed['group']=2-data['group']
        np.testing.assert_array_equal(balanced,p.sample_rows(np.random.default_rng(25),changed,'balanced',n))
        for path in range(4):
            for phase in range(3):
                chosen=balanced[(data['path'][balanced]==path)&(data['phase'][balanced]==phase)]
                values,counts=np.unique(chosen,return_counts=True)
                expected_rows=np.flatnonzero((data['path']==path)&(data['phase']==phase))
                np.testing.assert_array_equal(values,expected_rows)
                self.assertLess(np.max(np.abs(counts/counts.sum()-1/len(values))),.02)

    def test_uniform_fit_is_original_binary_nce(self):
        torch.set_num_threads(1);_,data=fixture()
        a=early.Critic(33);b=early.Critic(33)
        p.fit_rows(a,data,'uniform',34,steps=3)
        opt=torch.optim.Adam(b.parameters(),lr=.003)
        old.fit(b,opt,(data['x'],data['label']),3,34)
        self.assertEqual(old.parameter_sha(a),old.parameter_sha(b))
        self.assertFalse(any(x.requires_grad for x in a.parameters()))

    def test_query_actions_and_exact_remaining_horizon(self):
        records,_=fixture();q=e.query_records(records,4)
        for i,r in enumerate(records):
            h=r['action'].shape[1]
            for phase,t in zip(p.PHASES,[0,5,h//2]):
                np.testing.assert_array_equal(q[phase+'_state'][i],r['states'][0,t])
                np.testing.assert_array_equal(q[phase+'_action'][i],r['action'][0,t])
                self.assertEqual(q[phase+'_h'][i],h-t)

    def test_pairing_noise_screen_and_cluster_intervals(self):
        returns=np.repeat(np.array([.1,.2,.3,.4])[:,None],64,axis=1)
        pairs=e.ordering_pairs(returns,np.array([40,40,49,40]))
        self.assertEqual(pairs['all_same_h_pairs'],3);self.assertEqual(pairs['informative_pairs'],3)
        good=e.ordering(returns.mean(1),pairs,-returns.mean(1))
        self.assertEqual(good['accuracy'],1.);self.assertEqual(good['paired_accuracy_change'],1.)
        self.assertFalse(good['adequate'])
        tied=e.ordering_pairs(np.ones((4,64))*.2,np.full(4,40))
        self.assertEqual(tied['equal_pairs'],6);self.assertIsNone(e.ordering(np.zeros(4),tied)['accuracy'])
        noisy=np.array([[.1]*32+[.4]*32,[.2]*64])
        self.assertEqual(e.ordering_pairs(noisy,np.full(2,40))['informative_pairs'],0)
        comparison=e.paired_calibration(returns.mean(1),returns.mean(1),returns,np.zeros(4))
        np.testing.assert_array_equal(comparison['rmse_change_ci95'],[0.,0.])

    def test_native_context_interface_without_final_data(self):
        from crl.config import Config
        from crl.envs import make_env
        from scripts.collect_swamp_windy import make_windy_teacher
        env=make_env('point_two_route_swamp_windy_f4_v0',Config(env_name='point_two_route_swamp_windy_f4_v0'),seed=113000000)
        env.reset();policy=make_windy_teacher(env,np.random.default_rng(113000001),.05);memo={}
        self.assertEqual(env.max_episode_steps,50);self.assertEqual(env.active_prob,.3)
        previous=env._get_obs().copy()
        for _ in range(8):
            action=policy(env.state.copy(),env.goal.copy(),memo)
            env.step(action);current=env._get_obs().copy()
            np.testing.assert_array_equal(current[2:8],previous[:6]);np.testing.assert_array_equal(current[8:],old.GOAL)
            previous=current


if __name__=='__main__':unittest.main()
