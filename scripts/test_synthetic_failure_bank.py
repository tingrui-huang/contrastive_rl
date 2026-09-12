"""Objective, sampling and independently derived reference checks."""
import ast
import inspect
import unittest

import numpy as np
import torch

from scripts.synthetic_failure_bank import bank_losses, initialize, sample_data, train
from scripts.synthetic_failure_reference import (optimal_diagonal,optimal_response,
    population_reference,fixed_class_reference,quadrature)


class FailureBankTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_marginal_and_conditional_sampling(self):
        d,o=sample_data(3,8192,32768)
        np.testing.assert_array_equal(np.repeat(d[:,1],4),o[:,1])
        np.testing.assert_array_equal(d[:,0],d[:,1])
        self.assertTrue(np.all(np.abs(o)<=1));self.assertTrue(np.all(o[:,0]!=o[:,1]))
        self.assertLess(abs(float(o[:,0].mean())),.015)
        self.assertLess(abs(float(d[:,1].mean())),.02)
        self.assertLess(abs(float(np.corrcoef(o.T)[0,1])),.025)
        self.assertLess(abs(float(np.mean(np.abs(o[:,0]-o[:,1])))-2/3),.02)
        np.testing.assert_array_equal(sample_data(3,8192,32768)[1],o)

    def test_exact_two_terms_and_detached_bank(self):
        model=initialize(2);d,o=[torch.from_numpy(a) for a in sample_data(4,64,256)]
        bank=torch.tensor([-3.,-.5],dtype=torch.float64,requires_grad=True)
        total,diag,off=bank_losses(model,d,o,bank,.1)
        torch.testing.assert_close(total,diag+.1*off)
        expected=torch.minimum((model(o)+3)**2,(model(o)+.5)**2).mean()
        torch.testing.assert_close(off,expected)
        params=tuple(model.parameters())
        gd=torch.autograd.grad(diag,params,retain_graph=True)
        go=torch.autograd.grad(off,params,retain_graph=True)
        gt=torch.autograd.grad(total,params,retain_graph=True)
        for a,b,c in zip(gd,go,gt):torch.testing.assert_close(a+.1*b,c)
        self.assertIsNone(torch.autograd.grad(total,bank,allow_unused=True)[0])
        self.assertTrue(all(p.requires_grad for p in params))

    def test_reference_optimality_branch_and_population_integrals(self):
        z,w=np.polynomial.legendre.leggauss(128);w=w/2
        d=optimal_diagonal(z);m=(1+z*z)/2
        derivative=2*d+.2*(3+d-m)
        np.testing.assert_allclose(derivative,0,atol=1e-15)
        self.assertGreaterEqual(float(np.min(3+d-(1+np.abs(z)))),9/11)
        failure=(3+d)**2-2*(3+d)*m+1/3+z*z
        ref=population_reference()
        self.assertAlmostEqual(float(w@(d*d)),ref['diagonal_mse'],14)
        self.assertAlmostEqual(float(w@failure),ref['failure_mse'],13)
        self.assertAlmostEqual(float(w@(d*d+.1*failure)),ref['total'],14)
        for shift in [-.1,.1]:
            trial=(d+shift)**2+.1*((3+d+shift)**2-2*(3+d+shift)*m+1/3+z*z)
            np.testing.assert_allclose(trial-(d*d+.1*failure),1.1*shift**2,atol=1e-14)

    def test_feasible_reference_of_same_architecture(self):
        model=fixed_class_reference();x=np.linspace(-1,1,10001)
        with torch.no_grad():actual=model(torch.from_numpy(np.stack((x,x),-1))).numpy()
        bound=population_reference()['class_reference_max_diagonal_error']
        self.assertLessEqual(float(np.max(np.abs(actual-optimal_diagonal(x)))),bound+1e-14)
        q=quadrature(model,256);ref=population_reference()
        self.assertLess(abs(q['population_objective_gap']-ref['class_reference_population_gap']),1e-12)
        rng=np.random.default_rng(13);triples=rng.uniform(-1,1,(1000,3))
        a=optimal_response(triples[:,[0,2]]);b=optimal_response(triples[:,[1,2]])
        self.assertTrue(np.all(np.abs(a-b)<=np.abs(triples[:,0]-triples[:,1])+1e-14))

    def test_optimizer_has_no_reference_labels(self):
        tree=ast.parse(inspect.getsource(train))
        names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        self.assertFalse(names & {'optimal_response','optimal_diagonal','fixed_class_reference','two_losses'})
        model=initialize(0);d,o=[torch.from_numpy(a) for a in sample_data(8,32,128)]
        before=model(d).detach().clone()
        with torch.no_grad():model.conditioner[-1].bias[0].add_(.1)
        torch.testing.assert_close(model(d),before+.1)
        self.assertGreater(float(before.abs().max()),.001)


if __name__=='__main__':unittest.main()
