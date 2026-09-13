"""Checkpoint metadata and actual selected-set validation; zero stochastic draws."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ett.convex_action_transition import (ConvexActionTransition,save_convex_checkpoint,
    load_convex_checkpoint,emit_with_geometry,validate_selected_set)


class GeometryExperimentChecks(unittest.TestCase):
    def test_checkpoint_roundtrip_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for mode in ['rectangle','fork_segment']:
                model=ConvexActionTransition(None,geometry_mode=mode)
                path=root/(mode+'.npz');theta=np.linspace(-.2,.2,32)
                save_convex_checkpoint(path,model,theta)
                loaded=load_convex_checkpoint(None,path,require_geometry=True)
                self.assertEqual(loaded.geometry_mode,mode)
                np.testing.assert_array_equal(loaded.theta,theta.astype(np.float32))
            np.savez(root/'old.npz',theta=np.zeros(32),bound=1.)
            self.assertEqual(load_convex_checkpoint(None,root/'old.npz').geometry_mode,'rectangle')
            with self.assertRaises(ValueError):load_convex_checkpoint(None,root/'old.npz',require_geometry=True)
            np.savez(root/'broken.npz',theta=np.zeros(32),bound=1.,format_version=1)
            with self.assertRaises(ValueError):load_convex_checkpoint(None,root/'broken.npz')

    def test_segment_outside_reference_box_is_valid(self):
        state=np.tile([1.5,3.5],4).astype(np.float32)[None]
        anchor=np.array([[[2.4,3.4]]],np.float32)
        action=np.array([[-1.,-1.]],np.float32);xp=-action
        theta=np.tile([.4,0,0,.4],8).astype(np.float32)
        y,d=emit_with_geometry(theta,state,action,xp,anchor,geometry_mode='fork_segment')
        self.assertLess(float(y[0,0,1]),3.)
        self.assertEqual(float(d['box_low'][0,0,1]),3.)
        self.assertTrue(validate_selected_set(state,y,d))
        broken=np.array(y);broken[0,0,0]+=.1
        with self.assertRaises(AssertionError):validate_selected_set(state,broken,d)


if __name__=='__main__':unittest.main()
