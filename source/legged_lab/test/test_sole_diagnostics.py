import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/rsl_rl'))
from sole_diagnostics import sole_diagnostic


class TestSoleDiagnostics(unittest.TestCase):
    def test_external_edge_is_not_foot_crossing(self):
        h=np.full((5,11),-.035);h[:,-1]-=.12
        status,fraction,_=sole_diagnostic(h,150.)
        self.assertIn('脚外周圈',status);self.assertEqual(fraction,1.)

    def test_missing_rays_are_not_proof_of_edge(self):
        h=np.full((5,11),-.035);h[2,4]=np.inf
        status,_,missing=sole_diagnostic(h,150.)
        self.assertIn('无法确认',status);self.assertEqual(missing,1)

    def test_crossing_requires_load_and_interior_step(self):
        h=np.full((5,11),-.035);h[:,5:]-=.12
        self.assertIn('疑似跨边承重',sole_diagnostic(h,150.)[0])
        self.assertIn('不判踩边',sole_diagnostic(h,10.)[0])

    def test_tilted_contact_is_not_automatically_an_edge(self):
        h=np.tile(np.linspace(-.07,.02,11),(5,1))
        self.assertIn('未检出跨边',sole_diagnostic(h,150.)[0])

if __name__=='__main__':unittest.main()
