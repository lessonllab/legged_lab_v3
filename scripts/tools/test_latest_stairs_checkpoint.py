"""Checkpoint selection regression: a newer branch can have a smaller iteration."""
import os
import tempfile
import unittest
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('latest',Path(__file__).resolve().parents[1]/'latest_stairs_checkpoint.py')
latest=importlib.util.module_from_spec(spec);spec.loader.exec_module(latest)

class Selection(unittest.TestCase):
    def test_branch_and_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,iteration,stamp in [('reverse_height_v11',58999,100),('foothold_v12',55100,200),('foothold_v12_smoke',99999,300)]:
                path=root/name/f'model_{iteration}.pt';path.parent.mkdir();path.touch();os.utime(path,(stamp,stamp))
            self.assertEqual(latest.latest_checkpoint(root).parent.name,'foothold_v12')
            self.assertEqual(latest.latest_checkpoint(root,foothold=True).stem,'model_55100')

if __name__=='__main__':unittest.main()
