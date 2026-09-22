"""Select a recently saved formal 17-step policy; exclude smoke/evaluation runs."""
from pathlib import Path
import argparse

TAGS=('stairs17_crossing_from42700','instinct_edges_toe_v8','safe_landing_v9',
      'turning_rough_v10','reverse_height_v11','foothold_v12')


def latest_checkpoint(root=None,foothold=False):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]/'logs/rsl_rl/g1_amp_stairs_long'
    tags=('foothold_v12',) if foothold else TAGS
    runs=[p for p in root.iterdir() if p.is_dir() and 'smoke' not in p.name
          and any(tag in p.name for tag in tags)]
    candidates=[p for run in runs for p in run.glob('model_*.pt') if p.stem.split('_')[-1].isdigit()]
    if not candidates and foothold:
        fallback=root/'2026-09-16_20-26-52_reverse_height_v11/model_55000.pt'
        if fallback.is_file():return fallback.resolve()
    if not candidates:raise FileNotFoundError('No formal 17-step checkpoint found')
    # A new reward branch can start below the old branch's final iteration.
    return max(candidates,key=lambda p:(p.stat().st_mtime_ns,int(p.stem.split('_')[-1]))).resolve()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--foothold',action='store_true',help='Latest v12, or its evaluated v11 model_55000 baseline')
    args=p.parse_args()
    print(latest_checkpoint(foothold=args.foothold))
