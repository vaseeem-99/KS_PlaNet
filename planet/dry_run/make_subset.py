"""
make_subset.py
==============
Place at: planet/dry_run/make_subset.py

Creates tiny subsets of the KS and Kolmogorov datasets for the dry-run.
Defaults to 5 trajectories / pairs per split.

Run from planet/:
    python dry_run/make_subset.py

Or override sizes via env vars (used by docker_dry_run.sh):
    KS_TRAIN_N=5 KS_TEST_N=5 KO_TRAIN_N=5 ... python dry_run/make_subset.py
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'ks_dataset')

# ── Subset sizes (override via env vars or edit directly) ───────────────────
KS_TRAIN_N = int(os.environ.get('KS_TRAIN_N', 5))
KS_TEST_N  = int(os.environ.get('KS_TEST_N',  5))
KO_TRAIN_N = int(os.environ.get('KO_TRAIN_N', 5))
KO_VAL_N   = int(os.environ.get('KO_VAL_N',   5))
KO_TEST_N  = int(os.environ.get('KO_TEST_N',  5))
# ────────────────────────────────────────────────────────────────────────────


def subset_npz(in_path, out_path, n):
    if not os.path.exists(in_path):
        print(f'  SKIP (not found): {in_path}')
        return
    data = np.load(in_path, mmap_mode='r')
    sliced = {key: np.asarray(data[key][:n]) for key in data.files
              if hasattr(data[key], 'shape') and data[key].shape}
    for key in data.files:
        if key not in sliced:
            sliced[key] = np.asarray(data[key])
    np.savez(out_path, **sliced)
    sz = os.path.getsize(out_path) / 1e6
    print(f'  -> {os.path.basename(out_path):28s}  ({sz:.2f} MB)')


def subset_npy(in_path, out_path, n):
    if not os.path.exists(in_path):
        print(f'  SKIP (not found): {in_path}')
        return
    arr = np.load(in_path, mmap_mode='r')
    np.save(out_path, np.asarray(arr[:n]))
    sz = os.path.getsize(out_path) / 1e6
    print(f'  -> {os.path.basename(out_path):28s}  shape={arr[:n].shape}  ({sz:.2f} MB)')


def main():
    print(f'Creating subsets in {DATA_DIR}')
    print('=' * 60)

    print(f'\nKS  ({KS_TRAIN_N} train, {KS_TEST_N} test):')
    subset_npz(os.path.join(DATA_DIR, 'ks_train.npz'),
               os.path.join(DATA_DIR, 'ks_train_subset.npz'), KS_TRAIN_N)
    subset_npz(os.path.join(DATA_DIR, 'ks_test.npz'),
               os.path.join(DATA_DIR, 'ks_test_subset.npz'), KS_TEST_N)

    print(f'\nKolmogorov  ({KO_TRAIN_N} train, {KO_VAL_N} val, {KO_TEST_N} test):')
    for fname, n in [
        ('X_train.npy', KO_TRAIN_N), ('Y_train.npy', KO_TRAIN_N),
        ('X_val.npy',   KO_VAL_N),   ('Y_val.npy',   KO_VAL_N),
        ('X_test.npy',  KO_TEST_N),  ('Y_test.npy',  KO_TEST_N),
    ]:
        in_path  = os.path.join(DATA_DIR, fname)
        out_path = os.path.join(DATA_DIR, fname.replace('.npy', '_subset.npy'))
        subset_npy(in_path, out_path, n)

    print('\nDone.')


if __name__ == '__main__':
    main()
