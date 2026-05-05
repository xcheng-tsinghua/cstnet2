"""
Batch convert dataset text files to .npy for faster loading.

Usage:
    python data_utils/convert_txt_to_npy.py --root data/pcd_cstnet2/Param20K_Extend
"""
import argparse
import os
import time
import numpy as np


def iter_txt_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith('.txt'):
                yield os.path.join(dirpath, name)


def txt_to_npy_path(txt_path):
    return txt_path + '.npy'


def convert_one(txt_path, overwrite=False):
    npy_path = txt_to_npy_path(txt_path)
    if (not overwrite) and os.path.exists(npy_path):
        return False

    arr = np.loadtxt(txt_path)
    tmp_path = f'{npy_path}.tmp.{os.getpid()}.npy'
    np.save(tmp_path, arr)
    os.replace(tmp_path, npy_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=str, help='Dataset root directory')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild existing npy files')
    args = parser.parse_args()

    start = time.time()
    total = 0
    converted = 0

    for txt_path in iter_txt_files(args.root):
        total += 1
        if convert_one(txt_path, overwrite=args.overwrite):
            converted += 1
        if total % 200 == 0:
            print(f'processed {total} files, converted {converted}')

    elapsed = time.time() - start
    print(f'done. total={total}, converted={converted}, elapsed={elapsed:.2f}s')


if __name__ == '__main__':
    main()
