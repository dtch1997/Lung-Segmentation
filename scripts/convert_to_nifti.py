import pylidc as pl
import nibabel as nb
import numpy as np
import argparse
import pathlib
import warnings

from pylidc.utils import consensus

parser = argparse.ArgumentParser(description="Convert LIDC images to NIFTI1 format")
parser.add_argument('--savedir', help="Directory in which to save NIFTI1 files")
parser.add_argument('--overwrite', help="If True, overwrites existing files. Default: False", action='store_true')
parser.add_argument('--debug', help="If True, only converts one file for debugging purpose", action='store_true')

def numpy_to_nifti(array, path):
    image = nb.Nifti1Image(array, np.eye(4))
    image.to_filename(path)

def main():
    args = parser.parse_args()
    path = pathlib.Path(args.savedir)
    if args.debug:
        path = path / 'debug'
    print(f"Using {str(path)} as save directory")
    if path.exists() and path.is_dir():
        warnings.warn(f"Directory {str(path)} already exists.")
        if args.overwrite:
            print("Overwrite has been set. Continuing...")
        else:
            print("Terminating execution.")
            return
    else:
        path.mkdir(parents=True, exist_ok=True)

    if args.debug:
        scans = [pl.query(pl.Scan).first()]
    else:
        scans = pl.query(pl.Scan).all()
    for scan in scans:
        print(f"Converting patient {scan.patient_id}")
        vol = scan.to_volume() # (numpy array)
        mask = np.zeros(vol.shape, dtype=bool)
        nodules = scan.cluster_annotations()
        for nod in nodules:
            # Pad so that cmask is the whole volume
            cmask,_,_ = consensus(nod, clevel=0.5, pad=[(vol.shape[i],vol.shape[i]) for i in range(3)])
            mask = np.logical_or(mask, cmask)
        numpy_to_nifti(vol, path / f"{scan.patient_id}_volume.nii.gz")
        numpy_to_nifti(vol, path / f"{scan.patient_id}_segmask.nii.gz")

if __name__ == "__main__":
    main()
