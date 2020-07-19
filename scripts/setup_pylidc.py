import os
import argparse
import pathlib

parser = argparse.ArgumentParser(description="Write the config file for PYLIDC")
parser.add_argument("datadir", help="Path to LIDC data, ending in XXX/LIDC-IDRI")

def main():
    args = parser.parse_args()
    user = os.environ["USER"]
    pylidcrc_path = pathlib.Path("/home") / user / ".pylidcrc"

    with open(str(pylidcrc_path), 'w') as pylfile:
        pylfile.write("[dicom]\n")
        pylfile.write(f"path = {args.datadir}\n")
        pylfile.write("warn = True")

if __name__ == "__main__":
    main()
