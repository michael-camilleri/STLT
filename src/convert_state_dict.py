import torch
import argparse
from collections import OrderedDict

BASE_PATH = '${HOME}/Documents/DataSynced/PhD Project/Data/MRC Harwell/Models/STLT/Trained'

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(
        description="Convert Old-Style State-Dict to New Style"
    )
    arg_parser.add_argument(
        "-i",
        "--input",
        help=f"The input PTH file (relative to Base Path: {BASE_PATH})",
        required=True
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        help=f"Name for the output PTH file (relative to {BASE_PATH})",
        required=True
    )
    args = arg_parser.parse_args()

    in_pth = torch.load(args.input)
    out_pth = OrderedDict()
    for k, v in in_pth.items():
        k_new = k[5:] if k.startswith('stlt_backbone') else k
        out_pth[k_new] = v
    torch.save(out_pth, args.output)
