"""
Used to take a subsample of a sampledata file and check the number of inference sites
plus sites with missing data
"""

import argparse

import numpy as np
import tsinfer
import tskit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", default=None,
        help="A tsinfer sample file ending in '.samples")
    parser.add_argument("-o", "--output_file", default=None,
        help="An output ancestors file")
    parser.add_argument("-t", "--num_threads", type=int, default=0,
        help="The number of threads to use in inference")
    parser.add_argument("-e", "--engine", default='P',
        help="'C' or 'P' for the c engine or python engine")
    args = parser.parse_args()

    if args.engine == "C":
        engine = tsinfer.C_ENGINE
    elif args.engine == "P":
        engine = tsinfer.PY_ENGINE
    else:
        raise ValueError

    sd = tsinfer.load(args.input_file)

    anc = tsinfer.generate_ancestors(
        sd,
        path=args.output_file,
        engine=engine,
        num_threads=args.num_threads,
        progress_monitor=tsinfer.cli.ProgressMonitor(
            enabled=True, generate_ancestors=True),
    )
    
    full_len = np.logical_and(
        anc.ancestors_start[:][2:]==0, anc.ancestors_end[:][2:]==anc.num_sites)
    
    u, cnts = np.unique(anc.ancestors_time[:][2:][full_len], return_counts=True)
    
    print("{}/{} full length ancestors at {} unique times ({} single)".format(
        np.sum(full_len),
        len(full_len)
        len(u),
        np.sum(cnts==1),
        ))