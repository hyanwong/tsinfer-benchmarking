"""
Run standard simulations, both vanilla and from stdpopsim
"""
V_0_1_4 = "efbafff"  # commit hash of tsinfer v 0.1.4


import argparse
import sys
import collections
import time
import inspect
import argparse

import tqdm
import numpy as np
import pandas as pd
import msprime
import stdpopsim

import base


Stats = collections.namedtuple('Stats', ['time', 'num_edges', 'kc'])


def stat_compare(ts, tsinfer_module, use_position=False, precision=None):
    """
    Return tuple of time, #edges, kc
    """
    try:
        data = tsinfer_module.SampleData.from_tree_sequence(ts, use_times=False)
    except TypeError:
        data = tsinfer_module.SampleData.from_tree_sequence(ts)
    start = time.time()
    args = {'sample_data': data}
    if precision is not None:
        args['precision'] = precision
    if use_position:
        r = np.diff(data.sites_position[:][data.sites_inference[:] == 1])
        position_diffs = np.concatenate((
            [0.0],
            np.diff(data.sites_position[:][data.sites_inference[:] == 1])))
        args['recombination_rate'] = position_diffs / data.sequence_length
        print(np.min(r), np.max(r), np.mean(r), np.std(r))
    try:
        inferred = tsinfer_module.infer(**args).simplify()
    except (TypeError, UnboundLocalError):
        # Either rho not defined, or infer does not take recombination_rate as a param
        inferred = tsinfer_module.infer(data).simplify()
    end = time.time()
    return Stats(
        time=end - start,
        num_edges=inferred.num_edges,
        kc=base.ts_kc(ts, inferred))





if __name__ == "__main__":
    top_parser = argparse.ArgumentParser(description=__doc__)
    top_parser.add_argument(
        "-e", "--engine", default=tsinfer.C_ENGINE,
        help="The implementation to use.")

    subparsers = top_parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    parser = subparsers.add_parser(
        "test-commit", aliases=["tc"],
        help="Test one commit against another, and report time, #edges, and KC dist.")
    
    parser.add_argument("commits", nargs='+', help=(
        "A commit hash for tsinfer, or ''. Note that you can also specify 'master' "
        "here to get the most recent version. If blank (''), the local repo is saved in "
        "'_versions/tsinfer_', and the commit hash is NOT checked, which means you can "
        "use this repo for git bisections."))
    parser.add_argument("-g", "--genetic_distance", action="store_true",
        help="Should we allow genomic distances")
    parser.add_argument("-P", "--progress", action="store_true",
        help="Show a progress monitor.")
    parser.add_argument("-p", "--precision", type=int,
        help="Set a precision.")
    args = parser.parse_args()
    
    reps = 100
    for new_commit in args.commits:
        commits = [V_0_1_4, new_commit]
        commit_names = {}  # convert e.g. "master" into a hash
        data = collections.defaultdict(list)
        for commit in commits:
            tsinfer_module, commithash = base.import_tsinfer(commit)
            commit_names[commit] = commithash
            for seed in tqdm.trange(1, reps+1, desc=commit, disable=not args.progress):
                ts = msprime.simulate(
                    30, recombination_rate=10, mutation_rate=10, random_seed=seed)
                stats = stat_compare(
                    ts, tsinfer_module, use_position=args.genetic_distance,
                    precision=args.precision)
                data[commit].append(stats)
        diffs = []
        for d1, d2 in zip(data[commits[0]], data[commits[1]]):
            diffs.append(Stats(*[(a / b) for a, b in zip(d1, d2)]))
        df_diff = pd.DataFrame(diffs, columns = Stats._fields) * 100
        print("== {} vs {} {} positional info {} ==".format(
            commit_names[commits[0]],
            commit_names[commits[1]],
            ("with" if args.genetic_distance else "without"),
            ("(precision={})".format(args.precision) if args.precision else "")))
        print(pd.DataFrame.from_dict({
            "percent_improvement": df_diff.mean(axis=0), "stderr": df_diff.sem(axis=0)
            }))
