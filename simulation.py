"""
Run standard simulations, both vanilla and from stdpopsim

Compare: # edges, % compression, kc_distance

"""
V_0_1_4 = "efbafff"  # commit hash of tsinfer v 0.1.4


import argparse
import sys
import collections
import time
import inspect

import tqdm
import numpy as np
import pandas as pd
import msprime
import stdpopsim

import base

Stats = collections.namedtuple('Stats', ['time', 'num_edges', 'kc'])

def stat_compare(ts, tsinfer_module, use_position=False):
    """
    Return tuple of time, #edges, kc
    """
    try:
        data = tsinfer_module.SampleData.from_tree_sequence(ts, use_times=False)
    except TypeError:
        data = tsinfer_module.SampleData.from_tree_sequence(ts)
    start = time.time()

    if use_position:
        position_diffs = np.concatenate((
            [0.0],
            np.diff(data.sites_position[:][data.sites_inference[:] == 1])))
        rho = position_diffs / data.sequence_length
    try:
        inferred = tsinfer_module.infer(data, recombination_rate=rho).simplify()
    except (TypeError, UnboundLocalError):
        # Either rho not defined, or infer does not take recombination_rate as a param
        inferred = tsinfer_module.infer(data).simplify()

    end = time.time()
    return Stats(
        time=end - start,
        num_edges=inferred.num_edges,
        kc=base.ts_kc(ts, inferred))
    
    
if __name__ == "__main__":
    reps = 60
    commits = [V_0_1_4, "03ad4bd"]
    data = collections.defaultdict(list)
    for commit in commits:
        tsinfer_module = base.import_tsinfer(commit)
        for seed in tqdm.trange(1, reps, desc = commit):
            ts = msprime.simulate(
                30, recombination_rate=10, mutation_rate=10, random_seed=seed)
            stats = stat_compare(ts, tsinfer_module, use_position=False)
            data[commit].append(stats)
    diffs = []
    for d1, d2 in zip(data[commits[0]], data[commits[1]]):
        diffs.append(Stats(*[(a - b) for a, b in zip(d1, d2)]))

    df = pd.DataFrame(diffs, columns = Stats._fields)
    print("== {}-{} diffs ==".format(*commits))
    print(pd.DataFrame.from_dict({"mean": df.mean(axis=0), "stderr":df.sem(axis=0)}))