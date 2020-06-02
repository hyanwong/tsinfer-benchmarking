import multiprocessing
import argparse
import re
import sys


import tskit
import numpy as np

def run(filename):
    m = re.search(r'^(.*)_ma([-e\d\.]+)_ms([-e\d\.]+)_p(\d+).trees$', filename)
    if m is None:
        return None
    prefix = m.group(1)
    ma_mut = float(m.group(2))
    ms_mut = float(m.group(3))
    precision = int(m.group(4))
    ts = tskit.load(filename)
    sum = 0
    sum_sq = 0
    tot = 0

    for tree in ts.trees():
        for n in tree.nodes():
            sum += tree.num_children(n) * tree.span
            sum_sq += (tree.num_children(n) ** 2) * tree.span
            tot += tree.span
    mean = sum/tot
    var = sum_sq / tot - (mean ** 2) # can't be bothered to adjust for n
    return dict(
        prefix=prefix,
        ma_mut=ma_mut,
        ms_mut=ms_mut,
        precision=precision,
        mean=mean,
        var=var)
    
def process_files(filenames):
    prefix = set()
    ma_mut = []
    ms_mut = []
    precision = set()
    mean = []
    var = []
    with multiprocessing.Pool(40) as pool:
        for result in pool.imap_unordered(run, filenames):
            if result is not None:
                prefix.add(result['prefix'])
                ma_mut.append(result['ma_mut'])
                ms_mut.append(result['ms_mut'])
                precision.add(result['precision'])
                mean.append(result['mean'])
                var.append(result['var'] ** (1/2))
    if len(prefix) != 1:
        raise ValueError("You must pass in files with all the same prefix")
    else:
        prefix = prefix.pop()
    if len(precision) != 1:
        print("Multiple precisions: not necessarily a problem")    
    with open(prefix + ".num_poly", "wt") as file:
        print(
            "\t".join(['ma_mut', 'ms_mut', 'mean_num_poly', 'sd_num_poly']),
            file=file)
        for row in zip(ma_mut, ms_mut, mean, var):
            print("\t".join(str(r) for r in row), file=file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("args", nargs='+')
    args = p.parse_args()
    process_files(args.args)
