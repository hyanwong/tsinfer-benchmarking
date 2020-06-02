import multiprocessing
import argparse
import re
import sys
from tempfile import NamedTemporaryFile


import tskit
import numpy as np

import ARG_metrics
import ts_extras

def run(filename):
    m = re.search(r'^(.*)_ma([-e\d\.]+)_ms([-e\d\.]+)_p(\d+).trees$', filename)
    if m is None:
        return None
    prefix = m.group(1)
    ma_mut = float(m.group(2))
    ms_mut = float(m.group(3))
    precision = int(m.group(4))
    orig_ts = tskit.load(prefix + ".trees")
    inferred_ts = tskit.load(filename)

    with NamedTemporaryFile("wt") as nex1, NamedTemporaryFile("wt") as nex2:
        ts_extras.write_nexus_trees(orig_ts, nex1)
        nex1.flush()
        ts_extras.write_nexus_trees(inferred_ts.simplify(), nex2)
        nex2.flush()
        metrics = ARG_metrics.get_metrics(
            nex1.name, [nex2.name], randomly_resolve_inferred=True)

    return dict(
        prefix=prefix,
        ma_mut=ma_mut,
        ms_mut=ms_mut,
        precision=precision,
        metrics=metrics)
    
def process_files(filenames):
    prefix = set()
    ma_mut = []
    ms_mut = []
    precision = set()
    metrics = []
    metric_names = ARG_metrics.get_metric_names()
    with multiprocessing.Pool(40) as pool:
        for result in pool.imap_unordered(run, filenames):
            if result is not None:
                prefix.add(result['prefix'])
                ma_mut.append(result['ma_mut'])
                ms_mut.append(result['ms_mut'])
                precision.add(result['precision'])
                metrics.append(result['metrics'])
    if len(prefix) != 1:
        raise ValueError("You must pass in files with all the same prefix")
    else:
        prefix = prefix.pop()
    if len(precision) != 1:
        print("Multiple precisions: not necessarily a problem")    
    with open(prefix + ".kc_poly", "wt") as file:
        print(
            "\t".join(['ma_mut', 'ms_mut'] + metric_names),
            file=file)
        for r1, r2, m in zip(ma_mut, ms_mut, metrics):
            row = [r1, r2] + [m[k] for k in metric_names]
            print("\t".join(str(r) for r in row), file=file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("args", nargs='+')
    args = p.parse_args()
    process_files(args.args)
