"""
Use Dendropy to calculate the unweighted (i.e. topology only) Robinson Fould distance
between 2 tree sequences. Can be run in parallel along the lines of:

for f in data/mysim.*_rma*.trees; do python3 RFcalc.py data/mysim.trees $f -s 123 -v -o 100 & done

and RF distances can be extracted using something like

for f in data/mysim.*_rma*.trees.*RFdist; do echo -n $f | cat - $f | sed -r 's/.*rma(.*)_rms(.*)_p.*RFdist(.*)/\1\t\2\t\3/' ; done


"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # limit number of threads so we can run loads of these in parallel

import tempfile
import logging
import argparse
import itertools
import time

import tskit
import dendropy
import numpy as np

def main(original_ts, inferred_ts, metric, random_seed, output_tot = 1):
    if random_seed is not None:
        orig_ts = tskit.load(original_ts).simplify().randomly_split_polytomies(
            random_seed=random_seed)
        cmp_ts = tskit.load(inferred_ts).simplify().randomly_split_polytomies(
            random_seed=random_seed)
    else:
        orig_ts = tskit.load(original_ts).simplify()
        cmp_ts = tskit.load(inferred_ts).simplify()

    logging.info("Loaded initial tree sequences")
    assert orig_ts.sequence_length == cmp_ts.sequence_length
    seq_length = orig_ts.sequence_length

    if random_seed is None:
        suffix = "." + metric
    else:
        suffix = ".split." + metric


    if metric == "KC":
        kc = orig_ts.kc_distance(cmp_ts)
        with open(inferred_ts + suffix, "wt") as stat:
            print(kc, file=stat)
        logging.info(f"Saved data for '{inferred_ts}': KCdist = {kc}")

    elif metric == "RF":
        t_iter1 = orig_ts.trees()
        t_iter2 = cmp_ts.trees()
        rf_stat = 0
        pos = 0
        end1 = 0
        end2 = 0
        
        start = time.time()
        taxon_namespace = dendropy.Tree.get(
            string=orig_ts.first().newick(precision=0),
            schema="newick",
            rooting="force-rooted").taxon_namespace
        logging.info(
            f"Loaded 1 out of {orig_ts.num_trees} trees in {time.time()-start} sec")
    
        
        while True:
            if pos == seq_length:
                break
            if pos >= end1:
                t1 = next(t_iter1)
                end1 = t1.interval[1]
                if pos > 0 and (t1.index % (orig_ts.num_trees // output_tot)) == 0:
                    logging.debug("For {}, {:.0f}% done, RF ~= {}"
                        .format(
                            inferred_ts,
                            t1.index / orig_ts.num_trees * 100,
                            rf_stat/pos,
                        )
                    )
                    # save temporarily, so we can get stats even if not completed
                    with open(inferred_ts + suffix, "wt") as stat:
                        print(rf_stat/pos, file=stat)
            if pos >= end2:
                t2 = next(t_iter2)
                end2 = t2.interval[1]
    
            span = min(end1, end2) - pos
            orig_tree = dendropy.Tree.get(
                string=t1.newick(precision=0),
                schema="newick",
                rooting="force-rooted",
                taxon_namespace=taxon_namespace,
            )
            cmp_tree = dendropy.Tree.get(
                string=t2.newick(precision=0),
                schema="newick",
                rooting="force-rooted",
                taxon_namespace=taxon_namespace,
            )
            rf_stat += dendropy.calculate.treecompare.symmetric_difference(
                orig_tree, cmp_tree) * span
            pos = min(end1, end2)
    
        with open(inferred_ts + suffix, "wt") as stat:
            print(rf_stat / seq_length, file=stat)

        logging.info(f"Saved data for '{inferred_ts}': RFdist = {rf_stat / seq_length}")

    else:
        raise ValueError(f"Bad metric specified: {metric}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate the rooted Robinson Foulds distance between 2 tree seqs')
    parser.add_argument('orig_ts')
    parser.add_argument(
        'cmp_ts',
        help=(
            "the ts to compare against the original."
            " The stat will be saved under this name with a suffix '.RFdist'"))
    parser.add_argument(
        '--random_seed',
        '-s',
        type=int,
        default=None,
        help="If given, randomly split polytomies before calculating the RF dist",
    )
    parser.add_argument(
        '--output_tot',
        '-o',
        type=int,
        default=1,
        help=(
            "How many times to overwrite the output file during progress, to allow "
            "partially calculated stats to be used. Also determines the output progress "
            "if verbosity is >= 2."
            )
    )
    parser.add_argument('--metric', '-m', choices=["KC", "RF"], default="RF", 
        help='verbosity: output extra non-essential info')
    parser.add_argument('--verbosity', '-v', action="count", default=0, 
        help='verbosity: output extra non-essential info')
    
    args = parser.parse_args()
    if args.verbosity==0:
        logging.basicConfig(level=logging.WARNING)
    elif args.verbosity==1:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    elif args.verbosity>=2:
        logging.basicConfig(level=logging.DEBUG)

    main(args.orig_ts, args.cmp_ts, args.metric, args.random_seed, args.output_tot)
