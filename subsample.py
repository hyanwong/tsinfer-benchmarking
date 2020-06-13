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
    parser.add_argument("output_file", default=None,
        help="A tsinfer sample file ending in '.samples")
    # The _mrate parameter defaults set from analysis ot 1000G, see
    # https://github.com/tskit-dev/tsinfer/issues/263#issuecomment-639060101
    parser.add_argument("-n", "--num_samples", type=int, default=None,
        help="The number of samples to randomly pick from the input file")
    parser.add_argument("-p", "--percent_of_genome", type=float, default=10,
        help="The percent of the genome to include")
    parser.add_argument("-s", "--genome_start_percent", type=int, default=0,
        help="The genomic point at which to start the subsample, as a percentage of the"
            " total genome length")
    args = parser.parse_args()

    sd = tsinfer.load(args.input_file)
    num_samples = sd.num_samples if args.num_samples is None else args.num_samples
    assert num_samples <= sd.num_samples
    assert 0 < args.percent_of_genome <= 100
    assert args.percent_of_genome + args.genome_start_percent <= 100
    
    del_samples = np.random.choice(
        sd.num_samples, sd.num_samples-num_samples, replace=False)
    
    del_sites = np.ones(sd.num_sites, dtype=bool)
    start_keep = int(args.genome_start_percent/100.0 * sd.num_sites)
    end_keep = start_keep + int(args.percent_of_genome/100.0 * sd.num_sites)
    del_sites[np.arange(start_keep, end_keep)] = False
    
    
    small_sd = sd.delete(samples=del_samples, sites=del_sites, path=args.output_file)
    
    sites_inference = small_sd.sites_inference[:]
    inference_sites = np.sum(sites_inference)
    print(f"File now has {small_sd.num_samples} samples, {small_sd.num_sites} sites")
    missing_data_sites = 0
    missing_empty_sites = 0
    missing_singleton_sites = 0
    non_missing_inference_sites = 0
    for i, g in small_sd.genotypes():
        if np.any(g == tskit.NULL):
            missing_data_sites += 1
            s = sum(g[g != tskit.NULL])
            if s==0:
                assert sites_inference[i] == False
                missing_empty_sites += 1
            elif s==1:
                assert sites_inference[i] == False
                missing_singleton_sites += 1
        else:
            if sites_inference[i] == True:
                non_missing_inference_sites += 1
    print(
        f"{inference_sites} inference sites " +
        f"({non_missing_inference_sites} complete, with no mising data)")
    print(
        f"{missing_data_sites} sites with missing data " +
        f"of which {missing_empty_sites} empty and {missing_singleton_sites} singletons")
    