import os.path
import argparse
import collections
import re

import tskit
import numpy as np
import tsinfer
import stdpopsim

def run(params):
    """
    Run a single inference, with the specified rates
    """
    base_rec_prob = np.mean(params.rec_rate[1:])
    print("Starting {} {} with mean rho {}".format(
        params.ma_mut_rate, params.ms_mut_rate, base_rec_prob))
    prefix = None
    if params.sample_data.path is not None:
        assert params.sample_data.path.endswith(".samples")
        prefix = params.sample_data.path[0:-len(".samples")]
        inf_prefix = "{}_ma{}_ms{}_p{}".format(
            prefix,
            params.ma_mut_rate,
            params.ms_mut_rate,
            params.precision)
    anc = tsinfer.generate_ancestors(
        params.sample_data,
        num_threads=params.n_threads,
        path=None if inf_prefix is None else inf_prefix + ".ancestors",
        progress_monitor=tsinfer.cli.ProgressMonitor(1, 1, 0, 0, 0),
    )
    print(f"GA done (ma_mut:{params.ma_mut_rate} ms_mut{params.ms_mut_rate})")
    inferred_anc_ts = tsinfer.match_ancestors(
        params.sample_data,
        anc,
        num_threads=params.n_threads,
        precision=params.precision,
        recombination_rate=params.rec_rate,
        mutation_rate=base_rec_prob * params.ma_mut_rate,
        progress_monitor=tsinfer.cli.ProgressMonitor(1, 0, 1, 0, 0),
        )

    inferred_anc_ts.dump(path=inf_prefix + ".atrees")
    print(f"MA done (ma_mut:{params.ma_mut_rate} ms_mut{params.ms_mut_rate})")
    inferred_ts = tsinfer.match_samples(
        params.sample_data,
        inferred_anc_ts,
        num_threads=params.n_threads,
        precision=params.precision,
        recombination_rate=params.rec_rate,
        mutation_rate=base_rec_prob * params.ms_mut_rate,
        progress_monitor=tsinfer.cli.ProgressMonitor(1, 0, 0, 0, 1),
        )
    ts_path = inf_prefix + ".trees"
    inferred_ts.dump(path=ts_path)
    print(f"MS done (ma_mut:{params.ma_mut_rate} ms_mut{params.ms_mut_rate})")
    simplified_inferred_ts = inferred_ts.simplify()  # Remove unary nodes
    # Calculate mean num children (polytomy-measure) for internal nodes
    nc_sum = 0
    nc_sum_sq = 0
    nc_tot = 0
    for tree in simplified_inferred_ts.trees():
        for n in tree.nodes():
            n_children = tree.num_children(n)
            if n_children > 0:  # exclude leaves/samples
                nc_sum +=  n_children * tree.span
                nc_sum_sq += (n_children ** 2) * tree.span
                nc_tot += tree.span
    nc_mean = nc_sum/nc_tot
    nc_var = nc_sum_sq / nc_tot - (nc_mean ** 2) # can't be bothered to adjust for n

    return Results(
        ts_size=os.path.getsize(ts_path),
        ma_mut=params.ma_mut_rate,
        ms_mut=params.ms_mut_rate,
        edges=inferred_ts.num_edges,
        muts=inferred_ts.num_mutations,
        mean_node_children=nc_mean,
        var_node_children=nc_var,
        ts_path=ts_path)


Params = collections.namedtuple(
    "Params",
    "sample_data, rec_rate, ma_mut_rate, ms_mut_rate, precision, n_threads")

Results = collections.namedtuple(
    "Results",
    "ts_size, ma_mut, ms_mut, edges, muts, "
    "mean_node_children, var_node_children, ts_path")


def physical_to_genetic(recombination_map, input_physical_positions):
    map_pos = recombination_map.get_positions()
    map_rates = recombination_map.get_rates()
    map_genetic_positions = np.insert(np.cumsum(np.diff(map_pos) * map_rates[:-1]), 0, 0)
    return np.interp(input_physical_positions, map_pos, map_genetic_positions)


def setup_sample_file(filename):
    """
    Return a Thousand Genomes Project sample data file, the
    corresponding recombination rate array, a prefix to use for files, and None
    """
    if not filename.endswith(".samples"):
        raise ValueError("Sample data file must end with '.samples'")
    sd = tsinfer.load(filename)
    match = re.search(r'(chr\d+)', filename)
    if match:
        chr = match.group(1)
        print(f"Using {chr} from HapMapII_GRCh37 for the recombination map")
        map = stdpopsim.get_species("HomSap").get_genetic_map(id="HapMapII_GRCh37")
        if not map.is_cached():
            map.download()
        chr_map = map.get_chromosome_map(chr)
        inference_distances = physical_to_genetic(
            chr_map,
            sd.sites_position[:][sd.sites_inference])
        rho = np.concatenate(([0.0], np.diff(inference_distances)))
    else:
        inference_distances = sd.sites_position[:][sd.sites_inference]
        rho = np.concatenate(
            ([0.0], np.diff(inference_distances)/sd.sequence_length))
        
    return sd, rho, filename[:-len(".samples")], None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_file", default=None,
        help="A tsinfer sample file ending in '.samples'. If the filename contains chrNN"
            " where 'NN' is a number, assume this is a human samples file and use the"
            " appropriate recombination map from the thousand genomes project, otherwise"
            " use the physical distance between sites.")
    # The _mrate parameter defaults set from analysis ot 1000G, see
    # https://github.com/tskit-dev/tsinfer/issues/263#issuecomment-639060101
    parser.add_argument("-A", "--match_ancestors_mrate", type=float, default=1e-1,
        help="The recurrent mutation probability in the match ancestors phase,"
            " as a fraction of the average recombination probability between sites")
    parser.add_argument("-S", "--match_samples_mrate", type=float, default=1e-2,
        help="The recurrent mutation probability in the match samples phase,"
            " as a fraction of the average recombination probability between sites")
    parser.add_argument("-p", "--precision", type=int, default=None,
        help="The precision, as a number of decimal places, which will affect the speed"
            " of the matching algorithm (higher precision: lower speed). If None,"
            " calculate the smallest of the recombination rates or mutation rates, and"
            " use the negative exponent of that number plus four. E.g. if the smallest"
            " recombination rate is 2.5e-6, use precision = 6+4 = 10"
        )
    parser.add_argument("-t", "--num_threads", type=int, default=0,
        help="The number of threads to use in inference")
    args = parser.parse_args()
    

    samples, rho, prefix, ts = setup_sample_file(args.sample_file)
    if args.precision is None:
        precision = int(np.ceil(
            -min(
                np.min(np.log10(rho[1:])),
                np.log10(args.match_ancestors_mrate),
                np.log10(args.match_samples_mrate))))
    else:
        precision = args.precision

    params = Params(
        samples,
        rho,
        args.match_ancestors_mrate,
        args.match_samples_mrate,
        precision,
        args.num_threads)
    print(f"Running inference with {params}")
    with open(prefix + ".results", "wt") as file:
        result = run(params)
        print("\t".join(str(r) for r in result), file=file, flush=True)
