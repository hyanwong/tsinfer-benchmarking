import os.path
import argparse
import collections
import re
import time

import tskit
import msprime
import numpy as np
import tsinfer
import stdpopsim

Params = collections.namedtuple(
    "Params",
    "sample_data, rec_rate, ma_mis_rate, ms_mis_rate, cutoff_exponent, precision, "
    "num_threads")

Results = collections.namedtuple(
    "Results",
    "abs_ma_mis, abs_ms_mis, rel_ma_mis, rel_ms_mis, cutoff_exponent, "
    "precision, edges, muts, num_trees, "
    "kc, mean_node_children, var_node_children, process_time, ts_size, ts_path")

def run(params):
    """
    Run a single inference, with the specified rates
    """
    rho = params.rec_rate
    av_rho = np.quantile(rho, 0.5)
    ma_mis = av_rho * params.ma_mis_rate
    ms_mis = av_rho * params.ms_mis_rate

    if params.precision is None:
        # Smallest nonzero recombination rate
        min_rho = int(np.ceil(-np.min(np.log10(rho[rho > 0]))))
        # Smallest mean
        av_min = int(np.ceil(-np.log10(min(ma_mis, ms_mis))))
        precision = max(min_rho, av_min) + 3
    else:
        precision = params.precision

    print(
        f"Starting {params.ma_mis_rate} {params.ms_mis_rate}",
        f"with av rho {av_rho:.5g}",
        f"(mean {np.mean(rho):.4g}, median {np.quantile(rho, 0.5):.4g}, ",
        f"nonzero min {np.min(rho[rho > 0]):.4g}, ",
        f"2.5% quantile {np.quantile(rho, 0.025):.4g}) precision {precision}")
    prefix = None
    if params.sample_data.path is not None:
        assert params.sample_data.path.endswith(".samples")
        prefix = params.sample_data.path[0:-len(".samples")]
        inf_prefix = "{}_ma{}_ms{}_N{}_p{}".format(
            prefix,
            params.ma_mis_rate,
            params.ms_mis_rate,
            params.cutoff_exponent,
            precision)
    start_time = time.process_time()
    extra_params =  dict(num_threads=params.num_threads)
    if params.cutoff_exponent is not None:
        extra_params['cutoff_power'] = params.cutoff_exponent
    anc = tsinfer.generate_ancestors(
        params.sample_data,
        path=None if inf_prefix is None else inf_prefix + ".ancestors",
        progress_monitor=tsinfer.cli.ProgressMonitor(1, 1, 0, 0, 0),
        **extra_params,
    )
    print(f"GA done (cutoff exponent: {params.cutoff_exponent}")
    extra_params =  dict(
        num_threads=params.num_threads,
        recombination_rate=rho,
        precision=precision,
    )
    inferred_anc_ts = tsinfer.match_ancestors(
        params.sample_data,
        anc,
        mismatch_rate=ma_mis,
        progress_monitor=tsinfer.cli.ProgressMonitor(1, 0, 1, 0, 0),
        **extra_params,
    )
    inferred_anc_ts.dump(path=inf_prefix + ".atrees")
    print(f"MA done (ma_mis:{ma_mis}")
    inferred_ts = tsinfer.match_samples(
        params.sample_data,
        inferred_anc_ts,
        mismatch_rate=ms_mis,
        progress_monitor=tsinfer.cli.ProgressMonitor(1, 0, 0, 0, 1),
        **extra_params,
    )
    process_time = time.process_time() - start_time
    ts_path = inf_prefix + ".trees"
    inferred_ts.dump(path=ts_path)
    print(f"MS done: ms_mis rate = {ms_mis})")
    simplified_inferred_ts = inferred_ts.simplify()  # Remove unary nodes
    # Calculate mean num children (polytomy-measure) for internal nodes
    nc_sum = 0
    nc_sum_sq = 0
    nc_tot = 0
    root_lengths = collections.defaultdict(float)
    for tree in simplified_inferred_ts.trees():
        for n in tree.nodes():
            n_children = tree.num_children(n)
            if n_children > 0:  # exclude leaves/samples
                nc_sum +=  n_children * tree.span
                nc_sum_sq += (n_children ** 2) * tree.span
                nc_tot += tree.span
    nc_mean = nc_sum/nc_tot
    nc_var = nc_sum_sq / nc_tot - (nc_mean ** 2) # can't be bothered to adjust for n

    # Calculate span of root nodes in simplified tree


    # Calculate KC
    try:
        kc = simplified_inferred_ts.kc_distance(tskit.load(prefix+".trees"))
    except FileNotFoundError:
        kc = None
    return Results(
        abs_ma_mis=ma_mis,
        abs_ms_mis=ms_mis,
        rel_ma_mis=params.ma_mis_rate,
        rel_ms_mis=params.ms_mis_rate,
        precision=precision,
        edges=inferred_ts.num_edges,
        muts=inferred_ts.num_mutations,
        num_trees=inferred_ts.num_trees,
        kc=kc,
        cutoff_exponent=params.cutoff_exponent,
        mean_node_children=nc_mean,
        var_node_children=nc_var,
        process_time=process_time,
        ts_size=os.path.getsize(ts_path),
        ts_path=ts_path)


def physical_to_genetic(recombination_map, input_physical_positions):
    map_pos = recombination_map.get_positions()
    map_rates = recombination_map.get_rates()
    map_genetic_positions = np.insert(np.cumsum(np.diff(map_pos) * map_rates[:-1]), 0, 0)
    return np.interp(input_physical_positions, map_pos, map_genetic_positions)


def setup_sample_file(args):
    """
    Return a Thousand Genomes Project sample data file, the
    corresponding recombination rate array, a prefix to use for files, and None
    """
    filename = args.sample_file
    map = args.genetic_map
    if not filename.endswith(".samples"):
        raise ValueError("Sample data file must end with '.samples'")
    sd = tsinfer.load(filename)
    inference_pos = sd.sites_position[:][sd.sites_inference[:]]

    match = re.search(r'(chr\d+)', filename)
    if match or map is not None:
        if map is not None:
            chr_map = msprime.RecombinationMap.read_hapmap(map)
        else:
            chr = match.group(1)
            print(f"Using {chr} from HapMapII_GRCh37 for the recombination map")
            map = stdpopsim.get_species("HomSap").get_genetic_map(id="HapMapII_GRCh37")
            if not map.is_cached():
                map.download()
            chr_map = map.get_chromosome_map(chr)
        inference_distances = physical_to_genetic(chr_map, inference_pos)
        d = np.diff(inference_distances)
        rho = np.concatenate(([0.0], d))
    else:
        inference_distances = inference_pos
        d = np.diff(inference_distances)
        rho = np.concatenate(
            ([0.0], d/sd.sequence_length))

    return sd, rho, filename[:-len(".samples")], None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_file", default=None,
        help="A tsinfer sample file ending in '.samples'. If no genetic map is provided"
            " via the -m switch, and the filename contains chrNN"
            " where 'NN' is a number, assume this is a human samples file and use the"
            " appropriate recombination map from the thousand genomes project, otherwise"
            " use the physical distance between sites.")
    # The _mrate parameter defaults set from analysis ot 1000G, see
    # https://github.com/tskit-dev/tsinfer/issues/263#issuecomment-639060101
    parser.add_argument("-A", "--mismatch_ancestors", type=float, default=1,
        help="The mismatch probability in the match ancestors phase,"
            " as a fraction of the median recombination probability between sites")
    parser.add_argument("-S", "--mismatch_samples", type=float, default=1,
        help="The mismatch probability in the match samples phase,"
            " as a fraction of the median recombination probability between sites")
    parser.add_argument("-p", "--precision", type=int, default=None,
        help="The precision, as a number of decimal places, which will affect the speed"
            " of the matching algorithm (higher precision = lower speed). If None,"
            " calculate the smallest of the recombination rates or mismatch rates, and"
            " use the negative exponent of that number plus one. E.g. if the smallest"
            " recombination rate is 2.5e-6, use precision = 6+3 = 7")
    parser.add_argument("-t", "--num_threads", type=int, default=0,
        help="The number of threads to use in inference")
    parser.add_argument("-m", "--genetic_map", default=None,
        help="An alternative genetic map to be used for this analysis, in the format"
            "expected by msprime.RecombinationMap.read_hapmap")
    parser.add_argument("-x", "--cutoff_exponent", default=None, type=float,
        help="The value, x to be used as the exponenent of the freq, to shorten"
            "ancestor building")
    args = parser.parse_args()

    samples, rho, prefix, ts = setup_sample_file(args)

    params = Params(
        samples,
        rho,
        args.mismatch_ancestors,
        args.mismatch_samples,
        args.cutoff_exponent,
        args.precision,
        args.num_threads,
    )
    print(f"Running inference with {params}")
    with open(prefix + ".results", "wt") as file:
        result = run(params)
        print("\t".join(str(r) for r in result), file=file, flush=True)
