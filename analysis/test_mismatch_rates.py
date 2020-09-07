"""
Test the quality of inference, measured by num edges + num mutations, and (if
using simulated data) the KC distance
"""
import os.path
import argparse
import collections
import multiprocessing
import re
import time
import logging

import pandas as pd
import msprime
import tskit
import numpy as np
import stdpopsim #  Requires a version of msprime which allows gene conversion
import tsinfer

from error_generation import add_errors


logging.basicConfig()
logger = logging.getLogger(__name__)

def make_switch_errors(sample_data, switch_error_rate=0, random_seed=None, **kwargs):
    raise NotImplementedError


def simulate_human(random_seed=123, each_pop_n=10):
    logger.debug(
        f"Simulation Hom_sap using stdpopsim with 3x{each_pop_n} samples")
    species = stdpopsim.get_species("HomSap")
    contig = species.get_contig("chr20")
    r_map = contig.recombination_map
    model = species.get_demographic_model('OutOfAfrica_3G09')
    assert len(r_map.get_rates()) == 2  # Ensure a single rate over chr
    samples = model.get_samples(each_pop_n, each_pop_n, each_pop_n)
    engine = stdpopsim.get_engine('msprime')
    ts = engine.simulate(
        model, contig, samples,
        gene_conversion_rate=r_map.mean_recombination_rate * 10,
        gene_conversion_track_length=300,
        seed=random_seed)
    l = ts.sequence_length
    # cut down ts for speed
    return (
        ts.keep_intervals([[int(l * 3/20), int(l * 4/20)]]).trim(),
        f"data/OOA_sim_n{each_pop_n*3}_seed{random_seed}")

def test_sim(seed):
    ts = msprime.simulate(
        10,
        length=1000,
        mutation_rate=1e-2,
        recombination_rate=1e-2,
        random_seed=seed)
    return ts, f"data/test_sim{seed}"


def physical_to_genetic(recombination_map, input_physical_positions):
    map_pos = recombination_map.get_positions()
    map_rates = recombination_map.get_rates()
    map_genetic_positions = np.insert(np.cumsum(np.diff(map_pos) * map_rates[:-1]), 0, 0)
    return np.interp(input_physical_positions, map_pos, map_genetic_positions)


def setup_simulation(
    ts, prefix, random_seed=None, cheat_recombination=False, err=0, num_threads=1):
    """
    Take the results of a simulation and return a sample data file, some reconstructed
    ancestors, a recombination rate array, a prefix to use for files, and
    the original tree sequence.
    
    If "cheat_recombination" is true, multiply the recombination_rate for known
    recombination locations from the simulation by 20
    
    If 'err' is 0, we do not inject any errors into the haplotypes. Otherwise
    we add empirical sequencing error and ancestral allele polarity error
    
    """
    plain_samples = tsinfer.SampleData.from_tree_sequence(
        ts, use_times=False)
    if cheat_recombination:
        prefix += "cheat"
    if err == 0:
        sd = plain_samples.copy(path=prefix + ".samples")
    else:
        prefix += f"_ae{err}"
        sd = add_errors(
            plain_samples,
            err,
            random_seed=random_seed,
            path=prefix+".samples")

    anc = tsinfer.generate_ancestors(
        sd,
        num_threads=num_threads,
        path=prefix+".ancestors",
    )
    logger.info("GA done")

    inference_pos = anc.sites_position[:]

    rho = np.diff(anc.sites_position[:])/sd.sequence_length
    rho = np.concatenate(([0.0], rho))
    if cheat_recombination:
        breakpoint_positions = np.array(list(ts.breakpoints()))
        inference_positions = anc.sites_position[:]
        breakpoints = np.searchsorted(inference_positions, breakpoint_positions)
        # Any after the last inference position must be junked
        # (those before the first inference position make no difference)
        breakpoints = breakpoints[breakpoints != len(rho)]
        rho[breakpoints] *= 20
    return sd.path, anc.path, rho, prefix, ts

def setup_sample_file(args, num_threads=1):
    """
    Return a Thousand Genomes Project sample data file, the ancestors file, a
    corresponding recombination rate array, a prefix to use for files, and None
    """
    filename = args.sample_file
    map = args.genetic_map
    if not filename.endswith(".samples"):
        raise ValueError("Sample data file must end with '.samples'")
    base_filename = filename[:len(".samples")]
    sd = tsinfer.load(filename)

    anc = tsinfer.generate_ancestors(
        sd,
        num_threads=num_threads,
        path=base_filename + ".ancestors",
    )
    logger.info("GA done")

    inference_pos = anc.sites_position[:]

    match = re.search(r'(chr\d+)', filename)
    if match or map is not None:
        if map is not None:
            chr_map = msprime.RecombinationMap.read_hapmap(map)
        else:
            chr = match.group(1)
            logger.info(f"Using {chr} from HapMapII_GRCh37 for the recombination map")
            map = stdpopsim.get_species("HomSap").get_genetic_map(id="HapMapII_GRCh37")
            if not map.is_cached():
                map.download()
            chr_map = map.get_chromosome_map(chr)
        inference_distances = physical_to_genetic(chr_map, inference_pos)
        d = np.diff(inference_distances)
    else:
        inference_distances = sd.sites_position[:][sd.sites_inference]
        d = np.diff(inference_distances)/sd.sequence_length
    rho = np.concatenate(([0.0], d))
        
    if np.any(d==0):
        w = np.where(d==0)
        raise ValueError("Zero recombination rates at", w, inference_pos[w])

    return sd.path, anc.path, rho, filename[:-len(".samples")], None


Params = collections.namedtuple(
    "Params",
    "sample_file, anc_file, rec_rate, ma_mis_rate, ms_mis_rate, precision, num_threads")

Results = collections.namedtuple(
    "Results",
    "n, abs_ma_mis, abs_ms_mis, rel_ma_mis, rel_ms_mis, precision, edges, muts, "
    "num_trees, kc_poly, kc_split, arity_mean, arity_var, proc_time, ts_size, ts_path")

    
def run(params):
    """
    Run a single inference, with the specified rates
    """
    rho = params.rec_rate[1:]
    base_rec_prob = np.quantile(rho, 0.5)
    if params.precision is None:
        # Smallest recombination rate
        min_rho = int(np.ceil(-np.min(np.log10(rho[rho>0]))))
        # Smallest mean 
        av_min = int(np.ceil(-np.log10(
            min(1, params.ma_mis_rate, params.ms_mis_rate) * base_rec_prob)))
        precision = max(min_rho, av_min) + 3
    else:
        precision = params.precision
    ma_mis = base_rec_prob * params.ma_mis_rate
    ms_mis = base_rec_prob * params.ms_mis_rate
    logger.info(
        f"Starting {params.ma_mis_rate} {params.ms_mis_rate} " +
        f"with base rho {base_rec_prob:.5g} " +
        f"(mean {np.mean(rho):.4g} median {np.quantile(rho, 0.5):.4g} " +
        f"min {np.min(rho):.4g}, 2.5% quantile {np.quantile(rho, 0.025):.4g}) " +
        f"precision {precision}"
    )
    prefix = None
    assert params.sample_file.endswith(".samples")
    assert params.anc_file.endswith(".ancestors")
    samples = tsinfer.load(params.sample_file)
    ancestors = tsinfer.load(params.anc_file)
    prefix = params.sample_file[0:-len(".samples")]
    inf_prefix = "{}_rma{}_rms{}_p{}".format(
            prefix,
            params.ma_mis_rate,
            params.ms_mis_rate,
            precision)
    start_time = time.process_time()
    inferred_anc_ts = tsinfer.match_ancestors(
        samples,
        ancestors,
        num_threads=params.num_threads,
        precision=precision,
        recombination_rate=params.rec_rate,
        mismatch_rate=ma_mis)
    inferred_anc_ts.dump(path=inf_prefix + ".atrees")
    logger.info(f"MA done: abs_ma_mis rate = {ma_mis}")
    inferred_ts = tsinfer.match_samples(
        samples,
        inferred_anc_ts,
        num_threads=params.num_threads,
        precision=precision,
        recombination_rate=params.rec_rate,
        mismatch_rate=ms_mis)
    process_time = time.process_time() - start_time
    ts_path = inf_prefix + ".trees"
    inferred_ts.dump(path=ts_path)
    logger.info(f"MS done: abs_ms_mis rate = {ms_mis}")
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
    arity_mean = nc_sum/nc_tot
    arity_var = nc_sum_sq / nc_tot - (arity_mean ** 2) # can't be bothered to adjust for n

    # Calculate span of root nodes in simplified tree
    

    # Calculate KC
    try:
        kc_poly = simplified_inferred_ts.kc_distance(tskit.load(prefix+".trees"))
        logger.debug("KC poly calculated")
        polytomies_split_ts = simplified_inferred_ts.randomly_split_polytomies(
            random_seed=123)
        logger.debug("Polytomies split for KC calc")
        kc_split = polytomies_split_ts.kc_distance(tskit.load(prefix+".trees"))
        logger.debug("KC split calculated")
    except FileNotFoundError:
        kc_poly = kc_split = None
    return Results(
        n=inferred_ts.num_samples,
        abs_ma_mis=ma_mis,
        abs_ms_mis=ms_mis,
        rel_ma_mis=params.ma_mis_rate,
        rel_ms_mis=params.ms_mis_rate,
        precision=precision,
        edges=inferred_ts.num_edges,
        muts=inferred_ts.num_mutations,
        num_trees=inferred_ts.num_trees,
        kc_poly=kc_poly,
        kc_split=kc_split,
        arity_mean=arity_mean,
        arity_var=arity_var,
        proc_time=process_time,
        ts_size=os.path.getsize(ts_path),
        ts_path=ts_path)

def run_replicate(rep, args):
    """
    The main function that runs a parameter set
    """
    seed = rep+args.random_seed
    if len(args.precision) == 0:
        precision = [None]
    else:
        precision = args.precision
    nt = 2 if args.num_threads is None else args.num_threads
    if args.sample_file is None:
        logger.debug("Simulating human chromosome")
        sim = simulate_human(seed)
        sample_file, anc_file, rho, prefix, ts = setup_simulation(
            *sim,
            random_seed=seed,
            cheat_recombination=args.cheat_breakpoints,
            err=args.error,
            num_threads=nt,
        )
    else:
        logger.debug("Using provided sample data file")
        sample_file, anc_file, rho, prefix, ts = setup_sample_file(args, num_threads=nt)
    if ts is not None:
        ts.dump(prefix + ".trees")
    # Set up the range of params for multiprocessing
    errs = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00001])
    muts = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001])
    param_iter = (
        Params(sample_file, anc_file, rho, m, e, p, nt) for e in errs for m in muts for p in precision)
    results_filename = prefix + ".results"
    with open(results_filename, "wt") as file:
        print("\t".join(Results._fields), file=file, flush=True)
        if nt < 2:
            for p in param_iter:
                result=run(p)
                print("\t".join(str(r) for r in result), file=file, flush=True)
        else:
            with multiprocessing.Pool(40) as pool:
                for result in pool.imap_unordered(run, param_iter):
                    # Save to a results file.
                    # NB this can be pasted into R and plotted using
                    # d <- read.table(stdin(), header=T)
                    # d$rel_ma <- factor(d$ma_mis / d$ms_mis)
                    # ggplot() + geom_line(data = d, aes(x = ms_mis, y = edges+muts, color = rel_ma)) + scale_x_continuous(trans='log10')
                    print("\t".join(str(r) for r in result), file=file, flush=True)
    logger.info(f"Results saved to {results_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_file", nargs='?', default=None,
        help="A tsinfer sample file ending in '.samples'. If given, do not"
            "evaluate using a simulation but instead use (potentially) real"
            "data from the specified file. If no genetic map is provided"
            " via the -m switch, and the filename contains chrNN where"
            "'NN' is a number, assume this is a human samples file and use the"
            "appropriate recombination map from the thousand genomes project")
    parser.add_argument("-r", "--replicates", type=int, default=1)
    parser.add_argument("-s", "--random_seed", type=int, default=1)
    parser.add_argument("-e", "--error", type=float, default=0,
        help="Add sequencing and ancestral state error to the haplotypes before"
            "inferring. The value here gives the probability of ancestral state"
            "error")
    parser.add_argument("-c", "--cheat_breakpoints", action='store_true',
        help="Cheat when using simulated data by increasing the recombination"
            "probability in regions where there is a true breakpoint")
    parser.add_argument("-p", "--precision", nargs='*', type=int, default=[],
        help="The precision, as a number of decimal places, which will affect the speed"
            " of the matching algorithm (higher precision: lower speed). If not given,"
            " calculate the smallest of the recombination rates or mismatch rates, and"
            " use the negative exponent of that number plus four. E.g. if the smallest"
            " recombination rate is 2.5e-6, use precision = 6+4 = 10")
    parser.add_argument("-t", "--num_threads", type=int, default=None,
        help="The number of threads to use in each inference subprocess")
    parser.add_argument("-m", "--genetic_map", default=None,
        help="An alternative genetic map to be used for this analysis, in the format"
            "expected by msprime.RecombinationMap.read_hapmap")
    parser.add_argument("-v", "--verbosity", action="count", default=0,
        help="Increase the verbosity")
    args = parser.parse_args()

    log_level = logging.WARN
    if args.verbosity > 0:
        log_level = logging.INFO
    if args.verbosity > 1:
        log_level = logging.DEBUG
    logger.setLevel(log_level)

    for rep in range(args.replicates):
        logger.debug(f"Running replicate {rep}")
        run_replicate(rep, args)
