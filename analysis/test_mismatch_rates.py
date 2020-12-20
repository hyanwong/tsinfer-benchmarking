"""
Test the quality of inference, measured by num edges + num mutations, and (if
using simulated data) the KC distance
"""
import os.path
import argparse
import collections
import itertools
import multiprocessing
import re
import time
import logging
import json

import msprime
import tskit
import numpy as np
import stdpopsim #  Requires a version of msprime which allows gene conversion
import tsinfer

from error_generation import add_errors
import intervals

logging.basicConfig()
logger = logging.getLogger(__name__)


def make_switch_errors(sample_data, switch_error_rate=0, random_seed=None, **kwargs):
    raise NotImplementedError

def rnd_kc(params):
    ts, random_seed = params
    s = tskit.Tree.generate_star(
        ts.num_samples, span=ts.sequence_length, sample_lists=True)
    kc = 0
    for tree in ts.trees(sample_lists = True):
        kc += tree.span * tree.kc_distance(s.split_polytomies(
            random_seed=random_seed + tree.index, sample_lists=True)) 
    return kc / ts.sequence_length

def simulate_stdpopsim(
    species,
    model,
    contig,
    num_samples,
    mutation_file=None,
    seed=123,
    skip_existing=False,
    num_procs=1,
):
    base_fn = f"{model}_{contig}_n{num_samples}"
    tree_fn = f"{base_fn}_seed{seed}"
    logger.info(f"Using {species}:{contig} from stdpopsim using the {model} model")
    if skip_existing and os.path.exists(tree_fn + ".trees"):
        logger.info(
            f"Simulation file {tree_fn}.trees already exists, returning that.")
        return base_fn, tree_fn

    sample_data = None
    species = stdpopsim.get_species(species)
    model = species.get_demographic_model(model)
    num_pops = model.num_sampling_populations
    if num_samples < num_pops or num_samples % num_pops != 0:
        raise ValueError(
            f"num_samples must be an integer multiple of {num_pops} "
            f"(or 2 x {num_pops} if diploid sequencing error is injected)"
        )
    pop_n = num_samples // num_pops
    logger.info(
        f"Simulating {num_pops}x{pop_n} samples, seed {seed}, file prefix '{tree_fn}'."
    )
    contig = species.get_contig(contig)
    l = contig.recombination_map.get_sequence_length()
    if mutation_file is not None:
        logger.debug(f"Loading {mutation_file}")
        sample_data = tsinfer.load(mutation_file)
        if sample_data.sequence_length != l:
            raise ValueError(
                f"Mismatching sequence_length between simulation and {mutation_file}")
        # Reduce mutation rate to 0, as we will insert mutations later
        contig = stdpopsim.Contig(
            mutation_rate=0,
            recombination_map=contig.recombination_map,
            genetic_map=contig.genetic_map,
        )
    r_map = contig.recombination_map
    assert len(r_map.get_rates()) == 2  # Ensure a single rate over chr
    samples = model.get_samples(*([pop_n] * num_pops))
    engine = stdpopsim.get_engine('msprime')
    ts = engine.simulate(
        model, contig, samples,
        gene_conversion_rate=r_map.mean_recombination_rate * 10,
        gene_conversion_track_length=300,
        seed=seed)
    tables = ts.dump_tables()
    if sample_data is not None:
        pos = sample_data.sites_position[:]
        logger.info(
            f"Inserting {len(pos)} mutations at variable sites from {mutation_file}")
        for tree in ts.trees():
            positions = pos[np.logical_and(pos>=tree.interval[0], pos<tree.interval[1])]
            if len(positions) == 0:
                continue
            muts = list(zip(
                np.random.uniform(0, tree.total_branch_length, size=len(positions)),
                positions))
            muts.sort()
            tot = 0
            # place a mutation on a random branch, proportional to branch length
            try:
                for n in tree.nodes():
                    tot += tree.branch_length(n)
                    while muts[0][0] < tot:
                        _, position = muts.pop(0)
                        s = tables.sites.add_row(position=position, ancestral_state="0")
                        tables.mutations.add_row(node=n, site=s, derived_state="1")
            except IndexError:
                # No more mutations - go to next tree
                continue
        tables.sort()
        logger.debug(
            f"Inserted mutations at density {ts.num_mutations/ts.sequence_length}")
    interval = [int(l * 2/20), int(l * 2/20)+1e7] # 10Mb near the start, not centromeric
    tables.keep_intervals([interval])
    tables.trim()
    logger.debug(
        f"Cut down tree seq to  {interval} ({tables.sites.num_rows} sites) for speed")

    # Add info to the top-level metadata
    user_data = {}

    logger.info("Calculating the kc distance of the simulation against a flat tree")
    star_tree = tskit.Tree.generate_star(
            ts.num_samples, span=tables.sequence_length, record_provenance=False)
    user_data['kc_max'] = tables.tree_sequence().kc_distance(star_tree.tree_sequence)
    kc_array = []
    max_reps = 100
    ts = tables.tree_sequence()
    logger.info(
        f"Calculating KC distance of the sim against at most {max_reps} * {ts.num_trees}"
        f" random trees using {num_procs} parallel threads. This could take a while."
    )
    seeds = range(seed, seed + max_reps)
    with multiprocessing.Pool(num_procs) as pool:
        for i, kc in enumerate(pool.imap_unordered(
            rnd_kc, zip(itertools.repeat(ts), seeds))
        ):
            kc_array.append(kc)
            if i > 10:
                se_mean = np.std(kc_array, ddof=1)/np.sqrt(i)
                # break if SEM < 1/100th of mean KC. This can take along time
                if se_mean/np.average(kc_array) < 0.01:
                    logger.info(
                        f"Stopped after {i} replicates as kc_max_split deemed accurate.")
                    break
        user_data['kc_max_split'] = np.average(kc_array)

    if tables.metadata_schema != tskit.MetadataSchema({"codec":"json"}):
        if tables.metadata:
            raise RuntimeError("Metadata already exists, and is not JSON")
        tables.metadata_schema = tskit.MetadataSchema({"codec":"json"})
        tables.metadata = {}
    tables.metadata = {"user_data": user_data, **tables.metadata}
    tables.tree_sequence().dump(tree_fn + ".trees")
    return base_fn, tree_fn

def test_sim(seed):
    ts = msprime.simulate(
        10,
        length=1000,
        mutation_rate=1e-2,
        recombination_rate=1e-2,
        random_seed=seed)
    return ts, f"test_sim{seed}"


def setup_sampledata_from_simulation(
    prefix, random_seed, err=0, num_threads=1,
    cheat_breakpoints=False, use_sites_time=False, skip_existing=False):
    """
    Take the results of a simulation and return a sample data file, some reconstructed
    ancestors, a recombination rate array, a suffix to append to the file prefix, and
    the original tree sequence.
    
    If 'err' is 0, we do not inject any errors into the haplotypes. Otherwise
    we add empirical sequencing error and ancestral allele polarity error
    
    If "cheat_recombination" is True, multiply the recombination_rate for known
    recombination locations from the simulation by 20

    If "use_sites_time" is True, use the times
    
    If "skip_existing" is True, and the sample_data file and ancestors_file that were
    going to be generated already exist, then skip the actual simulation and just return
    those files and their data.
    """
    suffix = ""
    ts = tskit.load(prefix + ".trees")
    plain_samples = tsinfer.SampleData.from_tree_sequence(
        ts, use_sites_time=use_sites_time)
    if cheat_breakpoints:
        suffix += "cheat_breakpoints"
        logger.info("Cheating by using known breakpoints")
    if use_sites_time:
        suffix += "use_times"
        logger.info("Cheating by using known times")
    if err == 0:
        sd_path = prefix + suffix + ".samples"
        if skip_existing and os.path.exists(sd_path):
            logger.info(f"Simulation file {sd_path} already exists, loading that.")
            sd = tsinfer.load(sd_path)
        else:
            sd = plain_samples.copy(path=sd_path)  # Save the samples file
            sd.finalise()
    else:
        logger.info("Adding error")
        suffix += f"_ae{err}"
        sd_path = prefix + suffix + ".samples"
        if skip_existing and os.path.exists(sd_path):
            logger.info(f"Sample file {sd_path} already exists, loading that.")
            sd = tsinfer.load(sd_path)
        else:
            error_file = add_errors(
                plain_samples,
                err,
                random_seed=random_seed)
            sd = error_file.copy(path=prefix + suffix + ".samples")
            if use_sites_time:
                # Sites that were originally singletons have time 0, but could have been
                # converted to inference sites when adding error. Give these a nonzero time
                sites_time = sd.sites_time
                sites_time[sites_time == 0] = np.min(sites_time[sites_time > 0])/1000.0
                sd.sites_time[:] = sites_time
            sd.finalise()
    for attribute in ('sequence_length', 'num_samples', 'num_sites'):
        if getattr(sd, attribute) != getattr(ts, attribute):
            raise ValueError(
                f"{attribute} differs between original ts and sample_data: "
                f"{getattr(sd, attribute)} vs {getattr(ts, attribute)}")

    anc_path = prefix + suffix + ".ancestors"
    if skip_existing and os.path.exists(anc_path):
        logger.info(f"Ancestors file {anc_path} already exists, loading that.")
        anc = tsinfer.load(anc_path)
    else:
        anc = tsinfer.generate_ancestors(
            sd,
            num_threads=num_threads,
            path=anc_path,
        )
        logger.info("GA done")

    inference_pos = anc.sites_position[:]

    rho = 1e-8  # shouldn't matter what this is - it it relative to mismatch
    if cheat_breakpoints:
        raise NotImplementedError("Need to make a RateMap with higher r at breakpoints")
        breakpoint_positions = np.array(list(ts.breakpoints()))
        inference_positions = anc.sites_position[:]
        breakpoints = np.searchsorted(inference_positions, breakpoint_positions)
        # Any after the last inference position must be junked
        # (those before the first inference position make no difference)
        breakpoints = breakpoints[breakpoints != len(rho)]
        rho[breakpoints] *= 20
    return sd.path, anc.path, rho, suffix, ts

def setup_sample_file(base_filename, args, num_threads=1):
    """
    Return a sample data file, the ancestors file, a corresponding recombination rate
    (a single number or a RateMap), a prefix to use for files, and None
    """
    gmap = args.genetic_map
    sd = tsinfer.load(base_filename + ".samples")

    anc = tsinfer.generate_ancestors(
        sd,
        num_threads=num_threads,
        path=base_filename + ".ancestors",
    )
    logger.info("GA done")

    inference_pos = anc.sites_position[:]

    match = re.search(r'(chr\d+)', base_filename)
    if match or gmap is not None:
        if gmap is not None:
            logger.info(f"Using {gmap} for the recombination map")
            rho = intervals.read_hapmap(gmap)
        else:
            chr = match.group(1)
            logger.info(f"Using {chr} from HapMapII_GRCh37 for the recombination map")
            gmap = stdpopsim.get_species("HomSap").get_genetic_map(id="HapMapII_GRCh37")
            if not gmap.is_cached():
                gmap.download()
            filename = os.path.join(gmap.map_cache_dir, gmap.file_pattern.format(id=chr))
            rho = intervals.read_hapmap(filename)
    else:
        rho = 1e-8  # shouldn't matter what this is - it it relative to mismatch
        
    #if np.any(d==0):
    #    w = np.where(d==0)
    #    raise ValueError("Zero recombination rates at", w, inference_pos[w])

    return sd.path, anc.path, rho, "", None

# Parameters passed to each subprocess
Params = collections.namedtuple(
    "Params",
    "ts_file, sample_file, anc_file, rec_rate, ma_mis_ratio, ms_mis_ratio, precision, "
    "num_threads, kc_max, kc_max_split, seed, error, source, skip_existing"
)

    
def run(params):
    """
    Run a single inference, with the specified rates
    """
    precision = params.precision
    logger.info(
        f"Starting {params.ma_mis_ratio} {params.ms_mis_ratio}. Precision {precision}"
    )
    prefix = None
    assert params.sample_file.endswith(".samples")
    assert params.anc_file.endswith(".ancestors")
    samples = tsinfer.load(params.sample_file)
    ancestors = tsinfer.load(params.anc_file)
    start_time = time.process_time()
    prefix = params.sample_file[0:-len(".samples")]
    inf_prefix = "{}_rma{:g}_rms{:g}_p{}".format(
            prefix,
            params.ma_mis_ratio,
            params.ms_mis_ratio,
            precision)

    ats_path = inf_prefix + ".atrees"
    if params.skip_existing and os.path.exists(ats_path):
        logger.info(f"Ancestors ts file {ats_path} already exists, loading that.")
        inferred_anc_ts = tskit.load(ats_path)
        prov = json.loads(inferred_anc_ts.provenances()[-1].record.encode())
        if ancestors.uuid != prov['parameters']['source']['uuid']:
            logger.warning(
                "The loaded ancestors ts does not match the ancestors file. "
                "Checking the site positions, and will abort if they don't match!")
            # We might be re-running this, but the simulation file is the same
            # So double-check that the positions in the ats are a subset of those in the
            # used sample data file
            assert np.all(np.isin(
                inferred_anc_ts.tables.sites.position,
                samples.sites_position[:]))
            
    else:
        logger.info(f"MA running: will save to {ats_path}")
        inferred_anc_ts = tsinfer.match_ancestors(
            samples,
            ancestors,
            num_threads=params.num_threads,
            precision=precision,
            recombination_rate=params.rec_rate,
            mismatch_ratio=params.ma_mis_ratio)
        inferred_anc_ts.dump(ats_path)
        logger.info(f"MA done: mismatch ratio = {params.ma_mis_ratio}")

    ts_path = inf_prefix + ".trees"
    if params.skip_existing and os.path.exists(ts_path):
        logger.info(f"Inferred ts file {ts_path} already exists, loading that.")
        inferred_ts = tskit.load(ts_path)
        try:
            user_data = inferred_ts.metadata['user_data']
            try:
                assert np.allclose(params.kc_max, user_data['kc_max'])
            except (KeyError, TypeError):
                pass  # could be NaN e.g. if this is real data
            return user_data
        except (TypeError, KeyError):
            logging.warning("No metadata in {ts_path}: re-inferring these parameters")

    # Otherwise finish off the inference
    logger.info(f"MS running with {params.num_threads} threads: will save to {ts_path}")
    inferred_ts = tsinfer.match_samples(
        samples,
        inferred_anc_ts,
        num_threads=params.num_threads,
        precision=precision,
        recombination_rate=params.rec_rate,
        mismatch_ratio=params.ms_mis_ratio)
    process_time = time.process_time() - start_time
    logger.info(f"MS done: mismatch ratio = {params.ms_mis_ratio}")
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
    
    sim_ts_bytes = sim_ts_min_bytes = None
    kc_poly = kc_split = None
    
    if params.ts_file is not None:
        try:
            simulated_ts = tskit.load(params.ts_file + ".trees")
            logger.info(f"Calculating KC distances for {ts_path}")
            sim_ts_bytes = simulated_ts.nbytes
            sim_ts_min_bytes = simulated_ts.simplify(
                keep_unary=True, reduce_to_site_topology=True, filter_sites=False).nbytes
            kc_poly = simplified_inferred_ts.kc_distance(simulated_ts)
            logger.debug("KC poly calculated")
            kc_split = 0
            for interval, orig_tree, new_tree in simulated_ts.coiterate(
                simplified_inferred_ts, sample_lists=True
            ):
                kc_split += interval.span * orig_tree.kc_distance(
                    new_tree.split_polytomies(
                        random_seed=int(interval.left),
                        epsilon=1e-20,  # Smaller epsilon than used in path compression
                        sample_lists=True))
            kc_split /= simulated_ts.sequence_length
            logger.debug("KC split calculated")
        except FileNotFoundError:
            pass

    results = {
        'arity_mean': arity_mean,
        'arity_var': arity_var,
        'edges': inferred_ts.num_edges,
        'error': params.error,
        'kc_max_split': params.kc_max_split,
        'kc_max': params.kc_max,
        'kc_poly': kc_poly,
        'kc_split': kc_split,
        'muts': inferred_ts.num_mutations,
        'n': inferred_ts.num_samples,
        'num_sites': inferred_ts.num_sites,
        'num_trees': inferred_ts.num_trees,
        'precision': precision,
        'proc_time': process_time,
        'ma_mis_ratio': params.ma_mis_ratio,
        'ms_mis_ratio': params.ms_mis_ratio,
        'seed': params.seed,
        'sim_ts_min_bytes': sim_ts_min_bytes,
        'sim_ts_bytes': sim_ts_bytes,
        'source': params.source,
        'ts_bytes': inferred_ts.nbytes,
        'ts_path': ts_path,
    }
    # Save the results into the ts metadata - this should allow us to reconstruct the
    # results table should anything go awry, or if we need to add more
    tables = inferred_ts.dump_tables()
    if tables.metadata_schema != tskit.MetadataSchema({"codec":"json"}):
        if tables.metadata:
            raise RuntimeError("Metadata already exists in the ts, and is not JSON")
        tables.metadata_schema = tskit.MetadataSchema({"codec":"json"})
        tables.metadata = {}
    tables.metadata = {"user_data": results, **tables.metadata}
    tables.tree_sequence().dump(ts_path)
    return results


def run_replicate(rep, args):
    """
    The main function that runs a parameter set
    """
    params = {}  # The info to be passed though to each inference run
    params['num_threads'] = args.num_threads
    params['error'] = args.error
    params['source'] = args.source
    params['skip_existing'] = args.skip_existing_params
    params['seed'] = rep+args.random_seed
    precision = [None] if len(args.precision) == 0 else args.precision

    if args.source.count(":") >= 3:
        logger.debug("Simulating")
        details = args.source.split(":")
        base_name, ts_name = simulate_stdpopsim(
            species=details[0],
            contig=details[1],
            model=details[2],
            num_samples=int(details[3]),
            mutation_file=details[4] if len(details)>4 else None,
            seed=params['seed'],
            skip_existing=params['skip_existing'],
            num_procs=args.num_processes,
        )
        sample_file, anc_file, rho, suffix, ts = setup_sampledata_from_simulation(
            ts_name,
            random_seed=params['seed'],
            err=params['error'],
            num_threads=params['num_threads'],
            cheat_breakpoints=args.cheat_breakpoints,
            use_sites_time=args.use_sites_time,
            skip_existing=params['skip_existing'],
        )
        prefix = ts_name + suffix
        base_name += suffix
    else:
        logger.debug(f"Using provided sample data file {params['source']}")
        if not params['source'].endswith(".samples"):
            raise ValueError("Sample data file must end with '.samples'")
        prefix = params['source'][:-len(".samples")]
        sample_file, anc_file, rho, suffix, ts = setup_sample_file(
            prefix, args, params['num_threads'])
        ts_name = None
        base_name = prefix + suffix

    params['kc_max'], params['kc_max_split'] = None, None
    try:
        params['kc_max'] = ts.metadata['user_data']['kc_max']
        params['kc_max_split'] = ts.metadata['user_data']['kc_max_split']
    except (TypeError, KeyError, AttributeError):
        pass
    
    param_iter = [
        Params(ts_name, sample_file, anc_file, rho, ma_mr, ms_mr, p, **params)
            for ms_mr in args.match_samples_mismatch_ratio
                for ma_mr in args.match_ancestors_mismatch_ratio
                    for p in precision]
    treefiles = []
    results_filename = prefix + "_results.csv"
    with open(results_filename, "wt") as file:
        headers = []
        if args.num_processes < 2:
            for p in param_iter:
                result = run(p)
                if len(headers) == 0:
                    headers = list(result.keys())
                    print(",".join(headers), file=file)
                else:
                    if set(headers) != set(result.keys()):
                        logging.warning("Some differences in headers")
                result_str = [str(result.get(h, "")) for h in headers]
                print(",".join(result_str), file=file, flush=True)
                treefiles.append(result['ts_path'])
        else:
            num_procs = min(len(param_iter), args.num_processes)
            logger.info(
                f"Parallelising {len(param_iter)} parameter combinations "
                f"over {num_procs} processes")
            with multiprocessing.Pool(num_procs) as pool:
                for result in pool.imap_unordered(run, param_iter):
                    # Save to a results file.
                    if len(headers) == 0:
                        headers = list(result.keys())
                        print(",".join(headers), file=file)
                    else:
                        if set(headers) != set(result.keys()):
                            logging.warning("Some differences in headers")
                    result_str = [str(result.get(h, "")) for h in headers]
                    print(",".join(result_str), file=file, flush=True)
                    treefiles.append(result['ts_path'])
    logger.info(f"Results saved to {results_filename}")
    return base_name, treefiles

if __name__ == "__main__":


    # Set up the range of params for multiprocessing
    default_match_samples_mismatch_ratio = np.array(
        [1e4, 1e3, 1e2, 10, 5, 2, 1, 0.5, 0.1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5])
    default_match_ancestors_mismatch_ratio = np.array(
        [1e4, 1e3, 1e2, 10, 5, 2, 1, 0.5, 0.1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs='?',
        default="HomSap:chr20:OutOfAfrica_3G09:1500",
        help=
            "A string giving the source for the data used to test mismatch rates. "
            "If this contains at least 3 colons, it is taken as a specification for "
            "stspopsim in the form of species:contig:model:num_samples(:optional_file), "
            "where the optional_file, if given, is a sample_data file providing the "
            "sites used as targets for mutation. If the source contains one or no "
            "colons, it should be a tsinfer sample file ending in '.samples', in which "
            "case simulation is not perfomed, but instead the script uses (potentially) "
            "real data from the specified file. If no genetic map is provided "
            "via the -m switch, and the filename contains chrNN where "
            "'NN' is a number, assume this is a human samples file and use the "
            "appropriate recombination map from the thousand genomes project"
    )
    parser.add_argument("-r", "--replicates", type=int, default=1)
    parser.add_argument("-s", "--random_seed", type=int, default=1)
    parser.add_argument("-e", "--error", type=float, default=0,
        help="Add sequencing and ancestral state error to the haplotypes before"
            "inferring. The value here gives the probability of ancestral state"
            "error")
    parser.add_argument("-A", "--match_ancestors_mismatch_ratio", nargs='*', type=float,
        default=default_match_ancestors_mismatch_ratio,
        help = (
            "A list of values for the relative match_ancestors mismatch rate."
            "The rate is relative to the median recombination rate between sites")
    )
    parser.add_argument("-S", "--match_samples_mismatch_ratio", nargs='*', type=float,
        default=default_match_samples_mismatch_ratio,
        help =
            "A list of values for the relative match_samples mismatch rate. "
            "The rate is relative to the median recombination rate between sites"
    )
    parser.add_argument("-T", "--use_sites_time", action='store_true',
        help=
            "When using simulated data, cheat by using the times for sites (ancestors)"
            "from the simulation")
    parser.add_argument("-P", "--precision", nargs='*', type=int, default=[],
        help=
            "The precision, as a number of decimal places, which will affect the speed "
            "of the matching algorithm (higher precision: lower speed). If not given, "
            "calculate the smallest of the recombination rates or mismatch rates, and "
            "use the negative exponent of that number plus four. E.g. if the smallest "
            "recombination rate is 2.5e-6, use precision = 6+4 = 10"
    )
    parser.add_argument("-B", "--cheat_breakpoints", action='store_true',
        help=
            "When using simulated data, cheat by increasing the recombination"
            "probability in regions where there is a true breakpoint")
    parser.add_argument("-k", "--skip_existing_params", action='store_true',
        help=
            "If inference files exists with the same name, assume they skip the inference ")
    parser.add_argument("-t", "--num_threads", type=int, default=2,
        help=
            "The number of threads to use in each inference subprocess. "
    )
    parser.add_argument("-p", "--num_processes", type=int, default=40,
        help=
            "The number of processors that can be pooled to parallelise runs"
            "under different parameter values."
    )
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

    multiple_replicates = args.replicates > 1
    filenames = []
    for rep in range(args.replicates):
        # NB: this doesn't allow parallelising of replicates
        logger.debug(f"Running replicate {rep}")
        base_name, treenames = run_replicate(rep, args)
        filenames += treenames
    if multiple_replicates:
        
        header = None
        with open(base_name + "_results.csv", "wt") as final_file:
            for ts_name in filenames:
                metadata = json.loads(tskit.load(ts_name).metadata.decode())
                if header is None:
                    header = list(metadata.keys())
                    print(",".join(metadata.keys()), file=final_file)
                else:
                    if header != list(metadata.keys()):
                        raise ValueError(
                            f"Header '{header}' differs from {list(metadata.keys())}")
                print(",".join([str(v) for v in metadata.values()]), file=final_file)
        logger.info(f"Final results integrated into {base_name}_results.csv")
