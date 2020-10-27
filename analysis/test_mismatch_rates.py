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

import msprime
import tskit
import numpy as np
import stdpopsim #  Requires a version of msprime which allows gene conversion
import tsinfer

from error_generation import add_errors


# !! Delete this once https://github.com/tskit-dev/tskit/pull/815 is merged
def randomly_split_polytomies(
    ts,
    *,
    epsilon=None,
    squash_edges=True,
    record_provenance=True,
    random_seed=None,
):
    """
    Return a tree sequence with extra nodes and edges
    so that any node with greater than 2 children (i.e. a multifurcation
    or "polytomy") is resolved into successive bifurcations. For any
    multifucating node ``u`` with ``n`` children, the :math:`(2n - 3)!!`
    possible bifurcating topologies are produced with equal probability.
    """
    self = ts.dump_tables()  # Hack `self` to point to tables: allows straight code copy
    if epsilon is None:
        epsilon = 1e-10
    rng = np.random.default_rng(seed=random_seed)

    def is_unknown_time_array(a):
        np_unknown_time = np.float64(tskit.UNKNOWN_TIME)
        return a.view(np.uint64) == np_unknown_time.view(np.uint64)

    def resolve_polytomy(parent_node_id, child_ids, new_nodes_by_time_desc):
        """
        For a polytomy and list of child node ids, return a list of (child, parent)
        tuples, describing a bifurcating tree, rooted at parent_node_id, where the
        new_nodes_by_time_desc have been used to break polytomies. All possible
        topologies should be equiprobable.
        """
        nonlocal rng
        assert len(child_ids) == len(new_nodes_by_time_desc) + 2
        # Polytomies broken by sequentially splicing onto edges, so an initial edge
        # is required. This will always remain above the top node & is removed later
        edges = [
            [child_ids[0], None],
        ]
        # We know beforehand how many random ints are needed: generate them all now
        edge_choice = rng.integers(0, np.arange(1, len(child_ids) * 2 - 1, 2))
        tmp_new_node_lab = [parent_node_id] + new_nodes_by_time_desc
        assert len(edge_choice) == len(child_ids) - 1
        for node_lab, child_id, target_edge_id in zip(
            tmp_new_node_lab, child_ids[1:], edge_choice
        ):
            target_edge = edges[target_edge_id]
            # Insert in the right place, to keep edges in parent time order
            edges.insert(target_edge_id, [child_id, node_lab])
            edges.insert(target_edge_id, [target_edge[0], node_lab])
            target_edge[0] = node_lab
        top_edge = edges.pop()  # remove the edge above the top node
        assert top_edge[1] is None

        # Re-map the internal nodes IDs so they are used in time order
        real_node = iter(new_nodes_by_time_desc)
        node_map = {c: c for c in child_ids}
        node_map[edges[-1][1]] = parent_node_id  # last edge == oldest parent
        for e in reversed(edges):
            # Reversing along the edges, parents are in inverse time order
            for idx in (1, 0):  # look at parent (1) then child (0)
                if e[idx] not in node_map:
                    node_map[e[idx]] = next(real_node)
                e[idx] = node_map[e[idx]]
        assert len(node_map) == len(new_nodes_by_time_desc) + len(child_ids) + 1
        return edges

    edge_table = self.edges
    node_table = self.nodes
    # Store existing left, so we can change it if the edge is split
    existing_edges_left = edge_table.left
    # Keep other edge arrays etc. for fast read access
    existing_edges_right = edge_table.right
    existing_edges_parent = edge_table.parent
    existing_edges_child = edge_table.child
    existing_node_time = node_table.time

    # We can save a lot of effort if we don't need to check the time of mutations
    # We definitely don't need to check on the first iteration, a
    check_mutations = np.any(
        np.logical_not(is_unknown_time_array(self.mutations.time))
    )
    ts = self.tree_sequence()  # Only needed to check mutations
    tree_iter = ts.trees()  # ditto

    edge_table.clear()

    edges_from_node = collections.defaultdict(set)  # Active descendant edge ids
    nodes_changed = set()

    for interval, e_out, e_in in ts.edge_diffs(include_terminal=True):
        pos = interval[0]
        prev_tree = None if pos == 0 else next(tree_iter)

        for edge in itertools.chain(e_out, e_in):
            if edge.parent != tskit.NULL:
                nodes_changed.add(edge.parent)

        if check_mutations and prev_tree is not None:
            # This is grim. There must be a more efficient way.
            # It would also help if mutations were sorted such that all mutations
            # above the same node appeared consecutively, with oldest first.
            oldest_mutation_for_node = {}
            for site in prev_tree.sites():
                for mutation in site.mutations:
                    if not util.is_unknown_time(mutation.time):
                        oldest_mutation_for_node[mutation.node] = max(
                            oldest_mutation_for_node[mutation.node], mutation.time
                        )

        for parent_node in nodes_changed:
            child_edge_ids = edges_from_node[parent_node]
            if len(child_edge_ids) >= 3:
                # We have a previous polytomy to break
                parent_time = existing_node_time[parent_node]
                new_nodes = []
                child_ids = existing_edges_child[list(child_edge_ids)]
                left = None
                max_time = 0
                # Split existing edges
                for edge_id, child_id in zip(child_edge_ids, child_ids):
                    max_time = max(max_time, existing_node_time[child_id])
                    if check_mutations and child_id in oldest_mutation_for_node:
                        max_time = max(max_time, oldest_mutation_for_node[child_id])
                    if left is None:
                        left = existing_edges_left[edge_id]
                    else:
                        assert left == existing_edges_left[edge_id]
                    if existing_edges_right[edge_id] > interval[0]:
                        # make sure we carry on the edge after this polytomy
                        existing_edges_left[edge_id] = pos
                # Arbitrarily, if epsilon is not small enough, use half the min dist
                dt = min((parent_time - max_time) / (len(child_ids) * 2), epsilon)
                # Break this N-degree polytomy. This requires N-2 extra nodes to be
                # introduced: create them here in order of decreasing time
                new_nodes = [
                    node_table.add_row(time=parent_time - (i * dt))
                    for i in range(1, len(child_ids) - 1)
                ]
                # print("New nodes:", new_nodes, node_table.time[new_nodes])
                for new_edge in resolve_polytomy(parent_node, child_ids, new_nodes):
                    edge_table.add_row(
                        left=left, right=pos, child=new_edge[0], parent=new_edge[1],
                    )
                    # print("new_edge: left={}, right={}, child={}, parent={}"
                    #    .format(left, pos, new_edge[0], new_edge[1]))
            else:
                # Previous node was not a polytomy - just add the edges_out
                for edge_id in child_edge_ids:
                    if existing_edges_right[edge_id] == pos:  # is an out edge
                        edge_table.add_row(
                            left=existing_edges_left[edge_id],
                            right=pos,
                            parent=parent_node,
                            child=existing_edges_child[edge_id],
                        )

        for edge in e_out:
            if edge.parent != tskit.NULL:
                # print("REMOVE", edge.id)
                edges_from_node[edge.parent].remove(edge.id)
        for edge in e_in:
            if edge.parent != tskit.NULL:
                # print("ADD", edge.id)
                edges_from_node[edge.parent].add(edge.id)

        # Chop if we have created a polytomy: the polytomy itself will be resolved
        # at a future iteration, when any edges move into or out of the polytomy
        while nodes_changed:
            node = nodes_changed.pop()
            edge_ids = edges_from_node[node]
            # print("Looking at", node)

            if len(edge_ids) == 0:
                del edges_from_node[node]
            # if this node has changed *to* a polytomy, we need to cut all of the
            # child edges that were previously present by adding the previous
            # segment and left-truncating
            elif len(edge_ids) >= 3:
                for edge_id in edge_ids:
                    if existing_edges_left[edge_id] < interval[0]:
                        self.edges.add_row(
                            left=existing_edges_left[edge_id],
                            right=interval[0],
                            parent=existing_edges_parent[edge_id],
                            child=existing_edges_child[edge_id],
                        )
                    existing_edges_left[edge_id] = interval[0]
    assert len(edges_from_node) == 0
    self.sort()

    if squash_edges:
        self.edges.squash()
        self.sort()  # Bug: https://github.com/tskit-dev/tskit/issues/808

    return self.tree_sequence()

# Monkey-patch until https://github.com/tskit-dev/tskit/pull/815 is merged
tskit.TreeSequence.randomly_split_polytomies = randomly_split_polytomies        

logging.basicConfig()
logger = logging.getLogger(__name__)


def make_switch_errors(sample_data, switch_error_rate=0, random_seed=None, **kwargs):
    raise NotImplementedError


def simulate_human(random_seed=123, each_pop_n=500):
    logger.info(
        f"Simulating HomSap from stdpopsim: 3x{each_pop_n} samples, seed {random_seed}")
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
    return (
        # cut down ts for speed
        ts.keep_intervals([[int(l * 2/20), int(l * 5/20)]]).trim(),
        f"data/OOA_sim_n{each_pop_n*3}")

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
    ts, prefix, random_seed, err=0, num_threads=1,
    cheat_breakpoints=False, use_site_times=False):
    """
    Take the results of a simulation and return a sample data file, some reconstructed
    ancestors, a recombination rate array, a prefix to use for files, and
    the original tree sequence.
    
    If 'err' is 0, we do not inject any errors into the haplotypes. Otherwise
    we add empirical sequencing error and ancestral allele polarity error
    
    If "cheat_recombination" is True, multiply the recombination_rate for known
    recombination locations from the simulation by 20

    If "use_site_times" is True, use the times     
    """
    plain_samples = tsinfer.SampleData.from_tree_sequence(
        ts, use_times=use_site_times)
    if cheat_breakpoints:
        prefix += "cheat_breakpoints"
        logger.info("Cheating by using known breakpoints")
    if use_site_times:
        prefix += "use_times"
        logger.info("Cheating by using known times")
    if err == 0:
        # Save the samples file by copying
        sd = plain_samples.copy(path=prefix + ".samples")
    else:
        logger.info("Adding error")
        prefix += f"_ae{err}_seed{random_seed}"
        error_file = add_errors(
            plain_samples,
            err,
            random_seed=random_seed)
        sd = error_file.copy(path=prefix+".samples")
        if use_site_times:
            # Sites that were originally singletons have time 0, but could have been
            # converted to inference sites when adding error. Give these a nonzero time
            sites_time = sd.sites_time
            sites_time[sites_time == 0] = np.min(sites_time[sites_time > 0])/1000.0
            sd.sites_time[:] = sites_time
    sd.finalise()


    anc = tsinfer.generate_ancestors(
        sd,
        num_threads=num_threads,
        path=prefix+".ancestors",
    )
    logger.info("GA done")

    inference_pos = anc.sites_position[:]

    rho = np.diff(anc.sites_position[:])/sd.sequence_length
    rho = np.concatenate(([0.0], rho))
    if cheat_breakpoints:
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
    "sample_file, anc_file, rec_rate, ma_mis_rate, ms_mis_rate, precision, num_threads, "
    "kc_polymax, seed, error"
)

Results = collections.namedtuple(
    "Results",
    "n, abs_ma_mis, abs_ms_mis, rel_ma_mis, rel_ms_mis, precision, edges, muts, "
    "num_trees, kc_polymax, kc_poly, kc_split, arity_mean, arity_var, "
    "seed, error, proc_time, ts_size, ts_path"
)

    
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
    inferred_anc_ts.dump(inf_prefix + ".atrees")
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
    inferred_ts.dump(ts_path)
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
            random_seed=params.seed)
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
        error=params.error,
        precision=precision,
        edges=inferred_ts.num_edges,
        muts=inferred_ts.num_mutations,
        num_trees=inferred_ts.num_trees,
        kc_polymax=params.kc_polymax,
        kc_poly=kc_poly,
        kc_split=kc_split,
        arity_mean=arity_mean,
        arity_var=arity_var,
        seed=params.seed,
        proc_time=process_time,
        ts_size=os.path.getsize(ts_path),
        ts_path=ts_path)

def print_header(file):
    print("\t".join(Results._fields), file=file, flush=True)


def run_replicate(rep, args, header=True):
    """
    The main function that runs a parameter set
    """
    seed = rep+args.random_seed
    if len(args.precision) == 0:
        precision = [None]
    else:
        precision = args.precision
    nt = 2 if args.num_threads is None else args.num_threads
    kc_polymax = None
    if args.sample_file is None:
        logger.debug("Simulating human chromosome")
        sim = simulate_human(seed)
        sample_file, anc_file, rho, prefix, ts = setup_simulation(
            *sim,
            random_seed=seed,
            err=args.error,
            num_threads=nt,
            cheat_breakpoints=args.cheat_breakpoints,
            use_site_times=args.use_site_times,
        )
    else:
        logger.debug("Using provided sample data file")
        sample_file, anc_file, rho, prefix, ts = setup_sample_file(args, num_threads=nt)
    if ts is not None:
        ts.dump(prefix + ".trees")
        star_tree = tskit.Tree.unrank((0,0),ts.num_samples, span=ts.sequence_length)
        kc_polymax = ts.simplify().kc_distance(star_tree.tree_sequence)
    param_iter = (
        Params(sample_file, anc_file, rho, rma, rms, p, nt, kc_polymax, seed, args.error)
            for rms in args.match_samples_mismatch
                for rma in args.match_ancestors_mismatch
                    for p in precision)
    results_filename = prefix + ".results"
    with open(results_filename, "wt") as file:
        if header:
            print_header(file)
        if args.num_processes < 2:
            for p in param_iter:
                result=run(p)
                print("\t".join(str(r) for r in result), file=file, flush=True)
        else:
            with multiprocessing.Pool(args.num_processes) as pool:
                for result in pool.imap_unordered(run, param_iter):
                    # Save to a results file.
                    # NB this can be pasted into R and plotted using
                    # d <- read.table(stdin(), header=T)
                    # d$rel_ma <- factor(d$ma_mis / d$ms_mis)
                    # ggplot() + geom_line(data = d, aes(x = ms_mis, y = edges+muts, color = rel_ma)) + scale_x_continuous(trans='log10')
                    print("\t".join(str(r) for r in result), file=file, flush=True)
    logger.info(f"Results saved to {results_filename}")
    return results_filename

if __name__ == "__main__":


    # Set up the range of params for multiprocessing
    default_relative_match_samples_mismatch = np.array(
        [1e4, 1e3, 1e2, 10, 5, 2, 1, 0.5, 0.1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5])
    default_relative_match_ancestors_mismatch = np.array(
        [10, 5, 2, 1, 0.5, 0.1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-5])

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
    parser.add_argument("-A", "--match_ancestors_mismatch", nargs='*', type=float,
        default=default_relative_match_ancestors_mismatch,
        help = (
            "A list of values for the relative match_ancestors mismatch rate."
            "The rate is relative to the median recombination rate between sites")
    )
    parser.add_argument("-S", "--match_samples_mismatch", nargs='*', type=float,
        default=default_relative_match_samples_mismatch,
        help =
            "A list of values for the relative match_samples mismatch rate. "
            "The rate is relative to the median recombination rate between sites"
    )
    parser.add_argument("-T", "--use_site_times", action='store_true',
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
    parser.add_argument("-t", "--num_threads", type=int, default=None,
        help=
            "The number of threads to use in each inference subprocess. "
            "Normally, "
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
        filenames.append(run_replicate(rep, args, header=not multiple_replicates))
        print(filenames)
    if multiple_replicates:
        # Combine the replicates files together
        same_prefix = ""
        for char in zip(*filenames):
            if len(set(char)) > 1:
                break
            same_prefix += char[0]
        final_filename = same_prefix + ".results"
        if len(same_prefix) == 0 or final_filename in filenames:
            raise ValueError(
                f"{final_filename} invalid. Original results accessible in {filenames}")
        with open(final_filename, "wt") as final_file:
            print_header(final_file)
            for result_filename in filenames:
                with open(result_filename, "rt") as results_file:
                    for line in results_file:
                        final_file.write(line)
        logger.info(f"Final results integrated into {final_filename}")
