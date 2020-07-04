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

import pandas as pd
import msprime
import tskit
import numpy as np
import stdpopsim #  Requires a version of msprime which allows gene conversion
import tsinfer

def simulate_human(random_seed=123, each_pop_n=100):
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
        ts.keep_intervals([[int(l * 3/20), int(l * 6/20)]]).trim(),
        f"data/OOA_sim_seed{random_seed}")

def test_sim(seed):
    ts = msprime.simulate(
        10,
        length=1000,
        mutation_rate=1e-2,
        recombination_rate=1e-2,
        random_seed=seed)
    return ts, f"data/test_sim{seed}"

def make_seq_errors_genotype_model(g, error_probs):
    """
    Given an empirically estimated error probability matrix, resample for a particular
    variant. Determine variant frequency and true genotype (g0, g1, or g2),
    then return observed genotype based on row in error_probs with nearest
    frequency. Treat each pair of alleles as a diploid individual.
    """
    m = g.shape[0]
    frequency = np.sum(g) / m
    closest_row = (error_probs['freq']-frequency).abs().argsort()[:1]
    closest_freq = error_probs.iloc[closest_row]

    w = np.copy(g)
    
    # Make diploid (iterate each pair of alleles)
    genos = np.reshape(w,(-1,2))

    # Record the true genotypes (0,0=>0; 1,0=>1; 0,1=>2, 1,1=>3)
    count = np.sum(np.array([1,2]) * genos,axis=1)
    
    base_genotypes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    
    genos[count==0,:]=base_genotypes[
        np.random.choice(4,sum(count==0), p=closest_freq[['p00', 'p01','p01', 'p02']].values[0]*[1,0.5,0.5,1]),:]
    genos[count==1,:]=base_genotypes[[0,1,3],:][
        np.random.choice(3,sum(count==1), p=closest_freq[['p10', 'p11', 'p12']].values[0]),:]
    genos[count==2,:]=base_genotypes[[0,2,3],:][
        np.random.choice(3,sum(count==2), p=closest_freq[['p10', 'p11', 'p12']].values[0]),:]
    genos[count==3,:]=base_genotypes[
        np.random.choice(4,sum(count==3), p=closest_freq[['p20', 'p21', 'p21', 'p22']].values[0]*[1,0.5,0.5,1]),:]

    return(np.reshape(genos,-1))

    
def add_errors(sample_data, ancestral_allele_error=0, random_seed=None, **kwargs):
    if random_seed is not None:
        np.random.seed(random_seed)
    if sample_data.num_samples % 2 != 0:
        raise ValueError("Must have an even number of samples to inject error")
    error_probs = pd.read_csv("data/EmpiricalErrorPlatinum1000G.csv")
    n_variants = 0
    aa_error_by_site = np.zeros(sample_data.num_sites, dtype=np.bool)
    if ancestral_allele_error > 0:
        assert ancestral_allele_error <= 1
        n_bad_sites = round(ancestral_allele_error*sample_data.num_sites)
        # This gives *exactly* a proportion aa_error or bad sites
        # NB - to to this probabilitistically, use np.binomial(1, e, ts.num_sites)
        aa_error_by_site[0:n_bad_sites] = True
        np.random.shuffle(aa_error_by_site)
    new_sd = sample_data.copy(**kwargs)
    genotypes = new_sd.data["sites/genotypes"][:]  # Could be big
    alleles = new_sd.data["sites/alleles"][:]
    inference = new_sd.data["sites/inference"][:]
    
    for i, (ancestral_allele_error, v) in enumerate(zip(
            aa_error_by_site, sample_data.variants())):
        if ancestral_allele_error and len(v.site.alleles)==2:
            genotypes[i, :] = 1-v.genotypes
            alleles[i] = list(reversed(alleles[i]))
        genotypes[i, :] = make_seq_errors_genotype_model(
            genotypes[i, :], error_probs)
        if np.all(genotypes[i, :] == 1) or np.sum(genotypes[i, :]) < 2: 
            inference[i] = False
        else:
            inference[i] = True
    new_sd.data["sites/genotypes"][:] = genotypes
    new_sd.data["sites/alleles"][:] = alleles
    new_sd.data["sites/inference"][:] = inference
    return new_sd
            

def physical_to_genetic(recombination_map, input_physical_positions):
    map_pos = recombination_map.get_positions()
    map_rates = recombination_map.get_rates()
    map_genetic_positions = np.insert(np.cumsum(np.diff(map_pos) * map_rates[:-1]), 0, 0)
    return np.interp(input_physical_positions, map_pos, map_genetic_positions)


def setup_simulation(ts, prefix=None, random_seed=None, cheat_recombination=False, err=0):
    """
    Take the results of a simulation and return a sample data file, the
    corresponding recombination rate array, a prefix to use for files, and
    the original tree sequence.
    
    If "cheat_recombination" is true, multiply the recombination_rate for known
    recombination locations from the simulation by 20
    
    If 'err' is 0, we do not inject any errors into the haplotypes. Otherwise
    we add empirical sequencing error and ancestral allele polarity error
    
    """
    plain_samples = tsinfer.SampleData.from_tree_sequence(
        ts, use_times=False)
    if cheat_recombination and prefix is not None:
        prefix += "cheat"
    if err == 0:
        sd = plain_samples.copy(path=None if prefix is None else prefix+".samples")
    else:
        if prefix is not None:
            prefix += f"_ae{err}"
        sd = add_errors(
            plain_samples,
            err,
            random_seed=random_seed,
            path=None if prefix is None else prefix+".samples")
    sd.finalise()
    rho = np.diff(sd.sites_position[:][sd.sites_inference])/sd.sequence_length
    rho = np.concatenate(([0.0], rho))
    if cheat_recombination:
        breakpoint_positions = np.array(list(ts.breakpoints()))
        inference_positions = sd.sites_position[:][sd.sites_inference[:] == 1]
        breakpoints = np.searchsorted(inference_positions, breakpoint_positions)
        # Any after the last inference position must be junked
        # (those before the first inference position make no difference)
        breakpoints = breakpoints[breakpoints != len(rho)]
        rho[breakpoints] *= 20
    return sd, rho, prefix, ts

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
    else:
        inference_distances = sd.sites_position[:][sd.sites_inference]
        d = np.diff(inference_distances)/sd.sequence_length
    rho = np.concatenate(([0.0], d))
        
    if np.any(d==0):
        w = np.where(d==0)
        raise ValueError("Zero recombination rates at", w, inference_pos[w])

    return sd, rho, filename[:-len(".samples")], None


Params = collections.namedtuple(
    "Params",
    "sample_data, rec_rate, cutoff_power, trim_oldest, precision, num_threads")

Results = collections.namedtuple(
    "Results",
    "abs_ma_mis, abs_ms_mis, rel_ma_mis, rel_ms_mis, cutoff_power, trim_oldest, "
    "precision, edges, muts, num_trees, "
    "kc, mean_node_children, var_node_children, process_time, ts_size, ts_path")

    
def run(params):
    """
    Run a single inference, with the specified rates
    """
    rho = params.rec_rate[1:]
    base_rec_prob = np.quantile(rho, 0.5)
    ma_mis_rate = ms_mis_rate = 1.0
    if params.precision is None:
        # Smallest recombination rate
        min_rho = int(np.ceil(-np.min(np.log10(rho))))
        # Smallest mean 
        av_min = int(np.ceil(-np.log10(
            min(1, ma_mis_rate, ms_mis_rate) * base_rec_prob)))
        precision = max(min_rho, av_min) + 3
    else:
        precision = params.precision
    ma_mis = base_rec_prob * ma_mis_rate
    ms_mis = base_rec_prob * ms_mis_rate
    print(
        f"Starting {params.cutoff_power}, trim_oldest={params.trim_oldest}",
        f"with base rho {base_rec_prob:.5g}",
        f"(mean {np.mean(rho):.4g} median {np.quantile(rho, 0.5):.4g}",
        f"min {np.min(rho):.4g}, 2.5% quantile {np.quantile(rho, 0.025):.4g})",
        f"precision {precision}")
    prefix = None
    if params.sample_data.path is not None:
        assert params.sample_data.path.endswith(".samples")
        prefix = params.sample_data.path[0:-len(".samples")]
        inf_prefix = "{}_rma{}_rms{}_N{}_{}_p{}".format(
            prefix,
            ma_mis_rate,
            ms_mis_rate,
            params.cutoff_power,
            "trim" if params.trim_oldest else "norm",
            precision)
    start_time = time.process_time()
    anc = tsinfer.generate_ancestors(
        params.sample_data,
        cutoff_power=params.cutoff_power,
        trim_oldest=params.trim_oldest,
        num_threads=params.num_threads,
        path=None if inf_prefix is None else inf_prefix + ".ancestors",
    )
    print(f"GA done (rel_ma_mis:{ma_mis_rate}, rel_ms_mis:{ms_mis_rate})")
    inferred_anc_ts = tsinfer.match_ancestors(
        params.sample_data,
        anc,
        num_threads=params.num_threads,
        precision=precision,
        recombination_rate=params.rec_rate,
        mismatch_rate=ma_mis,
    )
    inferred_anc_ts.dump(path=inf_prefix + ".atrees")
    print(f"MA done: abs_ma_mis rate = {ma_mis}")
    inferred_ts = tsinfer.match_samples(
        params.sample_data,
        inferred_anc_ts,
        num_threads=params.num_threads,
        precision=precision,
        recombination_rate=params.rec_rate,
        mismatch_rate=ms_mis)
    process_time = time.process_time() - start_time
    ts_path = inf_prefix + ".trees"
    inferred_ts.dump(path=ts_path)
    print(f"MS done: abs_ms_mis rate = {ms_mis}")
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
        rel_ma_mis=ma_mis_rate,
        rel_ms_mis=ms_mis_rate,
        cutoff_power=params.cutoff_power,
        trim_oldest=params.trim_oldest,
        precision=precision,
        edges=inferred_ts.num_edges,
        muts=inferred_ts.num_mutations,
        num_trees=inferred_ts.num_trees,
        kc=kc,
        mean_node_children=nc_mean,
        var_node_children=nc_var,
        process_time=process_time,
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
    nt = 0 if args.num_threads is None else args.num_threads
    if args.sample_file is None:
        # Simulate
        sim = simulate_human(seed)
        samples, rho, prefix, ts = setup_simulation(
            *sim,
            random_seed=seed,
            cheat_recombination=args.cheat_breakpoints,
            err=args.error)
    else:
        samples, rho, prefix, ts = setup_sample_file(args)
    if ts is not None:
        ts.dump(prefix + ".trees")
    # Set up the range of params for multiprocessing
    cutoff_powers = np.array([np.inf, 6, 5, 4, 3, 2])
    trim_oldest = np.array([True, False])
    param_iter = (
        Params(samples, rho, c, t, p, nt)
            for c in cutoff_powers for t in trim_oldest for p in precision)
    with open(prefix + ".results", "wt") as file:
        print("\t".join(Results._fields), file=file, flush=True)
        with multiprocessing.Pool(40) as pool:
            for result in pool.imap_unordered(run, param_iter):
                # Save to a results file.
                # NB this can be pasted into R and plotted using
                # d <- read.table(stdin(), header=T)
                # d$rel_ma <- factor(d$ma_mis / d$ms_mis)
                # ggplot() + geom_line(data = d, aes(x = ms_mis, y = edges+muts, color = rel_ma)) + scale_x_continuous(trans='log10')
                print("\t".join(str(r) for r in result), file=file, flush=True)


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
    args = parser.parse_args()
    

    for rep in range(args.replicates):
        run_replicate(rep, args)
