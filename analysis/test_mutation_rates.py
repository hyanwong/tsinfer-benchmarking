"""
Test the quality of inference, measured by num edges + num mutations, and (if
using simulated data) the KC distance
"""
import os.path
import argparse
import collections
import multiprocessing

import msprime
import tskit
import numpy as np
import stdpopsim #  Requires a version of msprime which allows gene conversion
import tsinfer

def simulate_human(random_seed=123):
    species = stdpopsim.get_species("HomSap")
    contig = species.get_contig("chr20")
    r_map = contig.recombination_map
    model = species.get_demographic_model('OutOfAfrica_3G09')
    assert len(r_map.get_rates()) == 2  # Ensure a single rate over chr
    samples = model.get_samples(500, 500, 500)
    engine = stdpopsim.get_engine('msprime')
    ts = engine.simulate(
        model, contig, samples,
        gene_conversion_rate=r_map.mean_recombination_rate * 10,
        gene_conversion_track_length=300,
        random_seed=random_seed)
    l = ts.sequence_length
    # cut down ts for speed
    return (
        ts.keep_intervals([[int(l * 16/20), int(l * 17/20)]]).trim(),
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

    
def add_errors(sample_data, ancestral_allele_error=0):
    n_variants = 0
    aa_error_by_site = np.zeros(sample_data.num_sites, dtype=np.bool)
    if ancestral_allele_error > 0:
        assert ancestral_allele_error <= 1
        n_bad_sites = round(aa_error*sample_data.num_sites)
        logging.info("Adding ancestral allele polarity error: {}% ({}/{} sites) used in file {}"
            .format(aa_error * 100, n_bad_sites, ts.num_sites, fn))
        # This gives *exactly* a proportion aa_error or bad sites
        # NB - to to this probabilitistically, use np.binomial(1, e, ts.num_sites)
        aa_error_by_site[0:n_bad_sites] = True
        np.random.shuffle(aa_error_by_site)
    new_sd = sample_data.copy()
    for i, (ancestral_allele_error, g) in enumerate(zip(
            aa_error_by_site, sample_data.sites_genotypes[:])):
        if ancestral_allele_error:
            new_sd.data["sites/genotypes"][i, :] = g

def physical_to_genetic(recombination_map, input_physical_positions):
    map_pos = recombination_map.get_positions()
    map_rates = recombination_map.get_rates()
    map_genetic_positions = np.insert(np.cumsum(np.diff(map_pos) * map_rates[:-1]), 0, 0)
    return np.interp(input_physical_positions, map_pos, map_genetic_positions)


def setup_simulation(ts, prefix, cheat_recombination=False):
    """
    Take the results of a simulation and return a sample data file, the
    corresponding recombination rate array, a prefix to use for files, and
    the original tree sequence.
    
    If "cheat_recombination" is true, multiply the recombination_rate for known
    recombination locations from the simulation by 20
    
    """
    plain_samples = tsinfer.SampleData.from_tree_sequence(
        ts, use_times=False)
    # could inject error in here e.g.
    # sample_data = plain_samples.add_errors(..., path=***)
    sd = plain_samples.copy(path=prefix+".samples")
    sd.finalise()
    rho = np.diff(sd.sites_position[:][sd.sites_inference])/sd.sequence_length
    rho = np.concatenate(([0.0], rho))
    if cheat_recombination:
        breakpoint_positions = np.array(list(ts.breakpoints()))[1:-1]
        inference_positions = sd.sites_position[:][sd.sites_inference[:] == 1]
        breakpoints = np.searchsorted(inference_positions, breakpoint_positions)
        rho[breakpoints] *= 20
    return sd, rho, prefix, ts

def setup_TGP_chr20(prefix):
    """
    Return a Thousand Genomes Project sample data file, the
    corresponding recombination rate array, a prefix to use for files, and None
    """
    sd = tsinfer.load(prefix + ".samples")
    map = stdpopsim.get_species("HomSap").get_genetic_map(id="HapMapII_GRCh37")
    if not map.is_cached():
        map.download()
    chr20_map = map.get_chromosome_map("chr20")
    inference_distances = physical_to_genetic(
        chr20_map,
        sd.sites_position[:][sd.sites_inference])
    rho = np.concatenate(([0.0], np.diff(inference_distances)))
    return sd, rho, prefix, None


Params = collections.namedtuple(
    "Params",
    "sample_data, rec_rate, ma_mut_rate, ms_mut_rate, precision, n_threads")

Results = collections.namedtuple(
    "Results",
    "ts_path, ts_size, ma_mut, ms_mut, edges, muts, kc")

    
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
    )
    print(f"GA done (ma_mut:{params.ma_mut_rate} ms_mut{params.ms_mut_rate})")
    inferred_anc_ts = tsinfer.match_ancestors(
        params.sample_data,
        anc,
        num_threads=params.n_threads,
        precision=params.precision,
        recombination_rate=params.rec_rate,
        mutation_rate=base_rec_prob * params.ma_mut_rate)
    inferred_anc_ts.dump(path=inf_prefix + ".ancestors.trees")
    print(f"MA done (ma_mut:{params.ma_mut_rate} ms_mut{params.ms_mut_rate})")
    inferred_ts = tsinfer.match_samples(
        params.sample_data,
        inferred_anc_ts,
        num_threads=params.n_threads,
        precision=params.precision,
        recombination_rate=params.rec_rate,
        mutation_rate=base_rec_prob * params.ms_mut_rate)
    ts_path = inf_prefix + ".trees"
    inferred_ts.dump(path=ts_path)
    print(f"MS done (ma_mut:{params.ma_mut_rate} ms_mut{params.ms_mut_rate})")
    try:
        kc = inferred_ts.simplify().kc_distance(tskit.load(prefix+".trees"))
    except tskit.exceptions.FileFormatError:
        kc = None
    return Results(
        ts_path=ts_path,
        ts_size=os.path.getsize(ts_path),
        ma_mut=params.ma_mut_rate,
        ms_mut=params.ms_mut_rate,
        edges=inferred_ts.num_edges,
        muts=inferred_ts.num_mutations,
        kc=kc)

def run_replicate(seed):
    """
    The main function that runs a parameter set
    """
    samples, rho, prefix, ts = setup_simulation(*simulate_human(seed))
    #samples, rho, prefix, ts = setup_simulation(*simulate_human(seed), cheat_recombination=True)
    #samples, rho, prefix, ts = setup_TGP_chr20("data/1kg_chr20_small")
    prefix += "cheat"
    
    if ts is not None:
        ts.dump(prefix + ".trees")
    # Set up the range of params for multiprocessing
    errs = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
    rel_muts = np.array([2, 1, 0.5, 0.1, 0.01, 0.001])
    param_iter = (
        Params(samples, rho, m*e, e, 11, 2) for e in errs for m in rel_muts)
    with open(prefix + ".results", "wt") as file:
        print("\t".join(Results._fields), file=file, flush=True)
        with multiprocessing.Pool(40) as pool:
            for result in pool.imap_unordered(run, param_iter):
                # Save to a results file.
                # NB this can be pasted into R and plotted using
                # d <- read.table(stdin(), header=T)
                # d$rel_ma <- factor(d$ma_mut / d$ms_mut)
                # ggplot() + geom_line(data = d, aes(x = ms_mut, y = edges+muts, color = rel_ma)) + scale_x_continuous(trans='log10')
                print("\t".join(str(r) for r in result), file=file, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--replicates", type=int, default=1)
    parser.add_argument("-s", "--random_seed", type=int, default=123)
    parser.add_argument("-e", "--sequencing_error", type=float, default=0,
        help="Add some sequencing error to the haplotypes before inferring")
    args = parser.parse_args()
    

    for rep in range(args.replicates):
        run_replicate(rep+args.random_seed)
