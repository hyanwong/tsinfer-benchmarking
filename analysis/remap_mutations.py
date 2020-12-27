import argparse
import tskit
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)

def remapped_ts(ts, bad_ancestral_states=None, save_site_changes=None):
    """
    If you want to save a csv file of changed sites, provide save_site_changes as a path
    to a file
    """
    tables = ts.dump_tables()
    sites = tables.sites
    mutations = tables.mutations
    sites.clear()
    mutations.clear()
    v_iter = ts.variants()
    changed_states = 0
    corrected_AS = 0
    miscorrected_AS = 0
    if save_site_changes is not None:
        save_site_changes = open(save_site_changes, "wt")
        print("site_id,position,old_AS,new_AS", file=save_site_changes)
    for tree in ts.trees():
        for s in tree.sites():
            v = next(v_iter)
            new_anc_state, muts = tree.map_mutations(v.genotypes, v.alleles)
            site_id = sites.add_row(
                position=s.position,
                ancestral_state=new_anc_state,
                metadata=s.metadata,
            )
            mapping = {-1:-1}
            for i, m in enumerate(muts):
                mapping[i] = mutations.add_row(
                    site=site_id,
                    node=m.node,
                    time=m.time,
                    derived_state=m.derived_state,
                    parent=mapping[m.parent],
                    metadata=m.metadata)
            if s.ancestral_state != new_anc_state:                   
                changed_states += 1
                try:
                    if s.id in bad_ancestral_states:
                        # we might have corrected a bad AS
                        if bad_ancestral_states[s.id] == new_anc_state:
                            # Hurrah
                            corrected_AS += 1
                    else:
                        # we wrongly changed a good AS
                        miscorrected_AS += 1
                except TypeError:
                    # bad_ancestral_states is None: we didn't have an original ts
                    pass
                if save_site_changes is not None:
                    print(
                        f"{s.id},{s.position},{s.ancestral_state},{new_anc_state}",
                        file=save_site_changes,
                    )
                
    logger.info(
        f"{mutations.num_rows} new mutations vs {ts.num_mutations} old ones")
    if bad_ancestral_states is not None:
        logger.info(
            f"{corrected_AS}/{len(bad_ancestral_states)} bad ancestral states corrected "
            f"({miscorrected_AS}/{ts.num_sites-len(bad_ancestral_states)} miscorrected)"
        )
    if save_site_changes is not None:
        logger.info(
            f"{changed_states}/{ts.num_sites} ancestral states changed: "
            f"saved to {save_site_changes.name}")
    if 'user_data' in tables.metadata and 'muts' in tables.metadata['user_data']:
        metadata = tables.metadata.copy()
        user_meta = metadata['user_data']
        assert 'old_muts' not in user_meta
        assert 'old_ts_bytes' not in user_meta
        user_meta.update({
            'old_muts': user_meta['muts'],
            'muts':mutations.num_rows,
            'old_ts_bytes': user_meta['ts_bytes'],
            'ts_bytes':tables.nbytes,
        })
        tables.metadata = metadata
    return tables.tree_sequence()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Delete all mutations from a ts and re-place them via parsimony')
    parser.add_argument("ts")
    parser.add_argument("-s", "--save_site_changes", action="store_true",
        help="Should we save the changed sites to a file with the suffix _ASchanged.csv")
    parser.add_argument("-O", "--original_ts", default=None,
        help="The original tree sequence from which `ts` has been derived")
    parser.add_argument("-v", "--verbosity", action="count", default=0,
        help="Increase the verbosity")
    args = parser.parse_args()
    log_level = logging.WARN
    if args.verbosity > 0:
        log_level = logging.INFO
    if args.verbosity > 1:
        log_level = logging.DEBUG
    logger.setLevel(log_level)
    
    save_site_changes = None
    ts = tskit.load(args.ts)
    if args.save_site_changes:
        save_site_changes = args.ts.replace(".trees", "_ASchanged.csv")
    bad_ancestral_states = None
    if args.original_ts is not None:
        bad_ancestral_states = {}
        orig_ts = tskit.load(args.original_ts)
        if ((ts.sequence_length != orig_ts.sequence_length) or
            (ts.num_sites != orig_ts.num_sites) or
            (ts.num_samples != orig_ts.num_samples)
        ):
           raise ValueError(f"{args.ts} and {args.original_ts} describe different data")
        for orig_site, new_site in zip(orig_ts.sites(), ts.sites()):
            assert orig_site.position == new_site.position
            assert orig_site.id == new_site.id
            if orig_site.ancestral_state != new_site.ancestral_state:
                bad_ancestral_states[orig_site.id] = orig_site.ancestral_state
    new_ts = remapped_ts(ts, bad_ancestral_states, save_site_changes)
    new_ts.dump(args.ts.replace(".trees", ".remapped.trees"))
                                  