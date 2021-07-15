import argparse
import tskit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Quick hack: add an extra mutations column calculated by re-laying'
            ' mutations using parsimony. The modified'
            'csv is output to stdout, so do e.g. `python add_RF.py file.csv > new.csv`'
        )
    )
    parser.add_argument("csv_file")
    parser.add_argument("-c", "--column_containing_paths", default=-1,
        help=(
            "The column in the CSV containing the paths (can be negative, for pos from "
            "end). The value in this column is the path to the .trees file"
        )
    )
    args = parser.parse_args()
    with open(args.csv_file, "rt") as f:
        new_fields = ["", ""]
        for line_num, line in enumerate(f):
            fields = line.strip().split(",")
            try:
                ts = tskit.load(fields[args.column_containing_paths])
                tables = ts.dump_tables()
                tables.mutations.clear()
                parsimony_muts = 0
                tree_iter = ts.trees()
                tree = next(tree_iter)
                anc_states = []
                for v in ts.variants():
                    while v.site.position >= tree.interval.right:
                        tree = next(tree_iter)
                    anc_state, muts = tree.map_mutations(v.genotypes, v.alleles)
                    anc_states.append(anc_state)
                    for m in muts:
                        tables.mutations.append(
                            m.replace(parent=tskit.NULL, site=v.site.id))
                    parsimony_muts += len(muts)
                tables.compute_mutation_parents()
                tables.sites.packset_ancestral_state(anc_states)
                ts = tables.tree_sequence()
                new_fields[0] = str(parsimony_muts)
                new_fields[1] = str(ts.nbytes) 
            except FileNotFoundError:
                new_fields = ["", ""] if line_num>0 else ["parsimony_muts", "parsimony_nbytes"]
            # Add elements before the c'th one
            for f in new_fields:
                fields.insert(args.column_containing_paths, f)
            print(",".join(fields))
