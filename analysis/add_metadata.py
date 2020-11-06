# Read a .results file and add the results in it to the metadata in each tree sequence
import os
import csv
import json

import tskit

filename = "data/OutOfAfrica_3G09_sim_n18_seed1.results"


with open(filename) as csvfile:
     reader = csv.DictReader(csvfile, delimiter='\t')
     for row in reader:
        if 'ts_path' in row:
            ts = tskit.load(row['ts_path'])
            tables = ts.dump_tables()
            tables.metadata = json.dumps(row).encode()
            tables.tree_sequence().dump(row['ts_path'])  # Overwrite stored ts with the one with metadata
