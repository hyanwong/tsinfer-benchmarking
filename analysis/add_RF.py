import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Quick hack: add RF distances from files into a CSV file. The modified'
            'csv is output to stdout, so do e.g. `python add_RF.py > new_file.csv`'
        )
    )
    parser.add_argument("csv_file")
    parser.add_argument("-c", "--column_containing_paths", type=int, default=-1,
        help=(
            "The column in the CSV containing the paths (can be negative, for pos from "
            "end). Files with this path plus '.RF' or '.split.RF' will be searched for "
            "and opened to get data to add into the csv_file")
    )
    args = parser.parse_args()
    with open(args.csv_file, "rt") as f:
        new_fields = ["", ""]
        for line_num, line in enumerate(f):
            fields = line.strip().split(",")
            try:
                with open(fields[args.column_containing_paths]+".RFinfo", "rt") as rf:
                    new_fields[0] = rf.readline().strip()
            except:
                new_fields[0] = "" if line_num>0 else "RFinfo"
            try:
                with open(fields[args.column_containing_paths]+".split.RF", "rt") as rf:
                    new_fields[1] = rf.readline().strip()
            except:
                new_fields[1] = "" if line_num>0 else "RFsplit"
            # Add elements before the c'th one
            for f in new_fields:
                fields.insert(args.column_containing_paths, f)
            print(",".join(fields))
