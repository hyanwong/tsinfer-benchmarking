import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Quick hack: add RF distances from files into a CSV file')
    parser.add_argument("csv_file")
    parser.add_argument("-c", "--column_containing_paths", default=-1,
        help=
            "The column in the CSV containing the paths (can be negative, for pos from "
            "end) '.RF' and '.split.RF' will be appended to the value in this column."
    )
    args = parser.parse_args()
    with open(args.csv_file, "rt") as f:
        new_fields = ["", ""]
        for line_num, line in enumerate(f):
            fields = line.strip().split(",")
            try:
                with open(fields[-1]+".RF", "rt") as rf:
                    new_fields[0] = rf.readline().strip()
            except:
                new_fields[0] = "" if line_num>0 else "RF"
            try:
                with open(fields[-1]+".split.RF", "rt") as rf:
                    new_fields[1] = rf.readline().strip()
            except:
                new_fields[1] = "" if line_num>0 else "RFsplit"
            print(",".join(fields + new_fields))
