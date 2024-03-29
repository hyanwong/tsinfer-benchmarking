## This contains a recombination map for chr20 as produced below, but with all 0 values for
recombination rate (except the last) replaced by 0.001.


## Original README

This folder contains a build GRCh38 PLINK-format genetic map created by Xiaowen Tian
(tianx3@uw.edu).

The map was generated by lifting the build GRCh37 genetic map created by Adam Auton
(http://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh37.map.zip).

The conversion from build 37 to build 38 was performed using liftMap_b37tob38.py
(https://genome.sph.umich.edu/wiki/LiftOver#Lift_PLINK_format) which employs the
UCSC liftover tool to convert physical coordinates from GRCh37 to GRCh38.

After applying the liftover tool, SNPs with inconsistent order in GRCh37 and GRCh38
were removed using the following method:

1) Make two copies of the list of SNPs that were assigned GRCh38 positions by liftover.

2) Order the first list by GRCh38 position, and order the second list by GRCh37 position.

3) Calculate the difference in GRCh38 position of each pair of SNPs having the same rank
(i.e. list index) in the two lists.

4) Find the rank with the largest difference in step 3.  If there are multiple ranks with
the largest difference, select the first one. Select the SNP with this rank in the list
ordered by GRCh37 position, and remove this SNP from both lists.

5) Repeat steps 3 and 4 until all SNPs have the same order in both lists.

## LocusZoom Modifications

We modified the output to follow a format similar to LocusZoom's recomb rates, and
also included the calculated recombination rate.

`genetic_map_GRCh38_merged.tab` is the result of merging all genetic map files and adding in the
recombination rate calculated from the genetic map positions.

The file can be inserted into a locuszoom sqlite database file:

```bash
python2 bin/dbmeister.py --db data/database/locuszoom_hg38.db --recomb_rate genetic_map_GRCh38_merged.tab
```

If bugs or problems are found with this data, please file an issue at https://github.com/statgen/locuszoom-standalone
or comment on the original issue https://github.com/statgen/locuszoom-standalone/issues/1 in the same repository.
