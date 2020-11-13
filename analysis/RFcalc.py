"""
Use Dendropy to calculate the unweighted (i.e. topology only) Robinson Fould distance
between 2 tree sequences. This can be very slow. It can be run in parallel along the
lines of:

for f in data/mysim.*_rma*.trees; do python3 RFcalc.py data/mysim.trees $f -s 123 -v -o 100 & done

and RF distances can be extracted using something like

for f in data/mysim.*_rma*.trees.*RF; do echo -n $f | cat - $f | sed -r 's/.*rma(.*)_rms(.*)_p.*RF(.*)/\1\t\2\t\3/' ; done


"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # limit number of threads so we can run loads of these in parallel
import collections
import json
import tempfile
import logging
import argparse
import itertools
import time
import multiprocessing

import tskit
from tskit import provenance
import dendropy
import numpy as np



def randomly_split_polytomies(
    self,
    *,
    epsilon=None,
    squash_edges=True,
    record_provenance=True,
    random_seed=None,
):
    """
    Modifies the table collection in place, adding extra nodes and edges
    so that any node with greater than 2 children (i.e. a multifurcation
    or "polytomy") is resolved into successive bifurcations. This is identical
    to :meth:`TreeSequence.randomly_split_polytomies` but acts *in place* to
    alter the data in this :class:`TableCollection`. Please see
    :meth:`TreeSequence.randomly_split_polytomies` for a fuller description,
    and details of parameters.
    """
    if epsilon is None:
        epsilon = 1e-10
    rng = np.random.default_rng(seed=random_seed)

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
        np.logical_not(tskit.is_unknown_time(self.mutations.time))
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

        oldest_mutation_for_node = {}
        if check_mutations and prev_tree is not None:
             # It would also help if mutations were sorted such that all mutations
            # above the same node appeared consecutively, with oldest first.
            for site in prev_tree.sites():
                for mutation in site.mutations:
                    if not tskit.is_unknown_time(mutation.time):
                        if mutation.node in oldest_mutation_for_node:
                            oldest_mutation_for_node[mutation.node] = max(
                                oldest_mutation_for_node[mutation.node], mutation.time
                            )
                        else:
                            oldest_mutation_for_node[mutation.node] = mutation.time
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
                        left=left,
                        right=pos,
                        child=new_edge[0],
                        parent=new_edge[1],
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

    if record_provenance:
        parameters = {"command": "randomly_split_polytomies"}
        self.provenances.add_row(
            record=json.dumps(provenance.get_provenance_dict(parameters))
        )

tskit.TableCollection.randomly_split_polytomies = randomly_split_polytomies

def randomly_split_polytomies(
    self,
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

    Polytomies are split per node, not per tree, so that if an identical
    polytomy spans several trees, it will be randomly resolved into a single
    set of bifurcating splits. However, if on shifting to a new genomic
    region, the children of node ``u`` change, either being added or removed,
    an entirely new random resolution of the node will be applied to that
    region.

    Because a tree sequence requires that
    :ref:`parents be older than children<sec_valid_tree_sequence_requirements>`,
    the newly added nodes are inserted at a time fractionally younger than
    than the time of node ``u``. This can be controlled by the ``epsilon``
    parameter.

    :param epsilon: The maximum time between each newly inserted node. For a
        given polytomy, if possible, the ``n-2`` extra
        nodes are inserted at time ``epsilon`` apart from each other and from the
        time of node ``u``. By default, this is set to a very small time value
        (:math:`1e-10`). However, if there is a child node or a mutation above
        a child node whose time is very close to the time of ``u``, ``epsilon``
        may not be small enough. In this case, a smaller time interval is used
        so that the tables will still encode a valid tree sequence.
    :param bool squash_edges: If True (default), run :meth:`.squash()` at the
        end of the process. This can help to reduce the total number of extra
        edges produced.
    :param bool record_provenance: If True, add details of this operation to the
        provenance information of the returned tree sequence. (Default: True).
    :param int random_seed: The random seed. If this is None, a random seed will
        be automatically generated. Valid random seeds must be between 1 and
        :math:`2^32 − 1`.
    :return: A new tree sequence with polytomies split into random bifurcations.
    :rtype: .TreeSequence
    """
    tables = self.dump_tables()
    tables.randomly_split_polytomies(
        epsilon=epsilon,
        squash_edges=squash_edges,
        record_provenance=record_provenance,
        random_seed=random_seed,
    )
    return tables.tree_sequence()

tskit.TreeSequence.randomly_split_polytomies = randomly_split_polytomies



def node_encodings(ts):
    """
    Associate a binary-vector identifier to each node according to
    the identities of the leaves descended from the node.
    """
    # XXX: assumes samples == leaves
    n = ts.num_samples
    W = np.eye(n)
    f = lambda x: x
    return ts.general_stat(
        W,
        f,
        output_dim=n,
        mode="node",
        windows="trees",
        strict=False,
        polarised=True,
        span_normalise=True,
    ).astype(np.int8)


def branches_pnorm(bl1, bl2, p):
    """
    Robinson-Foulds family of L^p distances.
    """
    clades1 = bl1.keys()
    clades2 = bl2.keys()
    v = np.fromiter(
        itertools.chain(
            (bl1[cl] for cl in clades1 - clades2),
            (bl2[cl] for cl in clades2 - clades1),
            (bl1[cl] - bl2[cl] for cl in clades1 & clades2),
        ),
        dtype=float,
    )
    return np.linalg.norm(v, ord=p)


def branches_l0(bl1, bl2):
    """
    L1 distance (uinweighted Robinson-Foulds).
    """
    return branches_pnorm(bl1, bl2, 0)


def branches_l1(bl1, bl2):
    """
    L1 distance (weighted Robinson-Foulds).
    """
    return branches_pnorm(bl1, bl2, 1)


def branches_l2(bl1, bl2):
    """
    L2 (Euclidean) distance.
    This is the shortest path between two points in an embedding of
    BHV treespace into Euclidean space.  Note that this path is
    frequently outside BHV treespace.
    Amenta et al. (2007) lower bound on BHV treespace geodesic.
    http://doi.org/10.1016/j.ipl.2007.02.008
    See also the "Branch Score" or BLD (Branch Length Distance),
    Kuhner & Felsenstein (1994).
    http://doi.org/10.1093/oxfordjournals.molbev.a040126
    """
    return branches_pnorm(bl1, bl2, 2)


def branches_BHVub(bl1, bl2):
    """
    Distance through a strict consensus tree.
    Amenta et al. (2007) upper bound on BHV treespace geodesic.
    http://doi.org/10.1016/j.ipl.2007.02.008
    """
    clades1 = bl1.keys()
    clades2 = bl2.keys()
    u1 = np.fromiter((bl1[cl] for cl in clades1 - clades2), dtype=float)
    u2 = np.fromiter((bl2[cl] for cl in clades2 - clades1), dtype=float)
    u3 = np.fromiter((bl1[cl] - bl2[cl] for cl in clades1 & clades2), dtype=float)
    v = [(np.linalg.norm(u1) + np.linalg.norm(u2)), np.linalg.norm(u3)]
    return np.linalg.norm(v)


def rf_distance(ts1, ts2, dist_func):
    """
    Robinson-Foulds family of distances, calculated between two tree sequences.
    """
    assert ts1.num_samples == ts1.num_samples
    assert ts1.sequence_length == ts2.sequence_length

    ed1 = ts1.edge_diffs()
    ed2 = ts2.edge_diffs()
    try:
        (interval1, edges_out1, edges_in1) = next(ed1)
        (interval2, edges_out2, edges_in2) = next(ed2)
    except StopIteration:
        return 0

    def branches(tree, node_enc):
        b = dict()
        for u, enc in enumerate(node_enc):
            branch_length = tree.branch_length(u)
            if branch_length:
                b[enc.tobytes()] = branch_length
        return b

    trees1 = ts1.trees()
    trees2 = ts2.trees()
    nodes1 = iter(np.packbits(node_encodings(ts1), axis=-1))
    nodes2 = iter(np.packbits(node_encodings(ts2), axis=-1))
    branches1 = branches(next(trees1), next(nodes1))
    branches2 = branches(next(trees2), next(nodes2))

    def overlap(it1, it2):
        if it1[1] < it2[0] or it2[1] < it1[0]:
            return -1
        lo = max(it1[0], it2[0])
        hi = min(it1[1], it2[1])
        return hi - lo

    d = 0
    while True:
        span = overlap(interval1, interval2)
        assert span > 0
        d += span * dist_func(branches1, branches2)
        x = interval2[1] - interval1[1]
        if x >= 0:
            # advance through ts1
            try:
                (interval1, edges_out1, edges_in1) = next(ed1)
            except StopIteration:
                break
            # TODO update dict based on edges in/out
            branches1 = branches(next(trees1), next(nodes1))
        if x <= 0:
            # advance through ts2
            try:
                (interval2, edges_out2, edges_in2) = next(ed2)
            except StopIteration:
                break
            # TODO update dict based on edges in/out
            branches2 = branches(next(trees2), next(nodes2))

    d /= ts1.sequence_length
    return d



def run(params):
    original_ts, inferred_ts, metric, random_seed, output_tot, keep = params
    if random_seed is not None:
        orig_ts = tskit.load(original_ts).simplify().randomly_split_polytomies(
            random_seed=random_seed)
        cmp_ts = tskit.load(inferred_ts).simplify().randomly_split_polytomies(
            random_seed=random_seed)
    else:
        orig_ts = tskit.load(original_ts).simplify()
        cmp_ts = tskit.load(inferred_ts).simplify()

    logging.info("Loaded initial tree sequences")
    assert orig_ts.sequence_length == cmp_ts.sequence_length
    seq_length = orig_ts.sequence_length

    if random_seed is None:
        suffix = "." + metric
    else:
        suffix = ".split." + metric

    filename = inferred_ts + suffix
    if keep and os.path.exists(filename):
        logging.warning(
            "'--keep_existing' specified & file '{filename}' already exists. Aborting.")
        filename = None

    if metric == "KC":
        stat = orig_ts.kc_distance(cmp_ts)

    elif metric == "RFts":
        logging.info(f"Running ts-specific RF code")
        stat = rf_distance(orig_ts, cmp_ts, branches_l0)

    elif metric == "RF":
        t_iter1 = orig_ts.trees()
        t_iter2 = cmp_ts.trees()
        stat = 0
        pos = 0
        end1 = 0
        end2 = 0
        
        start = time.time()
        taxon_namespace = dendropy.Tree.get(
            string=orig_ts.first().newick(precision=0),
            schema="newick",
            rooting="force-rooted").taxon_namespace
        logging.info(
            f"Loaded 1 out of {orig_ts.num_trees} trees in {time.time()-start} sec")
    
        
        while True:
            if pos == seq_length:
                break
            if pos >= end1:
                t1 = next(t_iter1)
                end1 = t1.interval[1]
                if pos > 0 and (t1.index % (orig_ts.num_trees // output_tot)) == 0:
                    logging.debug("For {}, {:.0f}% done, RF ~= {}"
                        .format(
                            inferred_ts,
                            t1.index / orig_ts.num_trees * 100,
                            rf_stat/pos,
                        )
                    )
                    # save temporarily, so we can get stats even if not completed
                    with open(filename, "wt") as stat:
                        print(rf_stat/pos, file=stat)
            if pos >= end2:
                t2 = next(t_iter2)
                end2 = t2.interval[1]
    
            span = min(end1, end2) - pos
            orig_tree = dendropy.Tree.get(
                string=t1.newick(precision=0),
                schema="newick",
                rooting="force-rooted",
                taxon_namespace=taxon_namespace,
            )
            cmp_tree = dendropy.Tree.get(
                string=t2.newick(precision=0),
                schema="newick",
                rooting="force-rooted",
                taxon_namespace=taxon_namespace,
            )
            stat += dendropy.calculate.treecompare.symmetric_difference(
                orig_tree, cmp_tree) * span
            pos = min(end1, end2)
        stat /= seq_length

    else:
        raise ValueError(f"Bad metric specified: {metric}")

    return filename, stat        


def main(args):
    with multiprocessing.Pool(args.num_processes) as pool:
        for filename, stat in pool.imap_unordered(run, zip(
            itertools.repeat(args.orig_ts),
            args.cmp_ts,
            itertools.repeat(args.metric),
            itertools.repeat(args.random_seed),
            itertools.repeat(args.output_tot),
            itertools.repeat(args.keep_existing),
        )):
            if filename is not None:
                with open(filename, "wt") as file:
                    print(stat, file=file)
                logging.info(f"Saved data into '{filename}': {args.metric} = {stat}")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate the rooted Robinson Foulds distance between 2 tree seqs')
    parser.add_argument('orig_ts')
    parser.add_argument(
        'cmp_ts', nargs="+",
        help=(
            "the list of tree sequence files to compare against the original."
            " The stat will be saved under each name with a suffix '.RFdist'"))
    parser.add_argument(
        '--random_seed',
        '-s',
        type=int,
        default=None,
        help="If given, randomly split polytomies before calculating the RF dist",
    )
    parser.add_argument("-p", "--num_processes", type=int, default=40,
        help=
            "The number of processors that can be pooled to parallelise over files"
    )
    parser.add_argument(
        '--output_tot',
        '-o',
        type=int,
        default=1,
        help=(
            "How many times to overwrite the output file during progress, to allow "
            "partially calculated stats to be used. Also determines the output progress "
            "if verbosity is >= 2."
            )
    )
    parser.add_argument("-k", "--keep_existing", action='store_true',
        help=
            "If the file already exists, skip the calc ")
    parser.add_argument('--metric', '-m', choices=["KC", "RFts", "RF"], default="RF", 
        help='which metric to calculate')
    parser.add_argument('--verbosity', '-v', action="count", default=0, 
        help='verbosity: output extra non-essential info')
    
    args = parser.parse_args()
    if args.verbosity==0:
        logging.basicConfig(level=logging.WARNING)
    elif args.verbosity==1:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    elif args.verbosity>=2:
        logging.basicConfig(level=logging.DEBUG)

    main(args)