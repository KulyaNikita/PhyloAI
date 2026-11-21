"""Show a Newick tree in an interactive window (`phyloai vis-tree`). Needs ete3."""
import argparse


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("tree", help="Newick tree file (extended Newick; internal node names are kept)")
    args = parser.parse_args(argv)
    try:
        from ete3 import Tree, TreeStyle
    except ImportError:
        parser.error("this command needs ete3: conda install -c conda-forge ete3, or pip install 'phyloai[extras]'")

    # Load extended Newick tree (format=1 keeps internal node names)
    t = Tree(args.tree, format=1)

    # Optional: style nodes so internal names are clearly visible
    def layout(node):
        if not node.is_leaf():
            node.img_style["size"] = 6  # make internal nodes visible

    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.layout_fn = layout
    ts.show_branch_support = True  # if support values exist

    # Show tree (interactive window)
    t.show(tree_style=ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
