from ete3 import Tree, TreeStyle, NodeStyle
import sys

tree_name = sys.argv[1]

# Load extended Newick tree (format=1 keeps internal node names)
t = Tree(tree_name, format=1)

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