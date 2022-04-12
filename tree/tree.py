"""Binary tree with decision tree semantics and ASCII visualization."""
import dot2tex as d2t
from pathlib import Path
import pydot
import subprocess


class Node:
    """A decision tree node."""

    def __init__(self, num_samples, num_samples_per_class, predicted_class, gini=0, feature_index=None, threshold=None,
                 feature_index_occurrences=None, balanced_split=None):
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.gini = gini
        self.feature_index = feature_index
        self.feature_index_occurrences = feature_index_occurrences  # number of occurrences for each feature

        self.is_redundant = None  # True if is a redundant node

        # for each feature, +1 if feature x_i appears in x_i > j,
        # -1 if x_i appears in x_i <= j, and
        # 2 if x_if appears in both <= j and > j,
        # 0 otherwise
        self.feature_index_occurrences_redundant = None

        self.threshold = threshold

        # Whether the node split is balanced in terms of cost (product of pairs and weight)
        self.balanced_split = balanced_split

        self.left = None
        self.right = None

    def get_node_depth(self):
        # Assert that get_explainability_metrics was previously run
        return sum(x for x in self.feature_index_occurrences)

    def get_node_explainability(self):
        # Path to node less the redundant nodes
        # Assert that get_explainability_metrics was previously run
        return sum(abs(x) for x in self.feature_index_occurrences_redundant)

    def get_next_feature_occurrences_redundant(self, idx, direction):
        if direction != 'right' and direction != 'left':
            raise ValueError("Value of direction not permitted: ", direction)

        new_feature_list = self.feature_index_occurrences_redundant.copy()
        if self.feature_index_occurrences_redundant[idx] == 0:
            if direction == 'right':
                new_feature_list[idx] = 1
            else:
                new_feature_list[idx] = -1
        elif self.feature_index_occurrences_redundant[idx] == 1:
            if direction == 'right':
                new_feature_list[idx] = 1
            else:
                new_feature_list[idx] = 2
        elif self.feature_index_occurrences_redundant[idx] == -1:
            if direction == 'right':
                new_feature_list[idx] = 2
            else:
                new_feature_list[idx] = -1
        elif self.feature_index_occurrences_redundant[idx] == 2:
            # Do nothing +2 is the maximum
            pass
        else:
            raise ValueError("Some value not permitted on feature index redundant")
        return new_feature_list

    def debug(self, feature_names, class_names, show_details):
        """Print an ASCII visualization of the tree."""
        lines, _, _, _ = self._debug_aux(
            feature_names, class_names, show_details, root=True
        )
        for line in lines:
            print(line)

    def _debug_aux(self, feature_names, class_names, show_details, root=False):
        # See https://stackoverflow.com/a/54074933/1143396 for similar code.
        is_leaf = not self.right
        if is_leaf:
            lines = [class_names[self.predicted_class]]
        else:
            lines = [
                "{} < {:.2f}".format(feature_names[self.feature_index], self.threshold)
            ]
        if show_details:
            lines += [
                "feature_occurrences = {}".format(self.feature_index_occurrences),
                "feat_redund_occurr = {}".format(self.feature_index_occurrences_redundant),
                "gini = {:.2f}".format(self.gini),
                "balanced_split = {}".format(self.balanced_split),
                "samples = {}".format(self.num_samples),
                str(self.num_samples_per_class),
            ]
        width = max(len(line) for line in lines)
        height = len(lines)
        if is_leaf:
            lines = ["║ {:^{width}} ║".format(line, width=width) for line in lines]
            lines.insert(0, "╔" + "═" * (width + 2) + "╗")
            lines.append("╚" + "═" * (width + 2) + "╝")
        else:
            lines = ["│ {:^{width}} │".format(line, width=width) for line in lines]
            lines.insert(0, "┌" + "─" * (width + 2) + "┐")
            lines.append("└" + "─" * (width + 2) + "┘")
            lines[-2] = "┤" + lines[-2][1:-1] + "├"
        width += 4  # for padding

        if is_leaf:
            middle = width // 2
            lines[0] = lines[0][:middle] + "╧" + lines[0][middle + 1:]
            return lines, width, height, middle

        # If not a leaf, must have two children.
        left, n, p, x = self.left._debug_aux(feature_names, class_names, show_details)
        right, m, q, y = self.right._debug_aux(feature_names, class_names, show_details)
        top_lines = [n * " " + line + m * " " for line in lines[:-2]]
        # fmt: off
        middle_line = x * " " + "┌" + (n - x - 1) * "─" + lines[-2] + y * "─" + "┐" + (m - y - 1) * " "
        bottom_line = x * " " + "│" + (n - x - 1) * " " + lines[-1] + y * " " + "│" + (m - y - 1) * " "
        # fmt: on
        if p < q:
            left += [n * " "] * (q - p)
        elif q < p:
            right += [m * " "] * (p - q)
        zipped_lines = zip(left, right)
        lines = (
                top_lines
                + [middle_line, bottom_line]
                + [a + width * " " + b for a, b in zipped_lines]
        )
        middle = n + width // 2
        if not root:
            lines[0] = lines[0][:middle] + "┴" + lines[0][middle + 1:]
        return lines, n + m + width, max(p, q) + 2 + len(top_lines), middle

    def debug_pydot(self, output_file: str):
        pydot_graph = pydot.Dot("tree", graph_type="digraph", dpi=300)

        def get_node_name(node):
            return f"{node.get_node_depth()}.{node.num_samples}.{node.feature_index}.{node.threshold}." \
                   f"{node.feature_index_occurrences}.{node.feature_index_occurrences_redundant}"

        def get_pydot_node(node):
            node_name = get_node_name(node)

            fill_color = "none" if node.left or node.right else "gold"

            tex_label = f"$D{node.feature_index} \\leq {round(node.threshold, 3) + 0}$" if node.left or node.right \
                else "$\\begin{matrix}" + "\\text{Samples: }" + str(node.num_samples) + "\\\\" + \
                     "\\text{Class: }" + str(node.predicted_class) + "\\end{matrix}$"

            return pydot.Node(node_name, shape="box", fillcolor=fill_color, style="filled", texlbl=tex_label,
                              align="left")

        def visit_node(pydot_tree: pydot.Dot, node: Node, parent_node_name: str = "none"):
            node_name = get_node_name(node)

            if parent_node_name != "none":
                pydot_tree.add_edge(pydot.Edge(parent_node_name, node_name))

            if node.left:
                pydot_left_node = get_pydot_node(node.left)
                pydot_tree.add_node(pydot_left_node)
                visit_node(pydot_tree, node.left, node_name)
            if node.right:
                pydot_right_node = get_pydot_node(node.right)
                pydot_tree.add_node(pydot_right_node)
                visit_node(pydot_tree, node.right, node_name)

        # Call recursive function
        pydot_root_node = get_pydot_node(self)
        pydot_graph.add_node(pydot_root_node)
        visit_node(pydot_graph, self)

        # Save Results
        # Create dir if not exists
        dir_index = output_file.rfind("/")
        dir_name = output_file[:dir_index]
        filename = output_file[dir_index + 1:]
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        texcode = d2t.dot2tex(pydot_graph.to_string(), crop=True, autosize=True)
        # pydot_graph.write_png(output_file)
        output_file_tex = output_file + ".tex"
        with open(output_file_tex, 'w') as file:
            file.write(texcode)
        subprocess.run(["pdflatex", "-interaction=batchmode", "-output-directory", dir_name,
                        output_file_tex])
        subprocess.run(["convert", "-density", "300", f"{output_file}.pdf", "-quality", "100",
                        output_file])

    def get_explainability_metrics(self, num_features):
        wapl_by_node = []  # Weighted Path by leaf. List of pairs (depth, num_samples)
        wapl_redundant_by_node = []  # Same as wapl_by_node but discarding redundant features

        # Recursive function
        def visit_node(node: Node, _feature_index_occurs, _feature_index_redundant_occurs):

            # Copy lists from father
            node.feature_index_occurrences = _feature_index_occurs.copy()
            node.feature_index_occurrences_redundant = _feature_index_redundant_occurs.copy()

            # Update feature_index if it is not a leaf
            if node.left is not None and node.right is not None:
                node.feature_index_occurrences[node.feature_index] += 1

            left_node: Node = node.left
            right_node: Node = node.right

            if not left_node and not right_node:
                # leaf
                depth = node.get_node_depth()
                redundant_depth = node.get_node_explainability()
                num_samples = node.num_samples
                wapl_by_node.append((depth, num_samples))
                wapl_redundant_by_node.append((redundant_depth, num_samples))

            if left_node:
                next_index_occurs = node.get_next_feature_occurrences_redundant(node.feature_index, 'left')
                left_node.feature_index_occurrences_redundant = next_index_occurs
                visit_node(left_node, node.feature_index_occurrences.copy(), next_index_occurs.copy())
            if right_node:
                next_index_occurs = node.get_next_feature_occurrences_redundant(node.feature_index, 'right')
                right_node.feature_index_occurrences_redundant = next_index_occurs
                visit_node(right_node, node.feature_index_occurrences.copy(), next_index_occurs.copy())

        # Call recursive function
        self.feature_index_occurrences = [0] * num_features
        self.feature_index_occurrences_redundant = [0] * num_features
        visit_node(self, self.feature_index_occurrences.copy(), self.feature_index_occurrences_redundant.copy())

        # Calculate metrics
        max_depth = max(x[0] for x in wapl_by_node)
        max_redundant_depth = max(x[0] for x in wapl_redundant_by_node)

        total_samples = sum(x[1] for x in wapl_by_node)
        wapl_sum = 0
        for p in wapl_by_node:
            wapl_sum += p[0] * p[1]
        wapl = wapl_sum / total_samples

        wapl_redundant_sum = 0
        for p in wapl_redundant_by_node:
            wapl_redundant_sum += p[0] * p[1]
        wapl_redundant = wapl_redundant_sum / total_samples

        return max_depth, max_redundant_depth, wapl, wapl_redundant

    def get_unbalanced_splits(self):
        unbalanced_splits = [0]

        # Recursive function
        def visit_node(node: Node):
            left_node: Node = node.left
            right_node: Node = node.right
            unbalanced_splits[0] += 1 if node.balanced_split is False else 0
            if left_node:
                visit_node(left_node)
            if right_node:
                visit_node(right_node)

        visit_node(self)
        return unbalanced_splits[0]
