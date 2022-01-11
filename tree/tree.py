"""Binary tree with decision tree semantics and ASCII visualization."""


class Node:
    """A decision tree node."""

    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class, feature_index_occurrences=None,
                 feature_index_occurrences_redundant=None):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.feature_index_occurrences = feature_index_occurrences  # number of occurrences for each feature

        # for each feature, +1 if feature x_i appears in x_i > j,
        # -1 if x_i appears in x_i <= j, and
        # 2 if x_if appears in both <= j and > j,
        # 0 otherwise
        self.feature_index_occurrences_redundant = feature_index_occurrences_redundant

        self.threshold = 0
        self.left = None
        self.right = None

    def get_node_explainability(self):
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
