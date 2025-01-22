from math import log
from typing import Union, Optional
from tree import Tree


def calculate_tree_likelihood(newick_tree: Union[str, Tree], pattern: Optional[str] = None, verification_node_name:
                              Optional[str] = None, marginal_node_name: Optional[str] = None) -> None:
    """
        mode (str): `up` (default), 'up', 'down', 'marginal'.
    """
    newick_tree = Tree.check_tree(newick_tree)
    pattern_dict = newick_tree.get_pattern_dict(pattern)

    alphabet = Tree.get_alphabet_from_dict(pattern_dict)
    newick_tree.calculate_likelihood_for_msa(pattern, alphabet)
    alphabet_size = len(alphabet)

    print()
    if verification_node_name:
        newick_node = newick_tree.get_node_by_name(verification_node_name)
        if not newick_node:
            print('invalid node name')
        else:
            likelihood = 0
            qmatrix = newick_node.get_jukes_cantor_qmatrix(alphabet_size)
            for i in range(alphabet_size):
                for j in range(alphabet_size):
                    likelihood += (1 / alphabet_size * newick_node.up_vector[i] * newick_node.down_vector[j] *
                                   qmatrix[i, j])

            print(f'log-likelihood: {log(likelihood)}')
            print(f'likelihood: {likelihood}')

    if marginal_node_name:
        marginal_vector, likelihood = newick_tree.calculate_marginal(marginal_node_name, alphabet)
        print(f'marginal_vector: {marginal_vector}')
        print(f'log-likelihood: {log(likelihood)}')
        print(f'likelihood: {likelihood}')
