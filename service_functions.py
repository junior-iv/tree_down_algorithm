from math import log
from typing import List, Union, Tuple, Optional, Dict
from tree import Tree
import pandas as pd


def calculate_tree_likelihood_using_up_down_algorithm(alphabet: Union[Tuple[str, ...], str], newick_tree: Tree,
                                                      pattern_msa_dict: Dict[str, str], mode: str = 'up'
                                                      ) -> Tuple[List[float], float, float]:
    """
        mode (str): `up` (default), 'up', 'down'.
    """
    alphabet_size = len(alphabet)
    newick_node = newick_tree.root

    leaves_info = newick_node.get_list_nodes_info(True, 'pre-order', {'node_type': ['leaf']})

    len_seq = len(list(pattern_msa_dict.values())[0])
    likelihood, log_likelihood, log_likelihood_list = 1, 0, []
    for i_char in range(len_seq):
        nodes_dict = dict()
        for i in range(len(leaves_info)):
            node_name = leaves_info[i].get('node')
            character = pattern_msa_dict.get(node_name)[i_char]
            vector = [0] * alphabet_size
            vector[alphabet.index(character)] = 1
            nodes_dict.update({node_name: tuple(vector)})

        char_likelihood = newick_node.calculate_up(nodes_dict, alphabet)
        likelihood *= char_likelihood
        log_likelihood += log(char_likelihood)
        log_likelihood_list.append(log(char_likelihood))

        if mode == 'down':
            tree_info = newick_tree.get_tree_info()
            newick_node.calculate_down(tree_info, alphabet_size)

    return log_likelihood_list, log_likelihood, likelihood


def calculate_tree_likelihood(newick_tree: Union[str, Tree], pattern: Optional[str] = None, mode: str = 'up',
                              verification_node_name: Optional[str] = None, marginal_node_name: Optional[str] = None
                              ) -> None:
    """
        mode (str): `up` (default), 'up', 'down', 'marginal'.
    """
    newick_tree = Tree.check_tree(newick_tree)
    pattern_dict = newick_tree.get_pattern_dict(pattern)

    alphabet = Tree.get_alphabet_from_dict(pattern_dict)
    likelihood = calculate_tree_likelihood_using_up_down_algorithm(alphabet, newick_tree, pattern_dict, mode)
    alphabet_size = len(alphabet)

    if verification_node_name:
        nodes_info = newick_tree.get_list_nodes_info(True, 'pre-order')
        table = pd.Series([pd.Series(i) for i in nodes_info], index=[i.get('node') for i in nodes_info])
        for i in table:
            print(i)

        print(likelihood)
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
