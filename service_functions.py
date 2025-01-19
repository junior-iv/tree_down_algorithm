from math import log
from typing import List, Union, Tuple, Optional, Dict, Set
from tree import Tree, Node
import numpy as np
import pandas as pd


def find_dict_in_iterable(iterable: Union[List[Union[Dict[str, Union[float, np.ndarray, bool, str, List[float],
                          List[np.ndarray]]], 'Node']], Tuple[Dict[str, Union[float, bool, str, List[float],
                                                              Tuple[int, ...]]]]], key: str, value:
                          Optional[Union[float, bool, str, List[float]]] = None) -> Dict[str, Union[float, bool, str,
                                                                                         List[float], List[int],
                                                                                         Tuple[int, ...]]]:
    for index, dictionary in enumerate(iterable):
        if key in dictionary and (True if value is None else dictionary[key] == value):
            return dictionary


def get_alphabet(character_set: Set[str]) -> Tuple[str]:
    alphabets = ({'0', '1'}, {'A', 'C', 'G', 'T'},
                 {'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'})
    for alphabet in alphabets:
        if not character_set - alphabet:
            return tuple(alphabet)


def get_alphabet_from_dict(pattern_msa_dict: Dict[str, str]) -> Tuple[str]:
    character_list = []
    for sequence in pattern_msa_dict.values():
        character_list += [i for i in sequence]

    return get_alphabet(set(character_list))


def calculate_tree_likelihood_using_up_down_algorithm(alphabet: Tuple[str, ...], newick_tree: Tree, pattern_msa_dict:
                                                      Dict[str, str], mode: str = 'up'
                                                      ) -> Tuple[List[float], float, float]:
    """
        mode (str): `up` (default), 'up', 'down', 'marginal'.
    """
    alphabet_size = len(alphabet)
    newick_node = newick_tree.root

    leaves_info = newick_node.get_list_nodes_info(False, True, 'pre-order', {'node_type': ['leaf']})

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

        char_likelihood = Tree.calculate_up(newick_node, nodes_dict, alphabet)
        likelihood *= char_likelihood
        log_likelihood += log(char_likelihood)
        log_likelihood_list.append(log(char_likelihood))

        if mode == 'down':
            nodes_info = newick_tree.get_list_nodes_info(False, True, 'pre-order')
            tree_info = pd.Series([pd.Series(i) for i in nodes_info], index=[i.get('node') for i in nodes_info])
            Tree.calculate_down(newick_node, tree_info, alphabet_size)

    return log_likelihood_list, log_likelihood, likelihood


def get_pattern_dict(newick_tree: Tree, pattern: str) -> Dict[str, str]:
    list_nodes_info = newick_tree.get_list_nodes_info(False, True, 'pre-order', {'node_type': ['leaf']})
    pattern = pattern.strip()
    pattern_list = pattern.split()
    pattern_dict = dict()
    pattern_list_size = len(pattern_list)
    if pattern_list_size == 1:
        for i, node_info in enumerate(list_nodes_info):
            pattern_dict.update({node_info.get('node'): pattern[i]})
    else:
        for j in range(pattern_list_size // 2):
            if find_dict_in_iterable(list_nodes_info, 'node', pattern_list[j + j][1::]):
                pattern_dict.update({pattern_list[j + j][1::]: pattern_list[j + j + 1]})

    return pattern_dict


def calculate_tree_likelihood(newick_tree: Union[str, Tree], pattern: Optional[str] = None, mode: str = 'up',
                              verification_node_name: Optional[str] = None) -> None:
    """
        mode (str): `up` (default), 'up', 'down', 'marginal'.
    """
    newick_tree = Tree.check_tree(newick_tree)
    pattern_dict = get_pattern_dict(newick_tree, pattern)

    alphabet = get_alphabet_from_dict(pattern_dict)
    likelihood = calculate_tree_likelihood_using_up_down_algorithm(alphabet, newick_tree, pattern_dict, mode)

    if verification_node_name:
        nodes_info = newick_tree.get_list_nodes_info(False, True, 'pre-order')
        table = pd.Series([pd.Series(i) for i in nodes_info], index=[i.get('node') for i in nodes_info])
        for i in table:
            print(i)

        print(likelihood)
        node_info = table.get(verification_node_name)
        if node_info is None:
            print('invalid node name')
        else:
            likelihood = 0
            alphabet_size = len(alphabet)
            qmatrix = Tree.get_jukes_cantor_qmatrix(node_info.get('distance'), alphabet_size)
            for i in range(alphabet_size):
                for j in range(alphabet_size):
                    likelihood += (1 / alphabet_size * node_info.get('up_vector')[i] * node_info.get('down_vector')[j] *
                                   qmatrix[i, j])

            print(f'log-likelihood: {log(likelihood)}')
            print(f'likelihood: {likelihood}')
