from math import log
from typing import List, Union, Tuple, Optional, Dict, Set
from tree import Tree, Node
import numpy as np
from scipy.linalg import expm
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


def get_jukes_cantor_qmatrix(branch_length: float, alphabet_size: int) -> np.ndarray:
    qmatrix = np.ones((alphabet_size, alphabet_size))
    np.fill_diagonal(qmatrix, 1 - alphabet_size)
    qmatrix = qmatrix * 1 / (alphabet_size - 1)

    return expm(qmatrix * branch_length)


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


def calculate_up(newick_node: Node, nodes_dict: Dict[str, Tuple[int, ...]], alphabet: Tuple[str, ...]) -> Union[Tuple[
                                                      Union[List[np.ndarray], List[float]], float], float]:
    alphabet_size = len(alphabet)
    if not newick_node.children:
        newick_node.up_vector = list(nodes_dict.get(newick_node.name))
        newick_node.likelihood = np.sum([1 / alphabet_size * i for i in newick_node.up_vector])
        return newick_node.up_vector, newick_node.distance_to_father

    l_vect, l_dist = calculate_up(newick_node.children[0], nodes_dict, alphabet)
    r_vect, r_dist = calculate_up(newick_node.children[1], nodes_dict, alphabet)

    l_qmatrix = get_jukes_cantor_qmatrix(l_dist, alphabet_size)
    r_qmatrix = get_jukes_cantor_qmatrix(r_dist, alphabet_size)

    newick_node.up_vector = []
    for j in range(alphabet_size):
        freq_l = freq_r = 0
        for i in range(alphabet_size):
            freq_l += l_qmatrix[i, j] * l_vect[i]
            freq_r += r_qmatrix[i, j] * r_vect[i]
        newick_node.up_vector.append(freq_l * freq_r)

    nodes_dict.update({newick_node.name: newick_node.up_vector})

    newick_node.likelihood = np.sum([1 / alphabet_size * i for i in newick_node.up_vector])

    if newick_node.father:
        return newick_node.up_vector, newick_node.distance_to_father
    else:
        return newick_node.likelihood


def calculate_down(newick_node: Node, tree_info: pd.Series, alphabet_size: int) -> None:
    father = newick_node.father
    if not father:
        newick_node.down_vector = [1] * alphabet_size
        # newick_node.likelihood = np.float64(1)
        calculate_down(newick_node.children[0], tree_info, alphabet_size)
        calculate_down(newick_node.children[1], tree_info, alphabet_size)
        return

    brother_vector = b_qmatrix = None
    brothers = tuple(set(tree_info.get(father.name).get('children')) - {newick_node.name})
    if brothers:
        brother = tree_info.get(brothers[0])
        brother_vector = brother.get('up_vector')
        b_qmatrix = get_jukes_cantor_qmatrix(brother.get('distance'), alphabet_size)

    father_vector = father.down_vector
    f_qmatrix = get_jukes_cantor_qmatrix(father.distance_to_father, alphabet_size)
    newick_node.down_vector = []
    for j in range(alphabet_size):
        freq_b = sum([b_qmatrix[i, j] * brother_vector[i] for i in range(alphabet_size)]) if brother_vector else 1
        freq_f = sum([f_qmatrix[i, j] * father_vector[i] for i in range(alphabet_size)]) if father.father else 1
        newick_node.down_vector.append(freq_f * freq_b)

    if newick_node.children:
        calculate_down(newick_node.children[0], tree_info, alphabet_size)
        calculate_down(newick_node.children[1], tree_info, alphabet_size)


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


def calculate_tree_likelihood_using_up_down_algorithm(alphabet: Tuple[str, ...], newick_tree: Tree, pattern_msa_dict:
                                                      Dict[str, str], mode: str) -> Tuple[List[float], float, float]:
    alphabet_size = len(alphabet)
    newick_node = newick_tree.root

    leaves_info = newick_node.get_list_nodes_info(False, True, 'pre-order', {'node_type': ['leaf']})

    len_seq = len(list(pattern_msa_dict.values())[0])
    likelihood, log_likelihood, log_likelihood_list = 1, 0, []
    for i_char in range(len_seq):
        nodes_dict = dict()
        for i in range(len(leaves_info)):
            node_name = leaves_info[i].get('node')
            sequence = pattern_msa_dict.get(node_name)[i_char]
            frequency = [0] * alphabet_size
            frequency[alphabet.index(sequence)] = 1
            nodes_dict.update({node_name: tuple(frequency)})

        char_likelihood = calculate_up(newick_node, nodes_dict, alphabet)
        likelihood *= char_likelihood
        log_likelihood += log(char_likelihood)
        log_likelihood_list.append(log(char_likelihood))

        if mode == 'down':
            nodes_info = newick_tree.get_list_nodes_info(False, True, 'pre-order')
            tree_info = pd.Series([pd.Series(i) for i in nodes_info], index=[i.get('node') for i in nodes_info])
            calculate_down(newick_node, tree_info, alphabet_size)

    return log_likelihood_list, log_likelihood, likelihood


def calculate_tree_likelihood(newick_tree: Union[str, Tree], pattern: Optional[str] = None, mode: str = 'up',
                              verification_node_name: Optional[str] = None) -> None:
    """
        mode (str): `up` (default), 'up', 'down', 'marginal'.
    """
    if isinstance(newick_tree, str):
        newick_tree = Tree(newick_tree)
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
            qmatrix = get_jukes_cantor_qmatrix(node_info.get('distance'), alphabet_size)
            for i in range(alphabet_size):
                for j in range(alphabet_size):
                    likelihood += (1 / alphabet_size * node_info.get('up_vector')[i] * node_info.get('down_vector')[j] *
                                   qmatrix[i, j])

            print(f'log-likelihood: {log(likelihood)}')
            print(f'likelihood: {likelihood}')
