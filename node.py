from typing import Optional, Dict, Union, List, Tuple
from scipy.linalg import expm
import pandas as pd
import numpy as np


class Node:
    father: Optional['Node']
    children: List['Node']
    name: str
    distance_to_father: Union[float, np.ndarray]
    likelihood: Union[float, np.ndarray]
    up_vector: List[Union[float, np.ndarray]]
    down_vector: List[Union[float, np.ndarray]]
    marginal_vector: List[Union[float, np.ndarray]]
    probability_vector: List[Union[float, np.ndarray]]
    probable_character: str

    def __init__(self, name: Optional[str]) -> None:
        self.father = None
        self.children = []
        self.name = name
        self.distance_to_father = 0.0
        self.likelihood = 0.0
        self.up_vector = []
        self.down_vector = []
        self.marginal_vector = []
        self.probability_vector = []
        self.probable_character = ''

    def __str__(self) -> str:
        return self.get_name(True)

    def __dir__(self) -> list:
        return ['children', 'distance_to_father', 'father', 'name', 'up_vector', 'down_vector', 'likelihood',
                'marginal_vector', 'probability_vector', 'probable_character']

    def get_list_nodes_info(self, with_additional_details: bool = False, mode: Optional[str] = None, filters:
                            Optional[Dict[str, List[Union[float, int, str, List[float]]]]] = None, only_node_list:
                            bool = False) -> List[Union[Dict[str, Union[float, np.ndarray, bool, str, List[float],
                                                  List[np.ndarray]]], 'Node']]:
        """
        Retrieve a list of descendant nodes from a given node, including the node itself or retrieve a list of
        descendant nodes from the current instance of the `Tree` class.

        This function collects all child nodes of the specified `node`, including the `node` itself, or collects all
        child nodes of the current instance of the `Tree` class if `node` is not provided. The function
        returns these nodes names as a list.

        Args:
            with_additional_details (bool, optional): `False` (default).
            mode (Optional[str]): `None` (default), 'pre-order', 'in-order', 'post-order', 'level-order'.
            filters (Dict, optional):
            only_node_list (Dict, optional): `False` (default).
        Returns:
            list: A list of nodes names including the specified `node` (or the current instance's nodes  names) and its
                                    children.
        """
        list_result = []
        mode = 'pre-order' if mode is None or mode.lower() not in ('pre-order', 'in-order', 'post-order', 'level-order'
                                                                   ) else mode.lower()
        condition = with_additional_details or only_node_list

        def get_list(trees_node: Node) -> None:
            nonlocal list_result, filters, mode, condition

            nodes_info = trees_node.get_nodes_info()
            list_item = trees_node if only_node_list else nodes_info
            if trees_node.check_filter_compliance(filters, nodes_info):
                if mode == 'pre-order':
                    list_result.append(list_item if condition else trees_node.name)

                for i, child in enumerate(trees_node.children):
                    get_list(child)
                    if mode == 'in-order' and not i:
                        list_result.append(list_item if condition else trees_node.name)

                if not trees_node.children:
                    if mode == 'in-order':
                        list_result.append(list_item if condition else trees_node.name)

                if mode == 'post-order':
                    list_result.append(list_item if condition else trees_node.name)
            else:
                for child in trees_node.children:
                    get_list(child)

        if mode == 'level-order':
            nodes_list = [self]
            while nodes_list:
                newick_node = nodes_list.pop(0)
                if newick_node.check_filter_compliance(filters, newick_node.get_nodes_info()):
                    level_order_item = newick_node if only_node_list else newick_node.get_nodes_info()
                    list_result.append(level_order_item if condition else newick_node.name)

                for nodes_child in newick_node.children:
                    nodes_list.append(nodes_child)
        else:
            get_list(self)

        return list_result

    def get_nodes_info(self) -> Dict[str, Union[float, np.ndarray, bool, str, List[float], List[np.ndarray]]]:
        lavel = 1
        full_distance = [self.distance_to_father]
        father = self.father
        if father:
            father_name = father.name
            node_type = 'node'
            while father:
                full_distance.append(father.distance_to_father)
                lavel += 1
                father = father.father
        else:
            father_name = ''
            node_type = 'root'

        if not self.children:
            node_type = 'leaf'

        return {'node': self.name, 'distance': full_distance[0], 'lavel': lavel, 'node_type': node_type, 'father_name':
                father_name, 'full_distance': full_distance, 'children': [i.name for i in self.children], 'up_vector':
                self.up_vector, 'down_vector': self.down_vector, 'likelihood': self.likelihood, 'marginal_vector':
                self.marginal_vector, 'probability_vector': self.probability_vector, 'probable_character':
                self.probable_character}

    def get_node_by_name(self, node_name: str) -> Optional['Node']:
        if node_name == self.name:
            return self
        else:
            for child in self.children:
                newick_node = child.get_node_by_name(node_name)
                if newick_node:
                    return newick_node
        return None

    def calculate_marginal(self, qmatrix: np.ndarray, alphabet: Union[Tuple[str, ...], str]
                           ) -> Tuple[Union[List[np.ndarray], List[float]], Union[np.ndarray, float]]:
        alphabet_size = len(alphabet)
        self.marginal_vector = []
        for i in range(alphabet_size):
            marg = 0
            for j in range(alphabet_size):
                marg += qmatrix[i, j] * self.down_vector[j]
            self.marginal_vector.append(1 / alphabet_size * self.up_vector[i] * marg)

        likelihood = np.sum(self.marginal_vector)

        self.probability_vector = []
        for i in range(alphabet_size):
            self.probability_vector.append(self.marginal_vector[i] / likelihood)
        self.probable_character = alphabet[self.probability_vector.index(max(self.probability_vector))]

        return self.marginal_vector, likelihood

    def calculate_up(self, nodes_dict: Dict[str, Tuple[int, ...]], alphabet: Union[Tuple[str, ...], str]
                     ) -> Union[Tuple[Union[List[np.ndarray], List[float]], float], float]:
        alphabet_size = len(alphabet)
        if not self.children:
            self.up_vector = list(nodes_dict.get(self.name))
            self.likelihood = np.sum([1 / alphabet_size * i for i in self.up_vector])
            self.probable_character = alphabet[self.up_vector.index(max(self.up_vector))]
            return self.up_vector, self.distance_to_father

        l_vect, l_dist = self.children[0].calculate_up(nodes_dict, alphabet)
        r_vect, r_dist = self.children[1].calculate_up(nodes_dict, alphabet)

        l_qmatrix = self.children[0].get_jukes_cantor_qmatrix(alphabet_size)
        r_qmatrix = self.children[1].get_jukes_cantor_qmatrix(alphabet_size)

        self.up_vector = []
        for j in range(alphabet_size):
            prob_l = prob_r = 0
            for i in range(alphabet_size):
                prob_l += l_qmatrix[j, i] * l_vect[i]
                prob_r += r_qmatrix[j, i] * r_vect[i]
            self.up_vector.append(prob_l * prob_r)

        nodes_dict.update({self.name: self.up_vector})

        self.likelihood = np.sum([1 / alphabet_size * i for i in self.up_vector])

        if self.father:
            return self.up_vector, self.distance_to_father
        else:
            return self.likelihood

    def calculate_down(self, tree_info: pd.Series, alphabet_size: int) -> None:
        father = self.father
        if father:
            brother_vector = b_qmatrix = None
            brothers = tuple(set(tree_info.get(father.name).get('children')) - {self.name})
            if brothers:
                brother = father.get_node_by_name(brothers[0])
                if brother:
                    brother_vector = brother.up_vector
                    b_qmatrix = brother.get_jukes_cantor_qmatrix(alphabet_size)

            father_vector = father.down_vector
            f_qmatrix = father.get_jukes_cantor_qmatrix(alphabet_size)
            self.down_vector = []
            for j in range(alphabet_size):
                prob_b = sum(
                    [b_qmatrix[i, j] * brother_vector[i] for i in range(alphabet_size)]) if brother_vector else 1
                prob_f = sum([f_qmatrix[i, j] * father_vector[i] for i in range(alphabet_size)]) if father.father else 1
                self.down_vector.append(prob_f * prob_b)

            if self.children:
                self.children[0].calculate_down(tree_info, alphabet_size)
                self.children[1].calculate_down(tree_info, alphabet_size)
        else:
            self.down_vector = [1] * alphabet_size
            self.children[0].calculate_down(tree_info, alphabet_size)
            self.children[1].calculate_down(tree_info, alphabet_size)

    def get_jukes_cantor_qmatrix(self, alphabet_size: int) -> np.ndarray:
        qmatrix = np.ones((alphabet_size, alphabet_size))
        np.fill_diagonal(qmatrix, 1 - alphabet_size)
        qmatrix = qmatrix * 1 / (alphabet_size - 1)

        return expm(qmatrix * self.distance_to_father)

    def subtree_to_newick(self, with_internal_nodes: bool = False, decimal_length: int = 0
                          ) -> str:
        """This method is for internal use only."""
        node_list = self.children
        if node_list:
            result = '('
            for child in node_list:
                if child.children:
                    child_name = child.subtree_to_newick(with_internal_nodes, decimal_length)
                else:
                    child_name = child.name
                result += f'{child_name}:{str(child.distance_to_father).ljust(decimal_length, "0")},'
            result = f'{result[:-1]}){self.name if with_internal_nodes else ""}'
        else:
            result = f'{self.name}:{str(self.distance_to_father).ljust(decimal_length, "0")}'
        return result

    def get_name(self, is_full_name: bool = False) -> str:
        return (f'{self.subtree_to_newick() if self.children and is_full_name else self.name}:'
                f'{self.distance_to_father:.6f}')

    def add_child(self, child: 'Node', distance_to_father: Optional[float] = None) -> None:
        self.children.append(child)
        child.father = self
        if distance_to_father is not None:
            child.distance_to_father = distance_to_father

    def get_full_distance_to_leaves(self, return_list: bool = False) -> Union[List[float], float]:
        list_result = []
        children = [self]
        while children:
            child = children.pop(0)
            list_result.append({'node': child, 'distance': child.distance_to_father})
            for ch in child.children:
                children.append(ch)
        result = [i['distance'] for i in list_result]
        return result if return_list else sum(result)

    def get_full_distance_to_father(self, return_list: bool = False) -> Union[List[float], float]:
        list_result = []
        father = self
        while father:
            list_result.append({'node': father, 'distance': father.distance_to_father})
            father = father.father
        result = [i['distance'] for i in list_result]
        return result if return_list else sum(result)

    @staticmethod
    def check_filter_compliance(filters: Optional[Dict[str, List[Union[float, int, str, List[float]]]]], info: Dict[str,
                                Union[float, bool, str, list[float]]]) -> bool:
        permission = 0
        if filters:
            for key in filters.keys():
                for value in filters.get(key):
                    permission += sum(k == key and info[k] == value for k in info)
        else:
            permission = 1

        return bool(permission)
