import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from node import Node
from typing import Optional, List, Union, Dict, Tuple, Set
from Bio import Phylo


class Tree:
    root: Optional[Node]

    def __init__(self, data: Union[str, Node, None] = None, node_name: Optional[str] = None) -> None:
        if isinstance(data, str):
            self.newick_to_tree(data)
            if node_name and isinstance(node_name, str):
                Tree.rename_nodes(self, node_name)
        elif isinstance(data, Node):
            self.root = data
        else:
            self.root = Node('root')

    def __str__(self) -> str:
        return self.get_newick()

    def __len__(self) -> int:
        return self.get_node_count()

    def __eq__(self, other) -> bool:
        return str(self).lower() == str(other).lower()

    def __ne__(self, other) -> bool:
        return not self == other

    def __lt__(self, other) -> bool:
        return len(self) < len(other)

    def __le__(self, other) -> bool:
        return self < other or self == other or len(str(self)) < len(str(other))

    def __gt__(self, other) -> bool:
        return len(self) > len(other)

    def __ge__(self, other) -> bool:
        return self > other or self == other or len(str(self)) > len(str(other))

    def print_node_list(self, reverse: bool = False, with_additional_details: bool = False, mode: Optional[str] = None,
                        filters: Optional[Dict[str, List[Union[float, int, str, List[float]]]]] = None) -> None:
        """
        Print a list of nodes.

        This function prints a list of nodes. If the `reverse` argument is set to `True`, the list
        of nodes will be printed in reverse order. By default, `reverse` is `False`, so the list
        will be printed in its natural order.

        Args:
            reverse (bool, optional): If `True`, print the nodes in reverse order. If `False` (default),
                                      print the nodes in their natural order.
            with_additional_details (bool, optional): `False` (default)
            mode (Optional[str]): `None` (default), 'pre-order', 'in-order', 'post-order', 'level-order'.
            filters (Dict, optional):
        Returns:
            None: This function does not return any value; it only prints the nodes to the standard output.
        """
        data_structure = self.root.get_list_nodes_info(reverse, with_additional_details, mode, filters)

        str_result = ''
        for i in data_structure:
            str_result = f'{str_result}\n{i}'
        print(str_result, '\n')

    def get_tree_info(self, filters: Optional[Dict[str, List[Union[float, int, str, List[float]]]]] = None
                      ) -> pd.Series:
        nodes_info = self.get_list_nodes_info(False, True, 'pre-order', filters)

        return pd.Series([pd.Series(i) for i in nodes_info], index=[i.get('node') for i in nodes_info])

    def get_list_nodes_info(self, reverse: bool = False, with_additional_details: bool = False, mode: Optional[str] =
                            None, filters: Optional[Dict[str, List[Union[float, int, str, List[float]]]]] = None
                            ) -> List[Union[Dict[str, Union[float, np.ndarray, bool, str, List[float],
                                      List[np.ndarray]]], 'Node']]:
        """
        Args:
            reverse (bool, optional): If `True`, the resulting list of nodes will be in reverse order.
                                      If `False` (default), the nodes will be listed in their natural
                                      traversal order.
            with_additional_details (bool, optional): `False` (default).
            mode (Optional[str]): `None` (default), 'pre-order', 'in-order', 'post-order', 'level-order'.
            filters (Dict, optional):
        """
        return self.root.get_list_nodes_info(reverse, with_additional_details, mode, filters)

    def get_node_count(self, filters: Optional[Dict[str, List[Union[float, int, str, List[float]]]]] = None) -> int:
        """
        Args:
            filters (Dict, optional):
        """
        return len(self.get_list_nodes_info(False, True, None, filters))

    def get_node_by_name(self, name: str) -> Optional[Node]:

        return self.root.get_node_by_name(name)

    def get_newick(self, reverse: bool = False, with_internal_nodes: bool = False) -> str:

        """
        Convert the current tree structure to a Newick formatted string.

        This function serializes the tree into a Newick format, which is a standard format for representing
        tree structures. If the `reverse` argument is set to `True`, the order of the tree nodes in the
        resulting Newick string will be reversed. By default, `reverse` is `False`, meaning the nodes
        will appear in their natural order.

        Args:
            reverse (bool, optional): If `True`, reverse the order of the nodes in the Newick string.
                                      If `False` (default), preserve the natural order of the nodes.
            with_internal_nodes (bool, optional):

        Returns:
            str: A Newick formatted string representing the tree structure.
        """
        return f'{self.root.subtree_to_newick(reverse, with_internal_nodes)};'

    def find_node_by_name(self, name: str) -> bool:
        """
        Search for a node by its name in a tree structure.

        This function searches for a node with a specific name within a tree. If a root node is provided,
        the search starts from that node; otherwise, it searches from the default root of the tree.
        The function returns `True` if a node with the specified name is found, and `False` otherwise.

        Args:
            name (str): The name of the node to search for. This should be the exact name of the node
                        as a string.

        Returns:
            bool: `True` if a node with the specified name is found; `False` otherwise.
        """

        return name in self.root.get_list_nodes_info()

    def newick_to_tree(self, newick: str) -> Optional['Tree']:
        """
        Convert a Newick formatted string into a tree object.

        This function parses a Newick string, which represents a tree structure in a compact format,
        and constructs a tree object from it. The Newick format is often used in phylogenetics to
        describe evolutionary relationships among species.

        Args:
            newick (str): A string in Newick format representing the tree structure. The string
                              should be properly formatted according to Newick syntax.

        Returns:
            Tree: An object representing the tree structure parsed from the Newick string. The tree
                  object provides methods and properties to access and manipulate the tree structure.
        """
        newick = newick.replace(' ', '').strip()
        if newick.startswith('(') and newick.endswith(';'):

            len_newick = len(newick)
            list_end = [i for i in range(len_newick) if newick[i:i + 1] == ')']
            list_start = [i for i in range(len_newick) if newick[i:i + 1] == '(']
            list_children = []

            num = self.__counter()

            while list_start:
                int_start = list_start.pop(-1)
                int_end = min([i for i in list_end if i > int_start]) + 1
                list_end.pop(list_end.index(int_end - 1))
                node_name = newick[int_end: min([x for x in [newick.find(':', int_end), newick.find(',', int_end),
                                                 newick.find(';', int_end), newick.find(')', int_end)] if x >= 0])]
                distance_to_father = newick[int_end + len(node_name): min([x for x in [newick.find(',', int_end),
                                                                          newick.find(';', int_end), newick.find(')',
                                                                          int_end)] if x >= 0])]

                (visibility, node_name) = (True, node_name) if node_name else (False, 'nd' + str(num()).rjust(4, '0'))

                sub_str = newick[int_start:int_end]
                list_children.append({'children': sub_str, 'node': node_name, 'distance_to_father': distance_to_father,
                                      'visibility': visibility})

            list_children.sort(key=lambda x: len(x.get('children')), reverse=True)

            for i in range(len(list_children)):
                for j in range(i + 1, len(list_children)):
                    node_name = list_children[j].get('node') if list_children[j].get('visibility') else ''
                    list_children[i].update({'children': list_children[i].get('children').replace(
                        list_children[j].get('children') + node_name, list_children[j].get('node'))})
            for dict_children in list_children:
                if list_children.index(dict_children):
                    newick_node = self.get_node_by_name(dict_children.get('node'))
                    newick_node = newick_node if newick_node else self.__set_node(
                        f'{dict_children.get("node")}{dict_children.get("distance_to_father")}', num)
                else:
                    newick_node = self.__set_node(
                        f'{dict_children.get("node")}{dict_children.get("distance_to_father")}', num)
                    self.root = newick_node

                self.__set_children_list_from_string(dict_children.get('children'), newick_node, num)

            return self

    def get_html_tree(self, style: str = '', status: str = '') -> str:
        """This method is for internal use only."""
        return self.structure_to_html_tree(self.tree_to_structure(), style, status)

    def tree_to_structure(self, reverse: bool = False) -> dict:
        """This method is for internal use only."""
        return self.subtree_to_structure(self.root, reverse)

    def add_distance_to_father(self, distance_to_father: float = 0) -> None:
        def add_distance(newick_node: Node) -> None:
            nonlocal distance_to_father
            newick_node.distance_to_father += distance_to_father
            newick_node.distance_to_father = round(newick_node.distance_to_father, 12)
            for child in newick_node.children:
                add_distance(child)

        add_distance(self.root)

    def get_edges_list(self, reverse: bool = False) -> List[str]:
        list_result = []

        def get_list(newick_node: Node) -> None:
            nonlocal list_result
            if newick_node.father:
                list_result.append((newick_node.father.name, newick_node.name))
            for child in newick_node.children[::-1] if reverse else newick_node.children:
                get_list(child)

        get_list(self.root)

        return list_result

    def __set_children_list_from_string(self, str_children: str, father: Optional[Node], num) -> None:
        """This method is for internal use only."""
        str_children = str_children[1:-1] if str_children.startswith('(') and str_children.endswith(
            ')') else str_children
        lst_nodes = str_children.split(',')
        for str_node in lst_nodes:
            newick_node = self.__set_node(str_node.strip(), num)
            father.add_child(newick_node)

    def check_tree_for_binary(self) -> bool:
        nodes_list = self.get_list_nodes_info(False, True)
        for newick_node in nodes_list:
            for key in newick_node.keys():
                if key == 'children' and len(newick_node.get(key)) > 2:
                    return False
        return True

    def tree_to_table(self, sort_values_by: Optional[Tuple[str, ...]] = None, decimal_length: int = 8, columns: Optional
                      [Dict[str, str]] = None, filters: Optional[Dict[str, List[Union[float, int, str, List[float]]]]] =
                      None) -> pd.DataFrame:
        nodes_info = self.get_list_nodes_info(False, True, None, filters)
        columns = columns if columns else {'node': 'Name', 'father_name': 'Parent', 'distance': 'Distance to father',
                                           'children': 'child', 'lavel': 'Lavel', 'node_type': 'Node type',
                                           'full_distance': 'Full distance', 'up_vector': 'Up', 'down_vector': 'Down',
                                           'likelihood': 'Likelihood', 'marginal_vector': 'Marginal vector',
                                           'probability_vector': 'Probability vector', 'probable_character':
                                           'Probable character'}
        for node_info in nodes_info:
            for i in set(node_info.keys()) - set(columns.keys()):
                node_info.pop(i)
            if not node_info.get('father_name'):
                node_info.update({'father_name': 'root'})
            if columns.get('distance'):
                node_info.update({'distance': ' ' * (decimal_length // 2) if not node_info.get('distance') else
                                 f'{str(node_info.pop("distance")).ljust(decimal_length, "0")}'})
            for i in ('children', 'full_distance', 'up_vector', 'down_vector', 'marginal_vector', 'probability_vector'):
                if columns.get(i):
                    node_info.update({i: ' '.join(map(str, node_info.get(i)))})

        tree_table = pd.DataFrame([i for i in nodes_info], index=None)
        tree_table = tree_table.rename(columns=columns)
        tree_table = tree_table.reindex(columns=columns.values())

        return tree_table.sort_values(by=list(sort_values_by)) if sort_values_by else tree_table

    def calculate_marginal(self, newick_node: Optional[Union[Node, str]], alphabet: Union[Tuple[str, ...], str],
                           pattern: Optional[str] = None, node_name: Optional[str] = None
                           ) -> Tuple[Union[List[np.ndarray], List[float]], Union[np.ndarray, float]]:
        node_list, result = [], None
        if node_name and isinstance(node_name, str):
            Tree.rename_nodes(self, node_name)
        if not newick_node:
            node_list = self.root.get_list_nodes_info(False, False, None, {'node_type': ['node', 'root']}, True)
            if node_list:
                newick_node = np.random.choice(np.array(node_list))
        else:
            if isinstance(newick_node, str):
                newick_node = self.get_node_by_name(newick_node)
            node_list.append(newick_node)
        if not newick_node.up_vector and pattern:
            self.calculate_up(pattern, alphabet)
        if not newick_node.down_vector and newick_node.up_vector:
            self.calculate_down(alphabet)

        for current_node in node_list:
            qmatrix = current_node.get_jukes_cantor_qmatrix(len(alphabet))
            result = current_node.calculate_marginal(qmatrix, alphabet)

        return result

    def calculate_up(self, pattern: str, alphabet: Union[Tuple[str, ...], str]) -> Union[Tuple[Union[List[np.ndarray],
                                                                                         List[float]], float], float]:
        pattern_dict = self.get_pattern_dict(pattern, alphabet)

        return self.root.calculate_up(pattern_dict, alphabet)

    def calculate_down(self, alphabet: Union[Tuple[str, ...], str]) -> None:

        self.root.calculate_down(self.get_tree_info(), len(alphabet))

    def get_pattern_dict(self, pattern: str, alphabet: Optional[Union[Tuple[str, ...], str]] = None
                         ) -> Dict[str, Union[Tuple[int], str]]:
        list_nodes_info = self.get_list_nodes_info(False, True, 'pre-order', {'node_type': ['leaf']})
        pattern_list = pattern.strip().split()
        pattern_list_size, pattern_dict = len(pattern_list), dict()
        if pattern_list_size == 1:
            for i, node_info in enumerate(list_nodes_info):
                if alphabet:
                    value = [0] * len(alphabet)
                    value[alphabet.index(pattern[i])] = 1
                else:
                    value = pattern[i]
                pattern_dict.update({node_info.get('node'): tuple(value)})
        else:
            for j in range(pattern_list_size // 2):
                node_name = pattern_list[j + j][1::]
                if self.find_dict_in_iterable(list_nodes_info, 'node', node_name):
                    if alphabet:
                        value = [0] * len(alphabet)
                        value[alphabet.index(pattern_list[j + j + 1])] = 1
                    else:
                        value = pattern_list[j + j + 1]
                    pattern_dict.update({node_name: tuple(value)})

        return pattern_dict

    @staticmethod
    def tree_to_fasta(newick_tree: Union[str, 'Tree'], file_name: str = 'file.fasta', node_name: Optional[str] = None
                      ) -> None:
        if node_name and isinstance(node_name, str):
            Tree.rename_nodes(newick_tree, node_name)
        else:
            newick_tree = Tree.check_tree(newick_tree)

        Tree.make_dir(file_name)
        columns = {'node': 'Name', 'probable_character': 'Probable Character', 'children': 'child', 'lavel': 'Lavel',
                   'node_type': 'Node Type'}
        table = newick_tree.tree_to_table(('Node Type', 'Lavel', 'Name'), columns=columns)
        fasta_text = ''
        dict_table = table.to_dict()
        for i in table.T:
            fasta_text += f'>{dict_table.get("Name").get(i)}\n{dict_table.get("Probable Character").get(i)}\n'
        with open(file_name, 'w') as f:
            f.write(fasta_text[:-1])

    @staticmethod
    def tree_to_csv(newick_tree: Union[str, 'Tree'], file_name: str = 'file.csv', sep: str = '\t', sort_values_by:
                    Optional[Tuple[str, ...]] = None, decimal_length: int = 8, columns: Optional[Dict[str, str]] = None,
                    filters: Optional[Dict[str, List[Union[float, int, str, List[float]]]]] = None, node_name:
                    Optional[str] = None) -> None:
        if node_name and isinstance(node_name, str):
            Tree.rename_nodes(newick_tree, node_name)
        else:
            newick_tree = Tree.check_tree(newick_tree)

        Tree.make_dir(file_name)
        columns = columns if columns else {'node': 'Name', 'father_name': 'Parent', 'distance': 'Distance to father',
                                           'children': 'child'}
        table = newick_tree.tree_to_table(sort_values_by, decimal_length, columns, filters)
        table.to_csv(file_name, index=False, sep=sep)

    @staticmethod
    def tree_to_newick_file(newick_tree: Union[str, 'Tree'], file_name: str = 'tree_file.tree', with_internal_nodes:
                            bool = False, decimal_length: int = 8, node_name: Optional[str] = None) -> None:
        if node_name and isinstance(node_name, str):
            Tree.rename_nodes(newick_tree, node_name)
        else:
            newick_tree = Tree.check_tree(newick_tree)

        Tree.make_dir(file_name)
        newick_text = f'{newick_tree.root.subtree_to_newick(False, with_internal_nodes, decimal_length)};'
        with open(file_name, 'w') as f:
            f.write(newick_text)

    @staticmethod
    def tree_to_visual_format(newick_tree: Union[str, 'Tree'], file_name: str = 'tree_file.svg', file_extensions:
                              Optional[Union[str, Tuple[str, ...]]] = None, with_internal_nodes: bool = False,
                              show_axes: bool = False, node_name: Optional[str] = None) -> None:
        file_extensions = Tree.check_file_extensions_tuple(file_extensions, 'svg')
        if node_name and isinstance(node_name, str):
            Tree.rename_nodes(newick_tree, node_name)
        else:
            newick_tree = Tree.check_tree(newick_tree)

        Tree.make_dir(file_name)
        tmp_file = 'result_files/tmp_tree.tree'
        Tree.make_dir(tmp_file)
        Tree.tree_to_newick_file(newick_tree, tmp_file, with_internal_nodes)
        phylogenetic_tree = Phylo.read(tmp_file, 'newick')
        j = file_name[::-1].find('.')
        for file_extension in file_extensions:
            file_name = f'{file_name[:-(j + 1)]}.{file_extension}' if len(file_name) > j > -1 else (f'{file_name}.'
                                                                                                    f'{file_extension}')
            if file_extension == 'txt':
                with open(file_name, 'w') as f:
                    Phylo.draw_ascii(phylogenetic_tree, f)
            else:
                Phylo.draw(phylogenetic_tree, do_show=False)
                plt.axis('on' if show_axes else 'off')
                kwargs = {'format': file_extension, 'bbox_inches': 'tight', 'dpi': 300} if (
                        file_extension == 'svg') else {'format': file_extension}
                plt.savefig(file_name, **kwargs)
        os.remove(tmp_file)

    @staticmethod
    def tree_to_graph(newick_tree: Union[str, 'Tree'], file_name: str = 'graph', file_extensions: Optional[Union[str,
                      Tuple[str, ...]]] = None, node_name: Optional[str] = None) -> None:
        file_extensions = Tree.check_file_extensions_tuple(file_extensions, 'png')
        if node_name and isinstance(node_name, str):
            Tree.rename_nodes(newick_tree, node_name)
        else:
            newick_tree = Tree.check_tree(newick_tree)

        Tree.make_dir(file_name)
        columns = {'node': 'Name', 'father_name': 'Parent', 'distance': 'Distance to father'}
        table = newick_tree.tree_to_table(None, 0, columns)
        j = file_name[::-1].find('.')
        for file_extension in file_extensions:
            file_name = f'{file_name[:-(j + 1)]}.{file_extension}' if len(file_name) > j > -1 else (f'{file_name}.'
                                                                                                    f'{file_extension}')
            graph = nx.Graph()
            for row in table.values:
                graph.add_edge(row[1], row[0], length=(float(row[2]) if row[2] else 0.0))
            if 'png' in file_extension:
                nx.draw(graph, with_labels=True, font_color='Maroon', node_color='Gold', node_size=1000,
                        font_weight='bold')
                plt.savefig(file_name)
            if 'dot' in file_extension:
                nx.drawing.nx_pydot.write_dot(graph, file_name)

    @staticmethod
    def get_alphabet(search_argument: Union[Set[str], int, str]) -> Tuple[str, ...]:
        alphabets = (('0', '1'), ('A', 'C', 'G', 'T'),
                     ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                      'V'))
        if isinstance(search_argument, int):
            return tuple(alphabets[search_argument])
        if isinstance(search_argument, str):
            search_argument = set([i for i in search_argument])
        if isinstance(search_argument, set):
            for alphabet in alphabets:
                if not search_argument - set(alphabet):
                    return alphabet

    @staticmethod
    def get_alphabet_from_dict(pattern_msa_dict: Dict[str, str]) -> Tuple[str, ...]:
        character_list = []
        for sequence in pattern_msa_dict.values():
            character_list += [i for i in sequence]

        return Tree.get_alphabet(set(character_list))

    @staticmethod
    def find_dict_in_iterable(iterable: Union[List[Union[Dict[str, Union[float, np.ndarray, bool, str, List[float],
                              List[np.ndarray]]], 'Node']], Tuple[Dict[str, Union[float, bool, str, List[float], Tuple[
                                   int, ...]]]]], key: str, value: Optional[Union[float, bool, str, List[float]]] = None
                              ) -> Dict[str, Union[float, bool, str, List[float], List[int], Tuple[int, ...]]]:
        for index, dictionary in enumerate(iterable):
            if key in dictionary and (True if value is None else dictionary[key] == value):
                return dictionary

    @staticmethod
    def make_dir(path: str) -> None:
        path = '/'.join(path.split('/')[:-1])
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def check_tree(newick_tree: Union[str, 'Tree']) -> 'Tree':
        if isinstance(newick_tree, str):
            newick_tree = Tree(newick_tree)

        return newick_tree

    @staticmethod
    def check_file_extensions_tuple(file_extensions: Optional[Union[str, Tuple[str, ...]]] = None, default_value: str =
                                    'txt') -> Tuple[str, ...]:
        file_extensions = file_extensions if file_extensions else (default_value,)
        if isinstance(file_extensions, str):
            file_extensions = (file_extensions,)

        return file_extensions

    @staticmethod
    def check_newick(newick_text: str) -> bool:
        return newick_text and newick_text.startswith('(') and newick_text.endswith(';')

    @staticmethod
    def __set_node(node_str: str, num) -> Node:
        """This method is for internal use only."""
        if node_str.find(':') > -1:
            node_data: List[Union[str, int, float]] = node_str.split(':')
            node_data[0] = node_data[0] if node_data[0] else 'nd' + str(num()).rjust(4, '0')
            try:
                node_data[1] = float(node_data[1])
            except ValueError:
                node_data[1] = 0.0
        else:
            node_data = [node_str if node_str else 'nd' + str(num()).rjust(4, '0'), 0.0]

        newick_node = Node(node_data[0])
        newick_node.distance_to_father = float(node_data[1])
        return newick_node

    @staticmethod
    def __counter():
        """This method is for internal use only."""
        count = 0

        def sub_function():
            nonlocal count
            count += 1
            return count

        return sub_function

    @staticmethod
    def rename_nodes(newick_tree: Union[str, 'Tree'], node_name: str = 'N', fill_character: str = '0', number_length:
                     int = 0) -> 'Tree':
        newick_tree = Tree.check_tree(newick_tree)
        num = newick_tree.__counter()

        nodes_list = [newick_tree.root]
        while nodes_list:
            newick_node = nodes_list.pop(0)

            if newick_node.children:
                newick_node.name = f'{node_name}{str(num()).rjust(number_length, fill_character)}'
                for nodes_child in newick_node.children:
                    nodes_list.append(nodes_child)

        return newick_tree

    @classmethod
    def __get_html_tree(cls, structure: dict, status: str) -> str:
        """This method is for internal use only."""
        tags = (f'<details {status}>', '</details>', '<summary>', '</summary>') if structure['children'] else ('', '',
                                                                                                               '', '')
        str_html = (f'<li> {tags[0]}{tags[2]}{structure["name"].strip()} \t ({structure["distance_to_father"]}) '
                    f'{tags[3]}')
        for child in structure['children']:
            str_html += f'<ul>{cls.__get_html_tree(child, status)}</ul>\n' if child[
                'children'] else f'{cls.__get_html_tree(child, status)}'
        str_html += f'{tags[1]}</li>'
        return str_html

    @classmethod
    def get_robinson_foulds_distance(cls, tree1: Union['Tree', str], tree2: Union['Tree', str]) -> float:
        """This method is for internal use only."""
        tree1 = Tree(tree1) if type(tree1) is str else tree1
        tree2 = Tree(tree2) if type(tree2) is str else tree2

        edges_list1 = sorted(tree1.get_edges_list(), key=lambda item: item[1])
        edges_list2 = sorted(tree2.get_edges_list(), key=lambda item: item[1])

        distance = 0
        for newick_node in edges_list1:
            distance += 0 if edges_list2.count(newick_node) else 1
        for newick_node in edges_list2:
            distance += 0 if edges_list1.count(newick_node) else 1

        return distance / 2

    @classmethod
    def structure_to_html_tree(cls, structure: dict, styleclass: str = '', status: str = '') -> str:
        """This method is for internal use only."""
        return (f'<ul {f" class = {chr(34)}{styleclass}{chr(34)}" if styleclass else ""}>'
                f'{cls.__get_html_tree(structure, status)}</ul>')

    @classmethod
    def subtree_to_structure(cls, newick_node: Node, reverse: bool = False) -> dict:
        """This method is for internal use only."""
        dict_node = {'name': newick_node.name.strip(), 'distance_to_father': newick_node.distance_to_father}
        list_children = []
        if newick_node.children:
            for child in newick_node.children[::-1] if reverse else newick_node.children:
                list_children.append(cls.subtree_to_structure(child, reverse))
        dict_node.update({'children': list_children})
        return dict_node
