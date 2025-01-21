from tree import Tree
import numpy as np

if __name__ == '__main__':

    newick_text_1 = '((S1: 0.3, S2: 0.15):0.1,S3:0.4);'
    newick_text_2 = '((S1: 0.3, S2: 0.15):0.1,(S3:0.16,(S4:0.11,S5:0.73):0.9):0.4,S6:0.14);'
    newick_text_3 = '((S1:0.3,S2:0.15):0.1,(S3:0.16,(S4:0.11,S5:0.73):0.9):0.4);'

    columns = {'node': 'Name', 'up_vector': 'Up', 'down_vector': 'Down', 'probability_vector': 'Probability',
               'marginal_vector': 'Marginal', 'probable_character': 'Probable Character'}
    pattern_1 = '101'
    pattern_2 = '101010'
    pattern_3 = '10101'
    pattern_msa = '>S1\n0\n>S2\n1\n>S3\n0'
    newick_tree_1 = Tree(newick_text_1)
    newick_tree_2 = Tree(newick_text_2)
    newick_tree_3 = Tree(newick_text_3)
    Tree.rename_nodes(newick_tree_3)
    # # print(newick_tree_3.get_node_by_name('S4'))
    # list_nodes = newick_tree_3.root.get_list_nodes_info(False, False, None, {'node_type': ['node']}, True)
    # # nodes = self.tree_to_table(None, 8, {'node': 'node'}, {'node_type': ['node']}).get('node')
    # if list_nodes:
    #     newick_node = np.random.choice(np.array(list_nodes))
    # # print(newick_node)
    #
    # print(type(newick_tree_3.tree_to_table(None, 8, {'node': 'node'}, {'node_type': ['node']}).get('node')[0]))
    #
    # print(newick_tree_3.tree_to_table(None, 8, {'node': 'node'}, {'node_type': ['node']}).get('node1'))

    Tree.tree_to_graph(newick_tree_3, 'result_files/graph.txt', ('dot', 'png'), 'N')
    Tree.tree_to_newick_file(newick_tree_2, 'result_files/newick_tree.tree', True, 8, 'N')
    Tree.tree_to_visual_format(newick_tree_1, 'result_files/newick_tree.svg', ('svg', 'txt', 'png'), True, True, 'N')
    Tree.tree_to_csv(newick_tree_2, 'result_files/tree.csv', '\t', ('child', 'Name'), node_name='N')

    pattern_dict = newick_tree_3.get_pattern_dict(pattern_3)
    # alphabet = Tree.get_alphabet_from_dict(pattern_dict)
    alphabet = Tree.get_alphabet(0)
    print(f'newick_tree_3.calculate_marginal(None, alphabet): '
          f'{newick_tree_3.calculate_marginal(None, alphabet, pattern_3, "N")}')
    print(f'newick_tree_3.calculate_marginal("N2", alphabet): '
          f'{newick_tree_3.calculate_marginal("N3", alphabet, pattern_3, "N")}')
    print(f'newick_tree_3.root.likelihood: {newick_tree_3.root.likelihood}')
    Tree.tree_to_fasta(newick_tree_3, 'result_files/fasta_file.fasta')

    Tree.tree_to_csv(newick_tree_3, 'result_files/up_down_tree.csv', '\t', None, 0, columns, {'node_type': ['node']},
                     node_name='N')
    columns1 = {'node': 'Name', 'probable_character': 'Probable Character'}
    Tree.tree_to_csv(newick_tree_3, 'result_files/probable_character.csv', '\t', ('Name',), 0, columns1, node_name='N')
    table = newick_tree_3.get_tree_info()
    table2 = newick_tree_3.tree_to_table()

    print(table)
    print(table2)
    # print(table.get('N4').get('up_vector'))
    # print('')
    # print('!')
    # print('')
    # table = newick_tree_3.tree_to_table(None, 8, columns)
    # print(table.get('Up'))
    # print(table.get('Down'))
    # print(table.get('Probability'))
    # print(table.get('Marginal'))
    # print(table.get('Probable Character'))
    # print(table)
    # print(table.get('Name').to_dict().keys())
    # print(table.get('Name').to_dict().values())
    # print(table.get('Name').pop(0))
    # print(table.get('Name').pop(1))
    # print(table.get('Name').pop(2))
    # print(table.get('Name').to_dict().values())
