from tree import Tree
import service_functions as sf

if __name__ == '__main__':

    newick_text_1 = '((S1: 0.3, S2: 0.15):0.1,S3:0.4);'
    newick_text_2 = '((S1: 0.3, S2: 0.15):0.1,(S3:0.16,(S4:0.11,S5:0.73):0.9):0.4,S6:0.14);'
    newick_text_3 = '((S1:0.3,S2:0.15):0.1,(S3:0.16,(S4:0.11,S5:0.73):0.9):0.4);'
    newick_text_4 = ('(A:0.1,((E:0.1,(F1:0.1,F2:0.1):0.2):0.2,((D1:1.1,D2:0.1):0.12,'
                     '((B1:0.1,B2:0.1):0.2,C:0.1):0.2):0.2):0.2);')

    columns = {'node': 'Name', 'up_vector': 'Up', 'down_vector': 'Down', 'probability_vector': 'Probability',
               'marginal_vector': 'Marginal', 'probable_character': 'Probable Character'}
    pattern_1 = '101'
    pattern_2 = '010101'
    pattern_3 = '10101'
    pattern_4 = '111100011'
    pattern_msa = '>S1\n0\n>S2\n1\n>S3\n0'
    pattern_msa_2 = '>A\n'
    '1\n'
    '>B1\n'
    '1\n'
    '>B2\n'
    '1\n'
    '>C\n'
    '1\n'
    '>D1\n'
    '0\n'
    '>D2\n'
    '0\n'
    '>E\n'
    '0\n'
    '>F1\n'
    '1\n'
    '>F2\n'
    '1\n'
    pattern_msa_4 = ('>A\n11101010111000101011111011111\n>B1\n11111011000100110110000011111\n>B2\n'
                     '10110001111001110101010111111\n>C\n11011000000101111000101000001\n>D1\n'
                     '01011000111110101010101010001\n>D2\n01011000111110101010101010001\n>E\n'
                     '01010101100011111011011111011\n>F1\n10000101010110001101010101111\n>F2\n'
                     '10000101010000000101010101111')

    newick_tree_1 = Tree(newick_text_1)
    newick_tree_2 = Tree(newick_text_2)
    newick_tree_3 = Tree(newick_text_3)
    newick_tree_4 = Tree(newick_text_4)
    Tree.rename_nodes(newick_tree_3)

    Tree.tree_to_graph(newick_tree_3, 'result_files/graph.txt', ('dot', 'png', 'svg'), 'N')
    Tree.tree_to_newick_file(newick_tree_2, 'result_files/newick_tree.tree', True, 8, 'N')
    Tree.tree_to_visual_format(newick_tree_1, 'result_files/newick_tree.svg', ('svg', 'txt', 'png'), True, True, 'N')
    Tree.tree_to_csv(newick_tree_2, 'result_files/tree.csv', '\t', ('child', 'Name'), node_name='N')

    pattern_dict = newick_tree_2.get_pattern_dict(pattern_2)
    # alphabet = Tree.get_alphabet_from_dict(pattern_dict)
    alphabet = Tree.get_alphabet(0)
    print(f'newick_tree_2.calculate_marginal(None, alphabet): '
          f'{newick_tree_2.calculate_marginal(None, alphabet, pattern_2, "N")}')
    print(f'newick_tree_2.calculate_marginal("N3", alphabet): '
          f'{newick_tree_2.calculate_marginal("N2", alphabet, pattern_2, "N")}')
    print(f'newick_tree_2.root.likelihood: {newick_tree_2.root.likelihood}')
    Tree.tree_to_fasta(newick_tree_2, 'result_files/fasta_file.fasta')

    Tree.tree_to_csv(newick_tree_2, 'result_files/up_down_tree.csv', '\t', None, 0, columns, {'node_type': ['node',
                     'root', 'leaf']}, node_name='N')
    columns1 = {'node': 'Name', 'probable_character': 'Probable Character'}
    Tree.tree_to_csv(newick_tree_2, 'result_files/probable_character.csv', '\t', ('Name',), 0, columns1, node_name='N')
    table = newick_tree_2.get_tree_info()
    table2 = newick_tree_2.tree_to_table()
    print(table)
    print(table2)
    print(newick_tree_4.calculate_likelihood_for_msa(pattern_msa_4, alphabet))
    print(sf.calculate_tree_likelihood(newick_tree_2, pattern_2, 'S2', 'N3'))
