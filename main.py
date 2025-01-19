from tree import Tree
import service_functions as sf

if __name__ == '__main__':

    newick_text_1 = '((S1: 0.3, S2: 0.15):0.1,S3:0.4);'
    newick_text_2 = '((S1: 0.3, S2: 0.15):0.1,(S3:0.16,(S4:0.11,S5:0.73):0.9):0.4,S6:0.14);'
    newick_text_3 = '((S1:0.3,S2:0.15):0.1,(S3:0.16,(S4:0.11,S5:0.73):0.9):0.4);'

    columns = {'node': 'Name', 'up_vector': 'Up', 'down_vector': 'Down'}
    pattern_1 = '101'
    pattern_2 = '101010'
    pattern_msa = '>S1\n0\n>S2\n1\n>S3\n0'
    newick_tree_1 = Tree.rename_nodes(newick_text_1)
    newick_tree_2 = Tree.rename_nodes(newick_text_2)
    newick_tree_3 = Tree.rename_nodes(newick_text_3)
    Tree.tree_to_graph(newick_tree_3, 'result_files/graph.txt', ('dot', 'png'))
    Tree.tree_to_newick_file(newick_tree_2, 'result_files/newick_tree.tree', True, 8)
    Tree.tree_to_visual_format(newick_tree_3, 'result_files/newick_tree.svg', ('svg', 'txt', 'png'), True, True)
    Tree.tree_to_csv(newick_tree_2, 'result_files/tree.csv', '\t', ('child', 'Name'))

    pattern_dict = sf.get_pattern_dict(newick_tree_2, pattern_2)
    alphabet = sf.get_alphabet_from_dict(pattern_dict)
    sf.calculate_tree_likelihood_using_up_down_algorithm(alphabet, newick_tree_3, pattern_dict, 'down')
    Tree.tree_to_csv(newick_tree_3, 'result_files/up_down_tree.csv', '\t', None, 0, columns, {'node_type': ['node']})

    sf.calculate_tree_likelihood(newick_tree_1, pattern_msa, 'down', 'S2')
