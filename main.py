import service_functions as sf
from tree import Tree

# pattern_msa = ('>A'
#                '\n1'
#                '\n>B1'
#                '\n1'
#                '\n>B2'
#                '\n1'
#                '\n>C'
#                '\n1'
#                '\n>D1'
#                '\n0'
#                '\n>D2'
#                '\n0'
#                '\n>E'
#                '\n0'
#                '\n>F1'
#                '\n1'
#                '\n>F2'
#                '\n1')
# pattern_msa = ('>A'
#                '\n11101010111000101011111011111'
#                '\n>B1'
#                '\n11111011000100110110000011111'
#                '\n>B2'
#                '\n10110001111001110101010111111'
#                '\n>C'
#                '\n11011000000101111000101000001'
#                '\n>D1'
#                '\n01011000111110101010101010001'
#                '\n>D2'
#                '\n01011000111110101010101010001'
#                '\n>E'
#                '\n01010101100011111011011111011'
#                '\n>F1'
#                '\n10000101010110001101010101111'
#                '\n>F2'
#                '\n10000101010000000101010101111')


# newick_text = ('(A:0.1,((E:0.1,(F1:0.1,F2:0.1):0.2):0.2,((D1:1.1,D2:0.1):0.12,((B1:0.1,B2:0.1):0.2,C:0.1):0.2):0.2)'
#                ':0.2);')
# pattern_msa = ('>A'
#                '\n11101010111000101011111011111'
#                '\n>B1'
#                '\n11111011000100110110000011111'
#                '\n>B2'
#                '\n10110001111001110101010111111'
#                '\n>C'
#                '\n11011000000101111000101000001'
#                '\n>D1'
#                '\n01011000111110101010101010001'
#                '\n>D2'
#                '\n01011000111110101010101010001'
#                '\n>E'
#                '\n01010101100011111011011111011'
#                '\n>F1'
#                '\n10000101010110001101010101111'
#                '\n>F2'
#                '\n10000101010000000101010101111')


if __name__ == '__main__':
    newick_text_1 = '((S1: 0.3, S2: 0.15):0.1,S3:0.4);'
    newick_text_2 = '((S1: 0.3, S2: 0.15):0.1,(S3:0.16,(S4:0.11,S5:0.73):0.9):0.4,S6:0.14);'
    newick_text_3 = '((S1:0.3,S2:0.15):0.1,(S3:0.16,(S4:0.11,S5:0.73):0.9):0.4);'

    pattern = '101'
    pattern_msa = '>S1\n0\n>S2\n1\n>S3\n0'
    newick_tree_1 = Tree.rename_nodes(newick_text_1)

    sf.calculate_tree_likelihood(newick_tree_1, pattern_msa, 'down', 'S2')
    sf.print_tree_vectors(newick_tree_1, pattern)

    newick_tree_2 = Tree.rename_nodes(newick_text_2)
    Tree.tree_to_csv(newick_tree_2, 'result_files/tree.csv', '\t', ['child', 'Name'])

    newick_tree_3 = Tree.rename_nodes(newick_text_3)
    Tree.tree_to_newick_file(newick_tree_3, 'result_files/newick_tree.tree', True)

    # newick_text = (f'((S1:0.300000,S2:0.150000)N2:0.100000,(S3:0.160000,(S4:0.110000,S5:0.730000)N4'
    #                f':0.900000)N3:0.400000)N1;')
