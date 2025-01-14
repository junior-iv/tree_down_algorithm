from tree import Tree
import service_functions as sf
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

newick_text = '((S1: 0.3, S2: 0.15)N2:0.1,S3:0.4)N1;'

# pattern_msa = '010'
pattern_msa = ('>S1'
               '\n0'
               '\n>S2'
               '\n1'
               '\n>S3'
               '\n0')

if __name__ == '__main__':
    sf.calculate_tree_likelihood(newick_text, pattern_msa, 'down')
