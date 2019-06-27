import numpy as np
import sys
import csv

# THIS IS USED FOR OUR DATASET

# if len(sys.argv) < 3:
#     print("Not enough arguments.")
#     exit();
# input = open(sys.argv[1], 'r')
# output = open(sys.argv[2], 'wb')
# label_output = open(sys.argv[3], 'wb')
# label_prefix = '__label__'
#
# for line in input:
#     tokens = line.split('|')
#     label = tokens[3].lower()
#     sentence = tokens[5].lower()
#     output.write(label_prefix + label + ' ' + sentence)
#     label_output.write(label_prefix+label+'\n')
#
# input.close()
# output.close()
# label_output.close()


# THE FOLLOWING IS USED FOR PUBLIC RCT DATASET

def get_sentence_from_line(line):
    sentence = line.split('\t')[1].lower()
    return sentence


def get_label_from_line(line):
    return line.split('\t')[0]

filenames = ['pubmed-rct-master/PubMed_200k_RCT/train.txt', 'pubmed-rct-master/PubMed_200k_RCT/dev.txt', 'pubmed-rct-master/PubMed_200k_RCT/test.txt']

for file in filenames:
    fin = open(file, 'r')
    fout = open(file+'.preprocessed', 'w')
    writer = csv.writer(fout, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    text_chunk_in = ''
    for line in fin:
        if line.startswith('###'):
            text_chunk_in = ''
        elif line.isspace():
            chunk_lines = text_chunk_in.splitlines()
            for chunk_line in chunk_lines:
                sentence = get_sentence_from_line(chunk_line)
                label = get_label_from_line(chunk_line).lower()
                if 'objective' in label:
                        writer.writerow([0,sentence])
                elif 'background' in label:
                        writer.writerow([1,sentence])
                elif 'methods' in label:
                        writer.writerow([2,sentence])
                elif 'results' in label:
                        writer.writerow([3,sentence])
                elif 'conclusions' in label:
                        writer.writerow([4,sentence])
                else:
                    print("ERROR DETERMINE LABEL\n")
        else:
            text_chunk_in += line
    fin.close()
    fout.close()
