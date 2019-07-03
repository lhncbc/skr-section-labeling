from __future__ import unicode_literals
import glob, os, sys
import pathlib
import timeit
import itertools
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import subprocess
import spacy

data_directory = ""

def get_sentence_from_line(line, tok, token_prefix = ''):
    sentence = line.split('|')[7].lower()
    if token_prefix != '':
        #tokens = [t.text for t in tok.tokenizer(sentence)]
        tokens = sentence.split(' ')
        sentence = ' '.join([token_prefix + token for token in tokens])
    return sentence


def get_label_from_line(line):
    return line.split('|')[6].lower()

def write_chunk_output(text_chunk_in, included_context_sentence_offset, add_position_information, omit_labels, fout, tok):
    chunk_lines = text_chunk_in.splitlines()
    last_line_index = len(chunk_lines) - 1

        # iterate through lines in text chunk
    for chunk_line_number, chunk_line in enumerate(chunk_lines, start=0):

            sentence = get_sentence_from_line(chunk_line, tok)

            context_sentences = []

            # generate context sentences, prepend tokens with
            # specific strings based on context sentence offset
            for sentence_offset in included_context_sentence_offset:
                if not (0 <= chunk_line_number + sentence_offset <= last_line_index):
                    context_sentences.append('no__' + str(sentence_offset))
                else:
                    context_sentences.append(get_sentence_from_line( \
                                                chunk_lines[chunk_line_number+sentence_offset], tok, \
                                                token_prefix = str(sentence_offset) + '__'))



            if add_position_information:
                sentence = "sentence__{} of__{} {} {}\n".format( \
                    str(chunk_line_number), str(last_line_index), sentence, ' '.join(context_sentences))
            else:
                sentence = "{} {}\n".format( \
                    sentence, ' '.join(context_sentences))

            if omit_labels:
                fout.write(sentence)
            else:
                label = get_label_from_line(chunk_line).lower()

                if 'background' in label:
                        fout.write('__label__background\t'+sentence)
                elif 'methods' in label:
                        fout.write('__label__methods\t'+sentence)
                elif 'results' in label:
                        fout.write('__label__results\t'+sentence)
                elif 'conclusions' in label:
                        fout.write('__label__conclusions\t'+sentence)
                elif 'objective' in label:
                        fout.write('__label__objective\t'+sentence)

def preprocess_corpora(input_filenames, add_position_information = False, included_context_sentence_offset = [], \
                       excluded_chunk_ids = [], omit_labels = False, output_file_postfix = '.preprocessed'):
    tok = spacy.blank('en')
    for input_filename in input_filenames:

        fin = open(input_filename,'r')

        output_filename = input_filename + output_file_postfix
        fout = open(output_filename, 'w')

        text_chunk_in = ""
        chunk_id = ''

        # iterate through lines in input file
        for line in fin:
                tokens = line.split('|')
                if chunk_id == '':
                    chunk_id = tokens[0]
                if tokens[0] == chunk_id:
                    text_chunk_in += line
                else:
                    write_chunk_output(text_chunk_in, included_context_sentence_offset, add_position_information, omit_labels, fout, tok)
                    chunk_id = tokens[0]
                    text_chunk_in = ""
                    text_chunk_in += line

        write_chunk_output(text_chunk_in, included_context_sentence_offset, add_position_information, omit_labels, fout, tok)
        fin.close()
        fout.close()

def fasttext_train(train_filename, model_filename, params, pretrained_vector_file):
    if pretrained_vector_file:
        subprocess.call(['./fasttext', 'supervised', '-input', train_filename, '-output', \
            model_filename, '-pretrainedVectors', pretrained_vector_file, '-dim', str(params['dim']), '-wordNgrams', str(params['wordNgrams']), '-epoch', str(params['epoch']), '-verbose', '0'], shell=True)
        # ! fasttext supervised -input $train_filename -output $model_filename -pretrainedVectors $pretrained_vector_file -dim {params['dim']} -wordNgrams {params['wordNgrams']} -epoch {params['epoch']} -verbose 0
    else:
        subprocess.call(' '.join(['./fasttext', 'supervised', '-input', train_filename, '-output', \
            model_filename, '-dim', str(params['dim']), '-wordNgrams', str(params['wordNgrams']), '-epoch', str(params['epoch']), '-verbose', '1']), shell=True)
        # ! fasttext supervised -input $train_filename -output $model_filename -dim {params['dim']} -wordNgrams {params['wordNgrams']} -epoch {params['epoch']} -verbose 0


# def plot_confusion_matrix(cm, labels,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(labels))
#     plt.xticks(tick_marks, labels, rotation=45)
#     plt.yticks(tick_marks, labels)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

def fasttext_test(model_filename, test_filename, plot_matrix = False):
    y_true = []
    with open(test_filename, 'r') as fin:
        for line in fin:
            y_true.append(line.split('\t')[0].lower())

    # output = ! fasttext predict $model_filename $test_filename
    process = subprocess.Popen(['./fasttext', 'predict', model_filename, test_filename], stdout=subprocess.PIPE)
    output, err = process.communicate()
    #output = subprocess.check_output(' '.join(['./fasttext', 'predict', model_filename, test_filename]), shell=True)
    y_pred = []
    output = output.split(' ')
    output = output[0].split('\n')
    for line in output:
        if not line.isspace() and line:
            y_pred.append(line)

    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    print f1

    if plot_matrix:
        labels = ["__label__objective",
                  "__label__background",
                  "__label__methods",
                  "__label__results",
                  "__label__conclusions"]
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_true, y_pred, labels)
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, labels=labels, title='Confusion matrix, without normalization')
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, labels=labels, normalize=True, title='Normalized confusion matrix')
        plt.show()

    return precision, recall, f1


def train_and_test(train_filename, model_filename, test_filename, params, pretrained_vector_file=None, \
                   plot_matrix=False, verbose=False):
    if verbose:
        print("\n" + str(params))

    start_time = timeit.default_timer()
    #print train_filename
    #print model_filename
    fasttext_train(train_filename, model_filename, params, pretrained_vector_file)
    training_time = timeit.default_timer() - start_time

    precision, recall, f1 = fasttext_test(model_filename + '.bin', test_filename, plot_matrix)
    results = {'precision': precision, 'recall': recall, 'f1': f1, 'training_time': training_time}
    if verbose:
        print(str(results))

    return dict(results, **params)

#single file
#input_filenames = [sys.argv[1]]
#preprocess_corpora(input_filenames, add_position_information = True, included_context_sentence_offset = [-2, -1, +1, +2])

#all train, validation and test files
input_filenames = [data_directory + 'train.txt',
                   data_directory + 'valid.txt',
                   data_directory + 'test.txt']
preprocess_corpora(input_filenames, add_position_information = True, included_context_sentence_offset = [-2, -1, +1, +2])

param_grid = {'dim': [10,20,50,100,200], 'wordNgrams': [1,2,3,4], 'epoch': [1,2,3,4,5]}
test_results_df = pd.DataFrame(columns = ['precision', 'recall', 'training_time'] + (list(param_grid.keys())))

for params in ParameterGrid(param_grid):
    test_results_df = test_results_df.append(train_and_test(data_directory + "train.txt.preprocessed",\
                                "fasttext-model", \
                                data_directory + "valid.txt.preprocessed", params), ignore_index=True)


display(test_results_df.sort_values(['f1', 'training_time'], ascending=[False, True]))
