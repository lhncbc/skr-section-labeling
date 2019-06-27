

def get_sentence_from_line(line):
    sentence = line.split('|')[5].lower()
    return sentence


def get_label_from_line(line):
    return line.split('|')[3].lower()

def write_chunk_output(text_chunk_in, fout):
    chunk_lines = text_chunk_in.splitlines()
    last_line_index = len(chunk_lines) - 1

        # iterate through lines in text chunk
    for chunk_line_number, chunk_line in enumerate(chunk_lines, start=0):

        sentence = get_sentence_from_line(chunk_line)

        context_sentences = []

        if 0 <= chunk_line_number - 1 <= last_line_index:
            context_sentences.append(get_sentence_from_line(chunk_lines[chunk_line_number-1]))

        context_sentences.append(sentence)

        if 0 <= chunk_line_number + 1 <= last_line_index:
            context_sentences.append(get_sentence_from_line(chunk_lines[chunk_line_number+1]))


        sentence = "{}\n".format(' '.join(context_sentences))

        label = get_label_from_line(chunk_line).lower()

        if 'background' in label:
                fout.write('background|'+sentence)
        elif 'methods' in label:
                fout.write('methods|'+sentence)
        elif 'results' in label:
                fout.write('results|'+sentence)
        elif 'conclusions' in label:
                fout.write('conclusions|'+sentence)
        elif 'objective' in label:
                fout.write('objective|'+sentence)

def preprocess_corpora(input_filenames, output_file_postfix = '.preprocessed.context'):
    for input_filename in input_filenames:
        fin = open(input_filename,'r')
        output_filename = input_filename + output_file_postfix
        fout = open(output_filename, 'w')
        text_chunk_in = ""
        chunk_id = ''

        # iterate through lines in input file
        for line in fin:

                # lines starting with ### mark beginning of new abstract ('chunk')
                # if line.startswith('###'):
                #     chunk_id = line
                #     text_chunk_in = ''
                tokens = line.split('|')
                if chunk_id == '':
                    chunk_id = tokens[0]
                if tokens[0] == chunk_id:
                    text_chunk_in += line
                else:
                    write_chunk_output(text_chunk_in, fout)
                    chunk_id = tokens[0]
                    text_chunk_in = ""
                    text_chunk_in += line

                # otherwise, this is a line containing a labelled sentence
                # else:
                #         text_chunk_in += line
        write_chunk_output(text_chunk_in, fout)
        fin.close()
        fout.close()

data_directory = "200k/all_clean/"
input_filenames = [data_directory + '200kTrain_clean.txt', data_directory + '25kValidation_clean.txt', data_directory + '25kTest_clean.txt']
preprocess_corpora(input_filenames)
