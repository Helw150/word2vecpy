import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Pool, Value, Array

from Huffman import Vocab, VocabItem

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def init_net(dim, vocab_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    tmp = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)

def train_process(pid):
    # Set fi to point to the right chunk of training file
    start = vocab.bytes / num_processes * pid
    end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)
    fi.seek(start)
    #print('Worker %d beginning training at %d, ending at %d' % (pid, start, end))

    lr1 = starting_lr1

    word_count = 0
    last_word_count = 0

    while fi.tell() < end:
        line = fi.readline().strip()
        # Skip blank lines
        if not line:
            continue

        # Init sent, a list of indices of words in line
        sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])

        for sent_pos, token in enumerate(sent):
            if word_count % 10000 == 0:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count

                # Recalculate learning rate
                lr1 = starting_lr1 * (1 - float(global_word_count.value) / vocab.word_count)
                if lr1 < starting_lr1 * 0.0001: lr1 = starting_lr1 * 0.0001

                # Print progress info
                sys.stdout.write("\rLearning Rate: %f Progress: %d of %d (%.2f%%)" %
                                 (lr1, global_word_count.value, vocab.word_count,
                                  float(global_word_count.value) / vocab.word_count * 100))
                sys.stdout.flush()

            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win+1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] # Turn into an iterator?

            for context_word in context:
                # Init neu1e with zeros
                neu1e = np.zeros(dim)
            
            
                classifiers = zip(vocab[token].path, vocab[token].code)
                for target, label in classifiers:
                    z = np.dot(syn0[context_word], syn1[target])
                    p = sigmoid(z)
                    g = lr1 * (label - p)
                    neu1e += g * syn1[target]              # Error to backpropagate to syn0
                    syn1[target] += g * syn0[context_word] # Update syn1

                # Update syn0
                syn0[context_word] += neu1e

            word_count += 1

    # Print progress info
    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write("\rLearning Rate 1: %f Progress: %d of %d (%.2f%%)" %
                     (lr1, global_word_count.value, vocab.word_count,
                      float(global_word_count.value)/vocab.word_count * 100))
    sys.stdout.flush()
    fi.close()

def save(vocab, syn0, fo, binary):
    print('Saving model to', fo)
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n' % (len(syn0), dim))
        fo.write('\n')
        for token, vector in zip(vocab, syn0):
            fo.write('%s ' % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))

    fo.close()

def __init_process(*args):
    global vocab, syn0, syn1, table, dim, starting_lr1
    global win, num_processes, global_word_count, fi
    
    vocab, syn0_tmp, syn1_tmp, table, dim, starting_lr1, win, num_processes, global_word_count = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)

def train(fi, fo, dim, lr1, win, min_count, num_processes, binary):
    # Read train file to init vocab
    vocab = Vocab(fv, min_count)

    # Init net
    syn0, syn1 = init_net(dim, len(vocab))

    global_word_count = Value('i', 0)

    vocab.encode_huffman()

    # Begin training using num_processes workers
    t0 = time.time()
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(vocab, syn0, syn1, table, dim, lr1,
                          win, num_processes, global_word_count, fi))
    pool.map(train_process, range(num_processes))
    t1 = time.time()
    print("\n")
    print('Completed training. Training took', (t1 - t0) / 60, 'minutes')

    # Save model to file
    save(vocab, syn0, fo, binary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-vocab', help='Word Counts Files', dest='fv', required=True)
    parser.add_argument('-model', help='Output model file', dest='fo', required=True)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-lr1', help='Starting lr1', dest='lr1', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int) 
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5, type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=1, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0, type=int)
    #TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    args = parser.parse_args()

    train(args.fi, args.fv, args.fo, args.dim, args.lr1, args.win,
          args.min_count, args.num_processes, bool(args.binary))
