import sys
import argparse
import json
import numpy
import random

import utils as utils
import nets as nets

import chainer
import chainer.links as L

random.seed(3)

UNK, EOS = 0, 1

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', dest='input', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                    help='input file')
# parser.add_argument('-o', '--output', dest='output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
#                     help='output file')
parser.add_argument('-v', '--vocab', dest='vocab', default='',
                    help='vocab file')
parser.add_argument('-p', '--parameter', default='args.json',
                    help='parameter files args.json')
parser.add_argument('-m', '--model', dest='model', default='',
                    help='model file')
parser.add_argument('-sv', '--source_vocab', default='',
                    help='')
parser.add_argument('-tv', '--target_vocab', default='',
                    help='')
parser.add_argument('-wv', '--wordvector', action='store_true',
                    help='')
parser.add_argument('-', '--', dest='', default='',
                    help='')
args = parser.parse_args()


class MT:
    def __init__(self, model, source_vocab, target_vocab):
        self.model = model
        self.translate = model.model.translate
        self.sv = source_vocab
        self.tv = target_vocab
        self.rv_tv = {i: w for w, i in self.tv.items()}

    def __call__(self, line):
        input_lst = line.strip().split()

        xs = [self.convert_s2i(input_lst)]
        ys = [y.tolist() for y in self.translate(xs, max_length=10)]
        print(xs)
        print(ys)

        y = [self.convert_i2s(y) for y in ys]

        exit()


    def convert_s2i(self, s_lst):
        return numpy.array([self.sv.get(i, 0) for i in s_lst] + [1])

    def convert_i2s(self, t_lst):
        return ' '.join([self.rv_tv.get(i, '<UNK>') for i in t_lst])


def load_vocab(fi_name, vector=False):
    rd = {'<UNK>': 0, '<EOS>': 1}
    with open(fi_name) as fi:
        idx = 2
        for n, line in enumerate(fi):
            if vector and n == 0:
                continue
            l_lst = line.strip().split()
            rd.setdefault(l_lst[0], idx)
            idx += 1

    return rd


def main():
    p = json.load(open(args.parameter))
    sd = load_vocab(args.source_vocab, args.wordvector)
    td = load_vocab(args.target_vocab, args.wordvector)

    # Setup model
    if p.get('model') == 'gru':
        rnn_model = L.NStepGRU
        cell = False
    elif p.get('model') == 'lstm':
        rnn_model = L.NStepLSTM
        cell = True

    if p.get('attention') == 'standard':
        tr_model = nets.Standard
    elif p.get('attention') == 'global':
        tr_model = nets.GlobalAttention

    model = nets.Translator(tr_model(p.get('layer'), len(sd), len(td),
                            p.get('unit'), rnn_model, cell, same_vocab=p.get('SAME_VOCAB')))

    chainer.serializers.load_npz(args.model, model)
    mt = MT(model, sd, td)
    # x = mt('熱 さま シート')
    x = mt('アルミ ケース')
    print(x)

    exit()
    print('※分かち書き済みのテキストを入力')
    while True:
        query = input('>>')
        if query == 'exit':
            exit()
        mt(query)


if __name__ == '__main__':
    main()

