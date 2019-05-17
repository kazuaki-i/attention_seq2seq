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

UNK, EOS, BOS = 0, 1, 2

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
        self.translate = model.translate
        # self.translate = model.model.translate_beam_search
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
        print(y)


    def convert_s2i(self, s_lst):
        return numpy.array([self.sv.get(i, 0) for i in s_lst] + [1])

    def convert_i2s(self, t_lst):
        return ' '.join([self.rv_tv.get(i, '<UNK>') for i in t_lst])


def load_vocab(fi_name):
    rd = {'<UNK>': 0, '<EOS>': 1, '<BOS>': 2}
    with open(fi_name) as fi:
        idx = 3
        for n, line in enumerate(fi):
            l_lst = line.strip().split()
            if n == 0 and len(l_lst) == 2:
                continue
            rd.setdefault(l_lst[0], idx)
            idx += 1

    return rd


def main():
    p = json.load(open(args.parameter))
    sd = load_vocab(args.source_vocab)
    td = load_vocab(args.target_vocab)

    # Setup model
    if p.get('model') == 'gru':
        encoder, decoder = L.NStepGRU, L.NStepGRU
        cell, bi = False, False
    elif p.get('model') == 'lstm':
        encoder, decoder = L.NStepLSTM, L.NStepLSTM
        cell, bi = True, False
    elif p.get('model') == 'bigru':
        encoder, decoder = L.NStepBiGRU, L.NStepGRU
        cell, bi = False, True
    elif p.get('model') == 'bilstm':
        encoder, decoder = L.NStepBiLSTM, L.NStepLSTM
        cell, bi = True, True

    if p.get('attention') == 'standard':
        attention = False
    elif p.get('attention') == 'global':
        attention = True

    model = nets.Seq2Seq(p.get('layer'), len(sd), len(td), p.get('unit'), encoder, decoder,
                         attention=attention, cell=cell, bi=bi, feeding=p.get('feeding'), same_vocab=p.get('SAME_VOCAB'))

    chainer.serializers.load_npz(args.model, model)
    mt = MT(model, sd, td)
    x = mt('熱 さま シート')
    x = mt('アルミ ケース')
    x = mt('透明 テープ')
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

