import io
import progressbar
import numpy
import random

from glob import glob
from operator import itemgetter
from progressbar import ProgressBar
import chainer
from chainer import backend

UNK, EOS = 0, 1


def count_lines(path):
    with io.open(path, encoding='utf-8') as f:
        return sum([1 for _ in f])


def count_up_vocab(fi_name):
    rd = dict()
    with io.open(fi_name, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for w in words:
                rd[w] = rd.get(w, 0) + 1
    return rd


def write_vocab(word_ids, fo_name):
    with io.open(fo_name, 'w', encoding='utf-8') as f:
        for w, _ in sorted(word_ids.items(), key=itemgetter(1)):
            if w not in ['<UNK>', '<EOS>']:
                print(w, file=f)

    print('write vocab to {}'.format(fo_name))

    return None


def make_vocabulary(fi_name, fj_name=None, max_vocab_size=100000):
    td = count_up_vocab(fi_name)
    if fj_name is not None:
        td.update(count_up_vocab(fj_name))

    word_ids = dict()
    word_ids['<UNK>'], word_ids['<EOS>'] = 0, 1

    for n, (w, c) in enumerate(sorted(td.items(), key=itemgetter(1), reverse=True)):
        word_ids.setdefault(w, n+2)
        if max_vocab_size < n:
            break

    return word_ids


def load_vocabulary(path):
    with io.open(path, encoding='utf-8') as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids


def load_word2vec_model(fi_name, units):
    print('load {} word2vec model'.format(fi_name))
    with open(fi_name, encoding='utf-8') as fi:
        vocab = {'<UNK>': 0, '<EOS>': 1}
        vector = []
        for n, line in enumerate(fi):
            l_lst = line.strip().split()
            if n == 0:
                # vocabsize = int(l_lst[0])
                v_size = int(l_lst[1])
                assert(units == v_size)
                vector.append([random.uniform(-0.5, 0.5) for _ in range(v_size)])
                vector.append([random.uniform(-0.5, 0.5) for _ in range(v_size)])
            else:
                v = l_lst[0]
                vec = [float(i) for i in l_lst[1:]]
                vocab[v] = n + 1
                vector.append(vec)

    return vocab, numpy.array(vector, numpy.float32)


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    with io.open(path, encoding='utf-8') as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK)
                                 for w in words], numpy.int32)
            data.append(array)
    return data


def load_data_using_dataset_api(
        src_vocab, src_path, target_vocab, target_path, filter_func):

    def _transform_line(vocabulary, line):
        words = line.strip().split()
        return numpy.array(
            [vocabulary.get(w, UNK) for w in words], numpy.int32)

    def _transform(example):
        source, target = example
        return (
            _transform_line(src_vocab, source),
            _transform_line(target_vocab, target)
        )

    def _load_single_data_using_dataset_api(fs, ft):
        return chainer.datasets.TransformDataset(
            chainer.datasets.TextDataset([fs, ft], encoding='utf-8', filter_func=filter_func), _transform)

    s_path = glob(src_path)
    t_path = glob(target_path)
    p = ProgressBar(0, len(s_path))

    datasets = []
    for n, (fs, ft) in enumerate(zip(s_path, t_path)):
        p.update(n+1)
        datasets.append(_load_single_data_using_dataset_api(fs, ft))

    return chainer.datasets.ConcatenatedDataset(*datasets)


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


# @chainer.dataset.converter()
def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        src_xp = backend.get_array_module(*batch)
        xp = device.xp
        concat = src_xp.concatenate(batch, axis=0)
        sections = list(numpy.cumsum(
            [len(x) for x in batch[:-1]], dtype=numpy.int32))
        concat_dst = device.send(concat)
        batch_dst = xp.split(concat_dst, sections)
        return batch_dst

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}
