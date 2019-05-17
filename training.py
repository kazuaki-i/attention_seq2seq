import argparse
import datetime
import io
import json

import chainer
from chainer import backend
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from utils import *
from nets import *


UNK, EOS, BOS = 0, 1, 2


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--SOURCE', '-S', required=True, help='source sentence list')
    parser.add_argument('--TARGET', '-T', required=True, help='target sentence list')
    parser.add_argument('--SOURCE_VOCAB', '-SV', required=True, help='source vocabulary file')
    parser.add_argument('--TARGET_VOCAB', '-TV', help='target vocabulary file')
    parser.add_argument('--MAKE', '-M', dest='MAKE_VOCAB', action='store_true',
                        help='make vocabulary file from SOURCE and TARGET')
    parser.add_argument('--SAME', dest='SAME_VOCAB', action='store_true',
                        help='use same vocabulary between SOURCE and TARGET')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--resume', '-r', type=str,
                        help='resume the training from snapshot')
    parser.add_argument('--save', '-s', type=str,
                        help='save a snapshot of the training')
    parser.add_argument('--unit', '-u', type=int, default=512,
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--use-dataset-api', default=False,
                        action='store_true',
                        help='use TextDataset API to reduce CPU memory usage')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--max-source-vocabulary', type=int, default=100000,
                        help='maximum number of source vocabulary')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimum length of target sentence')
    parser.add_argument('--max-target-vocabulary', type=int, default=100000,
                        help='maximum number of target vocabulary')
    parser.add_argument('--max-target-sentence', type=int, default=10,
                        help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=4000,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evaluate the model with validation dataset')
    parser.add_argument('--snapshot-interval', type=int, default=4000,
                        help='number of iteration to save training snapshot')
    parser.add_argument('--model', '-model', default='gru', choices=['gru', 'lstm', 'bigru'],
                        help='Name of rnn model type.')
    parser.add_argument('--attention', '-attention', default='standard', choices=['standard', 'global'],
                        help='Name of attention model type.')
    parser.add_argument('--out', '-o', default='result', help='directory to output the result')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--feeding', action='store_true', help='apply input feeding to attention method')
    parser.add_argument('--progressbar', action='store_true', help='show training progressbar')
    parser.add_argument('--early-stop', action='store_true', help='use early stopping method')
    parser.add_argument('--snapshot-divide', action='store_true', help='save divide snapshot')
    parser.add_argument('--save-epoch', action='store_true', help='save model per epoch (not only best epoch)')
    parser.add_argument('--debug-mode', action='store_true', help='debug mode')
    parser.add_argument('--pre-wordvector', action='store_true', help='use pre-training word vector')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))
    print('')

    # Load pre-processed dataset

    print('[{}] Loading dataset... (this may take several minutes)'.format(datetime.datetime.now()))

    source_vector, target_vecotr = None, None
    if args.pre_wordvector:
        source_ids, source_vector =load_word2vec_model(args.SOURCE_VOCAB, args.unit)
        if args.SAME_VOCAB:
            target_ids, target_vecotr = source_ids, source_vector
        else:
            target_ids, target_vector =load_word2vec_model(args.TARGET_VOCAB, args.unit)

    elif args.MAKE_VOCAB:
        if args.SAME_VOCAB:
            source_ids = make_vocabulary(args.SOURCE, args.TARGET, max_vocab_size=args.max_source_vocabulary)
            target_ids = source_ids
        else:
            source_ids = make_vocabulary(args.SOURCE, max_vocab_size=args.max_source_vocabulary)
            target_ids = make_vocabulary(args.TARGET, max_vocab_size=args.max_target_vocabulary)
        write_vocab(source_ids, args.SOURCE_VOCAB), write_vocab(target_ids, args.TARGET_VOCAB)
    else:
        source_ids = load_vocabulary(args.SOURCE_VOCAB)
        target_ids = load_vocabulary(args.TARGET_VOCAB)

    print('source_vocab: {} target_vocab: {}'.format(len(source_ids), len(target_ids)))

    print('load data sets')
    if args.use_dataset_api:
        # if you get out of  CPU memory, this potion can reduce the memory usage
        def _filter_func(s, t):
            sl = len(s.strip().split())  # number of words in source line
            tl = len(t.strip().split())  # number of words in target line
            return (
                    args.min_source_sentence <= sl <= args.max_source_sentence and
                    args.min_target_sentence <= tl <= args.max_target_sentence)

        train_data = load_data_using_dataset_api(src_vocab=source_ids, src_path=args.SOURCE,
                                                 target_vocab=target_ids, target_path=args.TARGET,
                                                 filter_func=_filter_func)
    else:
        train_source = load_data(source_ids, args.SOURCE)
        train_target = load_data(target_ids, args.TARGET)

        train_data = [
            (s, t)
            for s, t in zip(train_source, train_target)
            if (args.min_source_sentence <= len(s) <= args.max_source_sentence
                and
                args.min_target_sentence <= len(t) <= args.max_target_sentence)
        ]

    dev_data = []
    if args.validation_source and args.validation_target:
        dev_source = load_data(source_ids, args.validation_source)
        dev_target = load_data(target_ids, args.validation_target)

        dev_data = [
            (s, t)
            for s, t in zip(dev_source, dev_target)
            if (args.min_source_sentence <= len(s) <= args.max_source_sentence
                and
                args.min_target_sentence <= len(t) <= args.max_target_sentence)
        ]

    print('[{}] Dataset loaded.'.format(datetime.datetime.now()))
    print('train: {} dev: {}'.format(len(train_data), len(dev_data)))

    # Setup model
    if args.model == 'gru':
        encoder, decoder = L.NStepGRU, L.NStepGRU
        cell, bi = False, False
    elif args.model == 'lstm':
        encoder, decoder = L.NStepLSTM, L.NStepLSTM
        cell, bi = True, False
    elif args.model == 'bigru':
        encoder, decoder = L.NStepBiGRU, L.NStepGRU
        cell, bi = False, True
    elif args.model == 'bilstm':
        encoder, decoder = L.NStepBiLSTM, L.NStepLSTM
        cell, bi = True, True

    if args.attention == 'standard':
        attention = False
    elif args.attention == 'global':
        attention = True

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backend.cuda.get_device_from_id(args.gpu).use()

    model = Seq2Seq(args.layer, len(source_ids), len(target_ids), args.unit, encoder, decoder,
                    src_embed_init=source_vector, target_embed_init=target_vecotr, attention=attention,
                    cell=cell, bi=bi, feeding=args.feeding, dropout=0.1, debug=args.debug_mode, same_vocab=args.SAME_VOCAB)

    if args.gpu >= 0:
        model.to_gpu()  # Copy the model to the GPU

    # Setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Setup iterator
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    updater = training.updaters.StandardUpdater(train_iter, optimizer, converter=convert)

    # Early Stopping
    if args.early_stop and dev_data:
        stop_trigger = training.triggers.EarlyStoppingTrigger(monitor='validation/main/loss', max_trigger=(args.epoch, 'epoch'))
    else:
        stop_trigger = (args.epoch, 'epoch')
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Take a snapshot
    if args.snapshot_divide:
        trainer.extend(
            extensions.snapshot(filename='snapshot_iter_{.updater.epoch}'), trigger=(args.snapshot_interval, 'iteration'))
        # extensions.snapshot(filename='snapshot_iter_{.updater.iteration}'), trigger=(args.snapshot_interval, 'iteration'))
    else:
        trainer.extend(extensions.snapshot(filename='snapshot_latest'), trigger=(args.snapshot_interval, 'iteration'))

    if dev_data:
        # Evaluate the model with the test dataset for each epoch
        trainer.extend(CalculatePerplexity(model, dev_data, 'validation/main'), trigger=(args.validation_interval, 'iteration'))
        trainer.extend(CalculatePerplexity(model, dev_data, 'validation/main'), trigger=(1, 'epoch'))
        monitor = ['epoch', 'iteration', 'main/loss', 'main/perp', 'validation/main/loss', 'validation/main/perp', 'elapsed_time']
    else:
        monitor = ['epoch', 'iteration', 'main/loss', 'main/perp', 'elapsed_time']

    trainer.extend(extensions.LogReport(trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(monitor), trigger=(args.log_interval, 'iteration'))

    # Save a model
    if args.save_epoch:
        trainer.extend(extensions.snapshot_object(model, 'epoch{.updater.epoch}_model.npz'), trigger=(1, 'epoch'))
    else:
        if dev_data:
            record_trigger = training.triggers.MinValueTrigger('validation/main/perp', (1, 'epoch'))
        else:
            record_trigger = training.triggers.MinValueTrigger('main/perp', (1, 'epoch'))
        trainer.extend(extensions.snapshot_object(model, 'best_model.npz'), trigger=record_trigger)


    # trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.iteration}'),
    #                trigger=(args.validation_interval, 'iteration'))
    if args.progressbar:
        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())

    import os
    # Save vocabulary and model's setting
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    # vocab_path = os.path.join(args.out, 'vocab.json')
    # model_path = os.path.join(args.out, 'best_model.npz')
    model_setup = args.__dict__
    model_setup['source_vocab_size'] = len(source_ids)
    model_setup['target_vocab_size'] = len(target_ids)
    model_setup['train_data_size'] = len(train_data)
    model_setup['dev_data_size'] = len(dev_data)
    # model_setup['datetime'] = current_datetime
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)

    if args.resume is not None:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer, strict=False)
    print('start training')

    trainer.run()

    if args.save is not None:
        # Save a snapshot
        chainer.serializers.save_npz(args.save, trainer)


if __name__ == '__main__':
    main()
