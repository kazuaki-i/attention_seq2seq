# seq2seq with attention

## About

This program is a global attention & standard sequence to sequence model corded by Chainer.

See detail

- [NEURAL MACHINE TRANSLATION
            BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/abs/1409.0473) 
- [Effective Approaches to Attention-based Neural Machine Translation](https://aclweb.org/anthology/D15-1166)

## How to learn

This program required below tokenized source and target data by space.

```
python training.py -S ./train.source -T ./traine.target -SV word2vec.model -TV word2vec.model \
-model gru --pre-wordvector --use-dataset-api --SAME --attention global
```

Important options are below:

- model: encoder and decoder rnn model
- attention: standard seq2seq or global attention
- feeding: use input feeding
- S: source training
- T: target training
- SV: source vocabulary file
- TV: target vocabulary file
- pre-wordvector: use pre-training word vector

## Generate

```
python generator.py -sv word2vec.model -tv word2vec.model -wv -m best_model.npz -p args.json
```
