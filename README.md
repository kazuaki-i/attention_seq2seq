# seq2seq with attention

## About

This program is a global attention & standard sequence to sequence model corded by Chainer

See detail [Global attention](https://arxiv.org/abs/1409.0473)

## How to learn

This program required below tokenized source and target data by space.

```
python ~/workspace/deploy/seq2seq/seq2seq.py -S ./train.source -T ./traine.target -SV word2vec.model -TV word2vec.model \
-model gru --pre-wordvector --use-dataset-api --SAME --attention global
```

Important options are below:

- model: encoder and decoder rnn model
- attention: standard seq2seq or global attention
- S: source training
- T: target training
- SV: source vocabulary file
- TV: target vocabulary file
- pre-wordvector: use pre-training word vector


