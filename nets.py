import numpy

from operator import itemgetter

import chainer
from chainer import backend
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

UNK, EOS, BOS = 0, 1, 2


class CalculatePerplexity(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, test_data, key, batch=500, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                loss, perp = self.model.forward(sources, targets, False)

                chainer.report({'{}/loss'.format(self.key): loss})
                chainer.report({'{}/perp'.format(self.key): perp})


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


# class Translator(chainer.Chain):
#     def __init__(self, model, dropout=0.1, debug=False):
#         super(Translator, self).__init__()
#         with self.init_scope():
#             self.model = model
#
#         self.dropout = dropout
#         self.debug = bool(debug)
#         self.cell = self.model.cell
#         self.eos = self.xp.array([EOS], numpy.int32)
#
#     def forward(self, xs, ys, train=True):
#         with chainer.using_config('debug', self.debug):
#             batch_size = len(xs)
#             os, ys_out = self.model(xs, ys)
#
#             concat_os = F.concat(os, axis=0)
#             concat_ys_out = F.concat(ys_out, axis=0)
#             loss = F.sum(F.softmax_cross_entropy(
#                 self.model.Wd(concat_os), concat_ys_out, reduce='no')) / batch_size
#
#             n_words = concat_ys_out.shape[0]
#             perp = self.xp.exp(loss.array * batch_size / n_words)
#
#             if train:
#                 chainer.report({'loss': loss}, self)
#                 chainer.report({'perp': perp}, self)
#                 return loss
#             else:
#                 return loss, perp



class Seq2Seq(chainer.Chain):
    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units, encoder, decoder,
                 src_embed_init=None, target_embed_init=None, attention=False, cell=False, bi=False,
                 dropout=0.1, debug=False, same_vocab=True):
        super(Seq2Seq, self).__init__()
        self.bi = 2 if bi else 1
        with self.init_scope():
            if not isinstance(src_embed_init,  type(numpy.array([]))):
                src_embed_init = chainer.initializers.Uniform(.25)
            self.embed_x = L.EmbedID(n_source_vocab, n_units, initialW=src_embed_init, ignore_label=-1)

            if same_vocab:
                self.embed_y = self.embed_x
            else:
                if not isinstance(target_embed_init,  type(numpy.array([]))):
                    target_embed_init = chainer.initializers.Uniform(.25)
                self.embed_y = L.EmbedID(n_target_vocab, n_units, target_embed_init, ignore_label=-1)
            self.encoder = encoder(n_layers, n_units, n_units, 0.1)
            self.decoder = decoder(n_layers, n_units, n_units*self.bi, 0.1)
            self.Wd = L.Linear(n_units*self.bi, n_target_vocab)
            self.Wa = L.Linear(n_units*2, n_units)
            self.attention_eh = L.Linear(n_units*self.bi, n_units)
            self.attention_hh = L.Linear(n_units, n_units)
            self.attention_hw = L.Linear(n_units*2, 1)

        self.n_layers = n_layers
        self.n_units = n_units
        self.attention = attention
        self.cell = cell
        self.n_inf = -100000000
        self.dropout = dropout
        self.debug = debug

    def masking(self, x):
        m = self.xp.sum(self.xp.absolute(x.data), axis=-1) > 0.
        m = m[:, :, :, None]
        m = self.xp.tile(m, (1, 1, 1, x.shape[-1]))
        return m

    def _input_molding(self, xs, ys):
        # if self.bi == 1:
        #     xs = [self.xp.array(x[::-1], numpy.int32) for x in xs]
        # else:
        #     xs = [self.xp.array(x, numpy.int32) for x in xs]
        xs = [self.xp.array(x[::-1], numpy.int32) for x in xs]

        eos = self.xp.array([EOS], numpy.int32)
        bos = self.xp.array([BOS], numpy.int32)

        ys_in = [F.concat([bos, self.xp.array(y, numpy.int32)], axis=0) for y in ys]
        ys_out = [F.concat([self.xp.array(y, numpy.int32), eos], axis=0) for y in ys]

        return xs, ys_in, ys_out

    def convert_hidden_layer(self, h):
        if self.bi > 1:
            hs = h.shape
            h = F.reshape(h, (self.bi, self.n_layers, hs[-2], hs[-1]))
            # h = F.sum(h, axis=0)
            h = F.transpose(h, (1, 2, 0, 3))
            h = F.reshape(h, (self.n_layers, hs[-2], hs[-1]*self.bi))
        return h

    def forward(self, xs, ys, train=True):
        with chainer.using_config('debug', self.debug), chainer.using_config('train', train):
            x_len_lst = [len(x) for x in xs]
            y_len_lst = [len(y) for y in ys]
            batch_size, x_len, y_len = len(xs), max(x_len_lst), max(y_len_lst) + 1

            print(F.concat(self.translate([xs[0]], max_length=10, leaveEOS=True), axis=0))
            # print(self.translate([xs[0]], max_length=10))
            print(ys[0])

            xs, ys_in, ys_out = self._input_molding(xs, ys)

            # Both xs and ys_in are lists of arrays.
            exs = sequence_embed(self.embed_x, xs)
            eys = sequence_embed(self.embed_y, ys_in)

            # basic encoder and decoder
            if self.cell:
                hx, c, xos = self.encoder(None, None, exs)
                hx = self.convert_hidden_layer(hx)
                _, _, yos = self.decoder(hx, c, eys)
            else:
                hx, xos = self.encoder(None, exs)
                hx = self.convert_hidden_layer(hx)
                _, yos = self.decoder(hx, eys)

            if self.attention:
                xo = F.pad_sequence(xos, padding=0.)
                yo = F.pad_sequence(yos, padding=0.)

                xo = F.reshape(xo, (batch_size, x_len, self.bi, self.n_units))
                yo = F.reshape(yo, (batch_size, y_len, self.bi, self.n_units))

                yo = self.global_attention(xo, yo, x_len, y_len, batch_size)
                yos = [F.reshape(h[:, :len(y), :], (len(y), h.shape[-1]))
                       for h, y in zip(F.split_axis(yo, batch_size, axis=0), ys_out)]

            concat_os = F.concat(yos, axis=0)
            concat_ys_out = F.concat(ys_out, axis=0)
            loss = F.sum(F.softmax_cross_entropy(
                self.Wd(concat_os), concat_ys_out, reduce='no')) / batch_size

            n_words = concat_ys_out.shape[0]
            perp = self.xp.exp(loss.array * batch_size / n_words)

            if train:
                chainer.report({'loss': loss}, self)
                chainer.report({'perp': perp}, self)
                return loss
            else:
                return loss, perp

    def global_attention(self, xo, yo, x_len, y_len, batch_size):
        def _batch_axis(v):
            return len(v.shape) - 1
        # shaping for calculate weighted average
        eh = xo
        hh = yo
        eh = F.repeat(eh, y_len, axis=1)
        hh = F.tile(hh, (1, x_len, 1, 1))

        h = F.concat([eh, hh], axis=-1)
        cond = self.xp.concatenate([self.masking(eh), self.masking(hh)], axis=-1)
        h = F.where(cond, F.tanh(h), self.xp.full(cond.shape, 0., self.xp.float32))

        # apply attention weight and soft max
        h = self.attention_hw(h, n_batch_axes=_batch_axis(h))

        h = F.reshape(h, (batch_size, x_len, y_len, self.bi))
        cond = self.xp.reshape(self.xp.all(cond, axis=-1), h.shape)

        h = F.where(cond, h, self.xp.full(h.shape, float(self.n_inf), self.xp.float32))
        h = F.softmax(h, axis=1)
        h = F.where(cond, h, self.xp.full(h.shape, 0., self.xp.float32))

        # Calculate weighted average
        h = h[:, :, :, :, None]
        h = F.repeat(h, self.n_units, axis=-1)

        xo = xo[:, :, None, :, :]
        xo = F.repeat(xo, y_len, axis=2)

        h = F.sum(h * xo, axis=1)

        # Apply weighted average to yo
        cond = self.xp.logical_and(self.masking(yo), self.masking(h))

        h = F.concat([yo, h], axis=-1)
        h = F.tanh(self.Wa(h, n_batch_axes=_batch_axis(h)))
        h = F.where(cond, h, self.xp.full(h.shape, 0., self.xp.float32))

        # Shaping input form
        h = F.reshape(h, (batch_size, y_len, self.n_units*self.bi))

        return h

    def eos_remover(self, result):
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs

    def translate(self, xs, max_length=10, leaveEOS=False):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x_len_lst = [len(x) for x in xs]
            batch_size, x_len = len(xs), max(x_len_lst)

            xs = [self.xp.array(x[::-1], numpy.int32) for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            if self.cell:
                h, c, xos = self.encoder(None, None, exs)
                h = self.convert_hidden_layer(h)
            else:
                h, xos = self.encoder(None, exs)
                h = self.convert_hidden_layer(h)

            xo = F.pad_sequence(xos, padding=0.)
            xo = F.reshape(xo, (batch_size, x_len, self.bi, self.n_units))

            ys = self.xp.full(batch_size, BOS, numpy.int32)
            y_len = 1
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)

                eys = F.split_axis(eys, batch_size, 0)
                if self.cell:
                    h, c, yos = self.decoder(h, c, eys)
                else:
                    h, yos = self.decoder(h, eys)

                if self.attention:
                    yo = F.pad_sequence(yos, padding=0.)
                    yo = F.reshape(yo, (batch_size, y_len, self.bi, self.n_units))

                    a = self.global_attention(xo, yo, x_len, y_len, batch_size)

                    yos = [F.reshape(y, (y_len, self.n_units*self.bi)) for y in F.split_axis(a, batch_size, axis=0)]

                cys = F.concat(yos, axis=0)
                wy = self.Wd(cys)
                ys = self.xp.argmax(wy.array, axis=1).astype(numpy.int32)
                result.append(ys)

        if leaveEOS:
            return result

        result = self.xp.concatenate([x[None, :] for x in result]).T

        return self.eos_remover(result)

    def beam_ranking(self, S, V, H, C, batch, beam_width, h_shape, c_shape):
        S = numpy.array(S).T
        asS = numpy.argsort(S, axis=1)[:, ::-1][:, :10].astype(numpy.int32)
        sS = numpy.sort(S, axis=1)[:, ::-1][:, :10].astype(numpy.float32)

        V = numpy.array(V, numpy.int32)

        TL = []
        l = len(V[0])
        for i in range(batch):
            aV = V[asS[i], :, i].tolist()
            TL.append([ [[x]*l, a] for n, (x, a) in enumerate(zip(asS[i], aV)) if n <= aV.index(a)][:beam_width])

        TL = numpy.array(TL)

        L = []
        for j in range(beam_width):
            idx = TL[:, :, 0, 0].T[j]
            vocab = TL[:, :, 1, :].T[:, j]

            ns = numpy.array([sS[n][i] for n, i in enumerate(idx)], numpy.float32)
            v = F.concat([H[i][:, n, :] for n, i in enumerate(idx)], axis=0)
            nh = F.reshape(v, h_shape)
            if self.cell:
                c = F.concat([C[i][:, n, :] for n, i in enumerate(idx)], axis=0)
                nc = F.reshape(c, c_shape)
                L.append([ns, vocab, nh, nc])
            else:
                L.append([ns, vocab, nh])

        return L



    # def translate_beam_search(self, xs, max_length=10, beam_width=3):
    #     with chainer.no_backprop_mode(), chainer.using_config('train', False):
    #         x_len_lst = [len(x) for x in xs]
    #         batch_size, x_len = len(xs), max(x_len_lst)
    #
    #         if self.bi == 1:
    #             xs = [self.xp.array(x[::-1], numpy.int32) for x in xs]
    #         else:
    #             xs = [self.xp.array(x, numpy.int32) for x in xs]
    #
    #         exs = sequence_embed(self.embed_x, xs, self.xp)
    #         if self.cell:
    #             h, c, xos = self.encoder(None, None, exs)
    #             pass
    #         else:
    #             h, xos = self.encoder(None, exs)
    #
    #         if self.bi == 1:
    #             xo = [x[::-1, :] for x in xos]
    #             xo = F.pad_sequence(xo, padding=0.)
    #             xo = xo[:, ::-1, :]
    #         else:
    #             xo = F.pad_sequence(xos, padding=0.)
    #         xo = F.reshape(xo, (batch_size, x_len, self.bi, self.n_units))
    #
    #         ps = self.xp.full(batch_size, 0, numpy.float32)
    #         py = numpy.array([[EOS]*batch_size], numpy.int32)
    #         L = [[ps, py, h]]*beam_width
    #         y_len = 1
    #
    #         for j in range(max_length):
    #             S, V, H, C = [], [], [], []
    #             for Y in L:
    #                 eys = self.embed_y(Y[1][-1])
    #                 eys = F.split_axis(eys, batch_size, 0)
    #
    #                 if self.cell:
    #                     h, c, yos = self.decoder(Y[2], Y[3], eys)
    #                 else:
    #                     h, yos = self.decoder(Y[2], eys)
    #
    #                 yo = F.pad_sequence(yos, padding=0.)
    #                 yo = F.reshape(yo, (batch_size, y_len, self.bi, self.n_units))
    #
    #                 h = self.global_attention(xo, yo, x_len, y_len, batch_size, x_len_lst)
    #
    #                 ys = [F.reshape(y, (y_len, self.n_units)) for y in F.split_axis(h, batch_size, axis=0)]
    #
    #                 cys = F.concat(ys, axis=0)
    #                 wy = self.W(cys)
    #                 score = self.xp.sort(wy.array, axis=1)[:, ::-1][:, :10].astype(numpy.float32)
    #                 ys = self.xp.argsort(wy.array, axis=1)[:, ::-1][:, :10].astype(numpy.int32)
    #
    #                 for i in range(beam_width):
    #                     S.append(numpy.array(Y[0] + numpy.log(score[:, i])))
    #                     a = numpy.concatenate([Y[1], [ys[:, i]]], axis=0)
    #                     V.append(a)
    #                     H.append(h)
    #                     if self.cell:
    #                         C.append(c)
    #
    #             cs = c.shape if self.cell else None
    #             L = self.utils.beam_ranking(S, V, H, C, batch_size, beam_width, h.shape, cs)
    #
    #     result = L[0][1].T[:, 1:]
    #
    #     return self.utils.eos_remover(result)
