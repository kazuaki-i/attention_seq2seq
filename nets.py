import numpy

from operator import itemgetter

import chainer
from chainer import backend
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

UNK, EOS = 0, 1


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


class Translator(chainer.Chain):
    def __init__(self, model, dropout=0.1, debug=False):
        super(Translator, self).__init__()
        with self.init_scope():
            self.model = model

        self.dropout = dropout
        self.debug = bool(debug)
        self.cell = self.model.cell
        self.eos = self.xp.array([EOS], numpy.int32)

    def forward(self, xs, ys, train=True):
        with chainer.using_config('debug', self.debug):
            batch_size = len(xs)
            os, ys_out = self.model(xs, ys)

            concat_os = F.concat(os, axis=0)
            concat_ys_out = F.concat(ys_out, axis=0)
            loss = F.sum(F.softmax_cross_entropy(
                self.model.W(concat_os), concat_ys_out, reduce='no')) / batch_size

            n_words = concat_ys_out.shape[0]
            perp = self.xp.exp(loss.array * batch_size / n_words)

            if train:
                chainer.report({'loss': loss}, self)
                chainer.report({'perp': perp}, self)
                return loss
            else:
                return loss, perp


class Utils:
    def __init__(self, embed_x, embed_y, n_units, cell=False):
        self.embed_x = embed_x
        self.embed_y = embed_y
        self.n_units = n_units
        self.cell = cell

    def eos_remover(self, result):
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs

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


class Attention(chainer.Chain):
    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units, rnn_model,
                 src_embed_init=None, target_embed_init=None, cell=False, same_vocab=True):
        super(Attention, self).__init__()
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
            self.encoder = rnn_model(n_layers, n_units, n_units, 0.1)
            self.decoder = rnn_model(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_target_vocab)
            self.AW = L.Linear(n_units*2, n_units)
            self.attention_eh = L.Linear(n_units, n_units)
            self.attention_hh = L.Linear(n_units, n_units)
            self.attention_hw = L.Linear(n_units, 1)

        self.n_layers = n_layers
        self.n_units = n_units
        self.cell = cell
        self.utils = Utils(self.embed_x, self.embed_y, n_units)

    def masking(self, x):
        m = self.xp.sum(self.xp.absolute(x.data), axis=-1) > 0.
        m = m[:, None]
        m = self.xp.tile(m, (1, x.shape[-1]))
        return m

    def forward(self, xs, ys):
        # Setup and embed xs
        input_len = [len(x) for x in xs]
        xs = self.xs_padding(xs)
        batch_size, max_len = len(xs), len(xs[0])

        # Seq2seq Encoder
        if self.cell:
            h, c, ox = self.encoder(None, None, xs)
        else:
            h, ox = self.encoder(None, xs)

        # Setup for decoder parameter
        oy = ox[-1]
        ox = F.reshape(F.concat(ox, axis=0), (batch_size, oy.shape[0], oy.shape[1]))
        eh = self.attention_eh(F.reshape(ox, (batch_size*max_len, self.n_units)))

        divider = self.xp.tile(self.xp.reshape(self.xp.array(input_len), (batch_size, 1, 1)), (1, max_len, self.n_units))
        ox = ox / divider

        # make masks
        mask1 = self.masking(eh)
        ml = [([False]*max_len + [True]*l)[-max_len:] for l in input_len]
        mask2 = self.xp.array(ml, self.xp.bool)

        # mask2 = self.xp.array([self.xp.array((, numpy.bool)
        zeros = self.xp.full(mask1.shape, 0., self.xp.float32)
        n_inf = self.xp.full(mask2.shape, float('-inf'), self.xp.float32)

        # Setup and embed ys
        ys_in, ys_out = self.ys_padding(ys)

        # Seq2seq decoder with global attention
        os = []
        for ys in ys_in:
            # Calculate attention
            a = self._attention(ys, h, ox, eh, batch_size, max_len, mask1, mask2, zeros, n_inf)

            # # Seq2seq Decoder
            if self.cell:
                h, c, oy = self.decoder(h, c, a)
            else:
                h, oy = self.decoder(h, a)

            oy = F.concat(oy, axis=0)
            os.append(oy)

        return os, ys_out

    def _attention(self, ys, h, ox, eh, batch_size, max_len, mask1, mask2, zeros, n_inf):
        # apply W_hh and shaping for eh calculation
        hh = F.reshape(h, (self.n_layers*batch_size, self.n_units))
        hh = self.attention_hh(hh)
        hh = F.tile(hh, (max_len, 1))

        # sum between hh & eh, and apply tanh
        hh = hh + eh
        hh = F.tanh(hh)
        hh = F.where(mask1, hh, zeros)

        # apply W_hw and softmax
        hw = self.attention_hw(hh)
        hw = F.reshape(hw, (batch_size, max_len))
        hw = F.where(mask2, hw, n_inf)
        hw = F.softmax(hw, axis=1)

        # shaping for ox and　multiplication hw and ox
        hw = F.reshape(hw, (batch_size, max_len, 1))
        hw = F.repeat(hw, self.n_units, axis=-1)
        ha = F.sum(hw * ox, axis=1)

        # attention layer's output
        ys = F.reshape(ys, (batch_size, self.n_units))
        ha = F.reshape(ha, (batch_size, self.n_units))
        a = F.concat([ys, ha], axis=-1)
        a = F.tanh(self.AW(a))

        # split for rnn
        a = F.split_axis(a, batch_size, axis=0)

        return a

    #
    # def _attention(self, ys, ox, eh, batch_size, max_len, mask1, mask2, zeros, n_inf):
    #
    #     hh = self.attention_hh(ys)
    #     hh = F.tile(hh, (max_len, 1))
    #
    #     h = hh + eh
    #     h = F.tanh(h)
    #
    #     zeros = self.xp.full(eh.shape, 0., self.xp.float32)
    #     h = F.where(mask1, h, zeros)
    #
    #     # Calculate hw with aligning hh shape to eh
    #     hw = self.attention_hw(h)
    #
    #
    #     print(hw)
    #
    #     # Softmax and Collect attention parameter
    #     ha = F.reshape(F.softmax(hw, axis=1), (batch_size, max_len, 1))
    #
    #     print(ha)
    #     ha = F.repeat(ha, self.n_units, axis=2)
    #
    #     oa = F.reshape(F.sum(ha * ox, axis=1), (batch_size, self.n_units))
    #     oa = F.split_axis(oa, oa.shape[0], axis=0)
    #     exit()
    #     return oa

    def xs_padding(self, xs):
        xs = [self.xp.array(x, numpy.int32) for x in xs]
        xs = F.pad_sequence(xs, padding=-1)
        xs = xs[:, ::-1]
        # mask = xs.data >= 0
        exs = self.embed_x(xs)
        exs = [F.reshape(x, x.shape[1:]) for x in F.split_axis(exs, len(xs), axis=0)]
        return exs

    def ys_padding(self, ys):
        eos = self.xp.array([EOS], self.xp.int32)
        ys_in = [F.concat([eos, self.xp.array(y, self.xp.int32)], axis=0) for y in ys]
        ys_in = self.embed_y(F.pad_sequence(ys_in, padding=-1))
        ys_in = F.split_axis(ys_in, ys_in.shape[1], axis=1)

        ys_out = [F.concat([self.xp.array(y, self.xp.int32), eos], axis=0) for y in ys]
        ys_out = F.pad_sequence(ys_out, padding=-1)

        return ys_in, ys_out


    def translate(self, xs, max_length=10):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # Setup and embed xs
            xs = self.xs_padding(xs)
            batch_size, max_len = len(xs), len(xs[0])

            # Seq2seq Encoder
            if self.cell:
                h, c, ox = self.encoder(None, None, xs)
            else:
                h, ox = self.encoder(None, xs)

            # Setup for decoder parameter
            oy = ox[-1]
            ox = F.reshape(F.concat(ox, axis=0), (batch_size, oy.shape[0], oy.shape[1]))
            eh = self.attention_eh(F.reshape(ox, (batch_size*max_len, self.n_units)))

            ys = self.xp.full(batch_size, EOS, numpy.int32)
            result = []
            for i in range(max_length):
                ys = self.embed_y(ys)
                oa = self._attention(ys, ox, eh, batch_size, max_len)
                # Seq2seq Decoder
                if self.cell:
                    h, c, oy = self.decoder(h, c, oa)
                else:
                    h, oy = self.decoder(h, oa)
                cys = F.reshape(F.concat(ys, axis=0), (batch_size, -1))
                wy = self.W(cys)
                ys = self.xp.argmax(wy.array, axis=1).astype(numpy.int32)
                result.append(ys)

                oy = F.concat(oy, axis=0)
                # result.append(oy)
        result = self.xp.concatenate([x[None, :] for x in result]).T

        return self.utils.eos_remover(result)

    def translate_beam_search(self, xs, max_length=10, beam_width=3):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = self.xs_padding(xs)
            batch_size, max_len = len(xs), len(xs[0])

            if self.cell:
                h, c, ox = self.encoder(None, None, xs)
            else:
                h, ox = self.encoder(None, xs)

            ps = self.xp.full(batch, 0, numpy.float32)
            py = self.xp.array([[EOS]*batch], numpy.int32)

            oy = ox[-1]
            ox = F.reshape(F.concat(ox, axis=0), (batch_size, oy.shape[0], oy.shape[1]))
            eh = self.attention_eh(F.reshape(ox, (batch_size*max_len, self.n_units)))

            L = [[ps, py, h]]*beam_width

            for j in range(max_length):
                S, V, H, C = [], [], [], []
                for Y in L:
                    # Calculate attention
                    ys = self.embed_y(Y[1][-1])
                    oa = self._attention(ys, ox, eh, batch_size, max_len)

                    if self.cell:
                        h, c, ys = self.decoder(Y[2], Y[3], F.split_axis(oa, oa.shape[0], axis=0))
                    else:
                        h, ys = self.decoder(Y[2], F.split_axis(oa, oa.shape[0], axis=0))

                    cys = F.concat(ys, axis=0)
                    wy = self.W(cys)
                    score = self.xp.sort(wy.array, axis=1)[:, ::-1][:, :10].astype(numpy.float32)
                    ys = self.xp.argsort(wy.array, axis=1)[:, ::-1][:, :10].astype(numpy.int32)

                    for i in range(beam_width):
                        S.append(numpy.array(Y[0] + numpy.log(score[:, i])))
                        a = numpy.concatenate([Y[1], [ys[:, i]]], axis=0)
                        V.append(a)
                        H.append(h)
                        if self.cell:
                            C.append(c)

                cs = c.shape if self.cell else None
                L = self.utils.beam_ranking(S, V, H, C, batch, beam_width, h.shape, cs)

        result = L[0][1].T[:, 1:]

        return self.utils.eos_remover(result)


class Standard(chainer.Chain):
    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units, rnn_model,
                 src_embed_init=None, target_embed_init=None, cell=False, same_vocab=True):
        super(Standard, self).__init__()
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
            self.encoder = rnn_model(n_layers, n_units, n_units, 0.1)
            self.decoder = rnn_model(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_target_vocab)
            self.model_name = 'gru'

        self.n_layers = n_layers
        self.n_units = n_units
        self.cell = cell
        self.utils = Utils(self.embed_x, self.embed_y, n_units)

    def forward(self, xs, ys):
        xs = [self.xp.array(x[::-1], numpy.int32) for x in xs]

        eos = self.xp.array([EOS], numpy.int32)
        ys_in = [F.concat([eos, self.xp.array(y, numpy.int32)], axis=0) for y in ys]
        ys_out = [F.concat([self.xp.array(y, numpy.int32), eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        # None represents a zero vector in an encoder.

        if self.cell:
            hx, _, _ = self.encoder(None, exs)
            _, _, os = self.decoder(hx, eys)
        else:
            hx, _ = self.encoder(None, exs)
            _, os = self.decoder(hx, eys)

        return os, ys_out

    def translate(self, xs, max_length=10):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            if self.cell:
                h, c, _ = self.encoder(None, None, exs)
            else:
                h, _ = self.encoder(None, exs)

            ys = self.xp.full(batch, EOS, numpy.int32)
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                if self.cell:
                    h, c, ys = self.decoder(h, c, eys)
                else:
                    h, ys = self.decoder(h, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.array, axis=1).astype(numpy.int32)
                result.append(ys)

        result = self.xp.concatenate([x[None, :] for x in result]).T

        return self.utils.eos_remover(result)

    def translate_beam_search(self, xs, max_length=10, beam_width=3):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)

            if self.cell:
                h, c, _ = self.encoder(None, None, exs)
                pass
            else:
                h, _ = self.encoder(None, exs)
            ps = self.xp.full(batch, 0, numpy.float32)
            py = numpy.array([[EOS]*batch], numpy.int32)

            L = [[ps, py, h]]*beam_width

            for j in range(max_length):
                S, V, H, C = [], [], [], []
                for Y in L:
                    eys = self.embed_y(Y[1][-1])
                    eys = F.split_axis(eys, batch, 0)

                    if self.cell:
                        h, c, ys = self.decoder(Y[2], Y[3], eys)
                    else:
                        h, ys = self.decoder(Y[2], eys)

                    cys = F.concat(ys, axis=0)
                    wy = self.W(cys)
                    score = self.xp.sort(wy.array, axis=1)[:, ::-1][:, :10].astype(numpy.float32)
                    ys = self.xp.argsort(wy.array, axis=1)[:, ::-1][:, :10].astype(numpy.int32)

                    for i in range(beam_width):
                        S.append(numpy.array(Y[0] + numpy.log(score[:, i])))
                        a = numpy.concatenate([Y[1], [ys[:, i]]], axis=0)
                        V.append(a)
                        H.append(h)
                        if self.cell:
                            C.append(c)

                cs = c.shape if self.cell else None
                L = self.utils.beam_ranking(S, V, H, C, batch, beam_width, h.shape, cs)

        result = L[0][1].T[:, 1:]

        return self.utils.eos_remover(result)

