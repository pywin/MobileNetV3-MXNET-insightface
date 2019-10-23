#-*- coding:utf-8 -*-
import mxnet as mx
from mxnet.gluon import nn
import symbol_utils
from config import config

model_config = {
    'large': [
        [3, 16, 16, False, 're', 1],
        [3, 64, 24, False, 're', 2],
        [3, 72, 24, False, 're', 1],
        [5, 72, 40, True, 're', 2],
        [5, 120, 40, True, 're', 1],
        [5, 120, 40, True, 're', 1],
        [3, 240, 80, False, 'hs', 2],
        [3, 200, 80, False, 'hs', 1],
        [3, 184, 80, False, 'hs', 1],
        [3, 184, 80, False, 'hs', 1],
        [3, 480, 112, True, 'hs', 1],
        [3, 672, 112, True, 'hs', 1],
        [5, 672, 160, True, 'hs', 2],
        [5, 960, 160, True, 'hs', 1],
        [5, 960, 160, True, 'hs', 1]
    ],
    'small': [
        [3, 16, 16, True, 're', 2],
        [3, 72, 24, False, 're', 2],
        [3, 88, 24, False, 're', 1],
        [5, 96, 40, True, 'hs', 1],
        [5, 240, 40, True, 'hs', 1],
        [5, 240, 40, True, 'hs', 1],
        [5, 120, 48, True, 'hs', 1],
        [5, 144, 48, True, 'hs', 1],
        [5, 288, 96, True, 'hs', 2],
        [5, 576, 96, True, 'hs', 1],
        [5, 576, 96, True, 'hs', 1],
    ]
}


class HardSwish(nn.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(HardSwish, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return x * F.minimum(F.maximum(x + 3, 0), 6) / 6.


class HardSigmoid(nn.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(HardSigmoid, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.minimum(F.maximum(x + 3, 0), 6) / 6.


class SEBlock(nn.HybridBlock):
    def __init__(self, in_size, r=4, prefix=None, params=None):
        super(SEBlock, self).__init__(prefix=prefix, params=params)
        self.excitation = nn.HybridSequential()
        self.excitation.add(nn.Dense(in_size // r, 'relu'))
        self.excitation.add(nn.Dense(in_size))
        self.excitation.add(HardSigmoid())

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = F.contrib.AdaptiveAvgPooling2D(data=x, output_size=(1, 1))
        y = self.excitation(y)
        y = F.expand_dims(F.expand_dims(y, axis=2), axis=3)
        z = F.broadcast_mul(x, y)
        return z


class BneckBlock(nn.HybridBlock):
    def __init__(self, kernel_size, expand_size, out_size, nolinear,
                 seblock, strides, prefix=None, params=None):
        super(BneckBlock, self).__init__(prefix=prefix, params=params)
        self.strides = strides
        self.seblock = seblock

        self.start_block = nn.HybridSequential()
        self.start_block.add(nn.Conv2D(expand_size, kernel_size=1, strides=1, padding=0,
                                       use_bias=False))
        self.start_block.add(nn.BatchNorm())

        self.start_block.add(nn.Conv2D(expand_size, kernel_size=kernel_size, strides=strides,
                                       padding=kernel_size // 2, groups=expand_size, use_bias=False))
        self.start_block.add(nn.BatchNorm())
        self.start_block.add(nolinear)

        self.end_block = nn.HybridSequential()
        self.end_block.add(nn.Conv2D(out_size, kernel_size=1, strides=1, padding=0,
                                     use_bias=False))
        self.end_block.add(nn.BatchNorm())
        self.end_block.add(nolinear)

        self.skip = nn.HybridSequential()
        self.skip.add(nn.Conv2D(out_size, kernel_size=1, strides=strides, padding=0,
                                use_bias=False))
        self.skip.add(nn.BatchNorm())

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = self.start_block(x)
        if self.seblock is not None:
            y = y * self.seblock(y)
        z = self.end_block(y)
        t = self.skip(x)
        out = t + z
        return out


class MobileNetV3(nn.HybridBlock):
    def __init__(self, mode, num_classes, prefix=None, params=None):
        super(MobileNetV3, self).__init__(prefix=prefix, params=params)

        self.mode = mode
        assert self.mode in ('large', 'small'), \
            "version is must one of (large, small)!!!"
        self.num_classes = num_classes

        with self.name_scope():
            self.first = nn.HybridSequential()
            self.first.add(
                nn.Conv2D(channels=16, kernel_size=3, strides=2, padding=1, use_bias=False)
            )
            self.first.add(nn.BatchNorm())
            self.first.add(HardSwish())

            self.bnecks = nn.HybridSequential()
            for kernel_size, exp_size, out_size, se, nl, strides in model_config[mode]:
                se = SEBlock(exp_size) if se else None
                if nl == 're':
                    nl = nn.Activation('relu')
                elif nl == 'hs':
                    nl = HardSwish()
                else:
                    raise "cannot use {} activation function".format(nl)

                self.bnecks.add(BneckBlock(kernel_size=kernel_size, expand_size=exp_size,
                                           out_size=out_size, nolinear=nl, seblock=se,
                                           strides=strides))

            self.last = nn.HybridSequential()
            if self.mode == 'small':
                self.last.add(nn.Conv2D(576, kernel_size=1, strides=1, use_bias=False))
                self.last.add(SEBlock(576))
                self.last.add(nn.GlobalAvgPool2D())
                self.last.add(nn.Conv2D(1280, kernel_size=1, strides=1, use_bias=False))
                self.last.add(nn.BatchNorm())
                self.last.add(HardSwish())
                self.last.add(nn.Conv2D(self.num_classes, kernel_size=1))
                self.last.add(nn.BatchNorm())
                self.last.add(HardSwish())
            else:
                self.last.add(nn.Conv2D(960, kernel_size=1, strides=1, use_bias=False))
                self.last.add(nn.BatchNorm())
                self.last.add(HardSwish())
                #self.last.add(nn.GlobalAvgPool2D())
                self.last.add(nn.Conv2D(1280, kernel_size=1, strides=1, use_bias=False))
                self.last.add(HardSwish())
                self.last.add(nn.Conv2D(self.num_classes, kernel_size=1))
            # self.last.add(nn.Flatten())

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.first(x)
        x = self.bnecks(x)
        out = self.last(x)
        return out


def get_symbol():
    num_classes = config.emb_size
    fc_type = config.net_output
    mode = 'large'
    data = mx.sym.Variable(name='data')
    data = data-127.5
    data = data * 0.0078125
    net = MobileNetV3(num_classes=num_classes,mode=mode)
    body = net(data)
    body = symbol_utils.get_fc1(body, num_classes, fc_type)
    return body
