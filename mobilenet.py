# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,unused-argument,missing-docstring
"""MobileNet and MobileNetV3, implemented in Gluon."""
__all__ = [
    'MobileNetV3',
    'mobilenet_v3_1_0',
    'get_mobilenet_v3']

__modify__ = 'wzw'
__modified_date__ = '19/05/19'

import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock


# Helpers
class RELU6(nn.HybridBlock):
    """Relu6 used in MobileNetV3."""

    def __init__(self, **kwargs):
        super(RELU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")

class HSwish(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return x*(F.clip(x+3, 0, 6, name='hswish')/6)

class HSigmoid(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HSigmoid, self).__init__(*kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x+3, 0, 6, name='hsigmoid')/6

# pylint: disable= too-many-arguments
def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, hswish=False, norm_layer=BatchNorm, norm_kwargs=None):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(norm_layer(scale=True, **({} if norm_kwargs is None else norm_kwargs)))
    if active:
        out.add(HSwish() if hswish else nn.Activation('relu'))

class SEBlock(nn.HybridBlock):
    def __init__(self, se_ratio, channels):
        super(SEBlock,self).__init__()
        self.se = nn.HybridSequential()
        self.se.add(nn.Conv2D(int(se_ratio*channels),kernel_size=1,padding=0))
        self.se.add(nn.Activation('relu'))
        self.se.add(nn.Conv2D(channels,kernel_size=1,padding=0))
        #self.se.add(nn.Activation('sigmoid'))
        self.se.add(HSigmoid())

    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.se(w)
        x = F.broadcast_mul(x,w)
        return x


class LinearBottleneck(nn.HybridBlock):
    r"""
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, in_channels, channels, exp_size, stride, use_se, kernel_size,hswish,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        self.use_se = use_se
        with self.name_scope():
            self.out = nn.HybridSequential()
            self.out2 = nn.HybridSequential()
            _add_conv(self.out,
                      exp_size,
                      hswish=hswish,
                      norm_layer=BatchNorm, norm_kwargs=None)
            _add_conv(self.out,
                      exp_size,
                      kernel=kernel_size,
                      stride=stride,
                      pad=(kernel_size-1)//2,
                      num_group=exp_size,
                      hswish=hswish,
                      norm_layer=BatchNorm, norm_kwargs=None)
            if self.use_se:
                self.se = SEBlock(0.25, exp_size)
            _add_conv(self.out2,
                      channels,
                      active=False,
                      hswish=hswish,
                      norm_layer=BatchNorm, norm_kwargs=None)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_se:
            out = self.se(out)
        out = self.out2(out)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out

class MobileNetV3Large(nn.HybridBlock):
    r"""
    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, multiplier=1.0, classes=1000,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(MobileNetV3Large, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                _add_conv(self.features, int(16 * multiplier), kernel=3,
                          stride=2, pad=1, hswish=True, norm_layer=BatchNorm, norm_kwargs=None)

                in_channels_group = [int(x * multiplier) for x in [16]*2 + [24] * 2
                                     + [40] * 3 + [80] * 4 + [112] * 3 + [160]]
                channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [40] * 3
                                  + [80] * 4 + [112] * 3 + [160] * 2]
                exp_size = [int(x*multiplier) for x in [16] + [64] + [72] * 2 + [120] * 2 + [240] + [200] + [184] * 2 + \
                           [480] + [672] * 3 + [960]]
                strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1]
                use_se = [False]*3 + [True]*3 + [False]*4 + [True]*5
                kernel_size = [3]*3 + [5]*3 + [3]*6 +[5]*3
                use_hswish = [False]*6 + [True]*9

                for in_c, c, exp, s, se, kz, hswish in zip(in_channels_group, channels_group, exp_size, strides, use_se, kernel_size, use_hswish):
                    self.features.add(LinearBottleneck(in_channels=in_c,
                                                       channels=c,
                                                       exp_size=exp,
                                                       stride=s,
                                                       use_se=se,
                                                       kernel_size=kz,
                                                       hswish=hswish,
                                                       norm_layer=BatchNorm, norm_kwargs=None))
                _add_conv(self.features,int(960*multiplier),hswish=True,
                          norm_layer=BatchNorm, norm_kwargs=None)
                self.features.add(nn.GlobalAvgPool2D())
                #self.features.add(HSwish())

                last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
                self.features.add(nn.Conv2D(last_channels,1,1))
                self.features.add(HSwish())

            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                self.output.add(
                    nn.Conv2D(classes, 1, use_bias=False, prefix='pred_'),
                    nn.Flatten()
                )

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

class MobileNetV3Small(nn.HybridBlock):
    r"""
    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, multiplier=1.0, classes=1000,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(MobileNetV3Small, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                _add_conv(self.features, int(16 * multiplier), kernel=3,
                          stride=2, pad=1, hswish=True, norm_layer=BatchNorm, norm_kwargs=None)

                in_channels_group = [int(x * multiplier) for x in [24]*3 + [40] * 4
                                     + [48] + [96] * 3]
                channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [40] * 3
                                  + [48] * 2 + [96] * 3]
                exp_size = [int(x*multiplier) for x in [16,72,88,96] + [240] * 2 + [120,144,288] + [576]*2]
                strides = [2] * 2 + [1,2] + [1] * 4 + [2] + [1]*2
                use_se = [True] + [False]*2 + [True]*8
                kernel_size = [3]*3 + [5]*8
                use_hswish = [False]*3 + [True]*8
                for in_c, c, exp, s, se, kz, hswish in zip(in_channels_group, channels_group, exp_size, strides, use_se, kernel_size, use_hswish):
                    self.features.add(LinearBottleneck(in_channels=in_c,
                                                       channels=c,
                                                       exp_size=exp,
                                                       stride=s,
                                                       use_se=se,
                                                       kernel_size=kz,
                                                       hswish = hswish,
                                                       norm_layer=BatchNorm, norm_kwargs=None))
                _add_conv(self.features,int(576*multiplier),hswish=True,
                          norm_layer=BatchNorm, norm_kwargs=None)
                #self.features.add(SEBlock(0.25,576))
                self.features.add(nn.GlobalAvgPool2D())
                #self.features.add(HSwish())

                last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
                _add_conv(self.features,last_channels,hswish=True,norm_layer=BatchNorm,norm_kwargs=None)

            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                self.output.add(
                    nn.Conv2D(classes, 1, use_bias=False, prefix='pred_'),
                    nn.Flatten()
                )

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_mobilenet_v3_large(multiplier, pretrained=False, ctx=cpu(),
                     root='~/.mxnet/models', norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    r"""

    Parameters
    ----------
    multiplier : float
        The width multiplier for controlling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    net = MobileNetV3Large(multiplier, norm_layer=BatchNorm, norm_kwargs=None, **kwargs)
    return net

def get_mobilenet_v3_small(multiplier, pretrained=False, ctx=cpu(),
                     root='~/.mxnet/models', norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    r"""

    Parameters
    ----------
    multiplier : float
        The width multiplier for controlling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    net = MobileNetV3Small(multiplier, norm_layer=BatchNorm, norm_kwargs=None, **kwargs)
    return net

if __name__ == '__main__':
    net = get_mobilenet_v3_small(0.75)
    import gluoncv as gcv
    #net = gcv.model_zoo.get_model('mobilenetv2_1.0',pretrained=True)
    net.initialize()
    net(mx.nd.zeros((1,3,224,224),ctx=mx.cpu(),dtype='float32'))
    net.save_parameters('mobilenet_v3.params')
    params = net.collect_params()
    total_param = 0
    for param in params:
        print(param)
        param_shape = params[param].shape
        cur_param = 1
        for s in param_shape:
            cur_param *= s
        total_param += cur_param
    print('total params:%.2fM'%(total_param/1024/1024))

    gcv.utils.export_block('mobilenet_v3',net,data_shape=(224,224,3),layout='CHW',preprocess=False)
    print('finish!')
