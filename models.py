import torch
import torch.nn as nn
import torch.nn.functional as F


# The model structure basically follows "https://github.com/sweetcocoa/DeepComplexUNetPyTorch",
# and the complex convolution and complex batch normalization is based on "https://github.com/litcoderr/ComplexCNN"


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        self.tconv_re = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.InstanceNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.InstanceNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_mode='zeros'):
        super().__init__()
        self.conv = ComplexConv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            padding_mode=padding_mode,
            bias=False
        )
        self.bn = ComplexBatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.deconv = ComplexConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False
        )
        self.bn = ComplexBatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DCUNet20(nn.Module):
    def __init__(self, channels=45, num_layers=20, padding_mode='zeros'):
        super().__init__()
        self.init_params(channels=channels, num_layers=num_layers)
        self.encoders, self.decoders = [], []
        self.enc_dec_depth = num_layers // 2
        for i in range(self.enc_dec_depth):
            block = Encoder(
                self.enc_params['channel'][i],
                self.enc_params['channel'][i + 1],
                kernel_size=self.enc_params['kernel_size'][i],
                stride=self.enc_params['stride'][i],
                padding=self.enc_params['padding'][i],
                padding_mode=padding_mode
            )
            self.add_module(f'enc_{i}', block)
            self.encoders.append(block)

            block = Decoder(
                self.dec_params['channel'][i] + self.enc_params['channel'][-i - 1],
                self.dec_params['channel'][i + 1],
                kernel_size=self.dec_params['kernel_size'][i],
                stride=self.dec_params['stride'][i],
            )
            self.add_module(f'dec_{i}', block)
            self.decoders.append(block)

        self.padding_mode = padding_mode
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        inters = []
        for i, enc in enumerate(self.encoders):
            inters.append(x)
            x = enc(x)

        for i, dec in enumerate(self.decoders):
            x = dec(x)
            if i < len(self.decoders) - 1:
                x = self.pad_n_cat(x, inters[-i - 1], dim=1)

        return x

    def pad_n_cat(self, dec_out, inter, dim=1):
        diff_row = inter.size(2) - dec_out.size(2)
        diff_col = inter.size(3) - dec_out.size(3)
        complex_pad = lambda x : torch.stack([
            F.pad(x[..., 0], (diff_col // 2, diff_col - diff_col // 2, diff_row // 2, diff_row - diff_row // 2)),
            F.pad(x[..., 1], (diff_col // 2, diff_col - diff_col // 2, diff_row // 2, diff_row - diff_row // 2))
        ], dim=-1)
        dec_out = complex_pad(dec_out)
        return torch.cat([dec_out, inter], dim=dim)

    def init_params(self, channels, num_layers):
        if num_layers == 20:
            self.enc_params = {
                'channel': [
                    1,
                    channels,
                    channels,
                    channels * 2,
                    channels * 2,
                    channels * 2,
                    channels * 2,
                    channels * 2,
                    channels * 2,
                    channels * 2,
                    128
                ],
                'kernel_size': [
                    (7, 1), (1, 7),
                    (7, 5), (7, 5),
                    (5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3)
                ],
                'stride': [
                    (1, 1), (1, 1),
                    (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1)
                ],
                'padding': [
                    (3, 0), (0, 3),
                    (3, 2), (3, 2),
                    (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)
                ]
            }
            self.dec_params = {
                'channel': [
                    0,
                    channels * 2,
                    channels * 2,
                    channels * 2,
                    channels * 2,
                    channels * 2,
                    channels * 2,
                    channels * 2,
                    channels,
                    channels,
                    1
                ],
                'kernel_size': [
                    (5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3),
                    (7, 5), (7, 5), 
                    (1, 1), (1, 1)
                ],
                'stride': [
                    (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), 
                    (1, 1), (1, 1)
                ],
            }
        else:
            raise ValueError('Invalid "num_layers", should be 10 or 20.')

class DeepConvolutionalUNet(nn.Module):
    def __init__(self, hidden_size=257):
        super().__init__()
        self.n_fft = (hidden_size - 1) * 2
        self.net = DCUNet20(channels=45, num_layers=20, padding_mode='zeros')

    def stft(self, x):
        return torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.n_fft // 4, 
            win_length=self.n_fft, 
            center=True, 
            normalized=False, 
            onesided=True,
            pad_mode='reflect',
            window=torch.hann_window(self.n_fft).to(x.device)
        )

    def istft(self, x):
        return torch.istft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.n_fft // 4, 
            win_length=self.n_fft, 
            center=True, 
            normalized=False, 
            onesided=True, 
            window=torch.hann_window(self.n_fft).to(x.device),
            length=(x.size(2) - 1) * self.n_fft // 4
        )

    def complex_multiply(self, m, y):
        mr, mi = m[..., 0], m[..., 1]
        yr, yi = y[..., 0], y[..., 1]
        return torch.stack([mr * yr - mi * yi, mr * yi + mi * yr], dim=-1)

    def forward(self, x):
        spec = self.stft(x) # B x D x T x 2
        spec = spec.unsqueeze(1) # B x 1 x D x T x 2
        e = self.net(spec) # B x 1 x D x T x 2
        # e = self.complex_multiply(e, spec) # B x 1 x D x T x 2
        e = e * spec
        e.squeeze_(1) # B x D x T x 2
        e = self.istft(e)
        return e
