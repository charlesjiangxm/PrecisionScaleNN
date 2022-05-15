import numpy as np
import random
import torch
import torch
from torch import Tensor
import torch.nn as nn

torch.set_printoptions(precision=2, threshold=100000, linewidth=1000, sci_mode=None)


class Conv2dLogQuant(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple[int, ...],
                 stride: tuple[int, ...],
                 padding: tuple[int, ...],
                 dilation: tuple[int, ...],
                 transposed: bool,
                 output_padding: tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = Tensor
        self.bias = Tensor

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3


def conv2d_torch(N, Cout, Cin, H, W, R, C, S,
                 weight, bias, inputs,
                 bias_on=True, device=torch.device('cpu')):
    weight = weight.to(device)
    bias = bias.to(device)
    inputs = inputs.to(device)

    m = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=(R, C), stride=S, bias=bias_on).to(device)
    assert m.weight.data.shape == weight.shape, "Error: model weight and new weight shape mis-match."
    m.weight.data = weight
    if bias_on:
        assert m.bias.data.shape == bias.shape, "Error: model bias and new bias shape mis-match."
        m.bias.data = bias
    out = m(inputs)

    return out


def conv2d_my(N, Cout, Cin, H, W, R, C, S,
              weight, bias, inputs,
              pad_on=False, bias_on=True, device=torch.device('cpu')):
    # load input to the target device
    inputs = inputs.to(device)
    weight = weight.to(device)
    if bias_on:
        bias = bias.to(device)

    # derived parameter: padding_on_height, padding_on_width, output_height, output_width
    Ph = int(np.ceil((H - 1) / 2)) if pad_on else 0
    Pw = int(np.ceil((W - 1) / 2)) if pad_on else 0
    Hout = int(np.ceil((H + 2 * Ph - R) / S) + 1)
    Wout = int(np.ceil((W + 2 * Pw - C) / S) + 1)

    # unfold the input and reshape to (N, Hout*Wout, Cin*R*C)
    unfold = nn.Unfold(kernel_size=(R, C), stride=S, padding=(Ph, Pw))
    inputs_im2col = unfold(inputs)  # (N, Cin*R*C, Hout*Wout)
    inputs_im2col = inputs_im2col.permute(0, 2, 1)  # (N, Hout*Wout, Cin*R*C)
    assert inputs_im2col.shape[1] == Hout * Wout, "Error: The Hout and Wout you calculate may be wrong."
    assert inputs_im2col.shape[2] == Cin * R * C, "Error: The Cin*R*C you calculate may be wrong."

    # marshales the weight from (Cout, Cin, R, C) to (Cin*R*C, Cout)
    weight_reshaped = weight.reshape(Cout, Cin * R * C)
    weight_reshaped = weight_reshaped.permute(1, 0)  # (Cin*R*C, Cout)

    # perform mmad
    out_3d = mmad_3d_inner(inputs_im2col, weight_reshaped, bias, bias_on)

    # marshals the output from (N, Hout*Wout, Cout) to (N, Cout, Hout, Wout)
    out_3d = out_3d.reshape(N, Hout, Wout, Cout)
    out_3d = out_3d.permute(0, 3, 1, 2)  # (N, Cout, Hout, Wout)

    return out_3d


def mmad_3d_inner(mat_a_3d, mat_b_2d, bias, bias_on=True):
    """
    perform mmad on a 3D input tensor and a 2D weight tensor using inner product
    mat_a_3d: (batch, M, K)
    mat_b_2d: (K, N)
    bias: (batch, N)
    """
    mat_out_3d = []
    for a_mat in mat_a_3d:
        o = torch.mm(a_mat, mat_b_2d)
        if bias_on:
            o = o + bias
        mat_out_3d.append(o)
    mat_out_3d = torch.stack(mat_out_3d)

    return mat_out_3d


def test_sanity_conv2d(N=1, Cout=1, Cin=3, H=5, W=5, R=3, C=3, S=1,
                       bias_on=True, pad_on=False, manual_seed=True,
                       device_0=torch.device('cpu'), device_1=torch.device('cpu')):
    if manual_seed:
        torch.random.manual_seed(0)
        random.seed(0)

    # data
    weight = torch.rand(Cout, Cin, R, C).to(device_0)
    if bias_on:
        bias = torch.rand(Cout).to(device_0)
    else:
        bias = None
    inputs = torch.rand(N, Cin, H, W).to(device_0)

    # results
    out_gold = conv2d_torch(N, Cout, Cin, H, W, R, C, S, weight, bias, inputs,
                            bias_on=bias_on, device=device_0)
    out_my = conv2d_my(N, Cout, Cin, H, W, R, C, S, weight, bias, inputs,
                       pad_on=pad_on, bias_on=bias_on, device=device_1)

    # result comparison
    diff = torch.abs(out_gold.to('cpu') - out_my.to('cpu'))
    assert torch.all(diff < 0.1), f"the out_gold and out_imp does not match, try with FP64 first. " \
                                  f"{Cin}, H:{H}, W:{W}, R:{R}, C:{C}"


def test_rand_con2d(repeat=1000, pad_on=False, bias_on=True, device='cpu'):
    # chose device
    device_0 = torch.device("cuda:0" if torch.cuda.is_available() and device != 'cpu' else "cpu")
    device_1 = torch.device("cuda:1" if torch.cuda.is_available() and device != 'cpu' else "cpu")

    # generating test vector sequences `repeat` times
    N_list = np.random.choice(np.arange(1, 4), repeat, replace=True)
    Cout_list = np.random.choice(np.arange(1, 256), repeat, replace=True)
    Cin_list = np.random.choice(np.arange(1, 256), repeat, replace=True)
    R_list = np.random.choice([3, 5, 7, 9, 11], repeat, replace=True)
    C_list = R_list
    H_list = R_list + np.random.choice(np.arange(1, 256), repeat, replace=True)
    W_list = H_list

    for i, (N, Cout, Cin, H, W, R, C) in enumerate(zip(N_list, Cout_list, Cin_list, H_list, W_list, R_list, C_list)):
        test_sanity_conv2d(N, Cout, Cin, H, W, R, C,
                           bias_on=bias_on, pad_on=pad_on, manual_seed=False, device_0=device_0, device_1=device_1)
        if i % (repeat//100) == 0:
            print(f"Processed {i}/{repeat}.")


if __name__ == '__main__':
    # test_rand_con2d(device='gpu')
    test_sanity_conv2d()
