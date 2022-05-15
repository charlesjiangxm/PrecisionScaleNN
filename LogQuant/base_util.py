import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils import dlpack
import onnx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cupy


def save_model(model, model_name='cifarnet', save_as_onnx=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # save a human readable onnx model for displaying the graph later
    if save_as_onnx:
        path = "saved_model/{}.onnx".format(model_name)
        torch.onnx.export(model,  # model being run
                          torch.randn(1, 3, 32, 32, requires_grad=False).to(device),  # model input
                          path,  # where to save the model
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch'},  # variable lenght axes
                                        'output': {0: 'batch'}})
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)
        print('ONNX model is saved to {}'.format(path))
    else:
        # save a pytorch model for training and fine tuning
        torch_path = "saved_model/{}.pth".format(model_name)
        torch.save(model, torch_path)  # model here can be a state = {}
        print('pytorch model is saved to {}'.format(torch_path))


def load_model(dir="saved_model", model_name='cifarnet', load_onnx=False):
    """
    :param load_onnx: Chose if you want to load the onnx or torch model
    """
    if load_onnx:
        return onnx.load('{}/{}.onnx'.format(dir, model_name))
    else:
        return torch.load('{}/{}.torch'.format(dir, model_name))


def plot_2d_hist_from_csv(load_path, save_path, epoch=0, draw_dense=True, bins=100):
    # construct data
    tensor = np.loadtxt(load_path, delimiter=',', skiprows=1)
    sparse_mat = tensor[epoch]
    dense_mat = sparse_mat[sparse_mat != 0]
    sparsity = 100. * (1.0 - len(dense_mat) / len(sparse_mat))

    # draw hist
    plt.hist(dense_mat if draw_dense else sparse_mat, density=True, bins=bins, facecolor='green', alpha=0.8)
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.title('Histogram of {}, epoch={}, sparsity={:.1f}'.format(load_path[4:-4], epoch, sparsity))
    plt.grid()
    plt.savefig(save_path)


def draw_fmap_hist(tensor_all_epoches, epoch_list, file_idx, file_name, draw_dense=True, bins=100, figure_size=(30, 30)):
    # canvas plan
    x_num_of_subplots = np.ceil(np.sqrt(len(epoch_list))).astype(np.int32)
    y_num_of_subplots = np.ceil(len(epoch_list) / x_num_of_subplots).astype(np.int32)
    color_scheme = 'C{}'.format(file_idx)
    fig = plt.figure(figsize=figure_size, clear=True)
    fig.suptitle(f'Histogram of {file_name[:-4]}', fontsize=1.5*sum(figure_size)/2)
    fig.text(0.5, 0.02, f'The Value Magnitude of {file_name[:-4]} (Num. = {len(tensor_all_epoches[0])})',
             ha='center', fontsize=figure_size[0])  # common x
    fig.text(0.02, 0.5, 'Probability', va='center', rotation='vertical', fontsize=figure_size[1])  # common y
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    axs = fig.subplots(x_num_of_subplots, y_num_of_subplots)

    for i, epoch in enumerate(epoch_list[1:]):
        # get the huffman compression ration of the matrix
        tensor_one_epoch = tensor_all_epoches[epoch]
        tensor_one_epoch_dense = tensor_one_epoch[tensor_one_epoch != 0]
        sparsity = 100. * (1.0 - len(tensor_one_epoch_dense) / len(tensor_one_epoch))

        # draw hist
        xpos = i // y_num_of_subplots
        ypos = i % y_num_of_subplots
        axs[xpos, ypos].hist(tensor_one_epoch_dense if draw_dense else tensor_one_epoch,
                             density=True, bins=bins, facecolor=color_scheme, alpha=0.8)
        axs[xpos, ypos].set_title(f'epoch {epoch} sparsity {sparsity:.1f}', fontsize=16)


def plot_batch_hist_from_csv(epoch_list):
    """
    :param draw_dense: Chose if you want to draw dense of sparse feature maps.
    :param bins: number of bins of the sub plots, increase it when you have more data.
    :param figure_size: (figure width, figure height) in inch, increase it when you have more subplots.
    :param epoch_list: list of epoch you want to plot. e.g. plot epoch [1,3,5,7].
    """
    # check if write-to directory exists
    shutil.rmtree('plot', ignore_errors=True)
    os.makedirs('plot')
    fname_under_plot = os.listdir('txt')[11:]

    for file_idx, file_name in enumerate(fname_under_plot):  # loop through all the files in the directory
        # mkdir and load tensor
        load_path = f'txt/{file_name}'
        save_path = f'plot/{file_name[:-4]}.jpeg'
        tensor_all_epoches = np.loadtxt(load_path, delimiter=',', skiprows=1)

        # draw one picture
        draw_fmap_hist(tensor_all_epoches, epoch_list, file_idx, file_name)

        # save and log
        plt.savefig(save_path)
        plt.close()
        print("finish saving to {}".format(save_path))


def save_as_csv(fmap_dict, epoch, idx=0, map_name='act'):
    for i, layer_name in enumerate(fmap_dict):

        # prepare data for each layer
        tensor_flat = np.array([fmap_dict[layer_name][idx].flatten()])
        txt_head = layer_name if epoch == 0 else ''

        # directly append each layer to a file
        with open('txt/{}-{}.txt'.format(map_name, layer_name), 'ab+') as f:
            np.savetxt(f, tensor_flat, delimiter=',', newline='\n', fmt='%.10f', header=txt_head)


def save_weight_as_csv(model, epoch):
    for name, param in model.named_parameters():
        if 'weight' in name:
            data = param.cpu().detach().numpy().flatten().reshape((1, -1))
            txt_head = name if epoch == 0 else ''

            # directly append each layer to a file
            with open('txt/{}.txt'.format(name), 'ab+') as f:
                np.savetxt(f, data, delimiter=',', newline='\n', fmt='%.10f', header=txt_head)


def plot_train_curve_from_csv(load_path='log/log.txt', save_path='log/train_curve.jpeg'):
    # construct data
    tensor = np.loadtxt(load_path, delimiter=',')
    epochs = tensor[:, 0].flatten()
    train_loss = tensor[:, 1].flatten()
    test_loss = tensor[:, 2].flatten()
    test_accuracy = tensor[:, 3].flatten()

    # draw hist
    plt.title('training curve')
    plt.xlabel('epoch')

    plt.plot(epochs, train_loss, 'C1-', label='train loss')
    plt.plot(epochs, test_loss, 'C3-', label='test loss')
    plt.legend(loc='upper left')
    plt.ylabel('loss')

    ax = plt.twinx()
    ax.plot(epochs, test_accuracy, 'C7--', label='test acc.')
    plt.legend(loc='upper right')
    plt.ylabel('acc.(%)')
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_similarity(plot_type, max_epoch):
    """
    :param plot_type: chose from 'similarity' or 'sparsity'
    """
    # create name
    assert plot_type == 'similarity' or plot_type == 'sparsity'
    plot_dir_name = f'plot_{plot_type}'
    fmap_dir_name = f'fmap_{plot_type}_csv'

    # remove previous data
    shutil.rmtree(plot_dir_name, ignore_errors=True)
    os.makedirs(plot_dir_name)
    fname_under_plot = os.listdir(fmap_dir_name)

    for file_idx, file_name in enumerate(fname_under_plot):  # loop through all the files in the directory
        # mkdir and load tensor
        load_path = f'{fmap_dir_name}/{file_name}'
        save_path = f'{plot_dir_name}/{file_name[:-4]}.jpeg'
        tensor_all_epoches = np.loadtxt(load_path, delimiter=',', skiprows=1).flatten()

        # make data frame
        epoch_idx = np.arange(tensor_all_epoches.size) / (tensor_all_epoches.size / max_epoch)
        data_frame = pd.DataFrame({'epoch': epoch_idx, f'{plot_type} (%)': tensor_all_epoches})

        # plot fiture
        plt.title(file_name)
        sns.scatterplot(data=data_frame, x='epoch', y=f'{plot_type} (%)')
        plt.savefig(f'{plot_dir_name}/{file_name[:-4]}.png')
        plt.clf()


