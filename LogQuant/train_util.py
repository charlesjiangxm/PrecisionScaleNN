from base_util import *
import torch
import torchvision
import torch.utils.data

# global training hyper-parameters and buffers
act = {}
grad_act = {}


def analysis_data(epoch, analysis_similarity=False, analysis_sparsity=True, analysis_dyn_range=True, plot_fmap=True):
    global act
    global grad_act

    def analysis(dict, dict_name):
        for key, value in dict.items():
            # calculating the data profile includes the similarity, sparsity, dynamic range, etc.
            similarity_all_batch = []
            sparsity_all_batch = []
            dyn_range_all_batch = []
            for i in range(len(value)):  # last batch may be fragmented
                if analysis_similarity:
                    # eliminate 0 in a
                    value_prev_no_zero = value[i - 1][value[i - 1] != 0]
                    value_after_no_zero = value[i][value[i - 1] != 0]

                    # compute similar data
                    similar_data = value_prev_no_zero[value_prev_no_zero == value_after_no_zero]
                    similarity = 100. * len(similar_data) / len(value[i])
                    similarity_all_batch.append(similarity)

                if analysis_sparsity:
                    tensor_flat = value[i].flatten()
                    sparsity = (1.0 - np.count_nonzero(tensor_flat) / tensor_flat.size) * 100.
                    sparsity_all_batch.append(sparsity)

                if analysis_dyn_range:
                    tensor = value[i]
                    tensor_min, tensor_max = np.amin(tensor), np.amax(tensor)
                    dyn_range = abs(tensor_max - tensor_min)
                    dyn_range_all_batch.append([tensor_min, tensor_max, dyn_range])

            # save the data profile to log
            if analysis_similarity:
                file_name = f'fmap_similarity_csv/{dict_name}-{key}.txt'
                if os.path.exists(file_name):
                    a_or_w = 'a'
                else:
                    a_or_w = 'w'
                with open(file_name, f'{a_or_w}b+') as f:
                    np.savetxt(f, similarity_all_batch, delimiter=',', newline='\n', fmt='%.4f')

            if analysis_sparsity:
                file_name = f'fmap_sparsity_csv/{dict_name}-{key}.txt'
                if os.path.exists(file_name):
                    a_or_w = 'a'
                else:
                    a_or_w = 'w'
                with open(file_name, f'{a_or_w}b+') as f:
                    np.savetxt(f, sparsity_all_batch, delimiter=',', newline='\n', fmt='%.4f')

            if analysis_dyn_range:
                file_name = f'fmap_dyn_range_csv/{dict_name}-{key}.txt'
                if os.path.exists(file_name):
                    a_or_w = 'a'
                else:
                    a_or_w = 'w'
                with open(file_name, f'{a_or_w}b+') as f:
                    np.savetxt(f, dyn_range_all_batch, delimiter=',', newline='\n')

            # plot feature maps per epoch, only plot the first batch
            if plot_fmap:
                # make dir and grid
                if not os.path.exists(f'plot_fmap/{dict_name}_{key}'):
                    os.mkdir(f'plot_fmap/{dict_name}_{key}')
                ret = torchvision.utils.make_grid(tensor=torch.Tensor(value[0])).numpy()

                # plot gray image
                gray_img_ch0 = ret[0]
                title = f"{key}-epoch{epoch}-batch0"
                plt.title(title)
                plt.xlabel(f"{value[0].shape[0]} images in a batch, plot channel is [0]/[{ret.shape[1]}]")
                plt.ylabel(f"batch average sparsity {sparsity_all_batch[0]:.2f}%")
                plt.imshow(gray_img_ch0, cmap='gray')
                plt.savefig(f'plot_fmap/{dict_name}_{key}/{title}.jpeg')
                plt.clf()

    # process
    analysis(act, 'act')
    analysis(grad_act, 'grad_act')


def hook_forward(name):
    def hook(module, input, output):
        if name not in act:
            act[name] = []
        act[name].append(output.cpu().detach().numpy())

    return hook


def hook_backward(name):
    def hook(module, grad_input, grad_output):
        """
        :param grad_input:  For conv, it stores (batch * layer input gradient, bias gradient)
                            For fc, it stores (bias gradient, batch * layer input gradient, weight gradient)
        :param grad_output: For conv, it stores (batch * layer output gradient)
                            For fc, it stores (batch * layer output gradient)
        """
        if name not in grad_act:
            grad_act[name] = []
        grad_act[name].append(grad_output[0].cpu().detach().numpy())

    return hook


def register_hook(model, backward_hook_on=True):
    """
    :param register: True - register hook, False - deregister hook
    """
    # clear the global buffer maintained in this file
    global act
    global grad_act
    act = {}
    grad_act = {}

    hook_handler = []
    for name, layer in model.module.named_children():
        hook_handler.append(layer.register_forward_hook(hook_forward(name)))
        if backward_hook_on:
            hook_handler.append(layer.register_backward_hook(hook_backward(name)))

    return hook_handler


def remove_hook(hook_handler):
    # clear the global buffer maintained in this file
    global act
    global grad_act
    act = {}
    grad_act = {}

    for handler in hook_handler:
        handler.remove()
