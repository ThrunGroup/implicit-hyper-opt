from tqdm import tqdm
import copy
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def save_learned(datas, is_mnist, batch_size, name, path='images'):
    """
    :param datas:
    :param is_mnist:
    :param batch_size:
    :param name:
    :param path:
    :return:
    """
    saturation_multiple = 1
    if not is_mnist:
        saturation_multiple = 1

    datas = torch.sigmoid(datas.detach() * saturation_multiple).cpu().numpy()
    col_size = 10
    row_size = batch_size // col_size
    if batch_size % row_size != 0:
        row_size += 1
    fig = plt.figure(figsize=(col_size, row_size))

    for i, data in enumerate(datas):
        ax = plt.subplot(row_size, col_size, i + 1)
        if is_mnist:
            plt.imshow(data[0], cmap='gray', interpolation='gaussian')
        else:
            plt.imshow(np.transpose(data, (1, 2, 0)), interpolation='gaussian')
        plt.xticks([])
        plt.yticks([])
        ax.set_aspect('auto')

    plt.subplots_adjust(wspace=0.05 * col_size / row_size, hspace=0.05)
    plt.draw()
    fig.savefig(f'{path}/{name}.pdf')
    plt.close(fig)


def save_images(images, labels, augment_net, args):
    """
    :param images:
    :param labels:
    :param augment_net:
    :param args:
    :return:
    """
    is_mnist = False
    if args.dataset == 'mnist':
        is_mnist = True
    num_save = 10
    save_learned(images[:num_save], is_mnist, num_save, 'image_Original', path=args.save_loc)

    num_sample = 10
    augs = torch.zeros(num_sample, num_save, images.shape[1], images.shape[2], images.shape[3])
    for i in range(num_sample):
        augs[i] = augment_net(images[:num_save], class_label=labels[:num_save])

    aug_1 = augment_net(images[:num_save], class_label=labels[:num_save])
    save_learned(aug_1, is_mnist, num_save, 'image_Augment', path=args.save_loc)

    aug_2 = augment_net(images[:num_save], use_zero_noise=True, class_label=labels[:num_save])
    save_learned(aug_2, is_mnist, num_save, 'image_Augment2', path=args.save_loc)

    std_augs = torch.std(augs, dim=0).cuda()
    std_augs = torch.log(std_augs)
    # print(torch.max(std_augs), torch.min(std_augs), torch.std(std_augs), torch.mean(std_augs))
    # std_augs = (std_augs - torch.mean(std_augs)) / torch.std(std_augs)
    save_learned(std_augs, is_mnist, num_save, 'image_AugmentDiff', path=args.save_loc)

    mean_augs = torch.mean(augs, dim=0).cuda()
    save_learned(mean_augs - images[:num_save], is_mnist, num_save, 'image_OriginalDiff', path=args.save_loc)


def graph_single_args(args, save_loc=None):
    """
    :param args:
    :return:
    """
    from train_augment_net_multiple import get_id
    if save_loc is None:
        args.save_loc = finetune_location + get_id(args)
    else:
        args.save_loc = save_loc

    args.load_baseline_checkpoint = None
    args.load_finetune_checkpoint = args.save_loc + '/checkpoint.pt'

    args.data_augmentation = False  # Don't use data augmentation for constructing graphs

    from train_augment_net2 import get_models
    model, train_loader, val_loader, test_loader, augment_net, reweighting_net, checkpoint = get_models(args)

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.cuda(), labels.cuda()

        save_images(images, labels, augment_net, args)


def init_ax(fontsize=24, nrows=1, ncols=1):
    """
    :param fontsize:
    :return:
    """
    font = {'family': 'Times New Roman'}
    mpl.rc('font', **font)
    mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['axes.grid'] = False

    fig = plt.figure(figsize=(6.4 / np.sqrt(nrows), 4.8 * nrows / np.sqrt(nrows)))
    axs = [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]
    for ax in axs:
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.tick_params(axis='y', which='both', left=False, right=False)
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    return fig, axs


def setup_ax(ax, do_legend=True, alpha=0.0, fontsize=24, legend_loc=None, handlelength=None):
    """
    :param ax:
    :param do_legend:
    :param alpha:
    :return:
    """
    if do_legend:
        ax.legend(fancybox=True, borderaxespad=0.0, framealpha=alpha, fontsize=fontsize,
                  loc=legend_loc, handlelength=handlelength)
    # ax.tick_params(axis='x', which='both', bottom=False, top=False)
    # ax.tick_params(axis='y', which='both', left=False, right=False)
    # ax.grid(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.tight_layout()
    return ax


def load_from_csv(path, do_train=True, do_val=False, do_test=False):
    """
    :param path:
    :param do_train:
    :param do_val:
    :param do_test:
    :return:
    """
    log_loc = 'log.csv'
    '''if do_test:
        log_loc = 'test_log.csv'
    elif do_val:
        log_loc = 'val_log.csv'
    elif do_train:
        log_loc = 'train_log.csv'
    else:
        print("Need do_train, do_val, or do_test to be True.")
        exit(1)'''

    with open(path + '/' + log_loc) as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        data = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                data[name].append(row[name])

    data['epoch'] = [int(i) for i in data['epoch']]
    data['iteration'] = [int(i) for i in data['iteration']]
    data['run_time'] = [float(i) for i in data['run_time']]  # TODO: Is float the correct data type?
    # TODO: (Convert time from unix timestamp to datetime?)
    data['hypergradient_cos_diff'] = [float(i) for i in data['hypergradient_cos_diff']]
    data['hypergradient_l2_diff'] = [float(i) for i in data['hypergradient_l2_diff']]
    if do_train:
        data['train_acc'] = [float(i) for i in data['train_acc']]
        data['train_loss'] = [float(i) for i in data['train_loss']]
    if do_val:
        data['val_acc'] = [float(i) for i in data['val_acc']]
        data['val_loss'] = [float(i) for i in data['val_loss']]
    if do_test:
        data['test_acc'] = [float(i) for i in data['test_acc']]
        data['test_loss'] = [float(i) for i in data['test_loss']]
    return data


def smooth_data(data, num_smooth):
    """
    :param data:
    :param num_smooth:
    :return:
    """
    # smoothed_data = np.zeros(data.shape)
    left_fraction = 1.0
    right_fraction = 1.0 - left_fraction
    smoothed_data = [np.mean(
        data[max(ind - int(num_smooth * left_fraction), 0): min(ind + int(num_smooth * right_fraction), len(data))])
        for ind in range(len(data))]
    return np.array(smoothed_data)


linewidth = 4


def graph_multiple_args(argss, ylabels, name_supplement='', xmin=0, legend_loc=None,
                        fontsize=12, handlelength=None):
    xlabels = ['iteration']  # , 'run_time']  # , 'epoch']

    fig, axs = init_ax(fontsize=fontsize, ncols=len(xlabels), nrows=len(ylabels))

    color_dict = {}
    data_dict = {ylabel: {} for ylabel in ylabels}
    min_xlim = 10e32
    for args in argss:
        temp_args = copy.deepcopy(args)
        temp_args.seed = float(1)  # A copy for storing the id of a set of different seeds

        from train_augment_net_multiple import get_id
        args.save_loc = finetune_location + get_id(args)
        try:
            args_data = load_from_csv(args.save_loc, do_val=True, do_test=True)
        except FileNotFoundError:
            print(f"Can't load {args.save_loc}")
            break
        label_index = 0
        for ylabel in ylabels:
            all_smoothed_data = []
            for xlabel in xlabels:
                smoothed_data = smooth_data(args_data[ylabel], num_smooth=20)[xmin:]
                if args.seed == 1:
                    label = get_id(args)
                    if ylabel == 'hypergradient_l2_diff' or ylabel == 'hypergradient_cos_diff':
                        if args.use_cg:
                            label = str(args.num_neumann_terms) + ' CG Steps'
                        else:
                            if args.num_neumann_terms == 1:
                                label = str(args.num_neumann_terms) + ' Neumann'
                            else:
                                label = str(args.num_neumann_terms) + ' Neumann'
                    if ylabel == 'hypergradient_l2_diff':
                        plot = axs[label_index].semilogy(args_data[xlabel][xmin:], smoothed_data, label=label,
                                                         alpha=1.0, linewidth=linewidth)
                    else:
                        plot = axs[label_index].plot(args_data[xlabel][xmin:], smoothed_data, label=label,
                                                     alpha=1.0, linewidth=linewidth)
                    plot = plot[0]
                    color_dict[get_id(args)] = plot.get_color()
                    data_dict[ylabel][get_id(args)] = []
                else:
                    if ylabel == 'hypergradient_l2_diff':
                        plot = axs[label_index].semilogy(args_data[xlabel][xmin:], smoothed_data, alpha=1.0,
                                                         color=color_dict[get_id(temp_args)], linewidth=linewidth)
                    else:
                        plot = axs[label_index].plot(args_data[xlabel][xmin:], smoothed_data, alpha=1.0,
                                                     color=color_dict[get_id(temp_args)], linewidth=linewidth)
                # smoothed_data = smooth_data(args_data[ylabel], num_smooth=20)
                '''all_smoothed_data += [smoothed_data]
                if args.seed == 1:
                    data_dict[ylabel][get_id(args)] = []
                elif args.seed == 3:
                    all_smoothed_data = np.array(all_smoothed_data)
                    mean = np.mean(all_smoothed_data, axis=0)
                    std = np.std(all_smoothed_data, axis=0)
                    plot = axs[label_index].errorbar(args_data[xlabel],
                                                 mean, std,
                                                 label=get_id(args), alpha=0.5)'''
                min_xlim = min(min_xlim, args_data[xlabel][-1])
                # if args.seed == 1:
                #    data_dict[ylabel][get_id(args)] = []
                data_dict[ylabel][get_id(temp_args)] += [smoothed_data[-1]]
                if False:  # ylabel[:3] == 'val' and xlabel == 'iteration':
                    diagnostic = f"y: {ylabel}"
                    # diagnostic += f", x: {xlabel}"
                    diagnostic += f", num_neumann: {args.num_neumann_terms}"
                    diagnostic += f", reg: {args.reg_weight}"
                    diagnostic += f", cg: {args.use_cg}"
                    diagnostic += f", seed: {args.seed}"
                    diagnostic += f", final value: {args_data[ylabel][-1]}"
                    print(diagnostic)

                label_index += 1
    # TODO: Store in array based on seed.
    # TODO: Compute mean and std_dev for each method
    # print(data_dict)
    label_index = 0
    for ylabel in data_dict:
        print(f"Value: {ylabel}")
        for id in data_dict[ylabel]:
            # mean = np.mean(data_dict[ylabel][id])
            # print(mean)
            # entry = np.asarray([np.array(x) for x in pre_entry])
            print(f"    ID: {id}")
            print(f"        mean = {np.mean(data_dict[ylabel][id]):.4f}, std = {np.std(data_dict[ylabel][id]):.4f}")
            print(f"        max = {np.max(data_dict[ylabel][id]):.4f}, min = {np.min(data_dict[ylabel][id]):.4f}")
        label_index += 1

    label_index = 0
    for ylabel in ylabels:
        for xlabel in xlabels:
            if label_index % len(xlabels) == 0:  # Only label the left size with y-labels
                # axs[label_index].set_ylabel(ylabel)
                pass
            else:
                axs[label_index].set_yticks([])

            if label_index > (len(ylabels) - 1) * (len(xlabels)) - 1:  # Only label the bottom with x-labels
                # axs[label_index].set_xlabel(xlabel)
                pass
            else:
                axs[label_index].set_xticks([])

            if ylabel[-3:] == 'acc':
                axs[label_index].set_ylim([0.92, 0.94])  # [.93, .965])
            elif ylabel[:3] in ['val', 'tes'] and ylabel[-4:] == 'loss':
                axs[label_index].set_ylim([.23, .35])  # [.17, .25])
            elif ylabel[-4:] == 'loss':
                axs[label_index].set_ylim([0.11, 0.17])
            elif ylabel == 'hypergradient_cos_diff':
                axs[label_index].set_ylim([0.0, 1.0])
            elif ylabel == 'hypergradient_l2_diff':
                axs[label_index].set_ylim([10e-4, 10e3])

            # if xmin is not None:
            #
            #    axs[label_index].set_xlim([xmin, axs[label_index].get_xlim()[-1]])
            label_index += 1

    axs = [setup_ax(ax, alpha=0.75, fontsize=fontsize, legend_loc=legend_loc,
                    handlelength=handlelength) for ax in axs[-1:]]

    name = f"./images/graph_multiple_args_{name_supplement}"
    fig.savefig(name + ".pdf", bbox_inches='tight')
    plt.close(fig)


def graph_final_multiple_args(argss, ylabels, name_supplement='', xmin=0, legend_loc=None,
                              fontsize=12, handlelength=None):
    fig, axs = init_ax(fontsize=fontsize, nrows=len(ylabels))

    cg_data = {ylabel: [] for ylabel in ylabels}
    neumann_data = {ylabel: [] for ylabel in ylabels}
    for args in argss:
        temp_args = copy.deepcopy(args)
        temp_args.seed = float(1)  # A copy for storing the id of a set of different seeds

        from train_augment_net_multiple import get_id
        args.save_loc = finetune_location + get_id(args)
        try:
            args_data = load_from_csv(args.save_loc, do_val=True, do_test=True)
        except FileNotFoundError:
            print(f"Can't load {args.save_loc}")
            break

        for ylabel in ylabels:
            smoothed_data = smooth_data(args_data[ylabel], num_smooth=10)[xmin:]
            print(smoothed_data[-1])
            if args.use_cg:
                cg_data[ylabel] += [smoothed_data[-1]]
            else:
                neumann_data[ylabel] += [smoothed_data[-1]]

    num_smooth = 10
    for label_index, ylabel in enumerate(ylabels):
        # axs[label_index].set_ylabel(ylabel)
        if ylabel == 'hypergradient_l2_diff':
            axs[label_index].semilogy(range(len(cg_data[ylabel])),
                                      smooth_data(cg_data[ylabel], num_smooth), label='CG', linestyle='--',
                                      linewidth=linewidth)
            axs[label_index].semilogy(range(len(neumann_data[ylabel])),
                                      smooth_data(neumann_data[ylabel], num_smooth), label='Neumann', linestyle='--',
                                      linewidth=linewidth)
        else:
            axs[label_index].plot(range(len(cg_data[ylabel])),
                                  smooth_data(cg_data[ylabel], num_smooth), label='CG', linestyle='--',
                                  linewidth=linewidth)
            axs[label_index].plot(range(len(neumann_data[ylabel])),
                                  smooth_data(neumann_data[ylabel], num_smooth), label='Neumann', linestyle='--',
                                  linewidth=linewidth)

    for label_index, ylabel in enumerate(ylabels):
        # if label_index % len(xlabels) == 0:  # Only label the left size with y-labels
        #    axs[label_index].set_ylabel(ylabel)
        # else:
        axs[label_index].set_yticks([])

        if label_index < (len(ylabels) - 1):
            axs[label_index].set_xticks([])

        if ylabel == 'hypergradient_cos_diff':
            axs[label_index].set_ylim([0.0, 1.0])
        elif ylabel == 'hypergradient_l2_diff':
            axs[label_index].set_ylim([10e-4, 10e3])

    axs = [setup_ax(ax, alpha=0.75, fontsize=fontsize, legend_loc=legend_loc,
                    handlelength=handlelength) for ax in axs[-1:]]

    name = f"./images/graph_final_multiple_args_{name_supplement}"
    fig.savefig(name + ".pdf", bbox_inches='tight')
    plt.close(fig)


finetune_location = './finetuned_checkpoints/'
if __name__ == '__main__':
    from train_augment_net_multiple import make_argss, get_id
    from train_augment_net2 import make_test_arg

    # test_args = argss[0]#make_test_arg()
    # print(f"Graphing individual args: {test_args}")
    # graph_single_args(test_args, save_loc='/h/lorraine/PycharmProjects/CG_IFT_test/finetuned_checkpoints/' + get_id(test_args))

    '''print(f"Graphing multiple args ablation")
    argss_ablation = make_argss()
    ylabels_ablation = ['val_loss', 'val_acc', 'test_loss', 'test_acc']
    graph_multiple_args(argss_ablation, ylabels_ablation, name_supplement='ablation')
    print("Finished graphing ablation!")'''

    print(f"Graphing multiple args hypergrad diff")
    from train_augment_net2 import multi_boston_args

    # TODO: Load multiple hypers/models for this experiment
    argss_inverse_compare = multi_boston_args()
    ylabels_inverse_compare = ['hypergradient_cos_diff', 'hypergradient_l2_diff']
    xmin = argss_inverse_compare[0].warmup_epochs + 20 + 1
    legend_loc = 2
    graph_multiple_args(argss_inverse_compare, ylabels_inverse_compare, name_supplement='hypergradient_error',
                        xmin=xmin, legend_loc=legend_loc, fontsize=20, handlelength=1.0)
    print("Finished graphing ablation!")

    print(f"Graphing final multiple args hypergrad diff")
    from train_augment_net2 import multi_boston_how_many_steps

    # TODO: Load multiple hypers/models for this experiment
    argss_inverse_compare = multi_boston_how_many_steps()
    ylabels_inverse_compare = ['hypergradient_cos_diff', 'hypergradient_l2_diff']
    graph_final_multiple_args(argss_inverse_compare, ylabels_inverse_compare, name_supplement='hypergradient_error',
                              fontsize=20, handlelength=1.0)
    print("Finished graphing ablation!")
