import torch

from multi_label_supervised_contrastive_losses import *
import torch.nn as nn
import matplotlib.pyplot as plt
from pca_plotting import plot_PCA_projected


lin = nn.Sequential(nn.Linear(128, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128)).to(DEVICE)
nn.init.normal_(lin[0].weight, mean=0.0, std=0.02)
nn.init.normal_(lin[3].weight, mean=0.0, std=0.02)


labels_num = 8
def get_rand_sample(label_idx=None):
    # if c_labels is None:
    #     c_labels = torch.rand((1, labels_num)) < 0.125
    # else:
    #     d_labels = torch.zeros((1, labels_num), dtype=torch.bool)
    #     for l in c_labels:
    #         d_labels[0][l] = True
    #     c_labels = d_labels ** 0.5
    # c_labels = c_labels.bool()
    #
    # item = torch.randn((1, 128))
    #
    # for j, l in enumerate(c_labels[0]):
    #     if l:
    #         item += j
    #
    # return item.float().to(DEVICE), c_labels.to(DEVICE)

    if label_idx is None:
        label_idx = torch.randint(0, 8, (1,)).item()
    label = torch.zeros((1, 8), dtype=torch.bool)
    label[0][label_idx] = True

    item = torch.randn((1, 128))
    # if label_idx < 5:
    #     item -= 2 * label_idx
    # else:
    #     item += 2 * (label_idx - 4)
    item *= (label_idx + 1) ** 2
    item = item.float()

    return item, label


def create_batch():
    b_item = []
    b_label = []
    for i in range(256):
        c_item, c_label = get_rand_sample()
        b_item.append(c_item)
        b_label.append(c_label)
    b_item = torch.cat(b_item, dim=0)
    b_label = torch.cat(b_label, dim=0)
    return b_item.to(DEVICE), b_label.to(DEVICE)

from itertools import chain, combinations
all_labels = [0, 1, 2, 3, 4, 5, 6, 7]

@torch.no_grad()
def plot_labels(b):
    lin.eval()
    print('plotting')
    # items = [[lin(get_rand_sample(c_labels=j)[0]) for _ in range(5)] for j in chain.from_iterable(combinations(all_labels, r) for r in range(len(all_labels) + 1))]
    items = [[lin(get_rand_sample(label_idx=j)[0]) for _ in range(10)] for j in all_labels]
    items = [torch.cat(arr, dim=0) for arr in items]
    plt.clf()
    plot_PCA_projected(items, all_labels)
    # for j, lst in enumerate(items):
    #     x = [item[0][0].item() for item in lst]
    #     y = [item[0][1].item() for item in lst]
    #     plt.scatter(x, y, label=j)
    # plt.legend()
    # # plt.savefig('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/test_plots/' + f"test_plot_{b}.png")
    # plt.savefig(f"test_plot_{b}.png")
    # plt.show()
    # print(items)

    lin.train()


cl_loss = MultiLabelSupervisedContrastiveLoss(temp=0.1, use_newer_loss=True, reg_batch_size=256, labels_num=8)
bce_loss = nn.BCELoss()

optim = torch.optim.Adam(lin.parameters(), lr=0.01)
optim.zero_grad()

print(torch.cuda.is_available())

for i in range(20000):

    inputs, labels = create_batch()
    labels = labels.float()
    # print(labels)
    if i == 0:
        print(inputs.shape)
        print(labels.shape)

    # TODO: put full batch in model (mim_model doesnt matter)
    outputs = lin(inputs)
    loss = cl_loss(outputs, labels)

    loss.backward()
    optim.step()
    optim.zero_grad()

    if i == 10:
        plot_labels(i)

    if i % 30 == 0:
        print(f'batch {i+1}, loss = {loss.item()}')
