import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from it_functions import mutual_information
from LeNet_mnist import LeNet

def mnist_analysis(batch_size: int=128):
    model = LeNet()
    model.load_state_dict(torch.load(
        "./model_checkpoints/lenet_mnist/lenet_mnist_ep1.pth",
        weights_only=True,
    ))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    MI_dict = { i: [] for i in range(6) }
    for data, _ in tqdm(loader, ncols=80):
        with torch.no_grad():
            out_list = model.layer_outputs(data)
            for i, out in enumerate(out_list):
                mi = mutual_information(data, out)
                MI_dict[i].append(float(mi))
    with open("mnist_out.txt", 'w') as f:
        print(MI_dict, file=f)
    return MI_dict


def mnist_classification_analysis(batch_size: int=128):
    model = LeNet()
    model.load_state_dict(torch.load(
        "./model_checkpoints/lenet_mnist/lenet_mnist_ep1.pth",
        weights_only=True,
    ))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    MI_dict = { i: [] for i in range(3) }
    for data, _ in tqdm(loader, ncols=80):
        with torch.no_grad():
            out_list = model.classifier_outputs(data)
            for i, out in enumerate(out_list):
                mi = mutual_information(data, out)
                MI_dict[i].append(float(mi))
    with open("mnist_out.txt", 'w') as f:
        print(MI_dict, file=f)
    return MI_dict


def plot_data(forward_arr):
    pos = [i+1 for i in range(len(forward_arr))]
    average_list = []
    for v in forward_arr:
        average_list.append(np.mean(v))

    _, ax = plt.subplots()
    _ = ax.boxplot(
        forward_arr,
        positions=pos,
        showfliers=False,
        widths=0.5,
    )
    plt.savefig("mnist_.png")
    plt.show()


from data_mnist import mnist_forward_dict_ep1_bs100_h5 as data1
from data_mnist import mnist_forward_dict_pt2_ep1_bs100_h5 as data2
if __name__ == "__main__":
    # data1 = mnist_analysis(batch_size=100)
    # data2 = mnist_classification_analysis(batch_size=100)

    mnist_forward_arr = [data1[i] for i in data1]
    for j in data2:
        mnist_forward_arr.append(data2[j])
    plot_data(mnist_forward_arr)
