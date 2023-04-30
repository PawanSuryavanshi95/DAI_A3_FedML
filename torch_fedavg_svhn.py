import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

import fedml
from fedml import FedMLRunner
from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist


def load_data(args):

    # Define batch size for data loaders
    batch_size = 10

    # Define transformations to be applied to data
    transform = transforms.Compose([
        transforms.Grayscale(),  # convert to grayscale
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    fedml.logging.info("Downloading SVHN Dataset ...")

    # Download and load the training set
    train_dataset = datasets.SVHN('data', split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN('data', split='test', download=True, transform=transform)

    train_dataset_1, train_dataset_2, train_dataset_3 = torch.utils.data.random_split(train_dataset, [len(train_dataset)//3, len(train_dataset)//3, len(train_dataset)//3])
    test_dataset_1, test_dataset_2, test_dataset_3 = torch.utils.data.random_split(test_dataset, [len(test_dataset)//3, len(test_dataset)//3, 1 + len(test_dataset)//3])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True)
    test_loader_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False)

    train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)
    test_loader_2 = torch.utils.data.DataLoader(test_dataset_3, batch_size=batch_size, shuffle=False)

    train_loader_3 = torch.utils.data.DataLoader(train_dataset_3, batch_size=batch_size, shuffle=True)
    test_loader_3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=batch_size, shuffle=False)

    fedml.logging.info("Preparing Dataset ...")
    
    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)
    class_num = 10

    train_data_local_num_dict = {}
    train_data_local_num_dict[0] = len(train_dataset_1)
    train_data_local_num_dict[1] = len(train_dataset_2)
    train_data_local_num_dict[2] = len(train_dataset_3)

    train_data_global = []
    for (images, labels) in train_loader:
        train_data_global.append((images, labels))

    test_data_global = []
    for (images, labels) in test_loader:
        test_data_global.append((images, labels))

    train_data_local_dict = {}

    train_data_local_dict[0] = []
    for (images, labels) in train_loader_1:
        train_data_local_dict[0].append((images, labels))

    train_data_local_dict[1] = []
    for (images, labels) in train_loader_2:
        train_data_local_dict[1].append((images, labels))

    train_data_local_dict[2] = []
    for (images, labels) in train_loader_3:
        train_data_local_dict[2].append((images, labels))

    test_data_local_dict = {}

    test_data_local_dict[0] = []
    for (images, labels) in test_loader_1:
        test_data_local_dict[0].append((images, labels))

    test_data_local_dict[1] = []
    for (images, labels) in test_loader_2:
        test_data_local_dict[1].append((images, labels))

    test_data_local_dict[2] = []
    for (images, labels) in test_loader_3:
        test_data_local_dict[2].append((images, labels))
    
    fedml.logging.info("Dataset Preparation Done !")
    
    """
    Please read through the data loader at to see how to customize the dataset for FedML framework.
    """
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    

    return dataset, class_num

class Digit_CNN_Model(nn.Module):
    def __init__(self):
        super(Digit_CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()
    args.client_num_in_total = 3

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = Digit_CNN_Model()

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()