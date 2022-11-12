import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
import cv2
from tqdm import tqdm
import numpy as np
import os
import shutil




def main():
    batch_size = 1
    epochs = 1
    nb_batch = 1
    lr = 1e-1
    # nb_batch = int(2791 / batch_size)
    savepath = f'losses_{epochs}_eps_{nb_batch}_batches_{batch_size}_batch_size_0_1_lr_bce'

    # Loader
    loader = LoopLoader(dset_path='./dataset_initial/temp/Seaquest-v5', batch_size=batch_size)

    # Model Initialization
    model = AE()

    # Copy this file to the output folder
    os.makedirs(savepath, exist_ok=True)
    shutil.copy2(__file__, savepath + '/test_autoencoder.py')

    # Validation using MSE Loss function
    # loss_function = torch.nn.MSELoss()
    loss_function = torch.nn.BCELoss()
    # loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = DiceLoss()
    # loss_function = GeneralizedDiceLoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = lr)#,
                                 #weight_decay = 1e-8)



    outputs = []
    losses = []
    for epoch in range(epochs):
        # Iterate over all images
        # for _ in range(len(loader.ds_folder)): # DON'T DO THIS! Blocks everythink
        for _ in tqdm(range(nb_batch)):
            # ary = loader.get_batch()
            img = np.random.rand(8, 3, 449, 449)
            ary = torch.from_numpy(img).to('cpu').float()

            # Output of Autoencoder
            reconstructed = model(ary)

            # Calculating the loss function
            # print(ary.shape)
            # print(reconstructed.shape)
            loss = loss_function(reconstructed, ary)

              # The gradients are set to zero,
              # the gradient is computed and stored.
              # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

              # Storing the losses in a list for plotting
            losses.append(loss.detach().numpy())
        out = (epoch, ary.detach().numpy(), reconstructed.detach().numpy())
        outputs.append(out)
        save_output(out, savepath)
        # print(losses)

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    print(losses)
    plt.plot(losses[1:])
    plt.savefig(savepath + '/loss')
    # save_outputs(outputs, savepath)
    plt.show()


# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        encoded_space_dim = 2048

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # nn.Linear(3 * 3 * 32, 128),
            nn.Linear(16384, encoded_space_dim * 2),
            nn.ReLU(True),
            nn.Linear(encoded_space_dim * 2, encoded_space_dim)
        )


        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, encoded_space_dim * 2),
            nn.ReLU(True),
            nn.Linear(encoded_space_dim * 2, 16384),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
        unflattened_size=(64, 16, 16))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3,
            stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3,
            stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2,
            padding=1, output_padding=1)
        )


    def forward(self, x):
        # Encode
        # print("Input", x.shape)
        x = self.encoder_cnn(x)
        # print("After encorder cnn", x.shape)
        # x = self.flatten(x)
        # # print("After encorder flatten", x.shape)
        # x = self.encoder_lin(x)
        # # print("After encorder linear", x.shape)
        #
        # # Decode
        # x = self.decoder_lin(x)
        # # print("After decoder linear", x.shape)
        # x = self.unflatten(x)
        # print("After decoder unflatten", x.shape)
        x = self.decoder_conv(x)
        # print("After decoder conv", x.shape)
        x = torch.sigmoid(x)
        return x





class LoopLoader():
    def __init__(self,
            dset_path,
            batch_size
        ):

        self.dset_path = dset_path
        self.batch_size = batch_size
        self.cuda_device = 'cpu'


        self.ds_folder = torch.utils.data.ConcatDataset([SegmentationDatasetFolder(
        path=f"{dset_path}/",
        cuda_device=self.cuda_device,
        loader=jpg_loader(self.cuda_device))
        ])


        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.ds_folder,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True)

        self.reload_iter()


    def reload_iter(self):
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.ds_folder,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True)
        self.loader_iter = iter(self.data_loader)


    def get_batch(self):
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.reload_iter()
            return next(self.loader_iter)



class SegmentationDatasetFolder(torchvision.datasets.DatasetFolder):
    def __init__(self, loader, path, cuda_device='cpu'):
        self.loader = loader
        self.cuda_device = cuda_device
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "*.jpg")
        self.data = []
        for img in file_list:
            self.data.append(img)
        self.num_samples = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = self.loader(img_path)

        return img



def jpg_loader(cuda_device):
    def the_loader(path):
        # Load data
        img = cv2.imread(path)
        # Move color channel forward
        img = img.transpose(2, 0, 1)
        # Ensure the size is correct
        img = np.pad(img, [(0,0), (0, 256), (0, 256)])[:,:256, :256]
        # Normalize
        img = np.divide(img, 255)
        # Send the tensor to the GPU/CPU depending on what device is available
        tensor = torch.from_numpy(img).to(cuda_device)
        return tensor.float()
    return the_loader


def save_output(output, save_path):
    # Extract values
    ep, ary, rec = output
    # Select firts image of each batch and move color channel
    ary = ary[0].transpose(1, 2, 0)
    rec = rec[0].transpose(1, 2, 0)
    # Generate names
    ary_name = save_path + '/orig_' + str(ep) + '.jpg'
    rec_name = save_path + '/rec_' + str(ep) + '.jpg'
    # Save images
    cv2.imwrite(rec_name, rec * 255)
    cv2.imwrite(ary_name, ary * 255)

# Save images
def save_outputs(outputs, save_path):
    # os.makedirs(save_path, exist_ok=True)
    for output in outputs:
        save_output(output, save_path)
    print(f'Images saved in {save_path}')





def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # input = flatten(input)
    input = input.flatten()
    # target = flatten(target)
    target = target.flatten()
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

class _AbstractDiceLoss(torch.nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = torch.nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = torch.nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # input = flatten(input)
        input = input.flatten()
        # target = flatten(target)
        target = target.flatten()
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())



if __name__ == "__main__":
    main()
