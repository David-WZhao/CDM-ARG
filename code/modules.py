import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from ConditionalDiffusion import ConditionalDiffusion,CrossAttention

torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CML(nn.Module):
    def __init__(self, X_dim, G_dim, z_dim,EMS_input_dim, antibiotic_count,mechanism_count,transfer_count,
                 ):
        super(CML, self).__init__()
        self.feature = nn.Sequential(
            # (batch * 1 * 1576 * 23) -> (batch * 32 * 1537 * 20)
            nn.Conv2d(1, 32, kernel_size=(40, 4), ),
            nn.LeakyReLU(),
            # (batch * 32 * 1537 * 20) -> (batch * 32 * 1533 * 19)
            nn.MaxPool2d(kernel_size=(5, 2), stride=1),
            # (batch * 32 * 1533 * 19) -> (batch * 64 * 1504 * 16)
            nn.Conv2d(32, 64, kernel_size=(30, 4)),
            nn.LeakyReLU(),
            # (batch * 64 * 1504 * 16) -> (batch * 128 * 1475 * 13)
            nn.Conv2d(64, 128, kernel_size=(30, 4)),
            nn.LeakyReLU(),
            # (batch * 128 * 1475 * 13) -> (batch * 128 * 1471 * 12)
            nn.MaxPool2d(kernel_size=(5, 2), stride=1),
            # (batch * 128 * 1471, 12) -> (batch * 256 * 1452 * 10)
            nn.Conv2d(128, 256, kernel_size=(20, 3)),
            nn.LeakyReLU(),
            # (batch * 256 * 1452 * 10) -> (batch * 256 * 1433 * 8)
            nn.Conv2d(256, 256, kernel_size=(20, 3)),
            nn.LeakyReLU(),
            # (batch * 256 * 1433 * 8) -> (batch * 256 * 1430 * 8)
            nn.MaxPool2d(kernel_size=(4, 1), stride=1),
            # (batch * 256 * 1430 * 8) -> (batch * 1 * 1411 * 6)
            nn.Conv2d(256, 1, kernel_size=(20, 3)),
            nn.LeakyReLU(),
            # (batch * 1 * 1411 * 6) -> (batch * 1 * 1410 * 6)
            nn.MaxPool2d(kernel_size=(2, 1), stride=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(8460, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, X_dim),
            nn.LeakyReLU()
        )
        # the projection leayer of ESM
        # self.esm_project = nn.Sequential(
        #     nn.Linear(EMS_input_dim, X_dim),
        #     nn.LeakyReLU()
        # )
        # the layer of CrossAttention
        self.cross_attn = CrossAttention(
            x_dim=X_dim,
            cond_dim=X_dim,  # the output dimension of cond_proj
            attn_dim=128
        )
        self.fakeLabel = FakeLabel(X_dim,antibiotic_count,mechanism_count,transfer_count).to(device)
        self.diffusion = ConditionalDiffusion(X_dim, [antibiotic_count,mechanism_count,transfer_count]).to(device)
        self.hidden = Hidden(X_dim, G_dim, z_dim)
        self.causal = Causal(z_dim, antibiotic_count,mechanism_count,transfer_count)

    def forward(self, seq_map, antibiotic_label, mechanism_label,transfer_label ,
                antibiotic_count,mechanism_count,transfer_count):
        x = self.feature(seq_map)
        x = torch.flatten(x, start_dim=1)
        H = self.fc(x)
        # aligned_ESM = self.esm_project(ESM_input)
        # aligned_representation = self.cross_attn(H, aligned_ESM)
        fakeAntibiocLable, fakeMechanismLable, fakeTransferLable = self.fakeLabel(H,
        antibiotic_label, mechanism_label,transfer_label,
        antibiotic_count,mechanism_count,transfer_count)
        fakeAntibiocLable.unsqueeze(-1)
        fakeMechanismLable.unsqueeze(-1)
        fakeTransferLable.unsqueeze(-1)

        labels = [fakeAntibiocLable, fakeMechanismLable, fakeTransferLable]
        G = self.diffusion(
            H,
            fakeAntibiocLable,
            fakeMechanismLable,
            fakeTransferLable,
            mode='generate'  # transfer to gernerate mode
        )
        diffusion_loss = self.diffusion(
            H,
            fakeAntibiocLable,
            fakeMechanismLable,
            fakeTransferLable,
            mode='train'  # transfer to train mode
        )

        # Z = self.hidden(H, G)
        Z = torch.cat((H, G), dim=1)
        transfer_pre, mechanism_pre, antibiotic_pre = self.causal(Z)

        return antibiotic_pre, mechanism_pre, transfer_pre, diffusion_loss


# hidden layer
class Hidden(nn.Module):
    def __init__(self, X_dim, G_dim, z_dim):
        super(Hidden, self).__init__()
        self.concat_dim = X_dim + G_dim
        self.hidden = nn.Sequential(
            nn.Linear(self.concat_dim, self.concat_dim),
            nn.LeakyReLU(),
            nn.Linear(self.concat_dim, z_dim),
        )

    def forward(self, X, G):
        input = torch.cat((X, G), dim=1)
        z = self.hidden(input)
        return z


# Causal Graph Module
class Causal(nn.Module):
    def __init__(self, input_dim, transfer_count, mechanism_count, antibiotic_count):
        super(Causal, self).__init__()
        self.transfer_layer = nn.Linear(input_dim, transfer_count)
        self.softmax = nn.Softmax(dim=1)

        self.mechanism_layer = nn.Linear(input_dim + transfer_count, mechanism_count)
        self.antibiotic_layer = nn.Linear(input_dim + transfer_count + mechanism_count, antibiotic_count)


    def forward(self, input):
        transfer_pre = self.softmax(self.transfer_layer(input))
        mechanism_pre = self.softmax(self.mechanism_layer(torch.cat((input, transfer_pre), dim=1)))
        antibiotic_pre = self.softmax(self.antibiotic_layer(torch.cat((input, transfer_pre, mechanism_pre), dim=1)))

        return antibiotic_pre, mechanism_pre, transfer_pre


# Prediction of peusdo labels
class FakeLabel(nn.Module):
    def __init__(self, X_dim,antibiotic_count, mechanism_count, transfer_count):
        super(FakeLabel, self).__init__()
        # prediction the labels of antibiotic, mechanicm and transferability
        self.antibiotic_layer = nn.Linear(X_dim, antibiotic_count)
        self.transfer_layer = nn.Linear(X_dim, transfer_count)
        self.mechanism_layer = nn.Linear(X_dim , mechanism_count)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input, antibiotic_label, mechanism_label, transfer_label,
                antibiotic_count, mechanism_count, transfer_count):
        mechanism_pre = self.softmax(self.mechanism_layer(input))
        antibiotic_pre = self.softmax(self.antibiotic_layer(input))
        transfer_pre = self.softmax(self.transfer_layer(input))

        # mixture of true labels and peusdo labels
        # transfer values to LongTensor
        antibiotic_tensor = torch.tensor(antibiotic_label, dtype=torch.long).squeeze()
        antibiotic_label_hot = F.one_hot(antibiotic_tensor, num_classes=antibiotic_count).to(input.device)
        mechanism_tensor = torch.tensor(mechanism_label, dtype=torch.long).squeeze()
        mechanism_label_hot = F.one_hot(mechanism_tensor, num_classes=mechanism_count).to(input.device)
        transfer_tensor = torch.tensor(transfer_label, dtype =torch.long).squeeze()

        transfer_label_hot = F.one_hot(transfer_tensor, num_classes=transfer_count).to(input.device)


        antibiotic_mixed = mix_predictions_and_labels(antibiotic_pre, antibiotic_label_hot)
        mechanism_mixed = mix_predictions_and_labels(mechanism_pre, mechanism_label_hot)
        transfer_mixed = mix_predictions_and_labels(transfer_pre, transfer_label_hot)
        if self.training:
            return antibiotic_mixed, mechanism_mixed, transfer_mixed
        else:
            return antibiotic_pre, mechanism_pre, transfer_pre

def mix_predictions_and_labels(predictions, labels):
    """
    Mix the predicted values and the true labels with a 50% probability
    Args:
        predictions: The probability distribution predicted by the model,shape [batch_size, num_classes]
        labels: Real labels encoded with one-hot,shape [batch_size, num_classes]
    Returns:
        The mixed tensor has the same shape as the input
    """
    rand_num = random.random()
    if rand_num < 0.5:
        return predictions
    else:
        return labels.float()



    return mixed



if __name__ == '__main__':
    batch_size = 4
    seq_map = torch.randn(batch_size, 1, 1576, 23).to(device)
    # print(seq_map.shape)
    antibiotic_count = 15
    mechanism_count = 6
    transfer_count = 2
    model = CML(5, 5, 5,   15, 6, 2).to(device)
    print(model)
    model.train()
    # print(model)
    transfer_label = torch.tensor([[0]]).to(device)
    mechanism_label = torch.tensor([[5]]).to(device)
    antibiotic_label = torch.tensor([[12]]).to(device)
    # print(transfer_label.shape)
    # exit()
    anti,mech, transfer, diffusion_loss = model.forward(seq_map,antibiotic_label, mechanism_label, transfer_label,15,6,2)
    # print(anti,mech, transfer, diffusion_loss)
    # model.forward(seq_map, torch.zeros(batch_size, 1).to(device),
    #                                                  torch.zeros(batch_size, 1).to(device),
    #                                                  torch.zeros(batch_size, 1).to(device),
    # antibiotic_count, mechanism_count, transfer_count).to(device)
    # print(transfer.shape, mech.shape, anti.shape)

    # print(torch.zeros(batch_size, 1).to(device))
    # tensor1 = torch.tensor([[1, 2, 3],[4,5,6]])
    # tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
    # cat = torch.cat((tensor1, tensor2), dim=1)
    # print(cat)
