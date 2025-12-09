import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from ConditionalDiffusion import ConditionalDiffusion, CrossAttention

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CML(nn.Module):
    def __init__(self, X_dim, G_dim, z_dim, EMS_input_dim, antibiotic_count, mechanism_count, transfer_count):
        super(CML, self).__init__()
        
        # CNN特征提取器
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(40, 4)),
            nn.BatchNorm2d(32),  # 优化: 添加BN
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(5, 2), stride=1),
            
            nn.Conv2d(32, 64, kernel_size=(30, 4)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 128, kernel_size=(30, 4)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(5, 2), stride=1),
            
            nn.Conv2d(128, 256, kernel_size=(20, 3)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.Conv2d(256, 256, kernel_size=(20, 3)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=1),
            
            nn.Conv2d(256, 1, kernel_size=(20, 3)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(8460, 256),
            nn.BatchNorm1d(256),  # 添加BN
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),  
            nn.Linear(256, X_dim),
            nn.BatchNorm1d(X_dim),
            nn.LeakyReLU()
        )
        
        self.esm_project = nn.Sequential(
            nn.Linear(EMS_input_dim, X_dim),
            nn.LeakyReLU()
        )
        
        self.cross_attn = CrossAttention(
            x_dim=X_dim,
            cond_dim=X_dim,
            attn_dim=128
        )
        
        self.fakeLabel = FakeLabel(X_dim, antibiotic_count, mechanism_count, transfer_count).to(device)
        self.diffusion = ConditionalDiffusion(X_dim, [antibiotic_count, mechanism_count, transfer_count]).to(device)
        self.hidden = Hidden(X_dim, G_dim, z_dim)
        self.causal = Causal(z_dim, antibiotic_count, mechanism_count, transfer_count)

    def forward(self, seq_map, antibiotic_label, mechanism_label, transfer_label,
                antibiotic_count, mechanism_count, transfer_count):
        x = self.feature(seq_map)
        x = torch.flatten(x, start_dim=1)
        H = self.fc(x)
        
        # 生成伪标签
        fakeAntibiocLable, fakeMechanismLable, fakeTransferLable = self.fakeLabel(
            H, antibiotic_label, mechanism_label, transfer_label,
            antibiotic_count, mechanism_count, transfer_count
        )
        
        # 扩散模型生成G
        G = self.diffusion(
            H,
            fakeAntibiocLable,
            fakeMechanismLable,
            fakeTransferLable,
            mode='generate'
        )
        
        diffusion_loss = self.diffusion(
            H,
            fakeAntibiocLable,
            fakeMechanismLable,
            fakeTransferLable,
            mode='train'
        )
        
        # 融合H和G
        Z = torch.cat((H, G), dim=1)
        
        # 因果预测
        antibiotic_pre, mechanism_pre, transfer_pre = self.causal(Z)
        
        return antibiotic_pre, mechanism_pre, transfer_pre, diffusion_loss


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


class Causal(nn.Module):
    def __init__(self, input_dim, antibiotic_count, mechanism_count, transfer_count):
        super(Causal, self).__init__()

        # Balanced dropout to prevent both NaN and overfitting
        self.transfer_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, transfer_count)
        )

        self.mechanism_layer = nn.Sequential(
            nn.Linear(input_dim + transfer_count, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, mechanism_count)
        )

        self.antibiotic_layer = nn.Sequential(
            nn.Linear(input_dim + transfer_count + mechanism_count, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, antibiotic_count)
        )

        self.softmax = nn.Softmax(dim=1)

        # Xavier initialization for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        transfer_pre = self.softmax(self.transfer_layer(input))
        mechanism_pre = self.softmax(self.mechanism_layer(torch.cat((input, transfer_pre), dim=1)))
        antibiotic_pre = self.softmax(self.antibiotic_layer(torch.cat((input, transfer_pre, mechanism_pre), dim=1)))

        return antibiotic_pre, mechanism_pre, transfer_pre


class FakeLabel(nn.Module):
    def __init__(self, X_dim, antibiotic_count, mechanism_count, transfer_count):
        super(FakeLabel, self).__init__()

        # Balanced dropout to prevent both NaN and overfitting
        self.antibiotic_layer = nn.Sequential(
            nn.Linear(X_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, antibiotic_count)
        )

        self.transfer_layer = nn.Sequential(
            nn.Linear(X_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, transfer_count)
        )

        self.mechanism_layer = nn.Sequential(
            nn.Linear(X_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, mechanism_count)
        )

        self.softmax = nn.Softmax(dim=1)

        # Xavier initialization for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, antibiotic_label, mechanism_label, transfer_label,
                antibiotic_count, mechanism_count, transfer_count):
        mechanism_pre = self.softmax(self.mechanism_layer(input))
        antibiotic_pre = self.softmax(self.antibiotic_layer(input))
        transfer_pre = self.softmax(self.transfer_layer(input))

        # 转换为one-hot
        antibiotic_tensor = torch.tensor(antibiotic_label, dtype=torch.long).squeeze(1)
        antibiotic_label_hot = F.one_hot(antibiotic_tensor, num_classes=antibiotic_count).to(input.device)

        mechanism_tensor = torch.tensor(mechanism_label, dtype=torch.long).squeeze(1)
        mechanism_label_hot = F.one_hot(mechanism_tensor, num_classes=mechanism_count).to(input.device)

        transfer_tensor = torch.tensor(transfer_label, dtype=torch.long).squeeze(1)
        transfer_label_hot = F.one_hot(transfer_tensor, num_classes=transfer_count).to(input.device)

        if self.training:
            temp_antibiotic = 0.5   # 50% true labels + 50% predictions
            temp_mechanism = 0.5    # 50% true labels + 50% predictions
            temp_transfer = 0.5     # 50% true labels + 50% predictions

            antibiotic_mixed = temp_antibiotic * antibiotic_label_hot.float() + (1 - temp_antibiotic) * antibiotic_pre
            mechanism_mixed = temp_mechanism * mechanism_label_hot.float() + (1 - temp_mechanism) * mechanism_pre
            transfer_mixed = temp_transfer * transfer_label_hot.float() + (1 - temp_transfer) * transfer_pre
            return antibiotic_mixed, mechanism_mixed, transfer_mixed
        else:
            return antibiotic_label_hot.float(), mechanism_label_hot.float(), transfer_label_hot.float()


def mix_predictions_and_labels(predictions, labels):
    """已废弃 - 使用温度混合策略"""
    rand_num = random.random()
    if rand_num < 0.5:
        return predictions
    else:
        return labels.float()
