import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 新增训练参数配置
class TrainingConfig:
    batch_size = 4
    learning_rate = 1e-3
    epochs = 100
    save_interval = 15  # 每多少epoch保存一次模型


# 损失函数定义（已包含在模型中，此处显式定义训练步骤）
def train_step(model, batch, optimizer, config):
    model.train()

    # 数据移动到设备
    H = batch["H"].to(device)
    antibiotic = batch["antibiotic"].to(device)
    mechanism = batch["mechanism"].to(device)
    transfer = batch["transfer"].to(device)

    # 梯度清零
    optimizer.zero_grad()

    # 前向传播（返回损失）
    loss = model(H, antibiotic, mechanism, transfer, mode='train')

    # 反向传播
    loss.backward()

    # 参数更新
    optimizer.step()

    return loss.item()


# 验证步骤
def validate(model, dataloader, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            H = batch["H"].to(device)
            antibiotic = batch["antibiotic"].to(device)
            mechanism = batch["mechanism"].to(device)
            transfer = batch["transfer"].to(device)

            loss = model(H, antibiotic, mechanism, transfer, mode='train')
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 自定义数据集类（示例结构）
class DiffusionDataset(Dataset):
    def __init__(self, num_samples=1000, input_dim=5):
        self.num_samples = num_samples
        self.input_dim = input_dim

        # 示例数据生成，实际使用时替换为真实数据
        self.H = torch.randn(num_samples, input_dim)
        self.antibiotic = torch.randn(num_samples, 15)
        self.mechanism = torch.randn(num_samples, 6)
        self.transfer = torch.randn(num_samples, 2)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "H": self.H[idx],
            "antibiotic": self.antibiotic[idx],
            "mechanism": self.mechanism[idx],
            "transfer": self.transfer[idx]
        }

class ConditionalDiffusion(nn.Module):
    def __init__(
            self,
            input_dim: int = 64,  # 初始特征H的维度
            cond_dims: list = [15, 6, 2],  # 条件维度 [antibiotic, mechanism, transfer]
            num_timesteps: int = 10,  # 扩散总时间步
            hidden_dim: int = 128  # 隐藏层维度

    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        # ---- 噪声调度参数 ----
        self.betas = self._linear_beta_schedule(num_timesteps).to(device)  # 线性噪声调度
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

        # ---- 条件融合模块 ----
        self.cond_proj = nn.Linear(sum(cond_dims), hidden_dim)

        # ---- 时间步编码器 ----
        self.time_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ---- 噪声预测网络 ----
        self.noise_predictor = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        # 新增交叉注意力层
        self.cross_attn = CrossAttention(
            x_dim=input_dim,
            cond_dim=hidden_dim,  # cond_proj的输出维度
            attn_dim=128
        )

    def _linear_beta_schedule(self, num_steps: int) -> torch.Tensor:
        """线性噪声调度"""
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, num_steps)

    def forward(
            self,
            H: torch.Tensor,
            antibiotic_cond: torch.Tensor,
            mechanism_cond: torch.Tensor,
            transfer_cond: torch.Tensor,
            mode: str = 'train'  # 'train'或'generate'
    ) -> torch.Tensor:
        """
        Args:
            H: 初始特征 [B, input_dim]
            antibiotic_cond: 抗生素条件 [B, cond_dims[0]]
            mechanism_cond: 机制条件 [B, cond_dims[1]]
            transfer_cond: 转移条件 [B, cond_dims[2]]
            mode: 模式（训练时需返回损失）
        Returns:
            生成的潜在表示 [B, input_dim] 或损失（训练模式）
        """
        # ---- 条件融合 ----
        cond = torch.cat([antibiotic_cond, mechanism_cond, transfer_cond], dim=1)
        cond_proj = self.cond_proj(cond)  # [B, hidden_dim]

        if mode == 'train':
            # 训练模式：执行加噪并预测噪声
            return self._train_forward(H, cond_proj)
        else:
            # 生成模式：执行完整去噪
            return self._generate(H, cond_proj)

    def _train_forward(self, H: torch.Tensor, cond_proj: torch.Tensor) -> torch.Tensor:
        """训练流程（加噪+噪声预测）"""
        # 随机采样时间步
        B = H.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=H.device)
        # 前向加噪
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(B, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t]).view(B, 1)
        noise = torch.randn_like(H)
        H_noisy = sqrt_alpha_cumprod * H + sqrt_one_minus_alpha_cumprod * noise

        # 时间编码
        t_emb = self.time_embed(timestep_embedding(t, self.input_dim))  # [B, hidden_dim]

        # 交叉注意力操作（替换原来的拼接）
        x_attn = self.cross_attn(H_noisy, cond_proj)  # [B, input_dim]
        # 合并特征、时间、条件
        combined = torch.cat([x_attn, t_emb], dim=1)

        # 预测噪声
        pred_noise = self.noise_predictor(combined)

        # 计算损失
        return F.mse_loss(pred_noise, noise)

    def _generate(self, H: torch.Tensor, cond_proj: torch.Tensor) -> torch.Tensor:
        """生成流程（去噪生成潜在表示）"""
        # 初始化为纯噪声（若需从H开始则修改此处）

        x = torch.randn_like(H).to(device)

        # 逐步去噪
        for t in reversed(range(self.num_timesteps)):
            # 时间编码
            t_batch = torch.full((x.shape[0],), t, device=x.device)
            t_emb = self.time_embed(timestep_embedding(t_batch, self.input_dim))

            # 交叉注意力操作（替换原来的拼接）
            x_attn = self.cross_attn(x, cond_proj)  # [B, input_dim]
            # 合并特征、时间、条件
            # 合并特征、时间、条件
            combined = torch.cat([x_attn, t_emb], dim=1)

            # 预测噪声
            pred_noise = self.noise_predictor(combined)

            # DDIM采样更新
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            sigma = (1 - alpha_cumprod).sqrt()

            x = (x - pred_noise * sigma) / alpha.sqrt()
            if t > 0:
                x += torch.sqrt(1 - alpha) * torch.randn_like(x)

        return x  # 去噪后的潜在表示


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """时间步正弦编码"""
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]

    if dim % 2 == 1:
        return torch.cat([torch.sin(emb), torch.cos(emb), torch.zeros_like(emb[:, :1])], dim=1)
    else:
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class CrossAttention(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int, attn_dim: int = 128):
        super().__init__()
        self.key = nn.Linear(x_dim, attn_dim, bias=False)
        self.query = nn.Linear(cond_dim, attn_dim, bias=False)
        self.value = nn.Linear(cond_dim, attn_dim, bias=False)
        self.proj = nn.Linear(attn_dim, x_dim)  # 将输出维度映射回x的维度
        self.attn_dim = attn_dim  # 显式保存为类属性

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, x_dim] 作为Key
        # cond: [B, cond_dim] 作为Query和Value
        B = x.shape[0]

        # 计算QKV
        K = self.key(x).view(B, 1, -1)  # [B, 1, attn_dim]
        Q = self.query(cond).view(B, 1, -1)  # [B, 1, attn_dim]
        V = self.value(cond).view(B, 1, -1)  # [B, 1, attn_dim]

        # 计算注意力分数
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.attn_dim ** 0.5)  # [B, 1, 1]
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 加权求和
        attended = torch.bmm(attn_weights, V)  # [B, 1, attn_dim]
        attended = attended.view(B, -1)  # [B, attn_dim]

        # 投影回原始维度
        return self.proj(attended)  # [B, x_dim]






