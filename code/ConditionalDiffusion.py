import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDiffusion(nn.Module):
    def __init__(
            self,
            input_dim: int = 128,
            cond_dims: list = [15, 6, 2],
            num_timesteps: int = 100,  # 优化: 从5增加到100
            hidden_dim: int = 256  # 优化: 从128增加到256
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        # 优化: 改用cosine噪声调度
        self.betas = self._cosine_beta_schedule(num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        # 优化: 增强条件融合模块
        self.cond_proj = nn.Sequential(
            nn.Linear(sum(cond_dims), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 时间步编码器
        self.time_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 优化: 加深噪声预测网络
        self.noise_predictor = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.cross_attn = CrossAttention(
            x_dim=input_dim,
            cond_dim=hidden_dim,
            attn_dim=128
        )

    def _cosine_beta_schedule(self, num_steps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine噪声调度 - 更平滑的噪声分布"""
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward(
            self,
            H: torch.Tensor,
            antibiotic_cond: torch.Tensor,
            mechanism_cond: torch.Tensor,
            transfer_cond: torch.Tensor,
            mode: str = 'train'
    ) -> torch.Tensor:
        # 条件融合
        cond = torch.cat([antibiotic_cond, mechanism_cond, transfer_cond], dim=1).float()
        cond_proj = self.cond_proj(cond)

        if mode == 'train':
            return self._train_forward(H, cond_proj)
        else:
            return self._generate(H, cond_proj)

    def _train_forward(self, H: torch.Tensor, cond_proj: torch.Tensor) -> torch.Tensor:
        """训练流程"""
        B = H.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=H.device)

        # 前向加噪
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(B, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t]).view(B, 1)
        noise = torch.randn_like(H)
        H_noisy = sqrt_alpha_cumprod * H + sqrt_one_minus_alpha_cumprod * noise

        # 时间编码
        t_emb = self.time_embed(timestep_embedding(t, self.input_dim))

        # 交叉注意力
        x_attn = self.cross_attn(H_noisy, cond_proj)
        combined = torch.cat([x_attn, t_emb], dim=1)

        # 预测噪声
        pred_noise = self.noise_predictor(combined)

        # 计算损失
        return F.mse_loss(pred_noise, noise)

    def _generate(self, H: torch.Tensor, cond_proj: torch.Tensor) -> torch.Tensor:
        """生成流程 - 优化的DDPM采样"""
        x = torch.randn_like(H).to(device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((x.shape[0],), t, device=x.device)
            t_emb = self.time_embed(timestep_embedding(t_batch, self.input_dim))

            x_attn = self.cross_attn(x, cond_proj)
            combined = torch.cat([x_attn, t_emb], dim=1)

            pred_noise = self.noise_predictor(combined)

            # 优化的DDPM采样
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
            x = coef1 * (x - coef2 * pred_noise)

            if t > 0:
                noise = torch.randn_like(x)
                alpha_cumprod_prev = self.alphas_cumprod_prev[t]
                sigma = torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * (1 - alpha))
                x = x + sigma * noise

        return x


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
        self.proj = nn.Linear(attn_dim, x_dim)
        self.attn_dim = attn_dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        K = self.key(x).view(B, 1, -1)
        Q = self.query(cond).view(B, 1, -1)
        V = self.value(cond).view(B, 1, -1)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.attn_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attended = torch.bmm(attn_weights, V)
        attended = attended.view(B, -1)

        return self.proj(attended)
