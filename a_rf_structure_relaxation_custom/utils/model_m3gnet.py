import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.data import Data

# M3GNet 风格的激活函数 (Swish / SiLU)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class RBFExpansion(nn.Module):
    """径向基函数扩展 (Radial Basis Function Expansion)"""
    def __init__(self, vmin=0, vmax=5.0, bins=50):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.centers = torch.linspace(vmin, vmax, bins)
        self.width = (vmax - vmin) / bins

    def forward(self, distance):
        # distance: [E]
        # return: [E, bins]
        centers = self.centers.to(distance.device)
        return torch.exp(-((distance.unsqueeze(-1) - centers) ** 2) / self.width ** 2)

class M3GBlock(nn.Module):
    """M3GNet 交互模块 (包含三体项)"""
    def __init__(self, dim, num_rbf):
        super().__init__()
        self.dim = dim

        # 边特征处理
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim + num_rbf + dim, dim), # input: node_i + rbf + node_j
            Swish(),
            nn.Linear(dim, dim),
            Swish()
        )

        # 节点更新
        self.node_mlp = nn.Sequential(
            nn.Linear(dim + dim, dim), # input: node + aggregated_edges
            Swish(),
            nn.Linear(dim, dim),
            Swish()
        )

        # 三体交互 (3-Body Gating)
        # 利用上下文平均向量来近似计算角度 theta_ijk
        self.three_body_mlp = nn.Sequential(
            nn.Linear(dim + 1, dim), # input: edge_feat + cos_theta
            Swish(),
            nn.Linear(dim, dim),
            nn.Sigmoid() # Gate
        )

    def forward(self, x, edge_index, edge_attr, edge_vec, edge_length):
        row, col = edge_index

        # 1. 边特征更新 (2-body)
        # Concatenate: x_i, rbf_ij, x_j
        edge_input = torch.cat([x[row], edge_attr, x[col]], dim=-1)
        m_ij = self.edge_mlp(edge_input)

        # 2. 三体交互 (3-body)
        # 计算局部平均向量 (Context Vector)
        num_nodes = x.shape[0]
        node_vec_sum = scatter(edge_vec, col, dim=0, dim_size=num_nodes, reduce='mean')

        # 广播回边
        ctx_vec = node_vec_sum[col]

        # 计算 cos(theta)
        dot = (edge_vec * ctx_vec).sum(dim=1)
        norm = edge_length * (ctx_vec.norm(dim=1) + 1e-6)
        cos_theta = (dot / norm).unsqueeze(-1).clamp(-1, 1) # [E, 1]

        # 三体门控: 根据角度调整边的权重
        # M3GNet 思想: m_ij' = m_ij * Gate(theta)
        # 这里把 cos_theta 拼接到 m_ij 上计算 gate
        g_ijk = self.three_body_mlp(torch.cat([m_ij, cos_theta], dim=-1))
        m_ij_prime = m_ij * g_ijk

        # 3. 聚合到节点
        aggr_edges = scatter(m_ij_prime, col, dim=0, dim_size=num_nodes, reduce='add')

        # 4. 节点更新
        x_new = self.node_mlp(torch.cat([x, aggr_edges], dim=-1))

        # 残差连接
        return x + x_new

class M3GNet_actor(nn.Module):
    r"""M3GNet Actor"""
    def __init__(self, input_dim, aggr, n_l1, n_l2, n_l3, n_l4, noise_clip):
        super().__init__()
        self.noise_clip = noise_clip

        # 隐藏层维度 (n_l3 作为主维度)
        hidden_dim = n_l3
        self.hidden_dim = hidden_dim

        # RBF 扩展
        self.rbf = RBFExpansion(vmax=5.0, bins=n_l2)

        # 初始 Embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, n_l1),
            Swish(),
            nn.Linear(n_l1, hidden_dim)
        )

        # 交互层 (相当于 Conv)
        # M3GNet 通常堆叠多层
        self.block1 = M3GBlock(hidden_dim, n_l2)
        self.block2 = M3GBlock(hidden_dim, n_l2)
        self.block3 = M3GBlock(hidden_dim, n_l2)

        # 输出头
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, n_l4),
            Swish(),
            nn.Linear(n_l4, 3), # 3D Force/Displacement
            nn.Tanh() # 限制范围 [-1, 1]
        )

    def forward(self, data, noise_scale=None):
        x, edge_index, edge_length = data.x, data.edge_index, data.edge_length
        forces_stack, forces_norm = data.forces_stack, data.forces_norm

        # 如果 data 里没有 edge_vec (方向向量)，需要重新计算
        if hasattr(data, 'edge_vec'):
            edge_vec = data.edge_vec
        else:
            pos = data.pos
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        # 1. RBF Embedding
        edge_attr = self.rbf(edge_length)

        # 2. Node Embedding
        x_in = torch.cat((x, forces_stack, forces_norm), dim=1)
        h = self.embedding(x_in)

        # 3. Interaction Blocks
        h = self.block1(h, edge_index, edge_attr, edge_vec, edge_length)
        h = self.block2(h, edge_index, edge_attr, edge_vec, edge_length)
        h = self.block3(h, edge_index, edge_attr, edge_vec, edge_length)

        # 4. Output
        out = self.out_mlp(h)

        if noise_scale is not None:
            epsilon = torch.randn_like(out) * noise_scale
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            out = out + epsilon

        return Data(x=out)

class M3GNet_critic(nn.Module):
    r"""M3GNet Critic"""
    def __init__(self, input_dim, aggr, n_l1, n_l2, n_l3, n_l4, n_l5):
        super().__init__()

        hidden_dim = n_l3

        self.rbf = RBFExpansion(vmax=5.0, bins=n_l2)

        # State Embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, n_l1),
            Swish(),
            nn.Linear(n_l1, hidden_dim)
        )

        # Action Embedding
        self.action_emb = nn.Linear(3, hidden_dim)

        # 融合后的维度还是 hidden_dim (通过加法或拼接投影)
        # 这里采用拼接后投影的方式
        self.merge = nn.Linear(hidden_dim * 2, hidden_dim)

        self.block1 = M3GBlock(hidden_dim, n_l2)
        self.block2 = M3GBlock(hidden_dim, n_l2)
        self.block3 = M3GBlock(hidden_dim, n_l2)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, n_l4),
            Swish(),
            nn.Linear(n_l4, 1) # Scalar Q-value
        )

    def forward(self, data, action):
        x, edge_index, edge_length = data.x, data.edge_index, data.edge_length
        forces_stack, forces_norm = data.forces_stack, data.forces_norm

        if hasattr(data, 'edge_vec'):
            edge_vec = data.edge_vec
        else:
            pos = data.pos
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        edge_attr = self.rbf(edge_length)

        # 1. State Feature
        x_in = torch.cat((x, forces_stack, forces_norm), dim=1)
        h_s = self.embedding(x_in)

        # 2. Action Feature
        h_a = self.action_emb(action.x)

        # 3. Merge
        h = self.merge(torch.cat([h_s, h_a], dim=-1))

        # 4. Blocks
        h = self.block1(h, edge_index, edge_attr, edge_vec, edge_length)
        h = self.block2(h, edge_index, edge_attr, edge_vec, edge_length)
        h = self.block3(h, edge_index, edge_attr, edge_vec, edge_length)

        # 5. Global Pooling (Readout)
        y = scatter(h, data.batch, dim=0, reduce='add') # Global Sum Pooling
        y = self.readout(y)

        return torch.squeeze(y, -1)