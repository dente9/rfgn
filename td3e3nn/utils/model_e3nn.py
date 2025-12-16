import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from e3nn import o3
import torch_scatter
import e3nn
from torch_scatter import scatter
from typing import Dict, Union
from torch_cluster import radius_graph
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists
from utils.env import ENV_INFO_DIM
ENV_INFO_DIM = ENV_INFO_DIM
torch.set_default_dtype(torch.float64)



# ==============================================================================
# 1. 基础组件
# ==============================================================================

class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x

# ==============================================================================
# 2. 基础网络类 (Network) - 修改支持后期融合 (Late Fusion)
# ==============================================================================
class Network(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_node_attr,
        layers,
        mul,
        lmax,
        max_radius,
        number_of_basis=10,
        radial_layers=1,
        radial_neurons=100,
        num_neighbors=1.,
        num_nodes=1.,
        reduce_output=True,
        late_fusion_dim=0, # [新增] 这里预留接口：允许在最后一步混入外部特征
        act=None,
        act_gates=None
    ) -> None:
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output
        self.late_fusion_dim = late_fusion_dim # 记录一下

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        if act is None: act = {1: torch.nn.functional.silu, -1: torch.tanh}
        if act_gates is None: act_gates = {1: torch.sigmoid, -1: torch.tanh}



        self.layers = torch.nn.ModuleList()

        # --- A. 构建主干网络 (Backbone) ---
        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
            irreps = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))

        # --- B. 构建输出层 (Output Layer) ---
        # [逻辑] 如果有 Late Fusion，最后一层的输入通道数要变大
        final_in_irreps = irreps
        if self.late_fusion_dim > 0:
            # 外部特征通常是标量 (0e)，所以在这里追加通道
            final_in_irreps += o3.Irreps(f"{self.late_fusion_dim}x0e")

        self.final_layer = Convolution(
            final_in_irreps, # 使用变大后的输入维度
            self.irreps_node_attr,
            self.irreps_edge_attr,
            self.irreps_out,
            number_of_basis,
            radial_layers,
            radial_neurons,
            num_neighbors
        )

        if self.late_fusion_dim > 0:
            self.fusion_weight = nn.Parameter(torch.tensor(0.01))

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        if 'edge_index' in data:
            edge_src = data['edge_index'][0]
            edge_dst = data['edge_index'][1]
            edge_vec = data['edge_vec']
        else:
            edge_index = radius_graph(data['pos'], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]

        return batch, edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]], late_feat=None) -> torch.Tensor:
        batch, edge_src, edge_dst, edge_vec = self.preprocess(data)
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='gaussian',
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and 'x' in data:
            x = data['x']
        else:
            x = data['pos'].new_ones((data['pos'].shape[0], 1))

        if self.input_has_node_attr and 'z' in data:
            z = data['z']
        else:
            z = data['pos'].new_ones((data['pos'].shape[0], 1))

        data["edge_attr"] = edge_attr
        data["edge_length_embedded"] = edge_length_embedded

        # 1. 跑主干网络 (只处理图信息)
        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        # 2. [融合点] 强硬拼接外部信息
        if self.late_fusion_dim > 0:
            if late_feat is None:
                raise ValueError("Model expects env_info (late_feat) but got None!")
            # 拼接: [原子特征, 环境特征]
            scaled_late_feat = late_feat * self.fusion_weight
            x = torch.cat([x, scaled_late_feat], dim=1)


        # 3. 跑输出层
        x = self.final_layer(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return x

class PeriodicNetwork_Pi(Network):
    def __init__(self, em_dim, noise_clip=0.2, scaled=False, expl_mode='state',
                 env_input_dim=ENV_INFO_DIM, **kwargs):

        # 【关键设置】关闭父类的强制融合，我们要自己控制融合方式
        kwargs['late_fusion_dim'] = 0
        kwargs['reduce_output'] = False

        super().__init__(**kwargs)

        self.scaled = scaled
        self.env_input_dim = env_input_dim
        self.em = nn.Linear(1, em_dim)
        self.noise_clip = noise_clip
        self.exploration_mode = expl_mode

        # 【环境门控网络】
        # 输入: env_info
        # 输出: 标量系数 (Scaling Factor)
        # 逻辑: 某些环境(如高压)可能需要小步长，某些环境需要大步长
        if self.env_input_dim > 0:
            self.env_gate = nn.Sequential(
                nn.Linear(env_input_dim, 64),
                nn.LayerNorm(64), # 稳定输入分布
                nn.SiLU(),
                nn.Linear(64, 1),
                nn.Softplus()     # 保证缩放系数恒为正 (>0)，防止反转方向
            )

    def forward(self, data, env_info=None, noise_scale=None):
        data_copy = data.clone()

        # [安全检查]
        if getattr(data_copy, 'batch', None) is None:
            data_copy.batch = torch.zeros(data_copy.num_nodes, dtype=torch.long, device=data_copy.x.device)

        # --- 1. 探索噪声与力特征处理 (保持原逻辑) ---
        if noise_scale is not None and self.exploration_mode == 'state':
            axis, angle = e3nn.o3.rand_axis_angle(1)
            angle *= 0.5 * noise_scale
            angle = torch.clamp(angle, -self.noise_clip, self.noise_clip)
            rot_matrix = e3nn.o3.axis_angle_to_matrix(axis, angle).to(data_copy.forces_stack.device)
            data_copy.forces_stack = torch.matmul(data_copy.forces_stack, rot_matrix[0])

            epsilon = (2 * torch.rand(data_copy.forces_norm.shape[0], 1) - 1) * noise_scale
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip).to(data_copy.forces_stack.device)
            data_copy.forces_norm *= (1 + epsilon)

        forces_ampl = F.leaky_relu(self.em(data_copy.forces_norm))
        data_copy.x = torch.hstack([data_copy.x, data_copy.forces_stack, forces_ampl])

        # --- 2. 纯粹的图卷积 (得到基础几何动作) ---
        # 此时不传入 late_feat，让 e3nn 专注处理几何等变性
        # raw_action: [Nodes, 3] (假设输出是 1x1o)
        raw_action = super().forward(data_copy, late_feat=None)

        # --- 3. 环境融合 (Scaling) ---
        final_action = raw_action

        if self.env_input_dim > 0:
            # 获取数据
            if env_info is None:
                env_info = getattr(data_copy, 'env_info', None)

            # [防御性逻辑] 确保一定有 Tensor 输入
            if env_info is not None:
                # 正常计算
                env_tensor = env_info.to(dtype=self.em.weight.dtype)
            else:
                # 补零: [Batch, env_input_dim]
                # 注意：这里需要构造 batch size 行，env_input_dim 列
                batch_size = int(data_copy.batch.max()) + 1
                env_tensor = torch.zeros((batch_size, self.env_input_dim),
                                         dtype=self.em.weight.dtype, device=data_copy.x.device)

            # 计算缩放系数 [Batch, 1]
            # 加上 0.5 或 1.0 的 bias，保证初始状态下网络依然能动，而不是被缩放到 0
            scale = self.env_gate(env_tensor) + 0.5

            # 广播到节点 [Nodes, 1]
            node_scale = scale[data_copy.batch]

            # 乘法融合：保留方向，调整幅度
            final_action = raw_action * node_scale

        # --- 4. 后处理 ---
        if noise_scale is not None and self.exploration_mode == 'action':
            epsilon = torch.randn_like(final_action) * noise_scale
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            final_action += epsilon

        if self.scaled:
            final_action = torch.tanh(final_action)

        return Data(x=final_action)


# ==============================================================================
# 2. Critic (Q) - 采用 Late Fusion (池化后拼接)
# ==============================================================================
class PeriodicNetwork_Q(Network):
    def __init__(self, em_dim, env_input_dim=ENV_INFO_DIM, **kwargs):
        # 【关键设置】关闭父类的强制融合
        kwargs['late_fusion_dim'] = 0
        kwargs['reduce_output'] = False

        super().__init__(**kwargs)

        self.em = nn.Linear(1, em_dim)
        self.em_act = nn.Linear(1, em_dim)
        self.env_input_dim = env_input_dim

        # 图特征的维度 (e3nn 输出的维度)
        graph_feat_dim = self.irreps_out.dim

        # 环境特征提取器
        if self.env_input_dim > 0:
            self.env_proj = nn.Sequential(
                nn.Linear(env_input_dim, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.Tanh() # 压缩到 [-1, 1] 以匹配 graph feature 的量级
            )
            # 最终打分层: 输入 = 图特征 + 环境特征
            self.tail_mlp = nn.Sequential(
                nn.Linear(graph_feat_dim + 64, 64),
                nn.SiLU(),
                nn.Linear(64, 1)
            )
        else:
            self.tail_mlp = nn.Linear(graph_feat_dim, 1)

    def forward(self, data, actions, env_info=None) -> torch.Tensor:
        data_copy = data.clone()

        # [安全检查]
        if getattr(data_copy, 'batch', None) is None:
            data_copy.batch = torch.zeros(data_copy.num_nodes, dtype=torch.long, device=data_copy.x.device)

        # --- 1. 动作与力的特征嵌入 (保持原逻辑) ---
        action = actions.x
        action_norm = action.norm(dim=1)
        action_norm_cor = action_norm + 1 * (action_norm == 0)
        action_stack = action / action_norm_cor[:, None]
        action_ampl = F.leaky_relu(self.em_act(action_norm.unsqueeze(1)))
        forces_ampl = F.leaky_relu(self.em(data_copy.forces_norm))

        data_copy.x = torch.hstack([
            data_copy.x,
            data_copy.forces_stack,
            forces_ampl,
            action_stack,
            action_ampl
        ])

        # --- 2. 纯粹的图卷积 ---
        # output: [Nodes, Features]
        output = super().forward(data_copy, late_feat=None)

        # --- 3. 全局池化 (Local -> Global) ---
        # graph_feat: [Batch, Features]
        graph_feat = torch_scatter.scatter_mean(output, data_copy.batch, dim=0)

        # --- 4. 环境融合 (Concatenation) ---
        if self.env_input_dim > 0:
            if env_info is None:
                env_info = getattr(data_copy, 'env_info', None)

            # [防御性逻辑]
            if env_info is not None:
                env_tensor = env_info.to(dtype=self.em.weight.dtype)
            else:
                batch_size = graph_feat.shape[0]
                env_tensor = torch.zeros((batch_size, self.env_input_dim),
                                         dtype=self.em.weight.dtype, device=graph_feat.device)

            # 投影环境特征 [Batch, 64]
            env_emb = self.env_proj(env_tensor)

            # 拼接: [Batch, Graph_Dim + 64]
            final_vec = torch.cat([graph_feat, env_emb], dim=1)

            # 打分
            q_value = self.tail_mlp(final_vec)
        else:
            q_value = self.tail_mlp(graph_feat)

        return torch.squeeze(q_value, -1)