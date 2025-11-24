# utils/calcs_func.py

定位: 项目的 "物理引擎接口"。

作用: 强化学习（RL）本身不懂物理，它只知道输出动作。这个文件负责给每一个晶体结构分配一个 "计算器 (Calculator)"。这个计算器负责根据原子的位置，算出真实的 能量 (Energy) 和 受力 (Forces)，反馈给 RL。

## 核心知识点：ASE Calculator (计算器)

- 输入: 原子坐标 (Positions) + 原子类型 (Numbers)。
- 输出: 系统的势能 (Potential Energy) + 每个原子的受力 (Forces) + 应力 (Stress)。

在这个项目中的角色:
它是 RL 环境中的 "裁判"。

RL Agent 移动原子 -> 环境调用 Calculator -> 算出受力 -> 如果受力变小了 -> 给 Agent 发奖励。

- EAM (Embedded Atom Method):
  这是一种经验势函数，计算速度非常快，专门用来算金属（如 Al, Fe, Cu 等）。它比量子力学计算（DFT）快几万倍，适合 RL 这种需要跑几百万步训练的场景。

# utils/utils.py

定位: 通用辅助函数库 (Utility Functions)。

定位：通用辅助函数库（UtilityFunctions

    1. 可视化：画出训练过程中的 Loss（损失）和 Reward（奖励）曲线。
    2. 存档管理：自动找到最新的模型存盘点（Checkpoint），方便断点续练。
    3. 数据平滑：把波动剧烈的训练数据平均化，让曲线更好看。
    4. 数据筛选：从 CSV 文件中挑选出特定的晶体结构。

# utils/convert_to_graph_e3nn.py

定位: 数据转换器 (Data Converter)。
输入: Pymatgen 的 Structure 对象（包含原子种类、坐标）和受力 Forces。

输出: PyTorch Geometric 的 Data 对象（包含节点特征 x、边索引 edge_index、相对位置向量 edge_vec）。

核心差异: 普通 GNN 只需要距离（标量），但 e3nn 需要相对位置向量（矢量），因为它要处理旋转等变性。

A. 原子特征嵌入 (Embedding) - "查字典"
代码的前半部分（关于 atomic_properties 的部分）是在做一个“元素周期表字典”。

B. 构图函数 to_graph - "连连看"

1.节点特征 (Node Features x):
x = torch.vstack([torch.Tensor(embedding[number]) for number in structure.atomic_numbers])
遍历结构里的每个原子，查字典，拼成一个大矩阵 x。
2.找邻居 (Neighbors):
以每个原子为圆心，画一个半径为 5Å 的球，球里面的其他原子就是邻居。这就定义了图的边 (Edge)。
计算边属性 (Edge Attributes)

# utils/model_e3nn.py

定位: 神经网络大脑 (Neural Network Architectures)。
作用:
Actor (Pi): 看到当前结构 -> 决定每个原子怎么动 (输出位移向量)。
Critic (Q): 看到当前结构 + 拟采取的动作 -> 打分 (这个动作好不好？能得多少分？)。
核心技术: e3nn (E(3) Equivariant Neural Networks)。
普通神经网络：如果你把输入的椅子旋转90度，它可能就不认识了，或者输出的位移向量没有跟着旋转。
e3nn：输入旋转，输出也自动、精确地跟着旋转。这对于物理模拟至关重要。

在 e3nn 中，所有数据都有明确的几何类型，叫 Irreps。

- "0e" (标量): 只有大小，没有方向。例如：能量、质量、原子种类。
- "1o" (矢量): 有大小，有方向。例如：速度、力、位移。
- "2e" (张量): 更复杂的几何量。

```
# 输入特征 Irreps
self.irreps_in = o3.Irreps("10x0e + 5x1o")
# 翻译：输入包含 10 个标量特征（如原子属性） 和 5 个矢量特征（如受力、速度）。
```

- PeriodicNetwork_Pi (Actor,td3)
  输入：现在的环境状况（晶体结构、当前的受力）。
  输出：动作指令。它告诉每一个原子：“你往东走 0.1 埃，你往西北走 0.05 埃”。
  特点：它的输出是一个向量场（每个原子对应一个 3D 箭头）。
- PeriodicNetwork_Q (Critic,td3)
  输入：现在的环境状况 + 打算做的动作。
  输出：一个分数 (Q-value)。它预测：“如果按照刚才那个动作走，未来的结构能有多稳定？”
  特点：它的输出是一个标量（只有一个数字，代表好坏）。
- Gaussian_actor (Actor,SAC )
  输入：状态。
  输出：动作的平均值和标准差
  指令：“我感觉应该往右移 0.1 埃左右（均值），但我不太确定，所以我允许你在 0.05 到 0.15 之间随机选一个值（方差）。”
- Q_func(Critic,SAC )
  输入：当前的晶体结构 (State) + 打算做的动作 (Action)。
  输出：一个分数 (Q-value)。

# utils/utils_model.py

- 定位: 基础网络模块库 (Base Network Modules)。
- 功能:
  1.提供积木: 定义了 Convolution（卷积）和 Gate（门控非线性）的组合方式。
  2.数据预处理: 把原始的 Graph 数据（坐标、边）加工成 e3nn 卷积层能吃的格式（球谐函数编码、距离高斯展开）。
  3.通用父类: Network 和 Network_basic 是 TD3 和 SAC 网络类的父亲。

在 e3nn 中，一个标准的图神经网络层通常由两部分组成：

- Convolution (线性卷积): 负责混合特征（混合不同原子、不同方向的信息）。
- Gate (门控非线性): 类似于普通神经网络里的 ReLU，但因为矢量（方向）不能直接做 ReLU（方向会被破坏），所以用“门控”机制。
  原理：用一个标量（0~1之间）去乘矢量。比如：标量算出来是 0.5，矢量长度就缩短一半。方向不变！
  Network (用于 TD3)
  Network_basic (用于 SAC)

# utils/td3.py

- TD3 算法实现
  作用:
  初始化 Actor 和 Critic 网络。
  定义 Loss Function (怎么算误差)。
  定义 Optimizer (怎么更新参数)。
  实现 Train Loop (训练主循环)。

A. 初始化 (__init__)
    1.网络 (Networks):
        self.ac: 包含 pi (Actor), q1, q2 (两个 Critics)。
        self.ac_targ: 目标网络 (Target Networks)。这是 TD3 的关键特征，它是主网络的一个“滞后备份”，用来稳定训练，防止Q值忽高忽低。
    2.优化器 (Optimizers):
        pi_optimizer: 只更新 Actor。
        q_optimizer: 只更新 Critics。
    3.经验回放池 (Replay Buffer):
        self.memory: 用来存历史数据的（State, Action, Reward, Next_State）。

B. 动作选择 (get_action)

# utils/replay_memory.py


* **定位** **:** **经验回放池 (Replay Buffer)**。
* **作用**:

  * **存储 (Record)**: 把环境里发生的所有事情(S,A,R,S′,D)**存下来。**
* **回放 (Sample)**: 训练时，随机抽出一批“往事”给 AI 学习。
