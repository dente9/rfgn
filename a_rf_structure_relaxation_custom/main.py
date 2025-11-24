import torch
import os
import pandas as pd
import numpy as np

# 引入项目模块
from utils.convert_to_graph_e3nn import to_graph
from utils.model_e3nn import PeriodicNetwork_Pi, PeriodicNetwork_Q
from utils.td3 import TD3Agent
from utils.env import Environment, get_sturct_lib_and_calcs_gen
from utils.utils import get_the_last_checkpoint, extract_number

# 1. 全局设置
# ------------------------------------------------------------------
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# 2. 配置参数 (对应你提供的命令行参数)
# ------------------------------------------------------------------
config = {
    # --- 路径与文件 ---
    "path_to_the_main_dir": "outputs/exp_name",  # --path_to_the_main_dir
    "structures_file": "structures/AlFe.csv",    # --structures_file
    "interval": [1, 2],                          # --interval (注意：Python切片是左闭右开，这里只选第1个结构)
    "env_name": "AlFe_cubic",                    # --env_name

    # --- 环境参数 ---
    "reward_func": "force",      # --reward_func
    "r_weights": [1, 1, 1],      # --r_weights
    "eps": 0.01,                 # --eps
    "r0": 1.5,                   # --r0
    "stop_numb": 50,             # (原代码默认值，虽然命令行没传，但Env需要)
    "trans_coef": 0.5,           # --trans_coef
    "with_weights": False,       # --with_weights

    # --- 神经网络参数 (Actor/Critic) ---
    "em_dim": 10,                # --em_dim
    "r_max": 5.0,                # --r_max
    "num_neighbors": 25,         # --num_neighbors
    "noise_clip": 0.1,           # --noise_clip
    "expl_mode": 'state',        # --expl_mode

    # Actor 具体参数
    "pi_n_layers": 2,            # --pi_n_layers
    "pi_mul": 20,                # --pi_mul
    "pi_lmax": 2,                # --pi_lmax

    # Critic 具体参数
    "q_n_layers": 2,             # --q_n_layers
    "q_mul": 20,                 # --q_mul
    "q_lmax": 2,                 # --q_lmax

    # --- Agent 训练参数 ---
    "random_seed": 5599,         # --random_seed
    "replay_size": int(1e6),     # --replay_size (1000000)
    "gamma": 0.9999,             # --gamma
    "polyak": 0.995,             # --polyak
    "pi_lr": 1e-05,              # --pi_lr
    "q_lr": 1e-05,               # --q_lr
    "batch_size": 100,           # --batch_size
    "start_steps": 0,            # --start_steps
    "update_after": 0,           # --update_after
    "update_every": 1,           # --update_every
    "target_noise": 0.05,        # --target_noise
    "policy_delay": 2,           # --policy_delay
    "noise": [0.2, 0.2],         # --noise

    # --- 训练循环控制 ---
    "train_ep": [800, 1000],     # --train_ep (800个回合，每回合1000步)
    "test_ep": [10, 100],        # --test_ep
    "save_result": True,         # --save_result
    "test_random": False,        # --test_random
    "test_every": 1000,          # --test_every
    "save_every": 1000,          # --save_every
    "start_iter": 0,             # --start_iter

    # --- 贪婪探索与辅助参数 ---
    "N_gr": int(1e6),            # --N_gr (1000000)
    "d_r_max": 0.0,              # --d_r_max
    "f_max": 0.001,              # --f_max
    "noise_level": 29.0,         # --noise_level
    "nfake": 10,                 # (默认值，你的命令里没传，补上以防报错)
    "path_load": None,           # (默认值)
    "path_weights": None,        # (默认值)
    "e_lim": None,               # (默认值)
    "net_lim": None              # (默认值)
}

# 3. 数据加载与环境准备
# ------------------------------------------------------------------
print(f"Loading structures from {config['structures_file']}...")
# 调用 utils.env 中的函数读取CSV并生成计算器
s_lib, calcs = get_sturct_lib_and_calcs_gen(config['structures_file'], interval=config['interval'])

env_kwargs = {
    "input_struct_lib": s_lib,
    "calculator_lib": calcs,
    "convert_to_graph_func": to_graph,
    "r0": config['r0'],
    "eps": config['eps'],
    "reward_func": config['reward_func'],
    "r_weights": config['r_weights'],
    "stop_numb": config['stop_numb']
}

# 4. 构建网络特征字典 (Network Config)
# ------------------------------------------------------------------
# 12x0e 代表原子属性输入的维度 (确保 atomic_properties.csv 匹配)
input_irreps = f"12x0e + 1x1o + {config['em_dim']}x0e"

actor_feat = {
    "em_dim": config['em_dim'],
    "irreps_in": input_irreps,
    "irreps_out": "1x1o",
    "noise_clip": config['noise_clip'],
    "irreps_node_attr": "0e",
    "layers": config['pi_n_layers'],
    "mul": config['pi_mul'],
    "lmax": config['pi_lmax'],
    "max_radius": config['r_max'],
    "num_neighbors": config['num_neighbors'],
    "expl_mode": config['expl_mode'],
    "reduce_output": False
}

# Critic 输入多一个 Action (1x1o) 和 Action Embedding (em_dim x 0e)
critic_input_irreps = f"{input_irreps} + 1x1o + {config['em_dim']}x0e"

critic_feat = {
    "em_dim": config['em_dim'],
    "irreps_in": critic_input_irreps,
    "irreps_out": "1x0e",
    "irreps_node_attr": "0e",
    "layers": config['q_n_layers'],
    "mul": config['q_mul'],
    "lmax": config['q_lmax'],
    "max_radius": config['r_max'],
    "num_neighbors": config['num_neighbors'],
    "reduce_output": True
}

ac_kwargs = {
    "net_actor": PeriodicNetwork_Pi,
    "net_critic": PeriodicNetwork_Q,
    "actor_feat": actor_feat,
    "critic_feat": critic_feat
}

# 5. 目录检查与权重处理
# ------------------------------------------------------------------
if not os.path.exists(config['path_to_the_main_dir']):
    os.makedirs(config['path_to_the_main_dir'])

# 这里的逻辑对应原代码的权重加载，因为你设置了 with_weights=False，这里 wf 为 None
wf = None
if config['with_weights'] and config['path_weights'] is not None:
    # (省略了复杂的读取逻辑，因为你的命令不需要)
    pass

# 6. 初始化 TD3 Agent
# ------------------------------------------------------------------
agent_params = {
    "env_fn": Environment,
    "env_kwards": env_kwargs,
    "ac_kwargs": ac_kwargs,
    "seed": config['random_seed'],
    "replay_size": config['replay_size'],
    "gamma": config['gamma'],
    "polyak": config['polyak'],
    "pi_lr": config['pi_lr'],
    "q_lr": config['q_lr'],
    "batch_size": config['batch_size'],
    "start_steps": config['start_steps'],
    "update_after": config['update_after'],
    "update_every": config['update_every'],
    "target_noise": config['target_noise'],
    "noise_clip": config['noise_clip'],
    "policy_delay": config['policy_delay'],
    "trans_coef": config['trans_coef'],
    "noise": config['noise'],
    "init_rewards_for_weights": wf,
    "with_weights": config['with_weights']
}

print("Initializing Agent...")
TD3_Agent = TD3Agent(**agent_params)

# 加载模型 (如果指定了 path_load)
if config['path_load'] == 'last':
    path_checkpoints = config['path_to_the_main_dir'] + "/checkpoints"
    if os.path.exists(path_checkpoints):
        config['path_load'] = get_the_last_checkpoint(path_checkpoints)
        config['start_steps'] = extract_number(config['path_load'])
        print(f"Resuming from step: {config['start_steps']}")

if config['path_load'] is not None and os.path.exists(config['path_load']):
    TD3_Agent.load_model(config['path_load'])

# 7. 保存配置并开始训练
# ------------------------------------------------------------------
# 保存 Agent 参数日志
if config['start_iter'] == 0:
    with open(config['path_to_the_main_dir'] + "/TD3_Agent_arguments.txt", 'w') as f:
        # 为了避免报错，简单转为字符串
        f.write(str(agent_params))

# 准备训练参数字典
train_args = {
    "train_ep": config['train_ep'],
    "test_ep": config['test_ep'],
    "path_to_the_main_dir": config['path_to_the_main_dir'],
    "env_name": config['env_name'],
    "e_lim": config['e_lim'],
    "net_lim": config['net_lim'],
    "save_result": config['save_result'],
    "start_iter": config['start_iter'],
    "test_random": config['test_random'],
    "test_every": config['test_every'],
    "save_every": config['save_every'],
    "N_gr": config['N_gr'],
    "d_r_max": config['d_r_max'],
    "f_max": config['f_max'],
    "noise_level": config['noise_level'],
    "nfake": config['nfake']
}

# 保存训练参数日志
with open(config['path_to_the_main_dir'] + "/TD3_Agent_train_arguments.txt", 'w') as f:
    f.write(str(train_args))

print("Start Training...")
TD3_Agent.train(**train_args)
print("Training Finished!")