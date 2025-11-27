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

# 2. 配置参数
# ------------------------------------------------------------------
config = {
    # --- 路径与文件 ---
    "path_to_the_main_dir": "outputs/alfe",
    "structures_file": "structures/AlFe.csv",
    "interval": [1, 2],
    "env_name": "AlFe_cubic",

    # --- 环境参数 ---
    "reward_func": "force",
    "r_weights": [1, 1, 1],
    "eps": 0.01,
    "r0": 1.5,
    "stop_numb": 50,
    "trans_coef": 0.5,
    "with_weights": False,

    # --- 神经网络参数 ---
    "em_dim": 10,
    "r_max": 5.0,
    "num_neighbors": 25,
    "noise_clip": 0.1,
    "expl_mode": 'state',

    # Actor 具体参数
    "pi_n_layers": 2,
    "pi_mul": 20,
    "pi_lmax": 2,

    # Critic 具体参数
    "q_n_layers": 2,
    "q_mul": 20,
    "q_lmax": 2,

    # --- Agent 训练参数 ---
    "random_seed": 5599,
    "replay_size": int(1e6),
    "gamma": 0.9999,
    "polyak": 0.995,
    "pi_lr": 1e-05,
    "q_lr": 1e-05,
    "batch_size": 100,
    "start_steps": 0,
    "update_after": 0,
    "update_every": 1,
    "target_noise": 0.05,
    "policy_delay": 2,
    "noise": [0.2, 0.2],

    # --- 训练循环控制 ---
    "train_ep": [800, 1000],
    "test_ep": [10, 100],
    "save_result": True,
    "test_random": False,
    # 【重点修改】：将测试频率从 1000 改为 200，这样你的 Test 图更新会变快
    "test_every": 200,
    "save_every": 1000,
    "start_iter": 0,

    # --- 贪婪探索与辅助参数 ---
    "N_gr": int(1e6),
    "d_r_max": 0.0,
    "f_max": 0.001,
    "noise_level": 29.0,
    "nfake": 10,
    "path_load": None,
    "path_weights": None,
    "e_lim": None,
    "net_lim": None
}

# 3. 数据加载与环境准备
# ------------------------------------------------------------------
print(f"Loading structures from {config['structures_file']}...")
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

# 4. 构建网络特征字典
# ------------------------------------------------------------------
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

# 5. 目录检查
# ------------------------------------------------------------------
if not os.path.exists(config['path_to_the_main_dir']):
    os.makedirs(config['path_to_the_main_dir'])

wf = None
if config['with_weights'] and config['path_weights'] is not None:
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

if config['path_load'] == 'last':
    path_checkpoints = config['path_to_the_main_dir'] + "/models"
    if os.path.exists(path_checkpoints):
        config['path_load'] = get_the_last_checkpoint(path_checkpoints)
        config['start_steps'] = extract_number(config['path_load'])
        print(f"Resuming from step: {config['start_steps']}")

if config['path_load'] is not None and os.path.exists(config['path_load']):
    TD3_Agent.load_model(config['path_load'])

# 7. 开始训练
# ------------------------------------------------------------------
if config['start_iter'] == 0:
    with open(config['path_to_the_main_dir'] + "/TD3_Agent_arguments.txt", 'w') as f:
        f.write(str(agent_params))

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
    "nfake": config['nfake'],
    "plot_every": 50
}

with open(config['path_to_the_main_dir'] + "/TD3_Agent_train_arguments.txt", 'w') as f:
    f.write(str(train_args))

print("Start Training...")
TD3_Agent.train(**train_args)
print("Training Finished!")