import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 引入项目模块
from utils.convert_to_graph_e3nn import to_graph
from utils.model_e3nn import PeriodicNetwork_Pi, PeriodicNetwork_Q
from utils.td3 import TD3Agent
from utils.env import Environment, get_sturct_lib_and_calcs_gen

# ==============================================================================
# 【用户配置区域】 (请修改这里的路径)
# ==============================================================================
# 1. 结构文件路径
STRUCT_FILE = "structures/AlFe.csv"

# 2. 模型权重文件路径 (.pth)
# 请修改为你想要测试的具体模型路径，例如: "outputs/alfe_experiment/models/td3_checkpoint_AlFe_cubic_1000"
# 如果不知道具体名字，填文件夹路径，我会自动找最新的
MODEL_PATH = "outputs/alfe_split_experiment/models"

# 3. 测试目标结构索引 (0是第一个，1是第二个)
TARGET_STRUCT_INDEX = 0

# 4. 测试次数 (对同一个结构进行多少次随机扰动测试)
NUM_TESTS = 5

# 5. 最大步数
MAX_STEPS = 100
# ==============================================================================

# 全局设置
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_latest_model(path):
    if os.path.isfile(path):
        return path
    files = glob.glob(os.path.join(path, "td3_checkpoint_*"))
    if not files:
        raise FileNotFoundError(f"No model found in {path}")
    # 按修改时间排序找最新的
    latest_file = max(files, key=os.path.getmtime)
    print(f"[System] Auto-selected latest model: {latest_file}")
    return latest_file

def main():
    print(f"--- Starting Inference Test on Structure #{TARGET_STRUCT_INDEX} ---")

    # 1. 加载数据 (只加载目标结构)
    # interval=[1, 2] 表示只取索引为 1 的那个结构
    print(f"Loading structure index {TARGET_STRUCT_INDEX} from {STRUCT_FILE}...")
    s_lib, calcs = get_sturct_lib_and_calcs_gen(STRUCT_FILE, interval=[TARGET_STRUCT_INDEX, TARGET_STRUCT_INDEX+1])

    if len(s_lib) == 0:
        print("Error: Structure not found. Check index and file.")
        return

    print(f"Structure loaded: {s_lib[0].formula} (Sites: {len(s_lib[0])})")

    # 2. 准备环境参数 (与 main.py 保持一致)
    # 注意：这里的参数必须和你训练时的一致，特别是神经网络参数
    em_dim = 10
    config_kwargs = {
        "structures_file": STRUCT_FILE,
        "env_name": "Inference_Env",
        "r_weights": [1, 1, 1],
        "eps": 0.01,
        "r0": 1.5,
        "stop_numb": 50,
        "em_dim": em_dim,
        "r_max": 5.0,
        "num_neighbors": 25,
        "noise_clip": 0.1,
    }

    env_kwargs = {
        "input_struct_lib": s_lib,
        "calculator_lib": calcs,
        "convert_to_graph_func": to_graph,
        "r0": config_kwargs['r0'],
        "eps": config_kwargs['eps'],
        "reward_func": "force",
        "r_weights": config_kwargs['r_weights'],
        "stop_numb": config_kwargs['stop_numb']
    }

    # 3. 构建网络 (必须与训练时结构一致)
    input_irreps = f"12x0e + 1x1o + {em_dim}x0e"
    # 注意：utils/td3.py 会自动根据 USE_ENV_DESCRIPTOR 修改 irreps，所以这里传原始的即可

    actor_feat = {
        "em_dim": em_dim,
        "irreps_in": input_irreps,
        "irreps_out": "1x1o",
        "noise_clip": 0.1,
        "irreps_node_attr": "0e",
        "layers": 2, "mul": 20, "lmax": 2,
        "max_radius": 5.0, "num_neighbors": 25,
        "expl_mode": 'state', "reduce_output": False
    }

    critic_input_irreps = f"{input_irreps} + 1x1o + {em_dim}x0e"
    critic_feat = {
        "em_dim": em_dim,
        "irreps_in": critic_input_irreps,
        "irreps_out": "1x0e", "irreps_node_attr": "0e",
        "layers": 2, "mul": 20, "lmax": 2,
        "max_radius": 5.0, "num_neighbors": 25,
        "reduce_output": True
    }

    ac_kwargs = {
        "net_actor": PeriodicNetwork_Pi,
        "net_critic": PeriodicNetwork_Q,
        "actor_feat": actor_feat,
        "critic_feat": critic_feat
    }

    # 4. 初始化 Agent 并加载权重
    agent = TD3Agent(
        env_fn=Environment,
        env_kwards=env_kwargs,
        ac_kwargs=ac_kwargs,
        batch_size=1, # 推理时不重要
        with_weights=False
    )

    # 加载模型
    final_model_path = load_latest_model(MODEL_PATH)
    agent.load_model(final_model_path)
    print("Model loaded successfully.")

    # 5. 开始测试循环
    results = [] # 存储每次测试的 [force_trajectory]

    plt.figure(figsize=(10, 6))
    plt.style.use('default') # 使用默认样式防止报错

    print(f"\nRunning {NUM_TESTS} tests on structure...")

    for i in range(NUM_TESTS):
        # Reset 环境 (加入随机扰动 trans_coef=0.5)
        # 我们直接调用 agent.env，不需要 test_env，因为这里只加载了一个结构
        # 设置不同的随机种子以产生不同的初始扰动
        np.random.seed(i + 100)

        o_raw, d, ep_ret, _, ep_len = agent.env.reset(trans_coef=0.5, num=0, correct=False), False, 0, 0, 0

        # 手动 Augment (如果开启了描述符)
        if hasattr(agent, '_augment_state'):
            o = agent._augment_state(o_raw)
        else:
            o = o_raw

        forces = []
        initial_f = o_raw.forces.norm(dim=1).max().item() if hasattr(o_raw, 'forces') else 0
        forces.append(initial_f)

        print(f"Test {i+1}: Initial Max Force = {initial_f:.4f} eV/A")

        # 逐步松弛
        done = False
        step_count = 0
        while not done and step_count < MAX_STEPS:
            # 获取动作 (无噪声)
            action = agent.get_action(o, noise_scale=None)

            # 环境步进
            o_raw, r, d, _, f, s = agent.env.step(action)

            # Augment
            if hasattr(agent, '_augment_state'):
                o = agent._augment_state(o_raw)
            else:
                o = o_raw

            forces.append(f)
            step_count += 1

            if d or s:
                done = True

        final_f = forces[-1]
        status = "Converged" if final_f < 0.05 else "Stopped/Limit"
        print(f"   -> Finished in {step_count} steps. Final Force: {final_f:.4f} ({status})")

        # 绘制曲线
        plt.plot(range(len(forces)), forces, 'o-', label=f'Run {i+1}', alpha=0.7, markersize=3)

    # 6. 设置图片格式并显示
    plt.axhline(y=0.01, color='r', linestyle='--', label='Target (0.01)')
    plt.yscale('log') # 对数坐标看力更清晰
    plt.xlabel('Steps')
    plt.ylabel('Max Force (eV/A)')
    plt.title(f'Relaxation Trajectories: Structure Index {TARGET_STRUCT_INDEX}')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()

    print("\nDisplaying plot...")
    plt.show()

if __name__ == "__main__":
    main()