import subprocess
import os
import sys

def run_cgcnn_training():
    """
    封装了 TD3_CGCNN 的启动命令。
    参数已严格对齐 main.py (e3nn版本) 的配置，以便进行公平的性能对比。
    """

    # 实验名称 (建议修改以区分不同实验)
    exp_name = "test_cgcnn_aligned_v1"

    # 定义所有参数，已根据 main.py 进行了数值对齐
    args = {
        # --- 路径与文件 ---
        "--path_to_the_main_dir": f"outputs/{exp_name}",
        "--structures_file": "structures/AlFe.csv",
        "--interval": "[1, 2]", # 对齐 main.py: train_interval=[1, 2]

        # --- 环境参数 ---
        "--env_name": "AlFe_cubic",
        "--reward_func": "hybrid", # 对齐 main.py: default="hybrid"
        "--r_weights": "[0.5, 2, 1]", # 对齐 main.py
        "--eps": "0.01",
        "--r0": "1.5",
        "--stop_numb": "50", # 对齐 main.py: 50 (原runner是1e10)
        "--trans_coef": "0.5", # 对齐 main.py: 0.5 (原runner是0.05)
        "--with_weights": "False",

        # --- 训练循环控制 ---
        "--train_ep": "[800, 1000]", # 对齐 main.py
        "--test_ep": "[10, 100]", # 对齐 main.py
        "--test_every": "500", # 对齐 main.py
        "--save_every": "1000",
        "--save_result": "True",
        "--test_random": "False",
        "--start_iter": "0",

        # --- Agent 训练参数 ---
        "--random_seed": "5599", # 对齐 main.py
        "--replay_size": "1000000", # 1e6
        "--gamma": "0.9999", # 对齐 main.py: 0.9999 (原runner是0.9)
        "--polyak": "0.995",
        "--pi_lr": "1e-05",
        "--q_lr": "1e-05",
        "--batch_size": "100",
        "--start_steps": "0", # 对齐 main.py: 0 (原runner是500)
        "--update_after": "0", # 对齐 main.py: 0 (原runner是100)
        "--update_every": "1",
        "--target_noise": "0.05",
        "--policy_delay": "2",
        "--noise": "[0.2, 0.2]", # 对齐 main.py: [0.2, 0.2] (原runner是[0.1, 0.01])
        "--noise_clip": "0.1",

        # --- 贪婪探索与辅助参数 ---
        "--N_gr": "1000000", # 对齐 main.py: 1e6
        "--d_r_max": "0.0",
        "--f_max": "0.001",
        "--noise_level": "29.0",
        "--nfake": "10", # 对齐 main.py: 10 (原runner是1e10)

        # --- CGCNN 特定网络参数 (保持默认或根据需要微调) ---
        "--num_node_features": "12",
        "--aggr_a": "mean",
        "--n_l_a": "[30,50,120,60]",
        "--aggr_c": "mean",
        "--n_l_c": "[30,50,120,20,60]",
    }

    # 构建命令列表
    # 添加 "-u" 参数强制 Python 使用无缓冲 (unbuffered) 输出
    cmd = [sys.executable, "-u", "TD3_CGCNN.py"]
    for key, value in args.items():
        cmd.append(key)
        cmd.append(value)

    print(f"Starting CGCNN Training (Aligned with main.py): {exp_name}...")
    print("="*60)
    print(" ".join(cmd))
    print("="*60)

    # 使用 Popen 实时获取输出
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )

        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()

        return_code = process.wait()

        if return_code != 0:
            print(f"\nTraining failed with exit code {return_code}")
        else:
            print("\nTraining finished successfully.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        process.terminate()
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    if not os.path.exists("TD3_CGCNN.py"):
        print("Error: TD3_CGCNN.py not found in current directory.")
    else:
        run_cgcnn_training()