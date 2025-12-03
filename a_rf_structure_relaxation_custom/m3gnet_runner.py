import subprocess
import os
import sys

def run_m3gnet_training():
    """
    封装了 TD3_M3GNet 的启动命令。
    参数已严格对齐 main.py (e3nn版本) 的配置。
    """

    # 实验名称
    exp_name = "test_m3gnet_aligned_v1"

    # 定义所有参数
    args = {
        # --- 路径与文件 ---
        "--path_to_the_main_dir": f"outputs/{exp_name}",
        "--structures_file": "structures/AlFe.csv",
        "--interval": "[1, 2]",

        # --- 环境参数 ---
        "--env_name": "AlFe_cubic",
        "--reward_func": "hybrid",
        "--r_weights": "[0.5, 2, 1]",
        "--eps": "0.01",
        "--r0": "1.5",
        "--stop_numb": "50",
        "--trans_coef": "0.5",
        "--with_weights": "False",

        # --- 训练循环控制 ---
        "--train_ep": "[800, 1000]",
        "--test_ep": "[10, 100]",
        "--test_every": "500",
        "--save_every": "1000",
        "--save_result": "True",
        "--test_random": "False",
        "--start_iter": "0",

        # --- Agent 训练参数 ---
        "--random_seed": "5599",
        "--replay_size": "1000000",
        "--gamma": "0.9999",
        "--polyak": "0.995",
        "--pi_lr": "1e-05", # 可尝试稍微调大，如 1e-4，因为M3GNet参数更多
        "--q_lr": "1e-05",
        "--batch_size": "100",
        "--start_steps": "0",
        "--update_after": "0",
        "--update_every": "1",
        "--target_noise": "0.05",
        "--policy_delay": "2",
        "--noise": "[0.2, 0.2]",
        "--noise_clip": "0.1",

        # --- 贪婪探索与辅助参数 ---
        "--N_gr": "1000000",
        "--d_r_max": "0.0",
        "--f_max": "0.001",
        "--noise_level": "29.0",
        "--nfake": "10",

        # --- M3GNet 网络参数 (Dimension Config) ---
        "--num_node_features": "12",
        # [64, 50, 64, 64] -> EmbeddingDim=64, RBF_bins=50, Hidden=64, OutHidden=64
        "--n_l_a": "[64, 50, 64, 64]",
        "--n_l_c": "[64, 50, 64, 64, 64]",
    }

    # 构建命令列表
    cmd = [sys.executable, "-u", "TD3_M3GNet.py"]
    for key, value in args.items():
        cmd.append(key)
        cmd.append(value)

    print(f"Starting M3GNet Training: {exp_name}...")
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
    if not os.path.exists("TD3_M3GNet.py"):
        print("Error: TD3_M3GNet.py not found. Please create it first.")
    else:
        run_m3gnet_training()