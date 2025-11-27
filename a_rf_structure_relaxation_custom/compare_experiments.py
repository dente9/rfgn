import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# ================= 配置区域 =================
# 输入两个实验的 outputs 文件夹路径
# 实验 A (基准路径，例如你当前的 outputs/alfe)
EXP_1_PATH = 'outputs/alfe'
# 实验 B (未来的路径，例如 outputs/alfe_with_desc)
EXP_2_PATH = 'outputs/alfe_new'

# 给它们起个名字，用于图例显示
EXP_1_NAME = 'Baseline (No Desc)'
EXP_2_NAME = 'With Descriptors'

# 平滑窗口 (用于 Step 数据，如 Force/Loss)
STEP_SMOOTH = 100
# 平滑窗口 (用于 Episode 数据，如 Reward/Length)
EP_SMOOTH = 20
# ===========================================

def load_experiment_data(exp_path):
    """读取所有三种日志文件：Step, Train(Episode), Test"""
    if not os.path.exists(exp_path):
        print(f"Warning: Path not found: {exp_path}")
        return None

    data = {}

    # 1. 读取 Step Log (steps_log.csv) - 微观步级数据
    step_log_path = os.path.join(exp_path, 'logs', 'steps_log.csv')
    if os.path.exists(step_log_path):
        try:
            df = pd.read_csv(step_log_path)
            data['step_log'] = df.sort_values('Total_Step')
        except Exception as e:
            print(f"Error reading steps_log: {e}")
            data['step_log'] = None
    else:
        data['step_log'] = None

    # 2. 读取 Train Log (df_..._train_...csv) - 宏观回合级数据
    train_files = glob.glob(os.path.join(exp_path, 'data', '*_train_*.csv'))
    if train_files:
        train_file = sorted(train_files)[-1] # 取最新的一个
        try:
            df = pd.read_csv(train_file)
            # 估算每个 Episode 结束时的 Total Step，以便与微观数据对齐 X 轴
            df['Estimated_Total_Step'] = df['Last_step_train'].cumsum()
            data['train_log'] = df
        except Exception as e:
            print(f"Error reading train csv: {e}")
            data['train_log'] = None
    else:
        data['train_log'] = None

    # 3. 读取 Test Log (df_..._test_...csv) - 验证集数据
    test_files = glob.glob(os.path.join(exp_path, 'data', '*_test_*.csv'))
    if test_files:
        test_file = sorted(test_files)[-1]
        try:
            data['test_log'] = pd.read_csv(test_file)
        except Exception as e:
            print(f"Error reading test csv: {e}")
            data['test_log'] = None
    else:
        data['test_log'] = None

    return data

def plot_comparison(data1, data2, name1, name2):
    if data1 is None and data2 is None:
        print("No data found for either experiment.")
        return

    # 设置绘图风格
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid') # 兼容旧版 matplotlib

    # 【修改点】：使用 constrained_layout=True 自动修复标题重叠
    fig, axes = plt.subplots(4, 2, figsize=(18, 18), constrained_layout=True)
    fig.suptitle(f'Comprehensive Comparison: {name1} vs {name2}', fontsize=18, weight='bold')

    # ============ ROW 1: Training Efficiency (Step Level) ============

    # 1. Max Force (Log Scale)
    ax = axes[0, 0]
    for d, name in [(data1, name1), (data2, name2)]:
        if d and d['step_log'] is not None:
            df = d['step_log']
            ax.plot(df['Total_Step'], df['Max_Force'].rolling(STEP_SMOOTH).mean(), label=name, alpha=0.8)
    ax.set_title(f'Training: Max Force (Moving Avg {STEP_SMOOTH})', fontsize=12)
    ax.set_ylabel('Force (eV/A)')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="-", alpha=0.3)

    # 2. Episode Length (From df_train)
    ax = axes[0, 1]
    for d, name in [(data1, name1), (data2, name2)]:
        if d and d['train_log'] is not None:
            df = d['train_log']
            x_axis = df.get('Estimated_Total_Step', df.index)
            ax.plot(x_axis, df['Last_step_train'].rolling(EP_SMOOTH).mean(), label=name, alpha=0.8)
    ax.set_title(f'Training: Steps per Episode (Moving Avg {EP_SMOOTH})', fontsize=12)
    ax.set_ylabel('Steps (Lower is Better)')
    ax.legend(loc='upper right')

    # ============ ROW 2: Model Stability (Losses) ============

    # 3. Critic Loss
    ax = axes[1, 0]
    for d, name in [(data1, name1), (data2, name2)]:
        if d and d['step_log'] is not None:
            df = d['step_log'].dropna(subset=['Loss_Q'])
            ax.plot(df['Total_Step'], df['Loss_Q'].rolling(STEP_SMOOTH).mean(), label=name, alpha=0.8)
    ax.set_title('Training: Critic Loss (Q)', fontsize=12)
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')

    # 4. Actor Loss
    ax = axes[1, 1]
    for d, name in [(data1, name1), (data2, name2)]:
        if d and d['step_log'] is not None:
            df = d['step_log'].dropna(subset=['Loss_Pi'])
            ax.plot(df['Total_Step'], df['Loss_Pi'].rolling(STEP_SMOOTH).mean(), label=name, alpha=0.8)
    ax.set_title('Training: Actor Loss (Pi)', fontsize=12)
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')

    # ============ ROW 3: Test Performance ============

    # 5. Test Max Force
    ax = axes[2, 0]
    for d, name in [(data1, name1), (data2, name2)]:
        if d and d['test_log'] is not None:
            df = d['test_log']
            ax.plot(df.index, df['Maximum_force'], 'o-', label=name, markersize=5, linewidth=2)
    ax.set_title('Test: Max Force (Validation)', fontsize=12)
    ax.set_yscale('log')
    ax.set_ylabel('Force (eV/A)')
    ax.set_xlabel('Test Iterations')
    ax.legend(loc='upper right')

    # 6. Test Efficiency
    ax = axes[2, 1]
    for d, name in [(data1, name1), (data2, name2)]:
        if d and d['test_log'] is not None:
            df = d['test_log']
            ax.plot(df.index, df['Last_step'], 'o-', label=name, markersize=5, linewidth=2)
    ax.set_title('Test: Steps to Relax (Validation)', fontsize=12)
    ax.set_ylabel('Steps')
    ax.set_xlabel('Test Iterations')
    ax.legend(loc='upper right')

    # ============ ROW 4: Overall Success (From df_train) ============

    # 7. Total Reward per Episode
    ax = axes[3, 0]
    for d, name in [(data1, name1), (data2, name2)]:
        if d and d['train_log'] is not None:
            df = d['train_log']
            x_axis = df.get('Estimated_Total_Step', df.index)
            ax.plot(x_axis, df['Total_reward'].rolling(EP_SMOOTH).mean(), label=name, alpha=0.8)
    ax.set_title(f'Training: Total Reward per Episode (Moving Avg {EP_SMOOTH})', fontsize=12)
    ax.set_ylabel('Reward')
    ax.set_xlabel('Total Training Steps')
    ax.legend(loc='lower right')

    # 8. Success Rate (Stop Label Analysis)
    ax = axes[3, 1]
    for d, name in [(data1, name1), (data2, name2)]:
        if d and d['train_log'] is not None:
            df = d['train_log']
            x_axis = df.get('Estimated_Total_Step', df.index)
            # Stop_label=True 表示失败/异常停止。 Success = Not Stop.
            success_rate = (~df['Stop_label_train'].astype(bool)).rolling(EP_SMOOTH).mean()
            ax.plot(x_axis, success_rate, label=name, alpha=0.8, linewidth=2)
    ax.set_title(f'Training: Success Rate (Moving Avg {EP_SMOOTH})', fontsize=12)
    ax.set_ylabel('Success Rate (0~1)')
    ax.set_xlabel('Total Training Steps')
    ax.legend(loc='lower right')
    ax.set_ylim(-0.05, 1.05)

    # 不再需要 plt.tight_layout()，因为 constrained_layout=True 已经处理了
    print("Plotting finished. Showing results...")
    plt.show()

if __name__ == "__main__":
    print(f"Loading experiment 1: {EXP_1_PATH} ...")
    d1 = load_experiment_data(EXP_1_PATH)

    print(f"Loading experiment 2: {EXP_2_PATH} ...")
    d2 = load_experiment_data(EXP_2_PATH)

    plot_comparison(d1, d2, EXP_1_NAME, EXP_2_NAME)