import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import sys
import numpy as np

def get_unique_labels(paths):
    """生成唯一的图例标签"""
    labels = []
    base_names = [os.path.basename(p) for p in paths]
    for i, path in enumerate(paths):
        name = base_names[i]
        if base_names.count(name) > 1:
            parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
            labels.append(f"{parent}/{name}")
        else:
            labels.append(name)
    return labels

def find_stable_step(df, threshold, window_size=100, fluctuation=0.1):
    """
    寻找首次稳定收敛点
    """
    candidates = df.index[df['Max_Force'] <= threshold].tolist()
    upper_bound = threshold + fluctuation
    max_step_idx = df.index.max()

    for idx in candidates:
        end_idx = min(idx + window_size, max_step_idx)
        future_window = df.loc[idx : end_idx, 'Max_Force']
        if future_window.max() <= upper_bound:
            step = df.loc[idx, 'Total_Step']
            val = df.loc[idx, 'Max_Force']
            return int(step), val
    return None, None

def compare_and_plot(csv_paths):
    # ================= 配置区域 =================
    thresholds = [1.0, 0.03, 0.01]
    fluctuation = 0.1
    stable_window = 100
    # ===========================================

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # 颜色和标记配置
    thresh_colors = ['gray', 'orange', 'red', 'purple', 'green']
    markers = ['o', 's', '*', 'D', '^']

    prop_cycle = plt.rcParams['axes.prop_cycle']
    curve_colors = prop_cycle.by_key()['color']

    exp_labels = get_unique_labels(csv_paths)

    print(f"\n{'='*100}")
    print(f"分析标准: 首次 <= 阈值，且后续 {stable_window} 步内波动不超过 +{fluctuation}")
    print(f"{'-'*100}")
    print(f"{'Experiment':<40} | {'Threshold':<10} | {'Stable Step':<12} | {'Force':<10} | {'Status'}")
    print(f"{'-'*100}")

    has_valid_data = False

    for idx, path in enumerate(csv_paths):
        if not os.path.exists(path):
            continue

        try:
            df = pd.read_csv(path)
            if 'Total_Step' not in df.columns or 'Max_Force' not in df.columns:
                continue

            has_valid_data = True
            exp_label = exp_labels[idx]
            line_color = curve_colors[idx % len(curve_colors)]

            # 1. 绘制实验曲线
            ax.plot(df['Total_Step'], df['Max_Force'],
                    label=exp_label, color=line_color, alpha=0.6, linewidth=1.5)

            # 2. 寻找并标记稳定点
            for t_idx, thresh in enumerate(thresholds):
                step_val, force_val = find_stable_step(df, thresh, window_size=stable_window, fluctuation=fluctuation)

                if step_val is not None:
                    print(f"{exp_label:<40} | {thresh:<10} | {step_val:<12} | {force_val:.5f}    | Converged")

                    # 动态选择标记样式
                    marker_style = markers[t_idx % len(markers)]
                    marker_size = 10 if marker_style != '*' else 16

                    # 画点
                    ax.plot(step_val, force_val, marker=marker_style, markersize=marker_size,
                            color=line_color, markeredgecolor='white', markeredgewidth=1.5, zorder=10)

                    # 【修改】现在对所有阈值点都添加文字标注
                    # 为了防止文字重叠，我们可以根据阈值的不同，稍微错开一点文字的位置
                    # xytext: (x偏移, y偏移)
                    offset_y = 10 if t_idx % 2 == 0 else -15

                    ax.annotate(f"{step_val}", xy=(step_val, force_val),
                                xytext=(5, offset_y), textcoords="offset points",
                                color=line_color, fontweight='bold', fontsize=9,
                                arrowprops=dict(arrowstyle="-", color=line_color, alpha=0.3))

                else:
                    print(f"{exp_label:<40} | {thresh:<10} | {'-':<12} | {'-':<10} | Unstable/Not Met")

        except Exception as e:
            print(f"[Error] {path}: {e}")

    print(f"{'='*100}\n")

    if not has_valid_data:
        return

    # ==============================================================================
    # 绘制阈值线
    # ==============================================================================
    for i, thresh in enumerate(thresholds):
        c = thresh_colors[i % len(thresh_colors)]
        ax.axhline(y=thresh, color=c, linestyle='--', linewidth=1.2, alpha=0.7, label=f'Threshold {thresh}')
        ax.text(0, thresh, f' {thresh}', color=c, va='bottom', fontweight='bold', fontsize=9)

    ax.set_title(f'Robust Convergence Analysis (Window={stable_window}, Fluctuation=+{fluctuation})', fontsize=16)
    ax.set_xlabel('Total Steps', fontsize=12)
    ax.set_ylabel('Max Force (eV/Å)', fontsize=12)
    ax.set_yscale('log')

    ax.legend(loc='upper right', frameon=True, framealpha=0.9, shadow=True)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', help='CSV file paths')

    if len(sys.argv) < 2:
        print("Usage: python plot_log.py file1.csv file2.csv")
        sys.exit(1)

    args = parser.parse_args()
    compare_and_plot(args.paths)