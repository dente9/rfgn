import torch
from copy import deepcopy
import numpy as np
from torch.optim import Adam
from utils.replay_memory import ReplayMemory
import itertools
from utils.utils import create_plots, aver_list
import os
from torch_geometric.data import Batch
import pandas as pd
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import csv
import re
import time
import matplotlib.pyplot as plt  # 【新增】用于在 train 中自定义绘图
import sys

# ==============================================================================
# 【配置区域】
# ==============================================================================
USE_ENV_DESCRIPTOR = True
DESC_DIM = 8
# ==============================================================================


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, n_gaussians=8):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = (offset[1] - offset[0]) * torch.ones_like(offset)
        self.register_buffer("offset", offset)
        self.register_buffer("widths", widths)

    def forward(self, dist):
        coeff = -0.5 / (self.widths ** 2)
        diff = dist.unsqueeze(-1) - self.offset
        y = torch.exp(coeff * torch.pow(diff, 2))
        return y

class Agent(nn.Module):
    r"""The class of TD3 Agent."""
    def __init__(self, net_actor, net_critic, actor_feat, critic_feat):
        super().__init__()
        self.device = torch.device("cpu" if sys.platform.startswith('win') else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q1 = net_critic(**critic_feat).to(self.device)
        self.q2 = net_critic(**critic_feat).to(self.device)
        self.pi = net_actor(**actor_feat).to(self.device)

    def act(self, o, noise_scale):
        with torch.no_grad():
             return self.pi(data = o.to(self.device), noise_scale = noise_scale)

class TD3Agent:
    r"""The class of TD3 Agent for structure relaxation process."""
    def __init__(self,
                 env_fn,
                 env_kwards = dict(),
                 test_env_kwards = None,
                 ac_kwargs = dict(),
                 seed=0,
                 replay_size=int(1e6),
                 gamma=0.99,
                 polyak=0.995,
                 pi_lr=1e-3,
                 q_lr=1e-3,
                 batch_size=100,
                 start_steps=10000,
                 update_after=1000,
                 update_every=50,
                 target_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 trans_coef = 0.1,
                 noise = [0.01, 0.001],
                 init_rewards_for_weights = None,
                 with_weights = False):

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device("cpu" if sys.platform.startswith('win') else ("cuda" if torch.cuda.is_available() else "cpu"))

        # --- Descriptor ---
        if USE_ENV_DESCRIPTOR:
            print(f"\n[Info] Environment Descriptor ENABLED. Adding {DESC_DIM} radial features.")

            self.r_max = ac_kwargs['actor_feat'].get('max_radius', 5.0)
            self.smearing = GaussianSmearing(start=0.0, stop=self.r_max, n_gaussians=DESC_DIM).to(self.device)

            def patch_irreps(irreps_str):
                match = re.search(r'(\d+)x0e', irreps_str)
                if match:
                    old_dim = int(match.group(1))
                    new_dim = old_dim + DESC_DIM
                    new_str = irreps_str.replace(f"{old_dim}x0e", f"{new_dim}x0e", 1)
                    return new_str
                else:
                    return f"{irreps_str} + {DESC_DIM}x0e"

            if 'actor_feat' in ac_kwargs:
                old_in = ac_kwargs['actor_feat']['irreps_in']
                ac_kwargs['actor_feat']['irreps_in'] = patch_irreps(old_in)
                print(f" -> Actor Input changed: {old_in}  ==>  {ac_kwargs['actor_feat']['irreps_in']}")

            if 'critic_feat' in ac_kwargs:
                old_in = ac_kwargs['critic_feat']['irreps_in']
                ac_kwargs['critic_feat']['irreps_in'] = patch_irreps(old_in)
                print(f" -> Critic Input changed: {old_in}  ==>  {ac_kwargs['critic_feat']['irreps_in']}")
        else:
            print("\n[Info] Environment Descriptor DISABLED.")
            self.smearing = None
        # ------------------

        self.env = env_fn(**env_kwards)

        if test_env_kwards is not None:
            print("[Info] Initializing Separate Test Environment...")
            self.test_env = env_fn(**test_env_kwards)
        else:
            self.test_env = self.env

        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_delay  = policy_delay
        self.polyak = polyak
        self.trans_coef = trans_coef
        self.noise = noise
        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.start_steps = start_steps
        self.test_labels = []

        self.ac = Agent(**ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)
        self.memory = ReplayMemory(buffer_capacity=replay_size, batch_size = batch_size)

        self.with_weights = False if len(env_kwards["input_struct_lib"]) == 1 else with_weights
        if self.with_weights:
            if init_rewards_for_weights is not None:
                self.rewards_for_weights = np.array(init_rewards_for_weights)
            else:
                self.rewards_for_weights = []
                for i in range(len(env_kwards["input_struct_lib"])):
                    o, _, _, _ = self.env.reset(self.trans_coef, i), False, 0, 0
                    if USE_ENV_DESCRIPTOR: o = self._augment_state(o)
                    _, _, _, _, f, _ = self.env.step(self.get_action(o, 0), 0)
                    self.rewards_for_weights.append(f)
                self.rewards_for_weights = np.array(self.rewards_for_weights)
            self.env.weights = self.rewards_for_weights/self.rewards_for_weights.sum()
        else:
            L = len(env_kwards["input_struct_lib"])
            self.env.weights = np.ones(L)/L

    def _augment_state(self, o):
        if not USE_ENV_DESCRIPTOR:
            return o

        dist = None
        if hasattr(o, 'edge_vec'):
            dist = o.edge_vec.norm(dim=-1)
        elif hasattr(o, 'edge_len'):
            dist = o.edge_len
        elif hasattr(o, 'edge_length'):
            dist = o.edge_length
        else:
            row, col = o.edge_index
            dist = (o.pos[row] - o.pos[col]).norm(dim=-1)

        if dist is None:
            return o

        edge_features = self.smearing(dist.to(self.device))
        row, col = o.edge_index
        node_descriptor = torch.zeros(o.num_nodes, DESC_DIM, device=self.device, dtype=o.x.dtype)
        node_descriptor.index_add_(0, row.to(self.device), edge_features)

        deg = torch.zeros(o.num_nodes, 1, device=self.device, dtype=o.x.dtype)
        ones = torch.ones(row.shape[0], 1, device=self.device, dtype=o.x.dtype)
        deg.index_add_(0, row.to(self.device), ones)
        deg = deg.clamp(min=1.0)
        node_descriptor = node_descriptor / deg

        o.x = torch.cat([o.x.to(self.device), node_descriptor], dim=1)
        return o

    def compute_loss_q(self, batch):
        device = self.device
        o =  Batch.from_data_list(batch["state"].tolist()).to(device)
        o2 = Batch.from_data_list(batch["next_state"].tolist()).to(device)
        a = Batch.from_data_list(batch["action"].tolist()).to(device)
        r = torch.FloatTensor(batch["reward"]).to(device)
        d = torch.FloatTensor(batch["done"]).to(device)

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        with torch.no_grad():
            pi_targ = self.ac_targ.pi(data = o2, noise_scale = self.target_noise)
            q1_pi_targ = self.ac_targ.q1(o2, pi_targ)
            q2_pi_targ = self.ac_targ.q2(o2, pi_targ)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q

    def compute_loss_pi(self, batch):
        device = self.device
        o = Batch.from_data_list(batch["state"].tolist()).to(device)
        a_pr = self.ac.pi(o)
        q2_pi = self.ac.q2(o, a_pr)
        return -q2_pi.mean()

    def update(self, data, timer):
        return_dict = {"loss_q": None, "loss_pi": None}
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        return_dict["loss_q"] = loss_q.detach().cpu().item()

        if (timer+1) % self.policy_delay == 0:
            for p in self.q_params:
                p.requires_grad = False
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()
            return_dict["loss_pi"] = loss_pi.detach().cpu().item()
            for p in self.q_params:
                p.requires_grad = True
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
        return return_dict

    def get_action(self, o, noise_scale):
        a = self.ac.act(o = o, noise_scale = noise_scale).detach().to('cpu')
        return a

    def update_weights(self, r_new, num):
        self.rewards_for_weights[num] = r_new
        self.env.weights = self.rewards_for_weights/self.rewards_for_weights.sum()

    def test_agent(self, num_test_episodes, max_test_steps, test_random = False, separate = False):
        # 1. 确保环境已初始化
        if self.test_env.current_ase_structure is None:
            self.test_env.reset(0.0)

        prev_state = self.test_env.current_ase_structure.copy()
        prev_calc = self.test_env.current_ase_structure.calc
        prev_num = self.test_env.num

        # 2. 初始化容器
        self.test_labels = []
        L = 1 if test_random else len(self.test_env.input_lib.keys())
        N_ep = num_test_episodes * L

        scores = np.zeros(N_ep)
        disc_scores = np.zeros(N_ep)
        last_steps = np.zeros(N_ep)
        forces_last_step = np.zeros(N_ep)

        success_flags = np.zeros(N_ep, dtype=int)
        stop_flags = np.zeros(N_ep, dtype=int)

        # ======================================================================
        # 【修改点1】：分组顺序测试
        # 0,0,0... 然后 1,1,1... 这样数据就是按结构分组的
        # ======================================================================
        global_idx = 0
        struct_indices = [0] if test_random else range(L) # 如果随机模式(L=1)只跑0，否则跑所有

        for num in struct_indices:
            actual_num = None if test_random else num

            for _ in range(num_test_episodes):
                # 保证每个结构的测试具有确定性，但不同次测试扰动不同
                # 使用 global_idx 作为种子偏移量
                np.random.seed(global_idx + 1000) # +1000防止和训练种子重叠

                o_raw, d, ep_ret, ep_disc_ret, ep_len = self.test_env.reset(self.trans_coef, actual_num, correct = False), False, 0, 0, 0
                if USE_ENV_DESCRIPTOR: o = self._augment_state(o_raw)
                else: o = o_raw

                self.test_labels.append(self.test_env.num) # 记录当前 label

                is_stopped = False
                while not(d or (ep_len == max_test_steps)):
                    o_raw, r, d, _, f, s = self.test_env.step(self.get_action(o, None))
                    if USE_ENV_DESCRIPTOR: o = self._augment_state(o_raw)
                    else: o = o_raw

                    ep_ret += r
                    ep_disc_ret += r*(self.gamma**ep_len)
                    ep_len += 1

                    if s:
                        is_stopped = True
                        break

                # 填入数据 (使用 global_idx 定位)
                scores[global_idx] = ep_ret
                last_steps[global_idx] = ep_len
                forces_last_step[global_idx] = f
                disc_scores[global_idx] = ep_disc_ret
                success_flags[global_idx] = 1 if d else 0
                stop_flags[global_idx] = 1 if is_stopped else 0

                global_idx += 1

        # 3. 数据返回
        if separate:
            data_to_save_test = {
                "Score": scores,
                "Last_step": last_steps,
                "Maximum_force": forces_last_step,
                "Disc_score": disc_scores,
                "Success_label": success_flags,
                "Stop_label": stop_flags
            }
        else:
            data_to_save_test = {
                "Score": scores.mean(),
                "Last_step": last_steps.mean(),
                "Maximum_force": forces_last_step.mean(),
                "Disc_score": disc_scores.mean(),
                "Score_std": scores.std(),
                "Last_step_std": last_steps.std(),
                "Maximum_force_std": forces_last_step.std(),
                "Disc_score_std": disc_scores.std(),
                "Test_labels": self.test_labels,
                "Score_med": np.median(scores),
                "Last_step_med": np.median(last_steps),
                "Maximum_force_med": np.median(forces_last_step),
                "Disc_score_med": np.median(disc_scores),
                "Success_Rate": success_flags.mean(),
                "Stop_Rate": stop_flags.mean()
            }

        self.test_env.current_ase_structure = prev_state
        self.test_env.num = prev_num
        self.test_env.current_ase_structure.calc = prev_calc
        return data_to_save_test

    def train(self, train_ep, test_ep, path_to_the_main_dir, env_name, test_every, start_iter = 0, save_every= 1000, e_lim = None, net_lim = None,
              save_result = True, test_random = False, N_gr = 30, d_r_max = 0.015, f_max = 0.1, noise_level = 10, nfake = 10,
              plot_every = 50):

        pi_losses, q_losses, max_force, local_reward, sticks = [], [], [], [], []
        t_total = 0

        df_test = pd.DataFrame(None, columns=['Score', 'Last_step', 'Maximum_force', "Disc_score", 'Score_std', 'Last_step_std', "Maximum_force_std", "Disc_score_std" , "Test_labels", 'Score_med', 'Last_step_med', "Maximum_force_med", "Disc_score_med", "Success_Rate", "Stop_Rate"])
        df_train = pd.DataFrame(None, columns=["Total_reward", "Last_step_train", "Stop_label_train", "Env_name", "Weights"])

        os.makedirs(os.path.join(path_to_the_main_dir, 'data'), exist_ok = True)
        writer = SummaryWriter(log_dir=os.path.join(path_to_the_main_dir, 'logs'))

        csv_log_path = os.path.join(path_to_the_main_dir, 'logs', 'steps_log.csv')
        os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
        with open(csv_log_path, 'w', newline='') as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(['Total_Step', 'Episode', 'Ep_Step', 'Reward', 'Max_Force', 'Loss_Q', 'Loss_Pi'])

        global_start_time = time.time()

        for i in range(train_ep[0]):
            ep_start_time = time.time()

            o_raw, ep_ret, ep_len = self.env.reset(self.trans_coef), 0, 0
            if USE_ENV_DESCRIPTOR: o = self._augment_state(o_raw)
            else: o = o_raw

            max_norm = []
            c_gr = 0

            for t in range(train_ep[1]):
                if t_total >= self.start_steps:
                    if c_gr == N_gr:
                        self.ac.pi.noise_clip  = noise_level*2
                        a = self.get_action(o, noise_level)
                        self.ac.pi.noise_clip = self.noise_clip
                        c_gr = 0
                    else:
                        a = self.get_action(o, ((self.noise[1] - self.noise[0])/train_ep[1]) * t + self.noise[0])

                    o2_raw, r, d, a2, f, s = self.env.step(a)
                    if USE_ENV_DESCRIPTOR: o2 = self._augment_state(o2_raw)
                    else: o2 = o2_raw
                else:
                    o2_raw, r, d, a2, f = self.env.fake_step()
                    if USE_ENV_DESCRIPTOR: o2 = self._augment_state(o2_raw)
                    else: o2 = o2_raw
                    s = False

                print(f"Episode {i+1}/{train_ep[0]} | Step {t+1}/{train_ep[1]} | Total {t_total} | Reward: {ep_ret:.2f} | Force: {f:.4f}")

                t_total +=1
                ep_ret += r
                ep_len += 1
                max_force.append(f)
                local_reward.append(ep_ret)

                writer.add_scalar('Train/Max_Force', f, t_total)
                writer.add_scalar('Train/Step_Reward', r, t_total)

                self.memory.record(o.to('cpu'), a2, r, o2.to('cpu'), d)

                if (t+1) % nfake == 0:
                    o2_f_raw, r_f, d_f, a_f, _ = self.env.fake_step()
                    if USE_ENV_DESCRIPTOR: o2_f = self._augment_state(o2_f_raw)
                    else: o2_f = o2_f_raw
                    self.memory.record(o.to('cpu'), a_f, r_f, o2_f.to('cpu'), d_f)

                o = o2

                current_q_loss = None
                current_pi_loss = None

                if t_total >= self.update_after and len(self.memory) >= self.batch_size and t_total % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = self.memory.sample()
                        losses = self.update(batch, t)
                        pi_losses.append(losses["loss_pi"])
                        q_losses.append(losses["loss_q"])

                        if losses["loss_q"] is not None:
                            current_q_loss = losses["loss_q"]
                            writer.add_scalar('Loss/Critic_Q', losses["loss_q"], t_total)
                        if losses["loss_pi"] is not None:
                            current_pi_loss = losses["loss_pi"]
                            writer.add_scalar('Loss/Actor_Pi', losses["loss_pi"], t_total)

                with open(csv_log_path, 'a', newline='') as log_file:
                    csv_writer = csv.writer(log_file)
                    q_val = current_q_loss if current_q_loss is not None else ''
                    pi_val = current_pi_loss if current_pi_loss is not None else ''
                    csv_writer.writerow([t_total, i+1, t+1, r, f, q_val, pi_val])

                max_norm.append(a2.x.norm(dim = 1).max().item())
                if np.array(max_norm)[-min(len(max_norm), 10):].mean() <= d_r_max and f >= f_max:
                    c_gr += 1
                else:
                    c_gr = 0

                if t_total% test_every == 0 and test_ep is not None:
                    print(f"\n>>> Running Test at Step {t_total} ...")

                    # 1. 获取分组顺序的详细数据
                    raw_test_data = self.test_agent(test_ep[0], test_ep[1], test_random, separate=True)

                    # ==========================================================
                    # 【修改点2】: 在同一张图里画出所有结构，行数 = 结构数
                    # ==========================================================
                    # 获取唯一的结构标签 (例如 [0, 1])
                    unique_labels = np.unique(self.test_labels)
                    num_structs = len(unique_labels)

                    # 创建画布: 行数=结构数, 列数=3 (Disc_score, Last_step, Max_force)
                    # figsize 随行数自动增高
                    fig, axes = plt.subplots(nrows=num_structs, ncols=3, figsize=(18, 5 * num_structs), constrained_layout=True)

                    # 统一字体
                    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 12}
                    plt.rc('font', **font)

                    # 如果只有一行，axes 是 1D 数组，需要把它变成 2D 方便索引 [row, col]
                    if num_structs == 1:
                        axes = np.array([axes])

                    # 遍历每个结构进行绘图
                    for row_idx, label_id in enumerate(unique_labels):
                        # 找到属于当前 label 的数据索引
                        mask = (np.array(self.test_labels) == label_id)

                        # 提取数据
                        y_score = raw_test_data['Disc_score'][mask]
                        y_step  = raw_test_data['Last_step'][mask]
                        y_force = raw_test_data['Maximum_force'][mask]

                        # X轴: 0, 1, 2... (第几次测试)
                        x_axis = range(len(y_score))

                        # Col 0: Disc Score
                        ax = axes[row_idx, 0]
                        ax.plot(x_axis, y_score, 'o-', label=f'Struct {label_id} Score')
                        ax.set_title(f"Struct {label_id}: Disc Score", fontsize=14)
                        ax.set_ylabel("Score")
                        ax.grid(True, alpha=0.3)

                        # Col 1: Last Step
                        ax = axes[row_idx, 1]
                        ax.plot(x_axis, y_step, 'o-', color='orange', label=f'Struct {label_id} Steps')
                        ax.set_title(f"Struct {label_id}: Steps to Relax", fontsize=14)
                        ax.set_ylabel("Steps")
                        ax.grid(True, alpha=0.3)

                        # Col 2: Max Force
                        ax = axes[row_idx, 2]
                        ax.plot(x_axis, y_force, 'o-', color='green', label=f'Struct {label_id} Force')
                        if e_lim: ax.set_ylim(e_lim) # 如果有配置Force限制
                        # 画阈值线
                        ax.axhline(y=self.env.eps, color='r', linestyle='--', alpha=0.5, label='Threshold')
                        ax.set_title(f"Struct {label_id}: Max Force", fontsize=14)
                        ax.set_ylabel("Force (eV/A)")
                        ax.grid(True, alpha=0.3)
                        ax.legend()

                    # 保存详细图片
                    save_dir = os.path.join(path_to_the_main_dir, 'figs')
                    if not os.path.exists(save_dir): os.makedirs(save_dir)
                    fig_name = f"_test_detailed_iter{t_total}.png"
                    fig.savefig(os.path.join(save_dir, env_name + fig_name))
                    plt.close(fig) # 关闭画布释放内存

                    # 3. 聚合数据用于 CSV 记录 (Mean)
                    agg_test_data = {
                        "Score": raw_test_data['Score'].mean(),
                        "Last_step": raw_test_data['Last_step'].mean(),
                        "Maximum_force": raw_test_data['Maximum_force'].mean(),
                        "Disc_score": raw_test_data['Disc_score'].mean(),
                        "Score_std": raw_test_data['Score'].std(),
                        "Last_step_std": raw_test_data['Last_step'].std(),
                        "Maximum_force_std": raw_test_data['Maximum_force'].std(),
                        "Disc_score_std": raw_test_data['Disc_score'].std(),
                        "Test_labels": self.test_labels,
                        "Score_med": np.median(raw_test_data['Score']),
                        "Last_step_med": np.median(raw_test_data['Last_step']),
                        "Maximum_force_med": np.median(raw_test_data['Maximum_force']),
                        "Disc_score_med": np.median(raw_test_data['Disc_score']),
                        "Success_Rate": raw_test_data['Success_label'].mean(),
                        "Stop_Rate": raw_test_data['Stop_label'].mean()
                    }

                    df_test = pd.concat([df_test, pd.DataFrame([agg_test_data])], ignore_index=True)
                    df_test.to_csv(f"{path_to_the_main_dir}/data/df_{env_name}_test_si{start_iter}.csv")

                    writer.add_scalar('Test/Score', agg_test_data['Score'], t_total)
                    writer.add_scalar('Test/Max_Force', agg_test_data['Maximum_force'], t_total)
                    writer.add_scalar('Test/Success_Rate', agg_test_data['Success_Rate'], t_total)
                    print(f">>> Test Finished. Success Rate: {agg_test_data['Success_Rate']*100:.1f}%\n")

                if save_result and (t_total % plot_every == 0):
                    self.save_plots(path_to_the_main_dir, env_name, start_iter,
                                    df_train, df_test, pi_losses, q_losses,
                                    net_lim, e_lim, sticks, max_force, local_reward, train_ep, test_ep)

                if t_total % save_every == 0:
                    self.save_model(path_to_the_main_dir, env_name, f"{i + start_iter}")

                if d or s:
                    sticks.append(t_total-1)
                    break
                if t + 1 == train_ep[1]:
                    sticks.append(t_total-1)

            ep_end_time = time.time()
            ep_duration = ep_end_time - ep_start_time
            total_duration = ep_end_time - global_start_time

            avg_step_time_ep = ep_duration / max(1, ep_len)
            avg_step_time_global = total_duration / max(1, t_total)

            print("-" * 80)
            print(f"Episode {i+1} Finished | Status: {'Done' if d else 'Stop' if s else 'MaxSteps'}")
            print(f"Time Used: {ep_duration:.2f}s (Total Elapsed: {total_duration/60:.2f} min)")
            print(f"Avg Time/Step: {avg_step_time_ep:.4f}s (This Ep) | {avg_step_time_global:.4f}s (Global)")
            print("-" * 80 + "\n")

            data_to_save_train = {"Total_reward":ep_ret, "Last_step_train":ep_len, "Stop_label_train":s, "Env_name":self.env.current_ase_structure.get_chemical_formula() + "_" + str(self.env.num), "Weights": self.env.weights}

            df_new_row = pd.DataFrame([data_to_save_train])
            if 'Stop_label_train' in df_new_row.columns:
                 df_new_row['Stop_label_train'] = df_new_row['Stop_label_train'].astype(bool)

            df_train = pd.concat([df_train, df_new_row], ignore_index=True)
            df_train.to_csv(f"{path_to_the_main_dir}/data/df_{env_name}_train_si{start_iter}.csv")

        writer.close()

    def save_plots(self, path_to_the_main_dir, env_name, start_iter,
                   df_train, df_test, pi_losses, q_losses,
                   net_lim, e_lim, sticks, max_force, local_reward, train_ep, test_ep):

        name = f"_train_start_iter{start_iter}.png"
        last_step_done, last_step_stop = [], []

        if "Stop_label_train" in df_train:
             for key, item in zip(df_train["Stop_label_train"].values, df_train["Last_step_train"].values):
                if key:
                    last_step_stop.append(item)
                    last_step_done.append(None)
                else:
                    last_step_done.append(item)
                    last_step_stop.append(None)

        pi_losses_clean = np.array(pi_losses)[pd.notna(pi_losses)]
        q_losses_avg    = aver_list(q_losses, self.policy_delay)

        data_list = {
            "Total reward of the episode": [["Total reward"], [df_train["Total_reward"].values], None, None, None],
            "Losses_Pi": [["Pi"], [pi_losses_clean], net_lim, None, (np.array(sticks) / self.policy_delay).astype(int)],
            "Losses_Q": [["Q"], [q_losses_avg], net_lim, None, (np.array(sticks) / self.policy_delay).astype(int)],
            "Last step of the episode": [["Last step done", "Last step stop"], [last_step_done, last_step_stop], None, "o", None],
            "Weights": [["Weights"], [self.env.weights], None, "o", None],
            "Max force": [["Max force"], [max_force], e_lim, None, sticks]
        }
        if train_ep[0] == 1:
            data_list["Total reward of the episode"][1][0] = local_reward

        create_plots(data_list=data_list, save=True, show=False,
                    path_to_the_main_dir=path_to_the_main_dir,
                    env_name=env_name, name=name, folder_name='figs')

        # 如果需要保留总趋势图，这里依然保留
        if test_ep is not None and not df_test.empty:
            name_test = f"_test_trend_start_iter{start_iter}.png"
            keys = df_test['Score'].notna()
            data_list_test = {
                "Disc_score": [["Disc_score"], [df_test['Disc_score'].values[keys]], None, "o-", None],
                "Last step test": [["Last step test"], [df_test['Last_step'].values[keys]], None, "o-", None],
                "Max force": [["Max force"], [df_test['Maximum_force'].values[keys]], e_lim, "o-", None],
                "Success Rate": [["Success Rate"], [df_test['Success_Rate'].values[keys]], [0, 1.05], "o-", None]
            }
            create_plots(data_list=data_list_test, save=True, show=False,
                        path_to_the_main_dir=path_to_the_main_dir,
                        env_name=env_name, name=name_test, folder_name='figs')

    def save_model(self, path_to_the_main_dir, env_name, suffix=""):
        ckpt_dir = os.path.join(path_to_the_main_dir, "models")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, "td3_checkpoint_{}_{}".format(env_name, suffix))
        print(f'>>> Saving models to {ckpt_path}')

        torch.save({'ac_pi': self.ac.pi.state_dict(),
                    'ac_pi_t' : self.ac_targ.pi.state_dict(),
                    'ac_q1': self.ac.q1.state_dict(),
                    'ac_q2': self.ac.q2.state_dict(),
                    'ac_q1_t': self.ac_targ.q1.state_dict(),
                    'ac_q2_t': self.ac_targ.q2.state_dict(),
                    'pi_optim': self.pi_optimizer.state_dict(),
                    'q_optim': self.q_optimizer.state_dict()}, ckpt_path)
        return ckpt_path

    def load_model(self, ckpt_path):
            print('Loading models from {}'.format(ckpt_path))
            if ckpt_path is not None:
                # 【修复】增加 map_location参数
                # 这样无论模型是在GPU还是CPU训练的，都会被正确加载到当前设备(self.device)
                checkpoint = torch.load(ckpt_path, map_location=self.device)

                self.ac.pi.load_state_dict(checkpoint['ac_pi'])
                self.ac_targ.pi.load_state_dict(checkpoint['ac_pi_t'])
                self.ac.q1.load_state_dict(checkpoint['ac_q1'])
                self.ac.q2.load_state_dict(checkpoint['ac_q2'])
                self.ac_targ.q1.load_state_dict(checkpoint['ac_q1_t'])
                self.ac_targ.q2.load_state_dict(checkpoint['ac_q2_t'])
                self.q_optimizer.load_state_dict(checkpoint['q_optim'])
                self.pi_optimizer.load_state_dict(checkpoint['pi_optim'])

    def save_results(self, *args, **kwargs):
        pass