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
import csv  # [新增] 用于写 CSV 日志

class Agent(nn.Module):
    r"""The class of TD3 Agent."""
    def __init__(self, net_actor, net_critic, actor_feat, critic_feat):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env =  env_fn(**env_kwards)
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
                assert len(init_rewards_for_weights) == len(env_kwards["input_struct_lib"])
                self.rewards_for_weights = np.array(init_rewards_for_weights)
            else:
                self.rewards_for_weights = []
                for i in range(len(env_kwards["input_struct_lib"])):
                    o, _, _, _ = self.env.reset(self.trans_coef, i), False, 0, 0
                    _, _, _, _, f, _ = self.env.step(self.get_action(o, 0), 0)
                    self.rewards_for_weights.append(f)
                self.rewards_for_weights = np.array(self.rewards_for_weights)
            self.env.weights = self.rewards_for_weights/self.rewards_for_weights.sum()
        else:
            L = len(env_kwards["input_struct_lib"])
            self.env.weights = np.ones(L)/L

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
        prev_state = self.env.current_ase_structure.copy()
        prev_calc = self.env.current_ase_structure.calc
        prev_num = self.env.num

        self.test_labels = []
        L = 1 if test_random else len(self.env.input_lib.keys())
        N_ep = num_test_episodes*L
        scores = np.zeros(N_ep)
        disc_scores = np.zeros(N_ep)
        last_steps = np.zeros(N_ep)
        forces_last_step = np.zeros(N_ep)

        for j in range(N_ep):
            np.random.seed(j)
            num = None if test_random else j % L
            o, d, ep_ret, ep_disc_ret, ep_len = self.env.reset(self.trans_coef, num, correct = False), False, 0, 0, 0
            self.test_labels.append(self.env.num)
            while not(d or (ep_len == max_test_steps)):
                o, r, d, _, f, _ = self.env.step(self.get_action(o, None))
                ep_ret += r
                ep_disc_ret += r*(self.gamma**ep_len)
                ep_len += 1
            scores[j] = ep_ret
            last_steps[j] = ep_len
            forces_last_step[j] = f
            disc_scores[j] = ep_disc_ret
            if self.with_weights:
                self.update_weights(f, self.env.num)

        if separate:
            data_to_save_test = {"Score": scores, "Last_step": last_steps, "Maximum_force": forces_last_step, "Disc_score": disc_scores}
        else:
            data_to_save_test = {"Score": scores.mean(), "Last_step": last_steps.mean(), "Maximum_force": forces_last_step.mean(), "Disc_score": disc_scores.mean(),
                                "Score_std": scores.std(), "Last_step_std": last_steps.std(), "Maximum_force_std": forces_last_step.std(), "Disc_score_std": disc_scores.std(), "Test_labels": self.test_labels,
                                "Score_med": np.median(scores), "Last_step_med": np.median(last_steps), "Maximum_force_med": np.median(forces_last_step), "Disc_score_med": np.median(disc_scores)}
        self.env.current_ase_structure = prev_state
        self.env.num = prev_num
        self.env.current_ase_structure.calc = prev_calc
        return data_to_save_test


    # 请确保文件头部有这些导入
    # import csv
    # from torch.utils.tensorboard import SummaryWriter

    def train(self, train_ep, test_ep, path_to_the_main_dir, env_name, test_every, start_iter = 0, save_every= 1000, e_lim = None, net_lim = None,
              save_result = True, test_random = False, N_gr = 30, d_r_max = 0.015, f_max = 0.1, noise_level = 10, nfake = 10,
              plot_every = 50):

        pi_losses, q_losses, max_force, local_reward, sticks = [], [], [], [], []
        t_total = 0
        df_test = pd.DataFrame(None, columns=['Score', 'Last_step', 'Maximum_force', "Disc_score", 'Score_std', 'Last_step_std', "Maximum_force_std", "Disc_score_std" , "Test_labels", 'Score_med', 'Last_step_med', "Maximum_force_med", "Disc_score_med"])
        df_train = pd.DataFrame(None, columns=["Total_reward", "Last_step_train", "Stop_label_train", "Env_name", "Weights"])

        # 建立目录
        os.makedirs(os.path.join(path_to_the_main_dir, 'data'), exist_ok = True)

        # TensorBoard
        writer = SummaryWriter(log_dir=os.path.join(path_to_the_main_dir, 'logs'))

        # CSV Log (初始化)
        csv_log_path = os.path.join(path_to_the_main_dir, 'logs', 'steps_log.csv')
        os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
        # 【修复1】使用 log_file 作为变量名，防止覆盖 f
        with open(csv_log_path, 'w', newline='') as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(['Total_Step', 'Episode', 'Ep_Step', 'Reward', 'Max_Force', 'Loss_Q', 'Loss_Pi'])

        for i in range(train_ep[0]):
            o, ep_ret, ep_len = self.env.reset(self.trans_coef), 0, 0
            max_norm = []
            c_gr = 0

            for t in range(train_ep[1]):
                print(f"--- [Episode {i+1}/{train_ep[0]}] [Step {t+1}/{train_ep[1]}] (Total: {t_total}) ---")

                if t_total >= self.start_steps:
                    if c_gr == N_gr:
                        self.ac.pi.noise_clip  = noise_level*2
                        a = self.get_action(o, noise_level)
                        self.ac.pi.noise_clip = self.noise_clip
                        c_gr = 0
                    else:
                        a = self.get_action(o, ((self.noise[1] - self.noise[0])/train_ep[1]) * t + self.noise[0])
                    o2, r, d, a2, f, s = self.env.step(a)
                else:
                    o2, r, d, a2, f = self.env.fake_step()
                    s = False

                t_total +=1
                ep_ret += r
                ep_len += 1
                max_force.append(f)
                local_reward.append(ep_ret)

                writer.add_scalar('Train/Max_Force', f, t_total)
                writer.add_scalar('Train/Step_Reward', r, t_total)

                self.memory.record(o.to('cpu'), a2, r, o2.to('cpu'), d)

                if (t+1) % nfake == 0:
                    o2_f, r_f, d_f, a_f, _ = self.env.fake_step()
                    self.memory.record(o.to('cpu'), a_f, r_f, o2_f.to('cpu'), d_f)

                o = o2

                current_q_loss = None
                current_pi_loss = None

                # Update models
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
                    print(f">>> Running Test at Step {t_total} ...")
                    data_to_save_test = self.test_agent(test_ep[0], test_ep[1], test_random)

                    # 【修复2】Pandas concat
                    df_test = pd.concat([df_test, pd.DataFrame([data_to_save_test])], ignore_index=True)

                    df_test.to_csv(f"{path_to_the_main_dir}/data/df_{env_name}_test_si{start_iter}.csv")
                    writer.add_scalar('Test/Score', data_to_save_test['Score'], t_total)
                    writer.add_scalar('Test/Max_Force', data_to_save_test['Maximum_force'], t_total)

                # Save Plots (main.py里配置了50，这里就是50)
                if save_result and (t_total % plot_every == 0):
                    self.save_plots(path_to_the_main_dir, env_name, start_iter,
                                    df_train, df_test, pi_losses, q_losses,
                                    net_lim, e_lim, sticks, max_force, local_reward, train_ep, test_ep)

                # Save Model
                if t_total % save_every == 0:
                    self.save_model(path_to_the_main_dir, env_name, f"{i + start_iter}")

                if d or s:
                    sticks.append(t_total-1)
                    break
                if t + 1 == train_ep[1]:
                    sticks.append(t_total-1)

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

        if test_ep is not None and not df_test.empty:
            name_test = f"_test_start_iter{start_iter}.png"
            keys = df_test['Score'].notna()
            data_list_test = {
                "Disc_score": [["Disc_score"], [df_test['Disc_score'].values[keys]], None, None, None],
                "Last step test": [["Last step test"], [df_test['Last_step'].values[keys]], None, None, None],
                "Max force": [["Max force"], [df_test['Maximum_force'].values[keys]], e_lim, None, None]
            }
            create_plots(data_list=data_list_test, save=True, show=False,
                        path_to_the_main_dir=path_to_the_main_dir,
                        env_name=env_name, name=name_test, folder_name='figs')

    def save_model(self, path_to_the_main_dir, env_name, suffix=""):
        ckpt_dir = os.path.join(path_to_the_main_dir, "models")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, "td3_checkpoint_{}_{}".format(env_name, suffix))
        print(f'>>> Saving models to {ckpt_path}') # 加个箭头更明显

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
            checkpoint = torch.load(ckpt_path)
            self.ac.pi.load_state_dict(checkpoint['ac_pi'])
            self.ac_targ.pi.load_state_dict(checkpoint['ac_pi_t'])
            self.ac.q1.load_state_dict(checkpoint['ac_q1'])
            self.ac.q2.load_state_dict(checkpoint['ac_q2'])
            self.ac_targ.q1.load_state_dict(checkpoint['ac_q1_t'])
            self.ac_targ.q2.load_state_dict(checkpoint['ac_q2_t'])
            self.q_optimizer.load_state_dict(checkpoint['q_optim'])
            self.pi_optimizer.load_state_dict(checkpoint['pi_optim'])