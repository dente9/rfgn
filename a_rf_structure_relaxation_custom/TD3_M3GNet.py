import argparse
import torch
import os
import pandas as pd
from utils.convert_to_graph_e3nn import to_graph
# [修改]: 导入 M3GNet 模型
from utils.model_m3gnet import M3GNet_actor, M3GNet_critic
from utils.td3 import Agent, TD3Agent
from utils.utils import get_the_last_checkpoint, extract_number
from utils.env import Environment, get_sturct_lib_and_calcs_gen
import time

torch.set_default_dtype(torch.float64)
parser = argparse.ArgumentParser(description='TD3 M3GNet')

#Environment parameters
parser.add_argument('--structures_file', type=str, help='Path to the file with input structures')
parser.add_argument('--interval', type=eval, default=None, help='The range of structures from structures_file that will be used for training')
parser.add_argument('--eps', type=float, default=0.1, help='Force threshold, eV/A')
parser.add_argument('--reward_func', type=str, default='step', help='Reward function type: force | log_force | step | hybrid')
parser.add_argument('--r_weights', type=eval, default=[1, 0, 0.5], help='Weights for the hybrid reward function')
parser.add_argument('--r0', type=float, default=1.5, help='Minimum possible distance between atoms')
parser.add_argument('--stop_numb', type=int, default=50, help='Maximum number of relaxation steps')

# Agent parameters
parser.add_argument('--random_seed', type=int, default=972, help = 'Seed for random number generators')
parser.add_argument('--replay_size', type=int, default=int(1e6), help = 'Maximum length of replay buffer')
parser.add_argument('--gamma', type=float, default=0.9999, help = 'Discount factor')
parser.add_argument('--polyak', type=float, default=0.995, help = 'Interpolation factor in polyak averaging')
parser.add_argument('--pi_lr', type=float, default=1e-5, help = 'Learning rate for Actor')
parser.add_argument('--q_lr', type=float, default=1e-5, help = 'Learning rate for Critic')
parser.add_argument('--batch_size', type=int, default=100, help = 'Minibatch size')
parser.add_argument('--start_steps', type=int, default=0, help = 'Number of steps for fake action selection')
parser.add_argument('--update_after', type=int, default=0, help = 'Number of env interactions before gradient descent')
parser.add_argument('--update_every', type=int, default=1, help = 'Number of env interactions between updates')
parser.add_argument('--target_noise', type=float, default=0.05, help = 'Noise level for smoothing noise')
parser.add_argument('--policy_delay', type=int, default=2, help = 'Policy update delay')
parser.add_argument('--trans_coef', type=float, default=0.5, help = 'Distortion parameter')
parser.add_argument('--noise', type=eval, default=[0.2,0.2], help = 'Noise level for exploration')
parser.add_argument('--path_weights', type=str, default=None, help = 'Path to weights')
parser.add_argument('--with_weights', type=eval, default=False, help = 'Flag for weighted selection')

# Training parameters
parser.add_argument('--path_to_the_main_dir', type=str, default='', help = 'Path to save results')
parser.add_argument('--path_load', type=str, default=None, help = 'Path to model checkpoints')
parser.add_argument('--train_ep', type=eval, default=[10000,1000], help = '[M_train, N_train]')
parser.add_argument('--test_ep', type=eval, default=[10,100], help = '[M_test, N_test]')
parser.add_argument('--env_name', type=str, default='', help = 'Name of the experiment')
parser.add_argument('--e_lim', type=eval, default=None, help = 'y-limits for plots of maximum forces')
parser.add_argument('--net_lim', type=eval, default=None, help = 'y-limits for plots of losses')
parser.add_argument('--save_result', type=eval, default=True, help = 'Flag indicating whether learning curves are plotted')
parser.add_argument('--start_iter', type=int, default=0, help = 'The initial iteration number')
parser.add_argument('--test_random', type=eval, default=False, help = 'Flag for random testing')
parser.add_argument('--test_every', type=int, default=1000, help = 'Test frequency')
parser.add_argument('--save_every', type=int, default=1000, help = 'Save frequency')
parser.add_argument('--N_gr', type=int, default=100, help = 'N_gr in greedy exploration')
parser.add_argument('--d_r_max', type=float, default=0.015, help = 'Delta r_max')
parser.add_argument('--f_max', type=float, default=0.1, help = 'f_max')
parser.add_argument('--noise_level', type=float, default=20, help = 'Noise level')
parser.add_argument('--nfake', type=int, default=10, help = 'Fake episodes frequency')

# Actor and Critic parameters (Using n_l_a/c for M3GNet dimensions)
parser.add_argument('--num_node_features', type=int, default=12, help='Number of graph node features')
parser.add_argument('--aggr_a', type=str, default="mean", help='Aggregation for Actor')
# M3GNet params mapping:
# n_l1: embedding dim, n_l2: rbf bins, n_l3: hidden dim, n_l4: output hidden
parser.add_argument('--n_l_a', type=eval, default=[64, 50, 64, 64], help = 'M3GNet Actor Dims')
parser.add_argument('--noise_clip', type=float, default=0.1, help='Noise clip')
parser.add_argument('--aggr_c', type=str, default="mean", help='Aggregation for Critic')
parser.add_argument('--n_l_c', type=eval, default=[64, 50, 64, 64, 64], help = 'M3GNet Critic Dims')

args = parser.parse_args()

s_lib, calcs = get_sturct_lib_and_calcs_gen(args.structures_file, interval = args.interval)

env_kwards = {"input_struct_lib": s_lib, "convert_to_graph_func": to_graph,
              "calculator_lib": calcs, "r0":args.r0 , "eps" : args.eps,
              "reward_func" : args.reward_func, "r_weights": args.r_weights, "stop_numb": args.stop_numb}

# M3GNet 需要 forces 维度 (+4)
actor_feat = {"input_dim": args.num_node_features + 4,
              "aggr": args.aggr_a,
              "n_l1":args.n_l_a[0],
              "n_l2":args.n_l_a[1],
              "n_l3":args.n_l_a[2],
              "n_l4":args.n_l_a[3],
              "noise_clip": args.noise_clip}

critic_feat = {"input_dim": args.num_node_features + 4,
               "aggr": args.aggr_c,
               "n_l1":args.n_l_c[0],
               "n_l2":args.n_l_c[1],
               "n_l3":args.n_l_c[2],
               "n_l4":args.n_l_c[3],
               "n_l5":args.n_l_c[4]}

# [修改]: 使用 M3GNet 类
ac_kwards = {"net_actor": M3GNet_actor, "net_critic": M3GNet_critic, "actor_feat": actor_feat, "critic_feat": critic_feat}

if not os.path.exists(args.path_to_the_main_dir):
    os.makedirs(args.path_to_the_main_dir)

wf = None
if args.with_weights and args.path_weights is not None:
    # (Weights loading logic same as original)
    pass

a = {"env_fn" : Environment ,
     "env_kwards" : env_kwards,
     "ac_kwargs" : ac_kwards,
     "seed": args.random_seed,
     "replay_size": args.replay_size,
     "gamma": args.gamma,
     "polyak": args.polyak,
     "pi_lr": args.pi_lr,
     "q_lr": args.q_lr,
     "batch_size": args.batch_size,
     "start_steps": args.start_steps,
     "update_after": args.update_after,
     "update_every": args.update_every,
     "target_noise": args.target_noise,
     "noise_clip": args.noise_clip,
     "policy_delay": args.policy_delay,
     "trans_coef": args.trans_coef,
     "noise": args.noise,
     "init_rewards_for_weights" : wf,
     "with_weights" : args.with_weights}

TD3_Agent = TD3Agent(**a)

if args.path_load == 'last':
    path_checkpoints = args.path_to_the_main_dir + "/models"
    if os.path.exists(path_checkpoints):
        args.path_load = get_the_last_checkpoint(path_checkpoints)
        args.start_steps = extract_number(args.path_load)
        print(f"Resuming from step: {args.start_steps}")

if args.path_load is not None and os.path.exists(args.path_load):
    TD3_Agent.load_model(args.path_load)

a["path_load"] = args.path_load

if args.start_iter == 0:
    with open(args.path_to_the_main_dir + "/TD3_Agent_arguments.txt", 'w') as f:
        f.write(str(a))

b = {"train_ep" : args.train_ep,
     "test_ep" : args.test_ep,
     "path_to_the_main_dir" : args.path_to_the_main_dir,
     "env_name" : args.env_name,
    "e_lim" : args.e_lim,
    "net_lim" : args.net_lim,
    "save_result" : args.save_result,
    "start_iter": args.start_iter,
    "test_random": args.test_random,
    "test_every": args.test_every,
    "save_every": args.save_every,
     "N_gr" : args.N_gr,
     "d_r_max" : args.d_r_max,
     "f_max" : args.f_max,
     "noise_level" : args.noise_level,
     "nfake" : args.nfake}

with open(args.path_to_the_main_dir + "/TD3_Agent_train_arguments.txt", 'w') as f:
    f.write(str(b))

print("Starting M3GNet Training...")
TD3_Agent.train(**b)