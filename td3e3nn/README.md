uv run python TD3_e3nn.py --path_to_the_main_dir outputs/fast_2_site_nodesc --structures_file "structures/AlFe.csv" --interval [1,2] --reward_func "hybrid" --env_name "AlFe_cubic" --eps 0.01 --start_iter 0 --random_seed 5599 --test_ep [10,100] --with_weights False --r_weights [0.5,2,1] --r0 1.5  --r_max 5 --em_dim 10 --noise_clip 0.1 --pi_n_layers 2 --pi_mul 20 --pi_lmax 2 --num_neighbors 25 --q_n_layers 2 --q_mul 20 --q_lmax 2 --replay_size 1000000 --gamma 0.9999 --polyak 0.98 --pi_lr 1e-04 --q_lr 3e-04 --batch_size 256 --start_steps 0 --update_after 0 --update_every 1 --target_noise 0.05 --policy_delay 2 --trans_coef 0.5 --noise [0.2,0.2] --train_ep [800,1000] --save_result True --test_random False --expl_mode 'state' --test_every 1000 --save_every 50 --N_gr 1000000 --d_r_max 0 --f_max 0.001 --noise_level 1


kill -9 $(lsof -t -i:6006)

uv run tensorboard --logdir outputt --host 0.0.0.0 --port 6006
