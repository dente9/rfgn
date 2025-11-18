uv v -p 3.10

.venv\Scripts\activate

uv pip install torch==2.5.1 --no-build-isolation

uv pip install setuptools

uv pip install -r .\requirements.txt --no-build-isolation

```bash
uv run python TD3_e3nn.py --path_to_the_main_dir 'outputs/$exp_name' --structures_file "structures/AlFe.csv" --interval [1,2] --reward_func "force" --env_name "AlFe_cubic" --eps 0.01 --start_iter 0 --random_seed 5599 --test_ep [10,100] --with_weights False --r_weights [1,1,1] --r0 1.5  --r_max 5 --em_dim 10 --noise_clip 0.1 --pi_n_layers 2 --pi_mul 20 --pi_lmax 2 --num_neighbors 25 --q_n_layers 2 --q_mul 20 --q_lmax 2 --replay_size 1000000 --gamma 0.9999 --polyak 0.995 --pi_lr 1e-05 --q_lr 1e-05 --batch_size 100 --start_steps 0 --update_after 0 --update_every 1 --target_noise 0.05 --policy_delay 2 --trans_coef 0.5 --noise [0.2,0.2] --train_ep [800,1000] --save_result True --test_random False --expl_mode 'state' --test_every 1000 --save_every 1000 --N_gr 1000000 --d_r_max 0 --f_max 0.001 --noise_level 29
```
