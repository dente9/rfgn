uv v -p 3.10

.venv\Scripts\activate

uv pip install torch==2.5.1 --no-build-isolation

uv pip install setuptools

uv pip install -r .\requirements.txt --no-build-isolation

```bash
uv run python TD3_e3nn.py --path_to_the_main_dir outputs/exp_name --structures_file "structures/AlFe.csv" --interval [1,2] --reward_func "force" --env_name "AlFe_cubic" --eps 0.01 --start_iter 0 --random_seed 5599 --test_ep [10,100] --with_weights False --r_weights [1,1,1] --r0 1.5  --r_max 5 --em_dim 10 --noise_clip 0.1 --pi_n_layers 2 --pi_mul 20 --pi_lmax 2 --num_neighbors 25 --q_n_layers 2 --q_mul 20 --q_lmax 2 --replay_size 1000000 --gamma 0.9999 --polyak 0.995 --pi_lr 1e-05 --q_lr 1e-05 --batch_size 100 --start_steps 0 --update_after 0 --update_every 1 --target_noise 0.05 --policy_delay 2 --trans_coef 0.5 --noise [0.2,0.2] --train_ep [800,1000] --save_result True --test_random False --expl_mode 'state' --test_every 1000 --save_every 1000 --N_gr 1000000 --d_r_max 0 --f_max 0.001 --noise_level 29
```

git clone https://github.com/dente9/rfgn.git

# TensorBoard 启动 & 连通性测试速查

## 1. 启动命令（容器 / 远程机）

```bash
# 杀掉旧进程
pkill -f tensorboard

# 重新启动：①秒级扫盘 ②允许外部映射 ③开启流式推送
tensorboard --logdir runs/sleep_test --host 0.0.0.0 --port 6006

tensorboard --logdir runs/outputs --port 6006
```

> 日志目录换成自己的；端口可改。

## 2. 本地端口映射（如有需要）

```bash
# 在本地电脑执行，把远程 6006 映射到本地 6006
ssh -L 6006:127.0.0.1:6006 user@远程IP
ssh -L 6006:127.0.0.1:6006 13161352780@xn-a.suanjiayun.com -p 2020
```

## 3. 连通性测试

| 目的               | 命令                              |
| ------------------ | --------------------------------- |
| 查看端口是否监听   | `lsof -i:6006`                  |
| 查看 TCP 监听状态  | `netstat -tulnp \| grep 6006`    |
| 确认 HTTP 服务正常 | `curl -I http://localhost:6006` |

期望输出（curl）：

```bash
HTTP/1.1 200 OK
Server: Werkzeug/3.x.x Python/3.x.x
...
```

## 4. 浏览器访问

- 地址：`http://127.0.0.1:6006`
- **首次或重启服务后务必「无痕窗口」或 Ctrl+Shift+R 强制刷新**，否则旧页面不会自动动。

netstat -ano | findstr ":6006"

Stop-Process -Id 29904 -Force
