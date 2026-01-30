1. 网站 : https://hpcgame.pku.edu.cn/org/2df8f692-0316-4682-80cd-591732b1eda6/contest/d61cbf2d-b554-43bf-a3de-aaa0d46264d9
2. register_buffer

register_buffer 用于在 PyTorch 模型中注册一个Tensor，它具有以下特性:

不是模型参数(Parameter): 它不会出现在 model.parameters() 中，也不会被优化器(Optimizer)更新.

属于模型状态(State): 它会出现在 model.state_dict() 中(除非设置 persistent=False).

设备同步: 当你调用 model.cuda() 或 model.to(device) 时，注册的 buffer 会自动跟着模型一起移动到相应的设备上.


3. Parameter 会被优化器更新.

4. 登陆服务器 :
```bash
ssh -p 2117 stu2400010766@119.167.167.34 
```

5. 压缩并传输文件 :
本地
```bash
cd ./data/cs336/llm-from-scratch-assignment1-basics/
scp -P 2117 ./cs336-spring2025-assignment-1-submission.zip \
  stu2400010766@119.167.167.34:/home/stu2400010766/ \
  cs336-spring2025-assignment-1-submission.zip
```
服务器解压


6. 值得一提的, 我们采用`wandb`记录日志
```bash
uv add wandb
uv run wandb login
```


8. `slurm`提交任务
```bash
nvidia-smi # 查看所有卡的状态, 一般会看到8张卡
srun --gres=gpu:1 --cpus-per-task=1 --mem=8G --pty /bin/bash # 申请 1 张 GPU
nvidia-smi # 确认身份, 只会看到属于你的那张卡
# 冒烟测试
uv run python -m scripts.train \
    --data_dir ./data \
    --save_ckp_path ./wandb_test_ckpts \
    --device cuda \
    --wandb_project "cs336-playground" \
    --wandb_run_name "run-1-demo" \
    --batch_size 16 \
    --context_len 256 \
    --train_steps 1000 \
    --val_interval 100 \
    --save_intervals 500 \
    --log_intervals 10
exit # 结束后退出
```

9. 走的时候记得 `exit` 退出



10. 规范:
- 创建新的文件记得打印**绝对路径**
- 用 `pathlib` 而非 `os` 方便维护
- 用 `argparse` 而不要改代码