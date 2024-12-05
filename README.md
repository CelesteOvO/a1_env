# a1_env
## Setup
```bash
python -m pip install -r requirements.txt
```

## Train
```bash
python train.py --run train
```
## Displaying Trained Models 

```bash
python train.py --run test --model_path <path to model zip file>
```
## 训练设备
当前使用cpu训练 

使用gpu训练 train.py 52,56,93行加上
```bash
device='cuda'
```

使用npu训练 添加
```bash
import torch_npu 
from torch_npu.contrib import transfer_to_npu
```
## 注意
mujoco版本不能是2开头的，如果mujoco版本是2开头的请卸载重装
```bash
pip uninstall mujoco
pip install mujoco
```
