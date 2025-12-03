import torch
import sys

# 定义一个配置类 (比全局变量更安全，防修改丢失)
class GlobalConfig:
    # 默认逻辑：如果是 Windows 就 CPU，否则看显卡
    DEVICE = torch.device("cpu" if sys.platform.startswith('win') else ("cuda" if torch.cuda.is_available() else "cpu"))
    USE_ENV_DESCRIPTOR = True
    DESC_DIM = 8