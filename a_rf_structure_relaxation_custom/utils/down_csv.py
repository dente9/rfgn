import pandas as pd
from pathlib import Path
import requests

CSV_URL = (
    "https://raw.githubusercontent.com/lenkakolenka/File/"
    "f75f040424669788d658bec8afe16f3e35d44d9e/atomic_properties.csv"
)
LOCAL_NAME = "atomic_properties.csv"

def download(url: str = CSV_URL, dst: str = LOCAL_NAME) -> Path:
    """把远程 csv 下载到本地，同名文件已存在则跳过"""
    dst = Path(dst)
    if dst.exists():
        print(f"[info] {dst.name} 已存在，跳过下载")
        return dst.resolve()

    print(f"[info] 正在下载 {url} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()          # 网络异常直接抛错
    dst.write_bytes(resp.content)
    print(f"[info] 已保存到 {dst}")
    return dst.resolve()

def get() -> pd.DataFrame:
    """返回下载后的 DataFrame"""
    csv_path = download()
    return pd.read_csv(csv_path)

# 当脚本直接运行时，简单测试
if __name__ == "__main__":
    df = get()
    print(df.head())