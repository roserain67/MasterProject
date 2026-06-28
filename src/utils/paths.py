"""项目路径自动定位工具"""
import os


def find_project_root():
    """从当前文件或 cwd 向上查找项目根目录（含 src/ 和 configs/ 的目录）."""
    # 先从本文件位置向上找
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "src")) and os.path.isdir(os.path.join(d, "configs")):
            return d
        d = os.path.dirname(d)

    # 再从 cwd 向上找
    d = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "src")) and os.path.isdir(os.path.join(d, "configs")):
            return d
        d = os.path.dirname(d)

    raise FileNotFoundError("无法定位项目根目录（需要包含 src/ 和 configs/）")


def project_path(*parts):
    """拼接项目根目录下的路径: project_path("configs", "pearl_default.yaml")"""
    return os.path.join(find_project_root(), *parts)
