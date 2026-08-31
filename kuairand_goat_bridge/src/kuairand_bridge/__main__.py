"""python -m kuairand_bridge 的启动入口。"""

from __future__ import annotations

import multiprocessing as mp

from .cli import main


if __name__ == "__main__":
    # Windows multiprocessing 使用 spawn。
    # freeze_support 防止子进程导入入口时重复启动整个主程序。
    mp.freeze_support()
    main()
