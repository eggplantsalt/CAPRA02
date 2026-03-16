"""CAPRA 数据读写层导出入口。

该层统一管理监督样本文件的读写接口。
"""

from experiments.robot.capra.io.supervision_io import SupervisionRecord, read_supervision_jsonl, write_supervision_jsonl

__all__ = ["SupervisionRecord", "read_supervision_jsonl", "write_supervision_jsonl"]
