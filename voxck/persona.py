"""人设文件加载：YAML frontmatter + markdown 正文。"""
from __future__ import annotations
from pathlib import Path
from typing import Any


def load(path: str | Path) -> dict[str, Any]:
    """返回 {meta: dict, prompt: str}。frontmatter 用极简解析，不引入 yaml 依赖。"""
    raw = Path(path).read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            body = parts[2]
            for line in parts[1].splitlines():
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    v = [x.strip() for x in v[1:-1].split(",") if x.strip()]
                meta[k.strip()] = v
    return {"meta": meta, "prompt": body.strip()}
