#!/usr/bin/env python3
"""Generate the project-authored long_2k common-user prompt text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


PARAGRAPHS = [
    (
        "周末旅行规划资料：目的地是一个适合亲子散步的湖边小城，市中心到湖区有直达公交，"
        "车程大约四十分钟。周六下午两点以后人流会增加，建议提前购买返程车票，并把"
        "儿童水杯、薄外套、充电宝和少量现金放在随身包里。"
    ),
    (
        "天气预报显示周六白天多云，傍晚可能有小雨，湖边风会比城区更明显。行程中可以"
        "安排室内书店、湖边步道和家庭餐厅三个停留点，这样下雨时也能快速调整，不需要"
        "临时寻找避雨地点。"
    ),
    (
        "餐饮方面，湖区东门附近有一家面馆和一家简餐店，面馆上菜快，适合孩子饿的时候"
        "优先选择；简餐店座位宽敞，但晚餐高峰可能排队二十分钟。预算按交通、饮料、晚餐"
        "和临时雨具计算，一家三口控制在三百元以内比较稳妥。"
    ),
    (
        "注意事项包括：不要把拍照时间排得太满，给孩子留出休息和上厕所的时间；傍晚六点"
        "后湖边灯光变暗，最好回到主路附近；如果孩子已经疲惫，就取消最后一个购物点，直接"
        "乘车回酒店。"
    ),
]


def build_text(repeats: int) -> str:
    lines = [
        "周末旅行规划资料",
        "",
        "请阅读下面重复整理的旅行资料，并在最后根据资料回答问题。",
        "",
    ]
    for i in range(repeats):
        lines.append(f"资料片段 {i + 1:03d}")
        lines.extend(PARAGRAPHS)
        lines.append("")
    lines.append(
        "请根据上面的资料，给一家三口安排一个周六下午到晚上的简短行程。"
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=36)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": build_text(args.repeats)}]
    args.out.write_text(json.dumps(messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
