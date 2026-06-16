#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Palette radar 35 segments
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

Color = Tuple[int, int, int]


@dataclass(frozen=True)
class RainClass:
    threshold: float   # seuil bas du segment (mm)
    rgb: Color


# 34 classes 
# Chaque classe couvre [threshold, threshold_suivant[.
RAIN_CLASSES: List[RainClass] = [
    RainClass(0.02,  (155, 190, 196)),  # #9BBEC4
    RainClass(0.04,  (102, 191, 199)),  # #66BFC7
    RainClass(0.06,  (126, 225, 240)),  # #7EE1F0
    RainClass(0.09,  ( 98, 235, 253)),  # #62EBFD
    RainClass(0.12,  ( 51, 170, 207)),  # #33AACF
    RainClass(0.16,  ( 19, 155, 228)),  # #139BE4
    RainClass(0.23,  ( 18, 117, 230)),  # #1275E6
    RainClass(0.32,  (  8,  38, 225)),  # #0826E1
    RainClass(0.4,   (  2, 254,   1)),  # #02FE01
    RainClass(0.6,   (  3, 237,   1)),  # #03ED01
    RainClass(0.9,   (  2, 221,   4)),  # #02DD04
    RainClass(1.1,   (  1, 207,   0)),  # #01CF00
    RainClass(1.2,   (  1, 192,   1)),  # #01C001
    RainClass(1.6,   (  1, 174,   2)),  # #01AE02
    RainClass(2.8,   (  1, 160,   0)),  # #01A000
    RainClass(3.2,   (  0, 143,   2)),  # #008F02
    RainClass(4.4,   (248, 239,   1)),  # #F8EF01
    RainClass(6.1,   (239, 208,   0)),  # #EFD000
    RainClass(8.5,   (234, 180,   0)),  # #EAB400
    RainClass(10.0,  (241, 148,   2)),  # #F19402
    RainClass(12.9,  (253, 114,   2)),  # #FD7202
    RainClass(18.0,  (252,  80,   1)),  # #FC5001
    RainClass(22.3,  (252,  41,   2)),  # #FC2902
    RainClass(30.2,  (251,   1,   1)),  # #FB0101
    RainClass(39.2,  (238,   1,   0)),  # #EE0100
    RainClass(50.1,  (210,   1,   4)),  # #D20104
    RainClass(63.6,  (196,   0,   0)),  # #C40000
    RainClass(80.7,  (172,   0,   0)),  # #AC0000
    RainClass(102.5, (251, 201, 252)),  # #FBC9FC
    RainClass(130.1, (229, 162, 230)),  # #E5A2E6
    RainClass(166.2, (202, 124, 198)),  # #CA7CC6
    RainClass(211.4, (178,  87, 180)),  # #B257B4
    RainClass(268.8, (151,  45, 152)),  # #972D98
    RainClass(341.9, (255, 185, 255)),  # #FFB9FF  (>= 341.9 mm)
]

# Seuil supérieur du dernier segment (bord droit de la barre)
MAX_THRESHOLD = 490.3


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def format_label(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def save_palette_svg(
    path: Path,
    width: int = 1000,
    bar_height: int = 30,
    title: str = "Précipitations 1h (mm)",
) -> None:
    n = len(RAIN_CLASSES)

    margin_left  = 20
    margin_right = 20
    margin_top   = 28
    sep_w        = 1.5
    tick_h       = 5
    label_offset = 10
    font_size    = 9

    bar_w  = width - margin_left - margin_right
    cell_w = bar_w / n

    label_h      = font_size + 2
    total_height = margin_top + bar_height + tick_h + label_offset + label_h + 6

    x0    = margin_left
    y_bar = margin_top
    y_bot = y_bar + bar_height

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{total_height}" '
        f'viewBox="0 0 {width} {total_height}">',
        f'  <rect width="{width}" height="{total_height}" fill="#e0e0e0"/>',
    ]

    # Titre
    lines.append(
        f'  <text x="{width / 2:.1f}" y="18" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="12" '
        f'font-weight="bold" text-anchor="middle" fill="#222">{title}</text>'
    )

    # Rectangles pleins
    for i, rc in enumerate(RAIN_CLASSES):
        x = x0 + i * cell_w
        lines.append(
            f'  <rect x="{x:.3f}" y="{y_bar}" '
            f'width="{cell_w:.3f}" height="{bar_height}" '
            f'fill="{rgb_to_hex(*rc.rgb)}"/>'
        )

    # Séparateurs noirs internes
    for i in range(1, n):
        x = x0 + i * cell_w
        lines.append(
            f'  <line x1="{x:.3f}" y1="{y_bar}" x2="{x:.3f}" y2="{y_bot}" '
            f'stroke="#000" stroke-width="{sep_w}"/>'
        )

    # Bordure externe
    lines.append(
        f'  <rect x="{x0}" y="{y_bar}" width="{bar_w:.3f}" height="{bar_height}" '
        f'fill="none" stroke="#000" stroke-width="1"/>'
    )

    # Tirets + labels — une seule ligne, toutes les frontières
    font = (
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{font_size}" '
        f'text-anchor="middle" fill="#111"'
    )
    y_label = y_bot + tick_h + label_offset

    # N labels aux frontières gauches + 1 label final à droite
    boundaries = [(x0 + i * cell_w, format_label(rc.threshold))
                  for i, rc in enumerate(RAIN_CLASSES)]
    boundaries.append((x0 + n * cell_w, format_label(MAX_THRESHOLD)))

    for x, label in boundaries:
        lines.append(
            f'  <line x1="{x:.3f}" y1="{y_bot}" x2="{x:.3f}" y2="{y_bot + tick_h}" '
            f'stroke="#555" stroke-width="0.8"/>'
        )
        lines.append(
            f'  <text x="{x:.3f}" y="{y_label}" {font}>{label}</text>'
        )

    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"SVG écrit dans : {path}  ({n} classes, cell_w={cell_w:.1f}px)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg",        type=Path, default=Path("palette_radar.svg"))
    parser.add_argument("--width",      type=int,  default=1000)
    parser.add_argument("--bar-height", type=int,  default=30)
    args = parser.parse_args()
    save_palette_svg(path=args.svg, width=args.width, bar_height=args.bar_height)


if __name__ == "__main__":
    main()
