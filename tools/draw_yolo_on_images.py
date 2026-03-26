"""
Desenha caixas YOLO (labels .txt) sobre as imagens correspondentes.

Layout esperado:
    <yolo_root>/images/train/*.png
    <yolo_root>/labels/train/*.txt   (mesmo stem que a imagem)

Por defeito grava em <yolo_root>/labeled_preview/<split>/ (não mistura com pastas de treino).

    python tools/draw_yolo_on_images.py --yolo-root dataset_labeltest_yolo
    python tools/draw_yolo_on_images.py --yolo-root dataset_yolo --yolo-root out/other_yolo
    python tools/draw_yolo_on_images.py --yolo-root dataset_labeltest_yolo --sidecar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_yolo_lines(text: str) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 5:
            continue
        boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def draw_boxes(
    im,
    boxes: list[tuple[int, float, float, float, float]],
    class_names: dict[int, str],
    line_thickness: int,
) -> None:
    H, W = im.shape[:2]
    colors = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 128, 0),
        (255, 0, 200),
        (200, 200, 0),
    ]
    auto_thick = max(line_thickness, max(1, min(W, H) // 400))
    for cls, cx, cy, bw, bh in boxes:
        x1 = int(round((cx - bw / 2) * W))
        y1 = int(round((cy - bh / 2) * H))
        x2 = int(round((cx + bw / 2) * W))
        y2 = int(round((cy + bh / 2) * H))
        color = colors[cls % len(colors)]
        cv2.rectangle(im, (x1, y1), (x2, y2), color, auto_thick, lineType=cv2.LINE_AA)
        label = class_names.get(cls, str(cls))
        cv2.putText(
            im,
            label,
            (x1, max(22, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            max(1, auto_thick - 1),
            cv2.LINE_AA,
        )


def process_root(
    yolo_root: Path,
    splits: list[str],
    sidecar: bool,
    line_thickness: int,
    class_names: dict[int, str],
) -> tuple[int, int]:
    written = 0
    skipped = 0
    for split in splits:
        img_dir = yolo_root / "images" / split
        lbl_dir = yolo_root / "labels" / split
        if not img_dir.is_dir():
            continue
        if sidecar:
            out_dir = img_dir
        else:
            out_dir = yolo_root / "labeled_preview" / split
            out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                continue
            if img_path.stem.endswith("_labeled"):
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                skipped += 1
                continue
            im = cv2.imread(str(img_path))
            if im is None:
                skipped += 1
                continue
            boxes = parse_yolo_lines(lbl_path.read_text(encoding="utf-8"))
            draw_boxes(im, boxes, class_names, line_thickness)
            if sidecar:
                out_path = out_dir / f"{img_path.stem}_labeled{img_path.suffix}"
            else:
                out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), im)
            written += 1
    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO labels desenhados sobre imagens")
    ap.add_argument(
        "--yolo-root",
        type=Path,
        action="append",
        dest="yolo_roots",
        required=True,
        help="Raiz do dataset YOLO (pode repetir para vários)",
    )
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="Pastas a processar")
    ap.add_argument(
        "--sidecar",
        action="store_true",
        help="Grava *_labeled.* junto a images/<split>/ em vez de labeled_preview/",
    )
    ap.add_argument("--line-thickness", type=int, default=2, help="Mínimo; escala com resolução")
    args = ap.parse_args()

    class_names = {0: "drone"}
    total_w = total_s = 0
    for root in args.yolo_roots:
        root = root.resolve()
        if not root.is_dir():
            print(f"[SKIP] não existe: {root}")
            continue
        w, s = process_root(root, args.splits, args.sidecar, args.line_thickness, class_names)
        total_w += w
        total_s += s
        mode = "sidecar" if args.sidecar else f"{root / 'labeled_preview'}"
        print(f"{root.name}: gravadas {w} imagens ({mode})  |  sem label/imagem: {s}")

    print(f"Total: {total_w} imagens com caixas.")


if __name__ == "__main__":
    main()
