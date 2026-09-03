#!/usr/bin/env python3
"""ETロボコンのコース全体図PDF（50%縮図）から片側コースのPNGを生成する．

PDFはリポジトリに含めていない（ETロボコン実行委員会の著作物のため）．
公式配布のPDFを手元に用意して実行すること．

使い方:
    python tools/コース画像生成.py コース全体図.pdf --side L --dpi 100

出力:
    assets/Lコース全体図.png （既定．--out で変更可）

DPI=100 のとき縮尺は 19.687 px/cm になる．
（PDF幅 7738.58pt = 実寸5460mm，50%縮図のため2倍して換算）
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# コース全体（L+R）の実寸 [cm]
COURSE_W_CM = 546.0
COURSE_H_CM = 364.0


def render_pdf(pdf_path: Path, dpi: int, workdir: Path) -> Path:
    """pdftoppm でPDF 1ページ目をPNGにラスタライズする．"""
    prefix = workdir / "page"
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1",
             str(pdf_path), str(prefix)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        sys.exit("pdftoppm が見つからない．poppler-utils を入れること．\n"
                 "  Ubuntu: sudo apt install poppler-utils\n"
                 "  Windows: https://github.com/oschwartz10612/poppler-windows")
    except subprocess.CalledProcessError as e:
        sys.exit(f"pdftoppm が失敗した: {e.stderr.decode(errors='replace')}")

    pngs = sorted(workdir.glob("page-*.png"))
    if not pngs:
        sys.exit("PNGが生成されなかった．PDFの内容を確認すること．")
    return pngs[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="コース全体図PDFから片側コースPNGを生成")
    ap.add_argument("pdf", type=Path, help="コース全体図のPDF（50%縮図）")
    ap.add_argument("--side", choices=["L", "R"], default="L", help="切り出す側（既定: L）")
    ap.add_argument("--dpi", type=int, default=100, help="ラスタライズ解像度（既定: 100）")
    ap.add_argument("--out", type=Path, default=None, help="出力PNGのパス")
    args = ap.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDFが見つからない: {args.pdf}")

    out = args.out or Path("assets") / f"{args.side}コース全体図.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        src = render_pdf(args.pdf, args.dpi, Path(td))
        img = Image.open(src)
        w, h = img.size

        px_per_cm = w / COURSE_W_CM
        aspect = (w / h) / (COURSE_W_CM / COURSE_H_CM)
        if not 0.98 < aspect < 1.02:
            print(f"[警告] 縦横比が想定と違う（比率 {aspect:.3f}）．"
                  f"PDFが全体図でない可能性がある．", file=sys.stderr)

        half = w // 2
        box = (0, 0, half, h) if args.side == "L" else (w - half, 0, w, h)
        img.crop(box).save(out, optimize=True)

    print(f"出力: {out}")
    print(f"サイズ: {half} x {h} px")
    print(f"縮尺: {px_per_cm:.4f} px/cm  "
          f"(= {half / px_per_cm:.1f}cm x {h / px_per_cm:.1f}cm)")


if __name__ == "__main__":
    main()
