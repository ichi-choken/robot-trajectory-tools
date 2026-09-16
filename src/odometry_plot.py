# -*- coding: utf-8 -*-
"""
ETロボコン2026 Lコース 軌跡プロット
- 同じフォルダ(CSV_DIR)内の最新CSVを自動選択 (pybricks-data-*.csv)
- odo_x, odo_y, o_deg 列を読み, background.png(正立向きのLコース)に重ねて描画
- 軸自体を反転(x:右→左に増加, y:上→下に増加)した180度回転表示なので,
  シート座標系の軌跡は正立コース上にそのまま重なる
- 実ログとの照合で確認済みの座標変換(2026-08-11のログで検証):
    plot_x = -odo_x   (odo_xは写真向きで左が正のため符号反転)
    plot_y = +odo_y   (写真向きで上が正, 単位cm)
    theta  = 90 + o_deg [deg]  (o_degは初期進行方向基準・反時計回り正)
- 出力: plot.png

必要: pandas, matplotlib, numpy
使い方: background.png と同じ場所に置いて python odometry_plot.py
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================= 設定 =================
CSV_DIR = "."            # CSVを探すフォルダ
CSV_GLOB = "pybricks-data-*.csv"       # 対象パターン
BACKGROUND = "background.png"
ODO_UNIT = "cm"          # ログのodo単位: "mm" / "cm" / "m" (実ログで確認済み)
X_SIGN = -1.0            # plot_x = X_SIGN * odo_x (検証済み: -1)
VEHICLE_WIDTH_CM = 12.5  # 車体幅の帯を軌跡に沿って描く (0で無効)
THETA_OFFSET_DEG = 90.0  # theta = o_deg + これ (検証済み: 90)
ARROW_STEP_CM = 10.0     # 矢印を打つ間隔(走行距離[cm]ごと)
ARROW_LEN_CM = 8.0       # 矢印の長さ[cm]
OUTPUT = "plot.png"

# ===== background.png の幾何定数(変更不要) =====
# 100dpiレンダリング, 実寸5460x3640mmの左半分(5374x7166px)
PX_PER_CM = 19.6865
X0_PX = 4960.0                     # ロボット座標原点のx[px](正立画像内)
Y0_PX = 727.5 - 10.0 * PX_PER_CM   # 原点y[px] = スタートライン手前10cm
IMG_W, IMG_H = 5374, 7166
# ============================================

UNIT_TO_CM = {"mm": 0.1, "cm": 1.0, "m": 100.0}[ODO_UNIT]


def find_latest_csv():
    files = glob.glob(os.path.join(CSV_DIR, CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"{CSV_DIR} に {CSV_GLOB} が見つかりません")
    return max(files, key=os.path.getmtime)


def load_log(path):
    # BOM付き・全値引用符付き・途中にヘッダ行再出現(BLE再接続)があっても読めるようにする
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    if "o_deg" not in df.columns and "o_degs" in df.columns:
        df = df.rename(columns={"o_degs": "o_deg"})
    need = ["odo_x", "odo_y", "o_deg"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"列が見つかりません: {missing} / 実際の列: {list(df.columns)}")
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # 再出現ヘッダ行はNaN化して除去
    df = df.dropna(subset=["odo_x", "odo_y"]).reset_index(drop=True)
    return df


def main():
    csv_path = find_latest_csv()
    print(f"対象CSV: {csv_path}")
    df = load_log(csv_path)

    x = X_SIGN * df["odo_x"].to_numpy() * UNIT_TO_CM
    y = df["odo_y"].to_numpy() * UNIT_TO_CM
    th = np.deg2rad(df["o_deg"].ffill().fillna(0.0).to_numpy() + THETA_OFFSET_DEG)

    # ---- 背景(正立コース)を, ロボット座標系の軸に載せる ----
    img = plt.imread(BACKGROUND)
    # 画像px -> ロボット座標[cm]: x_r=(X0-px)/k, y_r=(py-Y0)/k
    left = X0_PX / PX_PER_CM               # px=0
    right = (X0_PX - IMG_W) / PX_PER_CM    # px=IMG_W
    top = -Y0_PX / PX_PER_CM               # py=0
    bottom = (IMG_H - Y0_PX) / PX_PER_CM   # py=IMG_H

    fig, ax = plt.subplots(figsize=(9, 12), dpi=150)
    ax.imshow(img, extent=[left, right, bottom, top], zorder=0)
    ax.set_xlim(left, right)     # 右へ行くほどxが減る(180度回転表示)
    ax.set_ylim(bottom, top)     # 上へ行くほどyが減る

    # ---- 車体幅の帯 (線幅をデータ座標cmに合わせるため, レイアウト確定後に計算) ----
    fig.tight_layout()
    fig.canvas.draw()
    if VEHICLE_WIDTH_CM > 0:
        p0 = ax.transData.transform((0.0, 0.0))
        p1 = ax.transData.transform((1.0, 0.0))
        px_per_cm = abs(p1[0] - p0[0])
        lw_pts = VEHICLE_WIDTH_CM * px_per_cm * 72.0 / fig.dpi
        ax.plot(x, y, color="#4da6ff", lw=lw_pts, alpha=0.35,
                solid_capstyle="round", solid_joinstyle="round",
                zorder=1.5, label=f"body {VEHICLE_WIDTH_CM:g}cm")

    # ---- 軌跡 ----
    ax.plot(x, y, color="#d62728", lw=1.8, zorder=2, label="trajectory")

    # ---- 走行距離ベースで間引いて向き矢印 ----
    if len(x) > 1:
        dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        idx = [0]
        nxt = ARROW_STEP_CM
        for i, s in enumerate(dist):
            if s >= nxt:
                idx.append(i)
                nxt += ARROW_STEP_CM
        idx = np.array(idx)
        ax.quiver(
            x[idx], y[idx],
            ARROW_LEN_CM * np.cos(th[idx]), ARROW_LEN_CM * np.sin(th[idx]),
            angles="xy", scale_units="xy", scale=1,
            color="#1f4fd6", width=0.004, zorder=3, label="heading",
        )

    ax.plot(x[0], y[0], "o", color="lime", mec="k", ms=9, zorder=4, label="start")
    ax.plot(x[-1], y[-1], "s", color="orange", mec="k", ms=9, zorder=4, label="end")

    ax.set_aspect("equal")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title(os.path.basename(csv_path))
    ax.grid(alpha=0.25, ls="--")
    leg = ax.legend(loc="lower right", fontsize=8)
    for line in leg.get_lines():
        line.set_linewidth(min(line.get_linewidth(), 6))
    fig.savefig(OUTPUT)
    print(f"保存: {OUTPUT}")


if __name__ == "__main__":
    main()
