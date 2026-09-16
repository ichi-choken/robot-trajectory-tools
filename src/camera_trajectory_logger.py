# -*- coding: utf-8 -*-
"""
カレントフォルダ内で最新のmp4を読み込み，赤マーカーの重心を追跡して
軌跡CSVと軌跡画像を出力する．

変更点（旧: スクショ監視版 pose_logger2.py）:
- スクショフォルダ監視 → 最新mp4をVideoCaptureで読み込み
- タイムスタンプは実時間ではなく動画内時間 [ms]
- 緑HSV → 赤HSV（Hue折り返しのため2レンジ合成）
- 射影変換の4点は最初のフレーム上をクリックして指定（キャリブレーション）
- 実世界座標系は右上原点: x軸=右上→左上方向, z軸=右上→右下方向 [mm]
- 背景グリッドはフレームサイズに合わせて実行時に自動生成（1080p/4K両対応）
"""
import glob
import os
import csv
import time
import cv2
import numpy as np
from collections import deque

# ================= 設定 =================
VIDEO_DIR = "."          # mp4を探すフォルダ
OUT_CSV = "軌跡.csv"
OUT_TRACE_IMG = "軌跡画像.jpg"
OUT_MASK_IMG = "マスク確認.jpg"
BACKGROUND_PATH = "背景グリッド.png"  # 無ければ最初のフレームを背景にする

# ===== 座標変換モード =====
# "scale":       GUIなし．真下向き固定カメラ用．右上原点で
#                x = (画像幅 - cx) * MM_PER_PX, z = cy * MM_PER_PX
#                MM_PER_PX=1.0なら単位はpxのまま
# "perspective": 最初のフレームで4点クリックして射影変換（コースマット等，
#                実寸既知の四隅がある場合用）
CALIB_MODE = "scale"

# スケール係数 [mm/px]．AUTO_SCALE=Trueなら動画冒頭の基準紙から自動算出され，
# この値は基準紙が見つからなかった場合のフォールバックとしてのみ使われる
MM_PER_PX = 1.0

# ==== 基準紙による自動スケール較正 ====
# 15cm角の色つき折り紙をコース中央付近に置いておくと，動画の最初のフレームから
# 検出してMM_PER_PXを自動算出する．色だけでなく「正方形であること」でも
# 絞り込むので，多少似た色の物が写り込んでも誤検出しにくい
AUTO_SCALE = True
REF_SQUARE_MM = 150.0   # 基準紙の一辺 [mm]

# ==== 視差(高さ)補正 ====
# 基準紙は床面，赤マーカーはロボット上面にあり，マーカーの方がカメラに近い分
# 大きく写る．床面で求めたスケールをそのまま使うと距離が約10%過大になる
# (実測189.8cmに対し211cmと出た事例で確認済み)．
#   マーカー面スケール = 床面スケール × (CAM_HEIGHT - MARKER_HEIGHT) / CAM_HEIGHT
# メジャーで2つの高さを一度測って設定すること:
#   CAM_HEIGHT_MM:    床からカメラのレンズまでの高さ
#   MARKER_HEIGHT_MM: 床から赤マーカーの紙面までの高さ
# CAM_HEIGHT_MM=None の場合は補正なし(床面スケールのまま)で警告を出す
CAM_HEIGHT_MM = 1990.0     # 実測値（床からレンズまで: 2m−カメラ厚1cm）
MARKER_HEIGHT_MM = 161.0   # 実測値（床から赤マーカー紙面まで）
MARKER_DIAMETER_MM = 42.5  # 赤マーカーの直径．走行中の見かけ径からスケールを毎回検算する

# 高さ補正後も残る系統誤差の最終調整．1.0で無効．
# 基準紙の辺長はマスクの閾値処理で縁が1〜2px内側に食い込むため，
# 150mm角では2%程度スケールが過大に出やすい（実測との比較で確認）．
# 求め方: 既知距離を走らせ  SCALE_CORRECTION = 実測距離 ÷ 出力距離
SCALE_CORRECTION = 1.0

# 1段階目(発見用)の緩い閾値．辺長の確定はこの閾値ではなく，
# 検出領域の実際の色に適応した2段階目で行う（下のdetect_ref_square参照）．
# 動画は静止画より彩度が低く写るため，Sを厳しくすると紙の縁が削れて
# 辺長を過小評価し，スケールが過大になる
# 現在の設定: 青緑(ティール)の折り紙．実測 H=97, S=255, V=125．
# 緑〜青緑〜青を通しでカバーするので，この系統の紙なら色を変えても動く．
# 木目床(H≒15)や赤マーカー(H 0-10/165-180)とは十分離れている．
# 紙の色を大きく変えたときは，動画の1コマから紙のHSVを実測して調整すること
REF_LOWER = np.array([40, 70, 40])
REF_UPPER = np.array([125, 255, 255])

# --- perspective モード用設定 ---
# 実世界座標 [mm]（コース四隅など，クリック順に対応させる）
# クリック順: (1)右下 → (2)左下 → (3)左上 → (4)右上
# 座標系: 右上を原点(0,0)とし，x軸は右上→左上方向，z軸は右上→右下方向
DST_MM = np.float32([
    [0.0, 4260],     # (1)右下
    [3365, 4260],    # (2)左下
    [3365, 0.0],     # (3)左上
    [0.0, 0.0],      # (4)右上 = 原点
])

# クリックせず固定値を使う場合はここに4点を書き，USE_CLICK_CALIB=False
USE_CLICK_CALIB = True
SRC_FIXED = np.float32([
    [1600, 900],
    [300, 900],
    [300, 150],
    [1600, 150],
])

# 赤のHSV範囲（明所・暗所・接写の実写真3枚で検証済み）
# 重要: 同じ赤マーカーでも照明の色温度とスマホのホワイトバランス補正により
# Hueが大きくシフトする．実測では明所H≒177, 暗所H≒6（0/180をまたいで反転）．
# そのため両端のレンジが必須．片方だけだと逆の条件で完全に検出できなくなる．
# 木目床(H≒15-18, S≒110-131)との分離はSで行うので，Sの下限160は下げないこと．
# Hの上限10/下限165も床に近づくため広げない．暗所対応はVの80で確保している．
RED_RANGES = [
    (np.array([0, 160, 80]), np.array([10, 255, 255])),      # 暖色照明・暗所側
    (np.array([165, 160, 80]), np.array([180, 255, 255])),   # 昼白色・明所側
]
MIN_AREA = 30        # これ未満の輪郭はノイズとして無視 [px^2]
MAX_JUMP_PX = 80     # 前回検出位置からこれ以上飛んだら誤検出として棄却 [px]
                     # (0で無効) 30fpsで80pxなら実用上十分速い動きまで許容する．
                     # 検出が飛び飛びになる環境では大きめに
PROCESS_EVERY = 1    # nフレームに1回処理（重い場合は2や3に）
TRACE_LEN = 5000     # 軌跡の最大保持点数
# ========================================


def imread_u(path):
    """日本語パス対応のimread"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    """日本語パス対応のimwrite．
    他アプリがファイルを開いていて上書きできない場合は，時刻付きの
    別名に保存して知らせる（Windowsのフォトアプリ等はファイルをロックする）"""
    base, ext = os.path.splitext(path)
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        print(f"警告: {path} のエンコードに失敗")
        return None
    try:
        buf.tofile(path)
        return path
    except OSError:
        alt = f"{base}_{time.strftime('%H%M%S')}{ext}"
        try:
            buf.tofile(alt)
            print(f"警告: {path} を上書きできません（他のアプリで開いていませんか？）．"
                  f"{alt} に保存しました．")
            return alt
        except OSError as e:
            print(f"警告: {path} を保存できません: {e}")
            return None


def make_grid_background(w, h, mm_per_px=None):
    """グリッド背景を生成．

    mm_per_px を渡すとcm目盛りになり，原点は出力座標系と同じ右上
    （x: 右端0で左へ増加, z: 上端0で下へ増加）．
    渡さない場合は従来どおり画像px座標（左上原点）のグリッド．"""
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    s = max(1.0, w / 1920)          # 4K等でも読めるよう文字を拡大
    fs, th = 0.4 * s, max(1, int(s))

    def draw(xs, ys, color, thick, label=None):
        """xs, ys は (px位置, ラベル値) のリスト"""
        for xp, val in xs:
            cv2.line(img, (xp, 0), (xp, h), color, thick)
            if label is not None and 0 < xp < w:
                cv2.putText(img, label(val), (xp + 4, int(18 * s)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (120, 120, 120), th, cv2.LINE_AA)
        for yp, val in ys:
            cv2.line(img, (0, yp), (w, yp), color, thick)
            if label is not None and 0 < yp < h:
                cv2.putText(img, label(val), (4, yp - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (120, 120, 120), th, cv2.LINE_AA)

    if mm_per_px:
        # ---- cm目盛り（右上原点）----
        px_per_cm = 10.0 / mm_per_px
        for step_cm, color, thick, lab in [(5, (225, 225, 225), 1, None),
                                           (10, (190, 190, 190), 1, lambda v: f"{v:g}"),
                                           (50, (150, 150, 150), 2, None)]:
            d = step_cm * px_per_cm
            if d < 6:      # 目盛りが細かすぎるとつぶれるので省く
                continue
            xs = [(w - int(round(i * d)), i * step_cm)
                  for i in range(int(w / d) + 1)]
            ys = [(int(round(i * d)), i * step_cm)
                  for i in range(int(h / d) + 1)]
            draw(xs, ys, color, thick, lab)
        cv2.putText(img, "grid: 10cm (bold 50cm), origin = top-right",
                    (10, h - int(38 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (120, 120, 120), th, cv2.LINE_AA)
    else:
        # ---- px目盛り（左上原点）----
        for step, color, thick, lab in [(50, (225, 225, 225), 1, None),
                                        (100, (190, 190, 190), 1, lambda v: str(int(v))),
                                        (500, (150, 150, 150), 2, None)]:
            d = int(step * s)
            xs = [(x, x) for x in range(0, w + 1, d)]
            ys = [(y, y) for y in range(0, h + 1, d)]
            draw(xs, ys, color, thick, lab)
    return img


def detect_ref_square(frame):
    """基準紙(正方形)を検出し，(一辺px, 外接矩形) を返す．見つからなければ(None, None)

    2段階方式:
      1) 緩い色閾値+形状ゲート(充填率>0.8, 縦横比0.75-1.33)で候補を発見
      2) 候補領域の実際の色(H,Sの中央値)に適応した閾値で輪郭を精密化し，
         そのminAreaRectから辺長を確定する
    固定閾値だけだと照明・露出により紙の縁が削れて辺長を過小評価するため"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, REF_LOWER, REF_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 最小サイズは画像幅に比例させる．固定値だと，高解像度時に
    # ロボットの水色パーツなど小さな同系色を拾ってしまう
    min_side = max(20.0, 0.02 * frame.shape[1])
    min_area = min_side ** 2
    best = None
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if rw == 0 or rh == 0:
            continue
        if a / (rw * rh) > 0.8 and 0.75 < rw / rh < 1.33:
            if best is None or a > best[0]:
                best = (a, np.sqrt(rw * rh), c)
    if best is None:
        return None, None

    # ---- 2段階目: 候補領域の実際の色に適応して輪郭を精密化 ----
    _, side1, cont = best
    x, y, w0, h0 = cv2.boundingRect(cont)
    pad = int(0.4 * max(w0, h0))
    H, W = hsv.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w0 + pad), min(H, y + h0 + pad)
    roi = hsv[y0:y1, x0:x1]
    # 候補輪郭の内側の色を基準にする
    core = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(core, [cont], -1, 255, -1)
    core = core[y0:y1, x0:x1] > 0
    med_h = np.median(roi[:, :, 0][core])
    med_s = np.median(roi[:, :, 1][core])
    m2 = ((np.abs(roi[:, :, 0].astype(int) - med_h) <= 8)
          & (roi[:, :, 1] >= max(70, 0.5 * med_s))
          & (roi[:, :, 2] >= 40)).astype(np.uint8) * 255
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts2, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts2:
        c2 = max(cnts2, key=cv2.contourArea)
        rect2 = cv2.minAreaRect(c2)
        (_, _), (rw2, rh2), _ = rect2
        if rw2 > 0 and rh2 > 0:
            fill2 = cv2.contourArea(c2) / (rw2 * rh2)
            if fill2 > 0.8 and 0.75 < rw2 / rh2 < 1.33:
                # ---- 3段階目: サブピクセル辺長 ----
                # 2値化は縁のぼけの内側で切れて辺長を1〜3px過小評価する
                # (動画の色圧縮でこの傾向が強い)．そこで色の類似度マップの
                # 半値交差位置を線形補間で求め，縁位置を小数px精度で確定する
                side_sub = _subpixel_side(frame[y0:y1, x0:x1], m2, rect2)
                bx, by, bw, bh = cv2.boundingRect(c2)
                side_out = side_sub if side_sub is not None else float(np.sqrt(rw2 * rh2))
                return side_out, (x0 + bx, y0 + by, bw, bh)
    # 精密化に失敗したら1段階目の値を使う
    return float(side1), (x, y, w0, h0)


def _subpixel_side(roi_bgr, core_mask, rect):
    """紙色と床色を結ぶ色軸への射影(混合率に線形)の半値交差から辺長を求める．

    2値化やHueゲートはぼけた縁で切断位置が内側に寄り辺長を過小評価するが，
    射影値はぼけ・クロマ間引きに対して線形に振る舞うため，半値交差が
    幾何学的な縁位置に一致する"""
    bgr = roi_bgr.astype(np.float32)
    (rcx, rcy), (rw, rh), ang = rect
    if rw < rh:
        ang += 90.0
        rw, rh = rh, rw
    # 紙色: 収縮したコア領域の中央値 / 床色: 矩形の外側リングの中央値
    core = cv2.erode(core_mask, np.ones((7, 7), np.uint8))
    if core.sum() == 0:
        return None
    paper = np.median(bgr[core > 0], axis=0)
    ring = np.zeros(core_mask.shape, np.uint8)
    box = cv2.boxPoints(((rcx, rcy), (rw * 1.8, rh * 1.8), ang)).astype(np.int32)
    cv2.fillPoly(ring, [box], 255)
    box_in = cv2.boxPoints(((rcx, rcy), (rw * 1.25, rh * 1.25), ang)).astype(np.int32)
    cv2.fillPoly(ring, [box_in], 0)
    if ring.sum() == 0:
        return None
    floor = np.median(bgr[ring > 0], axis=0)
    axis = paper - floor
    denom = float(axis @ axis)
    if denom < 1e-6:
        return None
    val = ((bgr - floor) @ axis) / denom   # 床=0, 紙=1 の連続値

    M = cv2.getRotationMatrix2D((rcx, rcy), ang, 1.0)
    r = cv2.warpAffine(val, M, (val.shape[1], val.shape[0]))
    half = int(max(rw, rh) * 0.85)
    y0s, y1s = int(rcy - half), int(rcy + half)
    x0s, x1s = int(rcx - half), int(rcx + half)
    if y0s < 0 or x0s < 0 or y1s > r.shape[0] or x1s > r.shape[1]:
        return None
    win = r[y0s:y1s, x0s:x1s]
    th = 0.5

    def crossings(profiles):
        widths = []
        for prof in profiles:
            above = prof >= th
            if not above.any():
                continue
            i0, i1 = np.argmax(above), len(above) - np.argmax(above[::-1]) - 1
            if i0 == 0 or i1 >= len(prof) - 1:
                continue
            l = i0 - (prof[i0] - th) / max(prof[i0] - prof[i0 - 1], 1e-6)
            rr = i1 + (prof[i1] - th) / max(prof[i1] - prof[i1 + 1], 1e-6)
            widths.append(rr - l)
        return float(np.median(widths)) if len(widths) >= 5 else None

    cy0, cx0 = win.shape[0] // 2, win.shape[1] // 2
    band_h = max(3, int(rh * 0.3))
    band_w = max(3, int(rw * 0.3))
    width = crossings(win[cy0 - band_h:cy0 + band_h, :])
    height = crossings(win[:, cx0 - band_w:cx0 + band_w].T)
    if width is None or height is None:
        return None
    return float(np.sqrt(width * height))


def measure_disk_diameter(frame, mask, center):
    """赤マーカー(円)の見かけ直径[px]を線形色射影の半値面積から求める．
    失敗時はNone．基準紙と同じ理屈で，ぼけ・圧縮に対して不偏"""
    cx, cy = center
    a0 = cv2.countNonZero(mask)
    if a0 < 30:
        return None
    r_est = np.sqrt(a0 / np.pi)
    R = int(r_est * 3 + 8)
    H, W = mask.shape
    x0, y0 = max(0, cx - R), max(0, cy - R)
    x1, y1 = min(W, cx + R), min(H, cy + R)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    mroi = mask[y0:y1, x0:x1]
    core = cv2.erode(mroi, np.ones((3, 3), np.uint8))
    if core.sum() == 0:
        core = mroi
    k = int(r_est) | 1
    near = cv2.dilate(mroi, np.ones((max(3, k), max(3, k)), np.uint8))
    ring = cv2.dilate(near, np.ones((max(3, k), max(3, k)), np.uint8)) & ~near
    if ring.sum() == 0:
        return None
    marker = np.median(roi[core > 0], axis=0)
    floor = np.median(roi[ring > 0], axis=0)
    axis = marker - floor
    denom = float(axis @ axis)
    if denom < 1e-6:
        return None
    val = ((roi - floor) @ axis) / denom
    area = float(((val >= 0.5) & (near > 0)).sum())
    if area < 20:
        return None
    return 2.0 * np.sqrt(area / np.pi)


def find_latest_mp4(folder):
    files = glob.glob(os.path.join(folder, "*.mp4"))
    if not files:
        raise FileNotFoundError("mp4が見つかりません: " + os.path.abspath(folder))
    return max(files, key=os.path.getmtime)


def click_calibration(frame):
    """最初のフレーム上で4点クリックして射影変換元座標を得る"""
    points = []
    disp = frame.copy()
    order = ["(1)右下", "(2)左下", "(3)左上", "(4)右上"]
    win = "calibration (click 4 points / r: reset / q: quit)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(disp, p, 6, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        nxt = order[len(points)] if len(points) < 4 else "Enterで確定"
        cv2.putText(disp, f"next: {nxt}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == ord("q"):
            raise SystemExit("キャリブレーション中断")
        elif key in (13, 10) and len(points) == 4:  # Enter
            break
    cv2.destroyWindow(win)
    return np.float32(points)


def detect_red_center(frame):
    """赤マーカーの重心 (cx, cy) とマスクを返す．見つからなければ(None, mask)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in RED_RANGES:
        m = cv2.inRange(hsv, lower, upper)
        mask = m if mask is None else (mask | m)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 暗所ではマーカー内に検出漏れの穴が空きやすい．塞いでおかないと
    # 輪郭が分裂して面積が落ち，重心がフレームごとに飛んで軌跡が乱れる．
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None, mask
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None, mask
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])), mask


def save_trace_image(background, trace, last_center, ref_box, mm_per_px):
    """軌跡を背景グリッドに重ねて保存する．走行途中の更新にも使う"""
    vis = background.copy()
    points = list(trace)
    for i in range(1, len(points)):
        alpha = i / len(points)
        cv2.line(vis, points[i - 1], points[i], (0, int(255 * alpha), 0), thickness=2)
    if last_center is not None:
        cv2.circle(vis, last_center, 5, (0, 0, 255), -1)
    if CALIB_MODE == "scale":
        if ref_box is not None:
            bx, by, bw, bh = ref_box
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
            cv2.putText(vis, f"{REF_SQUARE_MM:.0f}mm ref", (bx, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(vis, f"{mm_per_px:.4f} mm/px", (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    imwrite_u(OUT_TRACE_IMG, vis)


def main():
    video_path = find_latest_mp4(VIDEO_DIR)
    print("読み込む動画:", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"fps={fps:.2f}, frames={total}")

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("最初のフレームを読めませんでした")
    h, w = first.shape[:2]

    # 座標変換の準備（モードに応じて to_real(cx, cy) -> (x, z) を定義）
    mm_per_px = MM_PER_PX   # perspectiveモードでは使わないが参照はされる
    ref_box = None
    if CALIB_MODE == "perspective":
        if USE_CLICK_CALIB:
            src = click_calibration(first)
        else:
            src = SRC_FIXED
        print("src:", src.tolist())
        perspective_M = cv2.getPerspectiveTransform(src, DST_MM)

        def to_real(cx, cy):
            real = cv2.perspectiveTransform(
                np.array([[[cx, cy]]], dtype=np.float32), perspective_M)
            return float(real[0][0][0]), float(real[0][0][1])
    else:  # "scale": 右上原点，x=右上→左方向，z=右上→下方向
        if AUTO_SCALE:
            # 冒頭30フレームから基準紙を探し，検出できた辺長の中央値でスケールを決める
            sides = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(30):
                ok2, fr = cap.read()
                if not ok2:
                    break
                side, box = detect_ref_square(fr)
                if side is not None:
                    sides.append(side)
                    ref_box = box
                    if len(sides) >= 5:
                        break
            if sides:
                side_px = float(np.median(sides))
                floor_scale = REF_SQUARE_MM / side_px
                if CAM_HEIGHT_MM:
                    kh = (CAM_HEIGHT_MM - MARKER_HEIGHT_MM) / CAM_HEIGHT_MM
                    mm_per_px = floor_scale * kh * SCALE_CORRECTION
                    print(f"基準紙を検出: 一辺{side_px:.1f}px → 床面 {floor_scale:.4f} mm/px "
                          f"→ 高さ補正×{kh:.4f} → 係数×{SCALE_CORRECTION:.4f} "
                          f"→ {mm_per_px:.4f} mm/px")
                else:
                    mm_per_px = floor_scale
                    print(f"基準紙を検出: 一辺{side_px:.1f}px → 床面 {floor_scale:.4f} mm/px")
                    print("警告: CAM_HEIGHT_MM 未設定のため高さ補正なし．マーカーは床より"
                          "カメラに近いため，距離が1割程度過大に出ます．カメラレンズと"
                          "マーカー面の床からの高さを測って設定してください．")
            else:
                print(f"警告: 基準紙が見つかりません．MM_PER_PX={MM_PER_PX} を使用します．"
                      "基準紙が映っているのに出ない場合はREF_LOWERのSを60程度に下げてください．")
        print(f"scaleモード: {mm_per_px:.4f} mm/px（1.0なら単位はpx）")

        def to_real(cx, cy):
            return (w - cx) * mm_per_px, cy * mm_per_px

    # 背景グリッドはスケール確定後に作る．
    # scaleモードでスケールが求まっていればcm目盛り(右上原点)，
    # それ以外はpx目盛りになる
    grid_mm_per_px = None
    if CALIB_MODE == "scale" and abs(mm_per_px - 1.0) > 1e-9:
        grid_mm_per_px = mm_per_px
    background = None
    if os.path.exists(BACKGROUND_PATH):
        bg = imread_u(BACKGROUND_PATH)
        if bg is not None and bg.shape[:2] == (h, w) and grid_mm_per_px is None:
            background = bg   # px目盛り時のみ既存ファイルを再利用
    if background is None:
        background = make_grid_background(w, h, grid_mm_per_px)
        imwrite_u(BACKGROUND_PATH, background)

    f = open(OUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["timestamp_ms", "x_mm", "z_mm", "img_x", "img_y"])

    trace = deque(maxlen=TRACE_LEN)
    frame_idx = 0
    detected = 0
    last_center = None
    last_mask = None
    marker_diams = []  # スケール検算用: マーカー見かけ径のサンプル
    rejected = 0      # 飛び値として棄却した数
    reject_run = 0    # 連続棄却カウント
    lost = 0          # 検出できなかったフレーム数

    # 最初のフレームも処理対象に戻す
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % PROCESS_EVERY != 0:
                frame_idx += 1
                continue
            timestamp_ms = int(frame_idx / fps * 1000)

            center, mask = detect_red_center(frame)
            last_mask = mask
            if center is not None:
                # 前回位置から不自然に飛んだ検出は誤検出とみなして捨てる．
                # ただし連続で棄却され続けると復帰できなくなるので，
                # 一定回数（15フレーム=0.5秒程度）続いたら受け入れて追従し直す．
                if (MAX_JUMP_PX > 0 and last_center is not None
                        and np.hypot(center[0] - last_center[0],
                                     center[1] - last_center[1]) > MAX_JUMP_PX
                        and reject_run < 15):
                    rejected += 1
                    reject_run += 1
                    center = None
                else:
                    reject_run = 0

            if center is not None:
                cx, cy = center
                trace.append(center)
                rx, rz = to_real(cx, cy)
                writer.writerow([timestamp_ms, rx, rz, cx, cy])
                detected += 1
                last_center = center
                if frame_idx % 15 == 0:   # 0.5秒毎にマーカー見かけ径を採取
                    d = measure_disk_diameter(frame, mask, center)
                    if d is not None:
                        marker_diams.append(d)
            else:
                lost += 1

            if frame_idx % 100 == 0:
                print(f"frame {frame_idx}/{total}  detected={detected}")
            if frame_idx % 60 == 0 and trace:
                # 走行中も定期的に軌跡画像を更新（途中経過の確認と，
                # 中断・エラー時にも画像が残るようにするため）
                save_trace_image(background, trace, last_center, ref_box, mm_per_px)
            frame_idx += 1
    except KeyboardInterrupt:
        print("中断されました．ここまでの結果を保存します．")
    finally:
        f.close()
        cap.release()

    save_trace_image(background, trace, last_center, ref_box, mm_per_px)
    if last_mask is not None:
        imwrite_u(OUT_MASK_IMG, last_mask)

    print(f"完了: {detected}/{frame_idx} フレームで検出 "
          f"(未検出 {lost}, 飛び値棄却 {rejected})")
    # ---- マーカー直径によるスケール検算 ----
    # マーカーはマーカー面そのものにあるので，見かけ径×スケールが実寸と
    # 合っていれば「基準紙スケール+高さ補正」の連鎖全体が正しい
    if CALIB_MODE == "scale" and marker_diams:
        d_px = float(np.median(marker_diams))
        d_mm = d_px * mm_per_px
        err = 100 * (d_mm / MARKER_DIAMETER_MM - 1)
        print(f"検算: マーカー見かけ径 中央値{d_px:.1f}px → {d_mm:.1f}mm "
              f"(実寸{MARKER_DIAMETER_MM}mm, 差{err:+.1f}%, n={len(marker_diams)})")
        if abs(err) > 5:
            print("警告: スケール検算が5%以上ずれています．基準紙の検出("
                  "軌跡画像.jpgの枠)，CAM_HEIGHT_MM/MARKER_HEIGHT_MMを確認してください．")
    if frame_idx and detected / frame_idx < 0.9:
        print("警告: 検出率が低いです．マスク確認.jpg を見て，マーカーが白く"
              "抜けていなければ RED_RANGES のSまたはVの下限を下げてください．")
    print("出力:", OUT_CSV, OUT_TRACE_IMG, OUT_MASK_IMG)


if __name__ == "__main__":
    main()