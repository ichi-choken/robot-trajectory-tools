# -*- coding: utf-8 -*-
"""
シミュレータ（athrill + Unity）のキャプチャ連番PNGを監視し，
機体LEDの色から位置を検出して実座標をCSVに記録する．

- 起動時にキャプチャフォルダ内の既存PNGを削除してから監視を始める
- 新しいPNGが増えるたびに，画像左上の領域でHSV検出 → 射影変換で実座標[mm]に変換
- 軌跡は固定の背景画像に重ねて保存する
- 終了は Ctrl-C

使い方（~/etrobo をカレントにして実行）:
    python src/pose_logger.py
"""
import csv
import glob
import os
import time
from collections import deque

import cv2
import numpy as np

# ================= 設定 =================
CAPTURE_DIR = "./raspike-athrill-v850e2m/sdk/workspace/capture/"
OUT_CSV = "./workspace/trajectory.csv"
RESULT_DIR = "./workspace/result/"
BACKGROUND_PATH = "./workspace/background.jpg"  # 軌跡を描く下地の画像

# キャプチャ画像から切り出す領域 [px]
CROP_W, CROP_H = 318, 401

# LED検出のHSV範囲
HSV_LOWER = np.array([62, 148, 245])
HSV_UPPER = np.array([68, 170, 255])

# 射影変換の4点（右下 → 左下 → 左上 → 右上）
# キャプチャの画角を変えたら書き換えること
SRC = np.float32([
    [318, 401],
    [0, 401],
    [0, 0],
    [318, 0],
])
# 対応する実座標 [mm]
DST = np.float32([
    [3365, 0.0],
    [0.0, 0.0],
    [0.0, 4260],
    [3365, 4260],
])

TRACE_LEN = 500  # 軌跡の最大保持点数
POLL_SEC = 0.1   # フォルダ監視の間隔 [s]
# ========================================


# 画像をすべて削除してから開始
paths = glob.glob(os.path.join(CAPTURE_DIR, "*.png"))
print(f"削除対象: {len(paths)}件")
for path in paths:
    os.remove(path)

os.makedirs(RESULT_DIR, exist_ok=True)

background = cv2.imread(BACKGROUND_PATH)
if background is None:
    raise FileNotFoundError(f"背景画像が読み込めません: {BACKGROUND_PATH}")

start_time = time.time()

f = open(OUT_CSV, "w", newline="")
writer = csv.writer(f)
writer.writerow(["timestamp", "x", "z"])

perspective_M = cv2.getPerspectiveTransform(SRC, DST)

# 画像座標の軌跡（描画用）
trace_result = deque(maxlen=TRACE_LEN)


def process(img, timestamp):
    top_left = img[0:CROP_H, 0:CROP_W]

    hsv = cv2.cvtColor(top_left, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    result = cv2.bitwise_and(top_left, top_left, mask=mask)
    cv2.imwrite(os.path.join(RESULT_DIR, "output.jpg"), result)

    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    print(f"Detected position in image: ({cx}, {cy})")

    trace_result.append((cx, cy))

    real = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), perspective_M)
    writer.writerow([timestamp, real[0][0][0], real[0][0][1]])
    f.flush()

    # 軌跡を背景画像に重ね描きして保存
    vis = background.copy()
    points = list(trace_result)
    for i in range(1, len(points)):
        alpha = i / len(points)
        color = (0, int(255 * alpha), 0)
        cv2.line(vis, points[i - 1], points[i], color, thickness=2)
    # 現在位置をマーク
    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(RESULT_DIR, "trace_result.jpg"), vis)


seen = set()
try:
    while True:
        files = sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.png")))
        for path in files:
            if path not in seen:
                seen.add(path)
                img = cv2.imread(path)
                if img is not None:
                    process(img, int((time.time() - start_time) * 1000))
        time.sleep(POLL_SEC)
except KeyboardInterrupt:
    f.close()
