# -*- coding: utf-8 -*-
"""Citra warna asli Himawari-9, kanal tampak.

   KENAPA ADA. Semua layer satelit yang lain itu turunan, suhu kecerahan,
   vektor gerak, titik panas. Yang ini citranya sendiri, gambar yang orang
   kenali sebagai "foto satelit". Dia juga satu satunya layer kita yang
   MEMPERLIHATKAN ASAP, sebab asap memang cuma terlihat di cahaya tampak,
   bukan di inframerah termal.

   KANALNYA EMPAT. Merah dari kanal 3 di 0,64 mikrometer, biru dari kanal 1
   di 0,47, dan hijaunya CAMPURAN, sebab kanal 2 di 0,51 itu terlalu condong
   ke biru dan kalau dipakai apa adanya daun terlihat kecoklatan. Dicampur
   sedikit kanal 4 di 0,86, tempat daun memantul kuat, dan daratan kembali
   hijau. Bobot 0,13 dipilih sesudah mencoba 0,07 dan 0,20.

   SIANG SAJA, dan itu tidak bisa ditawar. Tanpa matahari tidak ada yang
   dihamburkan. Alfanya memudar antara sudut zenit matahari 80 dan 88
   derajat, jadi batas siang malamnya melengkung di peta seperti aslinya,
   bukan terpotong lurus.

   KABUT BIRU DIBUANG DENGAN MENERA SENDIRI. Udara menghamburkan cahaya biru
   jauh lebih kuat daripada merah, dan tanpa dibetulkan seluruh laut
   tertutup selaput susu kebiruan. Tidak dipakai rumus Rayleigh penuh, sebab
   itu perlu tekanan permukaan dan sudut hamburan per piksel. Yang dipakai
   lebih sederhana dan menera sendiri tiap bingkai, persentil 2 di daerah
   yang sedang siang itu laut bersih, dan laut bersih pantulannya sudah
   diketahui. Selisihnya itulah tebal kabutnya, lalu dikurangkan.

   ONGKOSNYA. Kanal tampak jauh lebih besar daripada inframerah, sebab
   resolusinya 1 km dan kanal 3 malah 0,5 km. Satu slot empat kanal itu
   sekitar 535 MB, lawan 25 MB satu kanal inframerah. Membangun kisinya 54
   detik. Masih wajar untuk satu bingkai baru tiap kali jalan."""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import hsd
from . import himawari as H

LAYER = "citra_tampak"
VERSI_TAMPAK = 1

# Kanal dan resolusi berkasnya. Angka ini milik formatnya, bukan pilihan.
KANAL = {1: "R10", 2: "R10", 3: "R05", 4: "R10"}

BOBOT_NIR = 0.13        # bagian kanal 4 di dalam hijau campuran
GAMMA = 2.0

# Pantulan laut bersih yang diharapkan di tiap kanal, dipakai sebagai
# patokan waktu menera kabut biru.
LAUT_BERSIH = {1: 0.030, 2: 0.022, 3: 0.012}

SIANG_MULAI = 80.0      # sudut zenit matahari, mulai memudar
SIANG_AKHIR = 88.0      # habis, sudah malam
KOS_LANTAI = 0.15       # pembagi cos tidak boleh lebih kecil dari ini


def unduh(waktu: dt.datetime, band: int, folder: Path) -> list[Path]:
    import urllib.request
    import time as _t
    folder.mkdir(parents=True, exist_ok=True)
    jalur = []
    for s in H.SEGMEN:
        nama = "HS_H09_%s_B%02d_FLDK_%s_S%02d10.DAT.bz2" % (
            waktu.strftime("%Y%m%d_%H%M"), band, KANAL[band], s)
        t = folder / nama
        if not t.exists() or t.stat().st_size < 1000:
            url = "%s/AHI-L1b-FLDK/%s/%s" % (H.EMBER, waktu.strftime("%Y/%m/%d/%H%M"), nama)
            for coba in (1, 2):
                try:
                    with urllib.request.urlopen(url, timeout=900) as r:
                        t.write_bytes(r.read())
                    break
                except Exception as e:
                    if coba == 2:
                        print("    B%02d segmen %d gagal, %s" % (band, s, e))
                    else:
                        _t.sleep(2.0)
            else:
                continue
        jalur.append(t)
    return jalur


def kisi(jalur: list[Path]):
    """Albedo satu kanal di kisi lintang bujur kita."""
    if not jalur:
        return None, None
    h0 = hsd.baca(str(jalur[0]))
    kol, bar, sah, bentuk, redam = H._kisi(h0)
    out = np.full(bentuk, np.nan, np.float32)
    for p in jalur:
        h = hsd.baca(str(p))
        a = hsd.albedo(h)
        b0 = h["baris_awal"] - 1
        pilih = sah & (bar >= b0) & (bar < b0 + h["baris"])
        if pilih.any():
            out[pilih] = a[bar[pilih] - b0, kol[pilih]]
        del h, a
    return out, redam


def _kosinus_matahari(waktu: dt.datetime) -> np.ndarray:
    hari = waktu.timetuple().tm_yday
    dekl = math.radians(23.44) * math.sin(2 * math.pi * (284 + hari) / 365.0)
    lon = np.arange(H.LON0, H.LON1 + H.DERAJAT / 2, H.DERAJAT)
    lat = np.arange(H.LAT1, H.LAT0 - H.DERAJAT / 2, -H.DERAJAT)
    LON, LAT = np.meshgrid(lon, lat)
    jam = waktu.hour + waktu.minute / 60.0
    sj = np.radians(15.0 * (jam + LON / 15.0 - 12.0))
    kos = (np.sin(np.radians(LAT)) * math.sin(dekl)
           + np.cos(np.radians(LAT)) * math.cos(dekl) * np.cos(sj))
    return np.clip(kos, -1, 1).astype(np.float32)


def gambar(alb: dict, redam: np.ndarray, waktu: dt.datetime, tujuan: Path) -> float:
    """Menyusun warna asli dan menyimpannya. Memulangkan bagian yang siang."""
    kos = _kosinus_matahari(waktu)
    sza = np.degrees(np.arccos(kos))
    k = np.maximum(kos, KOS_LANTAI)
    siang = kos > 0.35

    ter = {}
    for i in (1, 2, 3):
        n = alb[i] / k
        v = n[siang & np.isfinite(n)]
        # Kalau yang siang terlalu sedikit, menera jadi tidak sah. Lebih baik
        # tidak membetulkan sama sekali daripada membetulkan dengan angka
        # yang diambil dari segelintir piksel di tepi senja.
        ter[i] = (float(np.percentile(v, 2)) - LAUT_BERSIH[i]) if v.size > 200000 else 0.0

    r = alb[3] / k - ter[3]
    b = alb[1] / k - ter[1]
    g = (1.0 - BOBOT_NIR) * (alb[2] / k - ter[2]) + BOBOT_NIR * (alb[4] / k)

    def regang(x):
        return np.clip(x, 0, 1) ** (1.0 / GAMMA)

    malam = np.clip((SIANG_AKHIR - sza) / (SIANG_AKHIR - SIANG_MULAI), 0, 1)
    rgba = np.zeros(r.shape + (4,), np.uint8)
    rgba[..., 0] = (regang(r) * 255).round()
    rgba[..., 1] = (regang(g) * 255).round()
    rgba[..., 2] = (regang(b) * 255).round()
    rgba[..., 3] = (malam * redam * 255).astype(np.uint8)
    rgba[~np.isfinite(alb[3])] = 0
    Image.fromarray(rgba, "RGBA").save(tujuan, "WEBP", quality=82, method=4,
                                       alpha_quality=100)
    return float(100.0 * (malam > 0.02).mean())
