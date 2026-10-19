# -*- coding: utf-8 -*-
"""Gerak awan Himawari-9, dijejak sendiri dari dua citra berurutan.

   KENAPA DIJEJAK SENDIRI, PADAHAL NOAA SUDAH PUNYA PRODUK AMV.

   Produk L2 Winds NOAA itu bagus dan sudah dipakai di sini sebelumnya, tapi
   dia BERHENTI DI SUDUT ZENIT SATELIT 69,5 DERAJAT. Di dalam kotak kita
   yang mulai 62 BT, itu berarti tidak ada satu pun angin di sebelah barat
   79,4 BT. Diperiksa langsung di berkasnya, di sana ada 72.252 titik
   sasaran dan TIDAK SATU PUN punya kecepatan yang sah, semuanya nilai
   pengisi. Jadi melonggarkan ambang mutu tidak akan menambah satu panah
   pun, batas itu batas produknya.

   Citra kanal 13 kita sendiri jalan terus sampai tepi piringan, dan dua
   bingkai berurutan memang sudah diunduh untuk layer Puncak Awan. Jadi
   menjejak sendiri itu gratis dari sisi unduhan, dan domainnya penuh.

   APA YANG DIUKUR, dan ini harus jujur. Yang dijejak itu GERAK PUNCAK AWAN
   yang terlihat inframerah, bukan angin pada satu ketinggian tertentu.
   Di satu kotak bisa saja ada sirus tinggi menutupi kumulus rendah, dan
   yang terbaca gerak si sirus. Produk NOAA memasang tekanan di tiap titik
   justru untuk memisahkan itu, dan kita tidak. Sebagai gantinya ini
   medan rapat yang penuh sedomain, dan itu yang cocok untuk digambar jadi
   partikel mengalir seperti angin GFS.

   MUTUNYA DIUKUR, bukan ditaksir. Lawan AMV NOAA QI 80 ke atas pada
   18 Oktober 2026 pukul 17:30 UTC, 11.233 titik beradu di kisi keluaran,
   korelasi u 0,86, korelasi v 0,78, RMSVD 8,3 m/detik, selisih arah
   median 10,1 derajat. Untuk penjejak satu kanal yang ketinggiannya cuma
   ditaksir dari suhu, itu wajar.

   AMBANG KONTRASNYA disetel dengan angka itu juga. Dinaikkan dari 1,2
   sampai 7,0 kelvin, ketepatannya TIDAK BERUBAH sama sekali, yang berubah
   cuma cakupan, 95 persen turun jadi 52 persen. Sebabnya masuk akal, di
   tempat NOAA punya angin memang ada awan berkontras, jadi yang dibuang
   ambang tinggi itu justru petak langit cerah yang tidak bisa diadu dengan
   apa apa. Dipilih 3,5 kelvin, cakupan 81 persen, cukup rapat di atas awan
   tapi tidak lagi menaburi laut gelap dengan panah berarah acak.

   TIDAK PERLU PUSTAKA BARU. Semuanya numpy, dan pencocokan kotaknya
   memakai gambar terpadu supaya jumlah kotak dihitung sekali jalan untuk
   seluruh citra, bukan kotak demi kotak. h5py yang dulu dipakai membaca
   NetCDF produk NOAA sudah tidak diperlukan lagi."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import himawari as H

LAYER = "amv_angin"       # kunci layer DIPERTAHANKAN, supaya tautan lama tetap hidup

# Citra dikasarkan dua kali sebelum dijejak. Di 0,05 derajat deraunya ikut
# terjejak dan kotak contohnya jadi mahal; di 0,1 derajat satu piksel itu
# 11,1 km, dan dalam 30 menit itu 6,2 m/detik per piksel.
KASAR = 2
DER_KASAR = H.DERAJAT * KASAR

PETAK = 15        # sisi kotak contoh, piksel kasar. 1,5 derajat.
CARI = 9          # geseran maksimal, piksel kasar. 0,9 derajat, kira kira 55 m/detik.

# Penjagaan mutu. Tanpa ini langit cerah ikut "terjejak" dan hasilnya medan
# derau yang kelihatan meyakinkan padahal isinya tebakan.
KONTRAS_MIN = 3.5     # simpangan baku suhu di dalam kotak, kelvin
ISI_MIN = 0.70        # bagian piksel sah yang harus ada di dalam kotak
TAJAM_MIN = 1.25      # puncaknya harus sekian kali lebih baik dari rerata skor

# Kisi keluaran. 0,4 derajat itu empat piksel kasar, jadi tiap sel keluaran
# merata ratakan sampai 25 jejak. Perataan itu sekaligus yang menghaluskan
# pembulatan piksel, jadi tidak perlu pencocokan sub piksel yang harganya
# dua kali lipat waktu jalan.
KISI = 0.4
SUMBER_MIN = 4        # sel keluaran butuh sekian jejak sah, kalau tidak dikosongkan


def _kasarkan(a: np.ndarray) -> np.ndarray:
    ny, nx = a.shape
    ny, nx = ny // KASAR * KASAR, nx // KASAR * KASAR
    b = a[:ny, :nx].reshape(ny // KASAR, KASAR, nx // KASAR, KASAR)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))


def _jumlah_kotak(a: np.ndarray, r: int) -> np.ndarray:
    """Jumlah dalam kotak (2r+1) kali (2r+1) lewat gambar terpadu.

    Tepinya DIJEPIT, bukan dilipat. Kotak di pinggir jadi lebih kecil, dan
    itu betul, sebab di luar citra memang tidak ada apa apa."""
    c = np.cumsum(np.cumsum(np.pad(a, ((1, 0), (1, 0))), 0), 1)
    ny, nx = a.shape
    y0 = np.clip(np.arange(ny) - r, 0, ny); y1 = np.clip(np.arange(ny) + r + 1, 0, ny)
    x0 = np.clip(np.arange(nx) - r, 0, nx); x1 = np.clip(np.arange(nx) + r + 1, 0, nx)
    return c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)] - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)]


def _jejak(a: np.ndarray, b: np.ndarray):
    """Geseran piksel terbaik tiap titik, plus penanda layak pakai.

    Dikerjakan PER GESERAN untuk seluruh citra sekaligus, bukan per kotak.
    Satu geseran itu satu pengurangan, satu pengkuadratan, dan dua gambar
    terpadu, semuanya di numpy. Ada 361 geseran yang dicoba, dan itu jauh
    lebih murah daripada 780 ribu kotak yang dicocokkan satu satu."""
    r = PETAK // 2
    sa, sb = np.isfinite(a), np.isfinite(b)
    a0 = np.where(sa, a, 0.0).astype(np.float32)

    # Kontras kotak di citra PERTAMA. Kotak yang datar tidak punya apa apa
    # untuk dicocokkan, dan skornya akan rendah di mana mana.
    n_a = _jumlah_kotak(sa.astype(np.float32), r)
    s_a = _jumlah_kotak(a0, r)
    s_a2 = _jumlah_kotak(a0 * a0, r)
    aman = n_a > 0
    rata = np.divide(s_a, n_a, out=np.zeros_like(s_a), where=aman)
    var = np.divide(s_a2, n_a, out=np.zeros_like(s_a2), where=aman) - rata * rata
    kontras = np.sqrt(np.clip(var, 0, None))

    penuh = (2 * r + 1) ** 2 * ISI_MIN
    bentuk = a.shape
    terbaik = np.full(bentuk, np.inf, np.float32)
    bdy = np.zeros(bentuk, np.int16)
    bdx = np.zeros(bentuk, np.int16)
    jum = np.zeros(bentuk, np.float32)
    cacah = 0

    for dy in range(-CARI, CARI + 1):
        for dx in range(-CARI, CARI + 1):
            bs = np.roll(np.roll(b, -dy, 0), -dx, 1)
            ss = np.roll(np.roll(sb, -dy, 0), -dx, 1)
            ok = sa & ss
            d2 = np.where(ok, (a - np.where(ss, bs, 0.0)) ** 2, 0.0).astype(np.float32)
            n = _jumlah_kotak(ok.astype(np.float32), r)
            s = _jumlah_kotak(d2, r)
            skor = np.divide(s, n, out=np.full_like(s, np.inf), where=n >= penuh)
            jum += np.where(np.isfinite(skor), skor, 0.0).astype(np.float32)
            cacah += 1
            lebih = skor < terbaik
            bdy = np.where(lebih, dy, bdy).astype(np.int16)
            bdx = np.where(lebih, dx, bdx).astype(np.int16)
            terbaik = np.where(lebih, skor, terbaik).astype(np.float32)

    rerata = jum / max(cacah, 1)
    tajam = np.divide(rerata, np.maximum(terbaik, 1e-6),
                      out=np.zeros_like(terbaik), where=terbaik > 0)
    # Geseran yang MENTOK di tepi jendela pencarian dibuang. Yang mentok itu
    # tandanya jawabannya mungkin di luar jendela, bukan tandanya dia pas.
    sah = (np.isfinite(terbaik) & (kontras >= KONTRAS_MIN) & (tajam >= TAJAM_MIN)
           & (np.abs(bdy) < CARI) & (np.abs(bdx) < CARI))
    return bdy, bdx, sah


def medan(bt0: np.ndarray, bt1: np.ndarray, detik: float):
    """Memulangkan u, v m/detik, dan suhu puncak awan di kisi keluaran.

    Yang NaN berarti di situ tidak ada yang bisa dijejak, biasanya langit
    cerah atau awan yang terlalu datar untuk dicocokkan."""
    A, B = _kasarkan(bt0), _kasarkan(bt1)
    bdy, bdx, sah = _jejak(A, B)
    ny, nx = A.shape
    lat = H.LAT1 - np.arange(ny) * DER_KASAR

    # Geseran piksel jadi meter per detik. Bujur menyempit mengikuti lintang,
    # jadi faktornya ikut kosinus. Sumbu y menghadap ke SELATAN, jadi v
    # dibalik tandanya.
    u = (bdx * DER_KASAR * 111320.0 * np.cos(np.radians(lat))[:, None] / detik)
    v = (-bdy * DER_KASAR * 110540.0 / detik)
    u = np.where(sah, u, np.nan).astype(np.float32)
    v = np.where(sah, v, np.nan).astype(np.float32)

    # Suhu puncak awan ikut dirata ratakan, tapi HANYA di sel yang jejaknya
    # sah. Yang menentukan warnanya itu ketinggian awan YANG DIJEJAK, bukan
    # ketinggian rata rata segala awan yang kebetulan ada di kotak itu.
    bt = np.where(sah, A, np.nan).astype(np.float32)

    NY = int(round((H.LAT1 - H.LAT0) / KISI)) + 1
    NX = int(round((H.LON1 - H.LON0) / KISI)) + 1
    k = int(round(KISI / DER_KASAR))
    U = np.full((NY, NX), np.nan, np.float32)
    V = np.full((NY, NX), np.nan, np.float32)
    BT = np.full((NY, NX), np.nan, np.float32)
    for J in range(NY):
        y = int(round(J * KISI / DER_KASAR))
        y0, y1 = max(0, y - k // 2), min(ny, y + k // 2 + 1)
        pu, pv, pb = u[y0:y1], v[y0:y1], bt[y0:y1]
        for I in range(NX):
            x = int(round(I * KISI / DER_KASAR))
            x0, x1 = max(0, x - k // 2), min(nx, x + k // 2 + 1)
            q = pu[:, x0:x1]
            n = int(np.isfinite(q).sum())
            if n >= SUMBER_MIN:
                U[J, I] = np.nanmean(q)
                V[J, I] = np.nanmean(pv[:, x0:x1])
                BT[J, I] = np.nanmean(pb[:, x0:x1])
    return U, V, BT


# Tiga lapis tekanan, tiga warna palet, SAMA PERSIS dengan sebelumnya
# supaya legenda di frontend tidak perlu diubah. Tinggi lime, tengah toska,
# rendah biru. Yang penting dibedakan memang lapisannya, bukan angka
# tekanannya, jadi sengaja bukan gradasi mulus.
LAPIS = [
    ((0, 400), (0xC5, 0xFF, 0x95)),
    ((400, 700), (0x5D, 0xEB, 0xD7)),
    ((700, 1100), (0x4F, 0xA8, 0xD8)),
]

# Ketinggian ditaksir dari SUHU PUNCAK AWAN, cara jendela inframerah yang
# paling sederhana. Angkanya atmosfer baku tropis, 400 hPa kira kira 250 K
# dan 700 hPa kira kira 279 K. Ini taksiran kasar, dan memang cuma dipakai
# memilih satu dari tiga warna, bukan dipajang sebagai angka tekanan.
BT_400 = 250.0
BT_700 = 279.0

# Jarak antar panah. Medannya 0,4 derajat, tapi menggambar panah tiap 0,4
# derajat berarti 49 ribu panah di satu gambar dan itu bubur. Tiap tiga sel
# jadi 1,2 derajat, kira kira 5.500 panah, sepadan dengan versi lama.
PANAH_TIAP = 3


def _lapis_warna(bt: float):
    if not np.isfinite(bt):
        return None
    pres = 300.0 if bt < BT_400 else (550.0 if bt < BT_700 else 850.0)
    for (a, b), w in LAPIS:
        if a <= pres < b:
            return w
    return None


def gambar(U, V, BT, tujuan: Path) -> int:
    """Menggambar panah gerak awan di kisi lintang bujur penuh.

    Panahnya digambar di kisi GAMBAR, yaitu 0,05 derajat, bukan di kisi
    medan, supaya ukuran dan ketajamannya sama dengan versi lama dan
    image_bounds di katalog tidak perlu berubah sama sekali."""
    from PIL import Image, ImageDraw

    lebar = int(round((H.LON1 - H.LON0) / H.DERAJAT)) + 1
    tinggi = int(round((H.LAT1 - H.LAT0) / H.DERAJAT)) + 1
    img = Image.new("RGBA", (lebar, tinggi), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    NY, NX = U.shape
    piksel_per_sel = KISI / H.DERAJAT
    n = 0
    for J in range(0, NY, PANAH_TIAP):
        for I in range(0, NX, PANAH_TIAP):
            u, v = U[J, I], V[J, I]
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            warna = _lapis_warna(BT[J, I])
            if warna is None:
                continue
            x = I * piksel_per_sel
            y = J * piksel_per_sel
            laju = float(np.hypot(u, v))
            if laju < 0.5:
                continue
            # Ekor di titiknya, kepala menunjuk ke arah tujuan. Sumbu y
            # gambar menghadap SELATAN, jadi v dibalik tandanya lagi.
            panjang = 6.0 + min(laju, 60.0) * 0.42
            dx = float(u) / laju * panjang
            dy = -float(v) / laju * panjang
            x2, y2 = x + dx, y + dy
            d.line([(x, y), (x2, y2)], fill=warna + (230,), width=2)
            th = math.atan2(dx, -dy)
            for sisi in (0.42, -0.42):
                d.line([(x2, y2),
                        (x2 - math.sin(th + sisi) * 4.5, y2 + math.cos(th + sisi) * 4.5)],
                       fill=warna + (230,), width=2)
            n += 1
    img.save(tujuan, "WEBP", quality=88, method=4, alpha_quality=100)
    return n
