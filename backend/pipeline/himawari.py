# -*- coding: utf-8 -*-
"""Suhu puncak awan dari Himawari-9, kanal 13.

   KENAPA INI ADA. Semua layer Smokewatch yang lain isinya MODEL, CAMS
   menghitung polutan seharusnya ada di mana. Satelit menunjukkan keadaan
   yang sedang terjadi, jadi modelnya bisa diadu dengan pengamatan. Kanal
   13 yang paling berguna sebab suhu puncak awan itu penanda langsung
   kedalaman konveksi, dan konveksi itu yang mengaduk polutan ke atas.

   BERKASNYA SALINAN DARI REPO ATMOSIGHT, dan sengaja dibuat sama persis
   supaya kedua app menampilkan parameter satelit yang identik. Kalau satu
   disetel, yang lain ikut disetel. Yang beda cuma dua paragraf pembuka
   ini dan di himawari_run.py.

   SUMBERNYA ember terbuka NOAA di AWS, bukan JAXA P-Tree. Dua duanya punya
   datanya, tapi P-Tree melarang mengedarkan ulang ke pihak ketiga sedangkan
   menayangkannya di situs publik itu pengedaran ulang. Ember NOAA tidak
   punya batasan itu dan tidak perlu akun sama sekali.

   TIDAK MEMAKAI SATPY. Yang diperlukan cuma satu kanal inframerah dan satu
   proyeksi geostasioner yang rumusnya tertutup, sedangkan satpy menyeret
   pyresample, xarray, dan dask. Di GitAction itu tiga menit pasang dan
   ratusan megabita, untuk pekerjaan yang muat di satu berkas.

   Pembaca HSD-nya ada di hsd.py, sebelah."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from . import hsd

EMBER = "https://noaa-himawari9.s3.amazonaws.com"
SAT = "h09"

# Domainnya DISAMAKAN dengan GFS dan CAMS, 62BT sampai 180BT, 33LS sampai
# 33LU. Pas sekali kebetulan, piringan Himawari mulai penuh di 61,9BT di
# khatulistiwa. Yang tidak kebagian cuma pojok barat laut dan barat daya,
# sebab tepi piringan itu melengkung, dan di situ sudutnya memang terlalu
# miring untuk dipakai.
LON0, LON1, LAT0, LAT1 = 62.0, 180.0, -33.0, 33.0
DERAJAT = 0.05                 # 5,5 km, kira kira dua kali resolusi asli 2 km

# TEPI BARAT. Versi pertama memotong keras di sudut zenit 80 derajat, dan itu
# SALAH. Yang terjadi, data berhenti di 69,3BT di khatulistiwa dan 73,0BT di
# lintang 33, padahal domain kita mulai di 62,0BT. Jadi sepuluh derajat paling
# barat hilang bukan karena satelitnya tidak melihat, tapi karena kupotong
# sendiri.
#
# Piringan Himawari-9 sebenarnya sampai 59,5BT di khatulistiwa dan 61,1BT di
# lintang 33, jadi SELURUH domain kita sebetulnya terlihat.
#
# Sekarang yang dipakai batas piringan sungguhan, dan mutunya yang turun di
# tepi disampaikan lewat ALFA YANG MEMUDAR, bukan lewat potongan. Di sudut 80
# derajat satu piksel sudah memanjang enam kali, di 85 derajat sebelas kali,
# jadi pembaca memang perlu tahu bahwa yang di tepi itu lebih kasar. Tapi
# memudar itu memberi tahu, memotong itu menyembunyikan.
ZEN_PUDAR_MULAI = 78.0         # sampai sini mutunya masih penuh
ZEN_PUDAR_AKHIR = 88.0         # di luar ini tinggal sisa paling redup
REDAM_MIN = 0.33

# Segmen piringan yang menyentuh pita lintang kita. Dihitung dari geometri,
# bukan ditebak, lalu dipatok supaya tidak perlu mengunduh sepuluh duanya.
# Lintang 33LU jatuh di baris 1076 dan 33LS di baris 4425, dari 5500 baris.
SEGMEN = range(2, 10)

# CAP VERSI GAMBAR. Dinaikkan tiap kali cara menggambarnya berubah, misalnya
# skala warna atau perlakuan tepi. Bingkai yang capnya beda DIGAMBAR ULANG
# walaupun berkasnya sudah ada.
#
# Tanpa ini perbaikan render tidak pernah sampai ke bingkai lama. Pemulihan
# dari cache membawa berkas lama, pemeriksa melihat berkasnya ada lalu
# melewatinya, dan yang terbit tetap yang lama. Sudah kejadian tepat sesudah
# tepi barat dibetulkan.
#   1  potongan keras di zenit 80
#   2  tepi dipudarkan sampai batas piringan
#   3  layer angin gerak awan ditambahkan
#   4  gerak awan tidak lagi dari produk AMV NOAA, dijejak sendiri dari
#      dua citra berurutan, jadi domainnya penuh sampai 62 BT
#   5  skala puncak awan jadi inferno sepuluh tingkat, -70 sampai -10
#   6  batas hangatnya naik ke -5, jadi sebelas tingkat
#   7  -5 jadi tingkat berwarna terakhir, beningnya ke -3, dan
#      tingkatnya dipangkas dari dua belas jadi TUJUH
VERSI_RENDER = 7

# IRAMA SEPULUH MENIT, DUA HARI KE BELAKANG. Diputuskan pemilik 21 Oktober
# 2026, menggantikan irama satu bingkai per hari yang berlaku sejak 19
# Oktober.
#
# Sepuluh menit itu irama asli Himawari, bukan angka pilihan kita. AHI
# memindai sepiringan bumi sekali tiap sepuluh menit, 144 kali sehari, dan
# sudah diperiksa langsung ke ember: pada 20 Oktober 2026 ada 142 slot,
# yang tidak ada cuma 02:40 dan 14:40 UTC sebab satelitnya membuang momentum
# roda reaksi dan berhenti memindai bumi.
#
# DUA HARI, BUKAN TIGA, dan itu batas ukuran bukan selera. Satu slot
# menghasilkan sekitar 0,9 MB untuk dua layer inframerah. Dua hari berarti
# 288 slot, tiga hari 432, dan GitHub Pages punya batas lunak 1 GB untuk
# SELURUH situs yang sudah berisi GFS dan ECMWF juga.
JEDA_MENIT = 10
JUM_HARI = 2

# Jarak aman dari jam sekarang sebelum sebuah slot boleh diambil. Lihat
# deret_slot() untuk angka ukurnya.
LATENSI_AMAN_MENIT = 20

# Jam patokan LAMA, dipertahankan CUMA untuk citra warna asli, lihat
# slot_tampak(). Angkanya dihitung, bukan dipilih sekenanya: pada 04:00 UTC
# atau 11:00 WIB, SELURUH domain kita kena matahari dari 62 sampai 180 BT,
# dan matahari paling tinggi di atas Indonesia dengan zenit rata rata 15,5
# derajat. Sejam sebelumnya 99,9 persen domain, sejam sesudahnya 99,6
# persen, lalu turun cepat, jam 09:00 WIB tinggal 92 persen dan jam 16:00
# WIB tinggal 53 persen.
JAM_PATOKAN = 4
MENIT_PATOKAN = 0

# Gerak awan tetap dijejak dari pasangan berjarak 30 MENIT, bukan dari dua
# hari berurutan. Dalam sehari awannya sudah berganti seluruhnya, jadi
# menjejak antar hari tidak berarti apa apa.
#
# Jaraknya juga TIDAK BOLEH dirapatkan jadi 10 menit sekadar supaya hemat.
# Di kisi yang sudah dikasarkan, satu piksel itu 11,1 km. Dalam 30 menit itu
# 6,2 m/detik per piksel, dalam 10 menit jadi 18,5 m/detik per piksel, dan
# pada kekasaran segitu hampir semua angin tropis jatuh di satu dua piksel
# yang sama.
JEDA_GERAK_MENIT = 30

LAYER = "cloudtop_temp"

# Skala derajat Celsius, INFERNO, TUJUH tingkat dari -70 sampai -5, lalu
# bening di -3.
#
# Jumlah tingkatnya diputuskan pemilik sesudah mencoba dua belas dan merasa
# terlalu ramai. Tujuh itu masih cukup untuk membedakan inti dari tepi, dan
# tiap selnya jadi cukup lebar untuk dibaca di legenda.
#
# JARAKNYA 11 DERAJAT, bukan angka bulat, dan itu akibat dua ujungnya yang
# dipatok. Pemilik memilih -70 sebagai yang terdingin dan -5 sebagai tingkat
# berwarna terakhir, dan membaginya jadi tujuh memberi 10,8 derajat per
# langkah yang dibulatkan jadi 11. Kalau suatu saat angka bulat lebih
# penting daripada ujungnya, -65 sampai -5 memberi langkah 10 persis, dan
# yang hilang cuma label -70 sebab yang lebih dingin dari -65 toh dipatok ke
# warna yang sama.
#
# RENTANGNYA DIPILIH DARI DATA. Percobaan pertama -90 sampai -70, dan itu
# diukur dulu pada tiga hari. Tidak ada satu pun piksel yang lebih dingin
# dari -90, dan yang lebih dingin dari -80 cuma 0,03 persen di hari terbaik.
# Delapan dari sepuluh tingkatnya tidak akan pernah terpakai dan yang tampil
# cuma 0,08 persen domain. Pemilik mengoreksi sendiri sesudah melihat
# angkanya, lalu batas hangatnya digeser dua kali sampai -5.
#
# Paling dingin ungu tua, naik lewat magenta dan merah ke oranye dan kuning,
# lalu memudar ke bening antara -5 dan -3 sehingga tepinya halus.
#
# ANGKA DI SINI WAJIB SAMA dengan legenda cloudtop_temp di app.js.
SKALA = [
    (-70, (0x36, 0x09, 0x61, 252)),
    (-59, (0x69, 0x16, 0x6e, 248)),
    (-48, (0x95, 0x28, 0x67, 244)),
    (-37, (0xc4, 0x43, 0x4f, 240)),
    (-26, (0xe8, 0x69, 0x2c, 234)),
    (-15, (0xfb, 0xa1, 0x0e, 228)),
    (-5, (0xfa, 0xde, 0x72, 220)),
    (-3, (0xfd, 0xf2, 0xcc, 0)),
]


def _ambil(url: str, timeout: int = 180) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


# _slot_ada() DIBUANG 21 Oktober 2026 bersama irama harian. Dia mendaftar
# ember untuk memastikan satu slot sudah mendarat, dan itu masuk akal waktu
# slotnya cuma lima sehari. Pada irama sepuluh menit daftarnya 288 slot, dan
# mendaftarkan tiap satunya berarti 288 permintaan cuma untuk menyusun
# rencana. Sekarang slot yang tidak ada gugur sendiri waktu unduhannya
# dicoba, lihat unduh().
#
# SATU JEBAKAN DARI SANA DISIMPAN DI SINI, sebab dia akan menggigit lagi
# kalau suatu saat ada yang mendaftar ember: awalannya berhenti di JAM dan
# TANPA garis miring. Kunci aslinya menyatukan jam dan menit,
# ".../18/0400/...", jadi awalan ".../18/04/" tidak pernah cocok dan
# daftarnya selalu kosong. Sudah kena dua kali, sekali memotong teks, sekali
# menambah garis miring.


def _bulat_slot(w: dt.datetime) -> dt.datetime:
    """Turunkan ke kelipatan JEDA_MENIT terdekat ke bawah."""
    return w.replace(minute=w.minute - w.minute % JEDA_MENIT, second=0, microsecond=0)


def deret_slot(hari: int = JUM_HARI) -> list[dt.datetime]:
    """Semua slot sepuluh menitan dalam `hari` terakhir, TANPA menyentuh ember.

    Dipisah dari pemeriksaan ketersediaan dengan sengaja. Daftarnya 288
    slot untuk dua hari, dan mendaftarkan tiap satunya ke ember berarti 288
    permintaan HTTP cuma untuk menyusun rencana. Yang dipakai gelung utama
    cukup rencana ini, dan slot yang ternyata tidak ada akan gugur sendiri
    waktu unduhannya dicoba.

    Slot paling muda yang diikutkan adalah LIMA BELAS MENIT ke belakang,
    bukan jam sekarang. Diukur 21 Oktober 2026, berkas pertama satu slot
    mendarat di ember 10,7 menit sesudah jam pengamatan dan yang terakhir
    14,1 menit. Jadi ada jendela sekitar tiga setengah menit ketika folder
    slotnya sudah ada tapi SEBAGIAN SEGMENNYA BELUM. Slot yang diambil di
    jendela itu menghasilkan citra berpita kosong tanpa galat apa pun,
    sebab unduh() melewati segmen yang gagal."""
    sekarang = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    muda = _bulat_slot(sekarang - dt.timedelta(minutes=LATENSI_AMAN_MENIT))
    n = hari * 24 * 60 // JEDA_MENIT
    return [muda - dt.timedelta(minutes=JEDA_MENIT * k) for k in range(n)]


def slot_tampak(hari: int = JUM_HARI) -> list[dt.datetime]:
    """Slot untuk citra warna asli, JAUH LEBIH JARANG daripada inframerah.

    Ini bukan kompromi gaya, ini batas yang terukur. Satu slot kanal tampak
    menuntut unduhan 588 MB, sebab kanal 3 sendiri 327 MB di resolusi 0,5
    km, sedangkan satu slot inframerah cuma 19 MB. Pada irama sepuluh menit
    selama dua hari itu 169 GB, dan runner GitHub tidak punya disk maupun
    waktu sebanyak itu.

    Lagi pula kanal tampak BUTUH MATAHARI. Di luar jam siang dia hitam,
    jadi sebagian besar slot sepuluh menitan cuma akan jadi bingkai gelap
    yang tidak memberi tahu apa apa.

    Jadi satu bingkai per hari di JAM_PATOKAN, jam yang seluruh domainnya
    terang penuh."""
    sekarang = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    ada = []
    for mundur in range(hari + 2):
        w = (sekarang - dt.timedelta(days=mundur)).replace(
            hour=JAM_PATOKAN, minute=MENIT_PATOKAN, second=0, microsecond=0)
        if w <= sekarang - dt.timedelta(minutes=LATENSI_AMAN_MENIT):
            ada.append(w)
        if len(ada) >= hari:
            break
    return ada


def unduh(waktu: dt.datetime, folder: Path) -> list[Path]:
    """Segmen satu slot. Daftar KOSONG kalau ada satu saja yang tidak utuh.

    SEGMEN YANG HILANG MEMBATALKAN SLOTNYA, tidak lagi dilewati. Dulu
    segmen gagal cuma dicetak lalu `continue`, dan citranya tetap digambar
    dari segmen yang ada. Hasilnya pita mendatar kosong selebar sepersepuluh
    piringan, tanpa galat, tanpa tanda apa apa di katalog.

    Dulu itu hampir tidak pernah kejadian sebab irama kita harian dan alur
    kerjanya jalan lebih dari satu jam sesudah slotnya. Sejak irama jadi
    sepuluh menit, slot termuda cuma dua puluh menit dari jam sekarang,
    sedangkan berkas terakhir satu slot baru mendarat 14,1 menit sesudah
    pengamatan. Jaraknya jadi tipis, dan slot setengah jadi bukan lagi
    kemungkinan yang jauh.

    Bingkai yang tidak digambar itu jujur, bingkai berlubang itu tidak."""
    folder.mkdir(parents=True, exist_ok=True)
    jalur = []
    for s in SEGMEN:
        nama = "HS_H09_%s_B13_FLDK_R20_S%02d10.DAT.bz2" % (waktu.strftime("%Y%m%d_%H%M"), s)
        tujuan = folder / nama
        if not tujuan.exists() or tujuan.stat().st_size < 1000:
            url = "%s/AHI-L1b-FLDK/%s/%s" % (EMBER, waktu.strftime("%Y/%m/%d/%H%M"), nama)
            try:
                tujuan.write_bytes(_ambil(url))
            except Exception as e:
                print("    segmen %d gagal, slot %s dibatalkan, %s"
                      % (s, waktu.strftime("%d %b %H:%M"), str(e)[:60]))
                return []
        jalur.append(tujuan)
    return jalur


def _kisi(h: dict):
    """Peta sekali pakai dari kisi lintang bujur ke piksel satelit."""
    lon = np.arange(LON0, LON1 + DERAJAT / 2, DERAJAT)
    lat = np.arange(LAT1, LAT0 - DERAJAT / 2, -DERAJAT)
    LON, LAT = np.meshgrid(lon, lat)
    rasio = h["rasio_pol"]
    d = np.radians(LON - h["sub_lon"])
    phi = np.arctan(rasio * np.tan(np.radians(LAT)))
    rl = h["rpol"] / np.sqrt(1.0 - (1.0 - rasio) * np.cos(phi) ** 2)
    r1 = h["rs"] - rl * np.cos(phi) * np.cos(d)
    r2 = -rl * np.cos(phi) * np.sin(d)
    r3 = rl * np.sin(phi)
    rn = np.sqrt(r1 * r1 + r2 * r2 + r3 * r3)
    # Tampak kalau titiknya menghadap satelit, uji hasil kali titik, bukan
    # tebakan sudut, jadi tepi piringannya persis.
    hadap = h["rs"] * np.cos(phi) * np.cos(d)
    tampak = hadap > rl
    # Sudut zenit satelit, dipakai MEMUDARKAN tepi, bukan memotongnya.
    kos = np.clip((hadap - rl) / np.where(rn > 0, rn, 1), -1, 1)
    zen = np.degrees(np.arccos(kos))
    x = np.degrees(np.arctan2(-r2, r1))
    y = np.degrees(np.arcsin(np.clip(-r3 / rn, -1, 1)))
    kol = np.rint(h["coff"] + x * (2.0 ** -16) * h["cfac"]).astype(np.int32)
    bar = np.rint(h["loff"] + y * (2.0 ** -16) * h["lfac"]).astype(np.int32)
    # Batas barisnya DARI HEADER, bukan angka mati. Piringan itu bujur
    # sangkar, jadi jumlah barisnya sama dengan jumlah kolomnya. Dulu di
    # sini tertulis 5500, benar untuk kanal inframerah 2 km, tapi kanal
    # tampak punya 11000 baris di 1 km dan 22000 di 0,5 km, dan angka mati
    # itu memotong separuh sampai tiga perempat piringannya. Ketahuan waktu
    # kanal tampak pertama kali digrid, isinya cuma 50 dan 9 persen.
    sah = tampak & (kol >= 0) & (kol < h["kolom"]) & (bar >= 0) & (bar < h["kolom"])
    redam = np.clip((ZEN_PUDAR_AKHIR - zen) / (ZEN_PUDAR_AKHIR - ZEN_PUDAR_MULAI), REDAM_MIN, 1.0)
    return kol, bar, sah, LON.shape, redam.astype(np.float32)


def bingkai(jalur: list[Path]) -> np.ndarray | None:
    """Suhu kecerahan kelvin di kisi lintang bujur, NaN di luar jangkauan."""
    if not jalur:
        return None
    h0 = hsd.baca(str(jalur[0]))
    kol, bar, sah, bentuk, redam = _kisi(h0)
    hasil = np.full(bentuk, np.nan, np.float32)
    for p in jalur:
        h = hsd.baca(str(p))
        tb = hsd.suhu_kecerahan(h).astype(np.float32)
        b0 = h["baris_awal"] - 1
        pilih = sah & (bar >= b0) & (bar < b0 + h["baris"])
        if pilih.any():
            hasil[pilih] = tb[bar[pilih] - b0, kol[pilih]]
    # suhu di luar nalar dibuang, bukan dipaksa masuk skala
    hasil[(hasil < 170) | (hasil > 340)] = np.nan
    return hasil, redam


def warnai(celsius: np.ndarray) -> np.ndarray:
    henti = np.array([s[0] for s in SKALA], float)
    warna = np.array([s[1] for s in SKALA], float)
    out = np.zeros(celsius.shape + (4,), np.float32)
    for c in range(4):
        out[..., c] = np.interp(celsius, henti, warna[:, c])
    out[~np.isfinite(celsius)] = 0
    return out.round().astype(np.uint8)


def simpan(kelvin: np.ndarray, tujuan: Path, redam: np.ndarray | None = None) -> None:
    rgba = warnai(kelvin - 273.15)
    if redam is not None:
        rgba[..., 3] = (rgba[..., 3] * redam).round().astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA")
    img.save(tujuan, "WEBP", quality=78, method=4, alpha_quality=100)


def image_bounds() -> list:
    """Kotak TEPI gambar, bukan pusat sel. Leaflet menarik tepi gambar ke
    kotak yang diberikan, jadi memberi pusat sel menggeser seluruh medan
    setengah sel. Jebakan yang sama sudah pernah kena di GFS."""
    s = DERAJAT / 2.0
    return [LON0 - s, LAT0 - s, LON1 + s, LAT1 + s]
