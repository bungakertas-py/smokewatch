# -*- coding: utf-8 -*-
"""Dua produk satelit untuk Smokewatch, dari Himawari-9.

   KENAPA DI SMOKEWATCH. CAMS itu MODEL, dia menghitung asap seharusnya ada
   di mana. Satelit menunjukkan asapnya SUNGGUH SUNGGUH ada di mana. Waktu
   karhutla, dua duanya bersebelahan itu cerita yang kuat, dan selisih di
   antara keduanya justru yang paling berguna.

   Yang dibuat dua.

   1. DEBU DAN ABU, dari selisih kanal 11, 13, dan 15. Jalan siang dan
      malam.

      NAMANYA SENGAJA BUKAN ASAP. Resep ini memang dipakai untuk debu dan
      abu vulkanik, dan dua duanya penting di Indonesia. Tapi ASAP
      KEBAKARAN hampir tidak terlihat di inframerah termal, butirannya
      terlalu halus untuk mengubah pancaran di 8 sampai 12 mikrometer.
      Yang terlihat jelas itu di cahaya tampak, dan kanal tampaknya
      beresolusi setengah kilometer sehingga satu bingkai saja 272 MB,
      terlalu berat untuk irama sejam. Jadi untuk kebakaran, yang
      diandalkan layer Titik Panas di sebelahnya.
   2. TITIK PANAS, dari kanal 7 di 3,9 mikrometer. Tiap sepuluh menit,
      dibanding VIIRS yang cuma dua kali sehari. Hasilnya INDIKATIF, bukan
      produk kebakaran resmi, dan itu disebut terang terangan di legenda.

   Pembaca HSD-nya salinan dari repo Atmosight, hsd.py di sebelah. Sengaja
   disalin, bukan dibuat pustaka bersama, sebab dua repo ini memang berdiri
   sendiri dan sudah menyalin beberapa hal lain dengan alasan yang sama.
   Kalau formatnya berubah, dua duanya harus disetel."""
from __future__ import annotations

import datetime as dt
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from . import hsd

EMBER = "https://noaa-himawari9.s3.amazonaws.com"

# Domain DISAMAKAN dengan CAMS, dibaca dari katalognya yang sedang hidup.
LON0, LON1, LAT0, LAT1 = 62.0, 180.0, -32.8, 32.8
DERAJAT = 0.05
SEGMEN = range(2, 10)
ZEN_PUDAR_MULAI, ZEN_PUDAR_AKHIR, REDAM_MIN = 78.0, 88.0, 0.33

LANGKAH_MENIT = 60      # sejam, bukan setengah jam. Empat kanal per bingkai,
JUM_BINGKAI = 6         # jadi bobotnya empat kali Atmosight.
VERSI_RENDER = 4

LAYER_ASAP = "debu_abu"
LAYER_API = "titik_panas"

# Ambang titik panas. Angkanya patokan yang lazim dipakai untuk pengindraan
# api geostasioner, dan sengaja KONSERVATIF supaya tidak memberi alarm palsu
# di tanah gersang tengah hari.
#   suhu 3,9 um mutlak harus tinggi, DAN selisihnya terhadap 10,4 um besar.
# Selisih itu yang membedakan api dari permukaan yang sekadar panas, sebab
# api memancar jauh lebih kuat di 3,9 um daripada di inframerah termal.
# Ambangnya BEDA SIANG DAN MALAM, dan itu bukan kehati hatian berlebihan.
# Di 3,9 mikrometer matahari ikut memantul, jadi permukaan yang sama
# terbaca jauh lebih panas tengah hari. Dengan satu ambang untuk sepanjang
# hari, jumlah titik melonjak dua puluh lima kali lipat antara pagi dan
# tengah hari, dan lonjakan itu matahari, bukan api.
API_BT39_SIANG = 322.0
API_BT39_MALAM = 305.0
API_SELISIH_MIN = 12.0
API_SELISIH_KUAT = 25.0


def _ambil(url: str, timeout: int = 300) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def slot_tersedia(batas: int = 300) -> list[dt.datetime]:
    sekarang = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    ada: list[dt.datetime] = []
    for mundur in range(8):
        jam = (sekarang - dt.timedelta(hours=mundur)).replace(minute=0, second=0, microsecond=0)
        pre = "AHI-L1b-FLDK/%s" % jam.strftime("%Y/%m/%d/%H")
        try:
            xml = _ambil("%s/?list-type=2&prefix=%s&delimiter=/" % (EMBER, pre), 60).decode("utf-8", "replace")
        except Exception as e:
            print("  daftar ember gagal untuk %s, %s" % (pre, e))
            continue
        for p in re.findall(r"<Prefix>([^<]+)</Prefix>", xml):
            m = re.search(r"/(\d{4})/$", p)
            if m:
                ada.append(jam.replace(minute=int(m.group(1)[2:])))
    return sorted(set(ada), reverse=True)[:batas]


def pilih_bingkai(slot: list[dt.datetime]) -> list[dt.datetime]:
    pakai = []
    for s in slot:
        if s.minute % LANGKAH_MENIT:
            continue
        pakai.append(s)
        if len(pakai) >= JUM_BINGKAI:
            break
    return pakai


def unduh_band(waktu: dt.datetime, band: int, folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    res = "R05" if band == 3 else "R10" if band in (1, 2, 4) else "R20"
    jalur = []
    for s in SEGMEN:
        nama = "HS_H09_%s_B%02d_FLDK_%s_S%02d10.DAT.bz2" % (
            waktu.strftime("%Y%m%d_%H%M"), band, res, s)
        tujuan = folder / nama
        if not tujuan.exists() or tujuan.stat().st_size < 1000:
            # SEKALI DIULANG. Ember sesekali memutus sambungan di tengah
            # jalan, dan satu segmen yang hilang meninggalkan pita kosong
            # melintang di gambarnya. Sudah kejadian sekali.
            url = "%s/AHI-L1b-FLDK/%s/%s" % (EMBER, waktu.strftime("%Y/%m/%d/%H%M"), nama)
            for coba in (1, 2):
                try:
                    tujuan.write_bytes(_ambil(url))
                    break
                except Exception as e:
                    if coba == 2:
                        print("    B%02d segmen %d gagal, %s" % (band, s, e))
                    else:
                        time.sleep(1.5)
            else:
                continue
        jalur.append(tujuan)
    return jalur


_peta = {}


def _kisi(h: dict):
    """Peta kisi ke piksel satelit, dihitung sekali lalu disimpan."""
    kunci = (h["kolom"], h["coff"], h["cfac"])
    if kunci in _peta:
        return _peta[kunci]
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
    hadap = h["rs"] * np.cos(phi) * np.cos(d)
    tampak = hadap > rl
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
    redam = np.clip((ZEN_PUDAR_AKHIR - zen) / (ZEN_PUDAR_AKHIR - ZEN_PUDAR_MULAI),
                    REDAM_MIN, 1.0).astype(np.float32)
    _peta[kunci] = (kol, bar, sah, LON.shape, redam)
    return _peta[kunci]


def suhu_kisi(jalur: list[Path]):
    """Suhu kecerahan satu kanal di kisi lintang bujur."""
    if not jalur:
        return None, None
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
    hasil[(hasil < 170) | (hasil > 400)] = np.nan
    return hasil, redam


def _regang(a, lo, hi, gamma=1.0):
    v = np.clip((a - lo) / (hi - lo), 0, 1)
    return v ** (1.0 / gamma) if gamma != 1.0 else v


def _zenit_matahari(waktu: dt.datetime):
    """Sudut zenit matahari di tiap sel kisi, derajat.

    Rumus deklinasi dan sudut jam yang biasa. Ketelitiannya cukup, yang
    diperlukan cuma membedakan siang dari malam."""
    hari = waktu.timetuple().tm_yday
    dekl = np.radians(23.44) * np.sin(2 * np.pi * (284 + hari) / 365.0)
    lon = np.arange(LON0, LON1 + DERAJAT / 2, DERAJAT)
    lat = np.arange(LAT1, LAT0 - DERAJAT / 2, -DERAJAT)
    LON, LAT = np.meshgrid(lon, lat)
    jam = waktu.hour + waktu.minute / 60.0
    sudut_jam = np.radians(15.0 * (jam + LON / 15.0 - 12.0))
    kos = (np.sin(np.radians(LAT)) * np.sin(dekl)
           + np.cos(np.radians(LAT)) * np.cos(dekl) * np.cos(sudut_jam))
    return np.degrees(np.arccos(np.clip(kos, -1, 1))).astype(np.float32)


def _kotak_rata(a, r):
    """Rata rata kotak (2r+1) lewat gambar jumlah, numpy saja.

    Dipakai menghitung latar sekitar untuk uji titik panas. Sengaja tidak
    memakai scipy supaya daftar pustaka di GitAction tetap dua baris."""
    b = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    m = np.isfinite(a).astype(np.float64)
    def jml(x):
        c = np.cumsum(np.cumsum(x, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        H, W = x.shape
        y0 = np.clip(np.arange(H) - r, 0, H); y1 = np.clip(np.arange(H) + r + 1, 0, H)
        x0 = np.clip(np.arange(W) - r, 0, W); x1 = np.clip(np.arange(W) + r + 1, 0, W)
        return (c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)]
                - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)])
    n = jml(m)
    return np.where(n > 0, jml(b) / np.maximum(n, 1), np.nan)


def gambar_asap(b11, b13, b15, redam, tujuan: Path) -> None:
    """RGB debu dan asap, resep baku pengindraan geostasioner.

    Merah selisih 12,4 dikurangi 10,4. Hijau selisih 10,4 dikurangi 8,6.
    Biru suhu 10,4 itu sendiri.

    ALFANYA BUKAN DARI KECERAHAN, melainkan dari CIRI AEROSOLNYA. Versi
    pertama memakai kekuatan warna begitu saja dan hasilnya seluruh peta
    tertutup biru pekat, sebab laut hangat yang bersih pun punya warna di
    resep ini. Untuk lapisan yang ditumpangkan di atas peta, yang harus
    kelihatan cuma yang memang berisyarat.

    Cirinya dua dan harus dua duanya. Selisih 12,4 dikurangi 10,4 yang
    POSITIF, itu tanda butiran aerosol, dan suhu 10,4 yang masih HANGAT,
    supaya awan tinggi yang tebal tidak ikut terwarnai."""
    r = _regang(b15 - b13, -4.0, 2.0)
    g = _regang(b13 - b11, 0.0, 15.0, gamma=2.5)
    b = _regang(b13, 261.0, 289.0)
    # Ambangnya dilonggarkan sesudah versi pertama terlalu ketat dan
    # petanya nyaris kosong seluruhnya. Yang dicari isyarat sedang pun
    # ikut terlihat, tapi langit bersih tetap bening.
    ciri_aerosol = np.clip((b15 - b13 + 0.9) / 1.6, 0, 1)
    bukan_awan = np.clip((b13 - 264.0) / 12.0, 0, 1)
    kuat = np.clip(ciri_aerosol * bukan_awan * 1.4, 0, 1)
    rgba = np.zeros(b13.shape + (4,), np.float32)
    rgba[..., 0] = r * 255
    rgba[..., 1] = g * 255
    rgba[..., 2] = b * 255
    rgba[..., 3] = kuat * 225 * redam
    rgba[~np.isfinite(b13)] = 0
    Image.fromarray(rgba.round().astype(np.uint8), "RGBA").save(
        tujuan, "WEBP", quality=80, method=4, alpha_quality=100)


def gambar_api(b07, b13, redam, tujuan: Path, waktu: dt.datetime | None = None) -> int:
    """Titik panas, uji BERLATAR, bukan ambang mutlak saja.

    Versi pertama cuma memakai ambang mutlak dan hasilnya gurun Australia
    tengah hari penuh titik palsu. Permukaan kering yang kepanasan memang
    bisa melewati ambang 3,9 mikrometer mana pun yang masuk akal.

    Yang membedakan api dari tanah panas itu KONTRASNYA TERHADAP SEKITAR.
    Api itu sumber kecil yang jauh lebih panas daripada tetangganya,
    sedangkan gurun panas itu hangat merata berkilo kilometer. Jadi tiap
    piksel diadu dengan rata rata latar di sekitarnya, dan yang lolos cuma
    yang menonjol di dua duanya, suhu 3,9 um dan selisih 3,9 dikurangi
    10,4.

    Hasilnya tetap INDIKATIF. Ini bukan produk kebakaran resmi, ambangnya
    kami tetapkan sendiri, dan itu disebut di keterangan layernya."""
    selisih = b07 - b13
    latar_b07 = _kotak_rata(b07, 12)
    latar_sel = _kotak_rata(selisih, 12)
    menonjol = ((b07 - latar_b07) >= 8.0) & ((selisih - latar_sel) >= 6.0)
    if waktu is not None:
        siang = _zenit_matahari(waktu) < 85.0
        ambang = np.where(siang, API_BT39_SIANG, API_BT39_MALAM)
    else:
        ambang = API_BT39_SIANG
    masuk_akal = np.isfinite(selisih) & (b07 >= ambang) & (selisih >= API_SELISIH_MIN)
    panas = masuk_akal & menonjol
    # Yang sangat panas DAN sangat menonjol ditandai lebih kuat.
    kuat = panas & (selisih >= API_SELISIH_KUAT) & ((b07 - latar_b07) >= 15.0)
    rgba = np.zeros(b13.shape + (4,), np.uint8)

    def tebalkan(m, r):
        out = m.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                out |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        return out

    # Satu piksel satelit itu satu piksel kisi, dan di peta se-Indonesia
    # titik sebesar itu tidak akan terlihat. Digemukkan jadi cakram kecil.
    rgba[tebalkan(panas & ~kuat, 2)] = (0xF5, 0xA9, 0x1E, 225)
    rgba[tebalkan(kuat, 3)] = (0xE4, 0x23, 0x20, 240)
    Image.fromarray(rgba, "RGBA").save(tujuan, "WEBP", quality=90, method=4, alpha_quality=100)
    return int(panas.sum())


def image_bounds() -> list:
    s = DERAJAT / 2.0
    return [LON0 - s, LAT0 - s, LON1 + s, LAT1 + s]
