# -*- coding: utf-8 -*-
"""Menjalankan sumber data SATELIT, folder dan katalognya sendiri.

   SENGAJA tidak menyisip ke katalog CAMS. Satelit itu sumber tersendiri,
   domainnya sendiri, waktunya sendiri, dan parameternya sendiri. Kalau
   ditumpangkan ke CAMS, garis waktu ramalan dan garis waktu pengamatan
   berebut satu penggeser, dan pembaca yang menggeser ke besok mendapat
   peta kosong. Di frontend dia muncul sebagai pilihan di panel Model dan
   Satelit, bukan sebagai parameter tambahan CAMS.

   BERKASNYA SALINAN DARI REPO ATMOSIGHT, lihat catatan di himawari.py.

   Bingkai yang sudah pernah dibuat TIDAK diulang. Alur kerjanya memulihkan
   keluaran lama dari cache, jadi biasanya tiap jalan cuma satu atau dua
   bingkai baru yang perlu diunduh."""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.pipeline import himawari as H
from backend.pipeline import himawari_gerak as G
from backend.pipeline import himawari_tampak as T

AKAR = Path(__file__).resolve().parents[1]
# Folder keluaran boleh ditimpa lewat argumen. Di repo ini data/output itu
# symlink WSL yang tidak bisa dibaca python Windows, jadi uji lokal perlu
# jalan lain. Lewat ARGUMEN, bukan lingkungan, sebab proses Windows yang
# dijalankan dari WSL tidak mewarisi variabel lingkungan begitu saja.
# Di GitAction dua duanya dibiarkan kosong dan jalurnya kembali ke bawaan.
KELUARAN = Path(sys.argv[1]) if len(sys.argv) > 1 else AKAR / "data" / "output" / "himawari"
SEMENTARA = Path(sys.argv[2]) if len(sys.argv) > 2 else AKAR / "data" / "tmp_himawari"


def nama_bingkai(w: dt.datetime) -> str:
    return "%s_%s_preview.webp" % (H.LAYER, w.strftime("%Y%m%d_%H%M"))


def nama_tampak(w: dt.datetime) -> str:
    return "%s_%s_preview.webp" % (T.LAYER, w.strftime("%Y%m%d_%H%M"))


def nama_angin(w: dt.datetime) -> str:
    return "%s_%s_preview.webp" % (G.LAYER, w.strftime("%Y%m%d_%H%M"))


def versi_lama() -> int:
    """Cap versi gambar yang tersimpan di katalog, 0 kalau belum ada."""
    kat = KELUARAN / "catalog.json"
    if not kat.exists():
        return 0
    try:
        return int(json.loads(kat.read_text("utf-8")).get("versi_render", 0))
    except Exception:
        return 0


def main() -> int:
    KELUARAN.mkdir(parents=True, exist_ok=True)
    gambar_ulang = versi_lama() != H.VERSI_RENDER
    if gambar_ulang:
        print("cap versi gambar berubah, %d jadi %d, semua bingkai digambar ulang"
              % (versi_lama(), H.VERSI_RENDER))
    urut = sorted(H.deret_slot())
    if not urut:
        print("deret slot kosong, berhenti tanpa mengubah apa apa")
        return 0
    print("irama           : tiap %d menit, %d hari, %d slot"
          % (H.JEDA_MENIT, H.JUM_HARI, len(urut)))
    print("rentang         : %s sampai %s UTC"
          % (urut[0].strftime("%d %b %H:%M"), urut[-1].strftime("%d %b %H:%M")))


    import numpy as np

    # SIMPANAN KISI SEBENTAR, dan ini memangkas setengah unduhan.
    #
    # Penjejakan gerak butuh kisi dari 30 menit sebelumnya. Pada irama
    # harian pasangan itu memang slot asing yang harus diunduh sendiri.
    # Pada irama sepuluh menit dia SLOT TETANGGA, tiga langkah ke belakang
    # di deret yang sama, dan gelung ini berjalan maju jadi kisinya baru
    # saja dipegang.
    #
    # Tanpa simpanan, tiap slot mengunduh dua kali, dirinya dan
    # pasangannya, yaitu 38 MB bukan 19 MB. Untuk 288 slot itu 5,5 GB jadi
    # 11 GB, dan waktunya ikut dua kali lipat.
    #
    # Isinya dibatasi empat, cukup menjangkau 30 menit ke belakang pada
    # irama sepuluh menit, dan satu kisi itu sekitar 12 MB jadi simpanannya
    # tidak pernah lewat 50 MB.
    simpanan: dict = {}

    def kisi_slot(w, sebab):
        """Kisi suhu satu slot, dari simpanan kalau ada, kalau tidak diunduh."""
        if w in simpanan:
            return simpanan[w]
        print("  %s mengunduh %d segmen, %s"
              % (w.strftime("%d %b %H:%M"), len(H.SEGMEN), sebab))
        folder = SEMENTARA / w.strftime("%Y%m%d_%H%M")
        try:
            hasil = H.bingkai(H.unduh(w, folder))
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        hasil = hasil if hasil else (None, None)
        simpanan[w] = hasil
        for tua in sorted(simpanan)[:-4]:
            del simpanan[tua]
        return hasil

    # JATAH SLOT BARU PER JALAN. Tanpa ini pengisian pertama tidak akan
    # pernah selesai, dia cuma akan kena batas enam jam GitHub Actions lalu
    # mati di tengah dan mengulang dari awal di jalan berikutnya.
    #
    # Terukur dari jalan nyata 21 Oktober 2026, 18 bingkai selesai dalam
    # 13,7 menit, yaitu 0,81 menit per slot. Dua hari berarti 288 slot,
    # yaitu sekitar 3,9 jam, jadi SEBENARNYA MUAT dalam satu jalan.
    #
    # Angka 1,6 menit yang sempat dipakai di sini diukur SEBELUM simpanan
    # kisi dipasang, waktu tiap slot masih mengunduh dua kali.
    #
    # Jatahnya tetap ada, dan bukan karena waktunya kurang. Cache keluaran
    # baru disimpan SESUDAH masak selesai, jadi satu jalan yang gugur di
    # jam keempat membuang seluruh kemajuannya. Dengan jatah, tiap jalan
    # menyimpan apa yang sudah jadi, dan jalan berikutnya melanjutkan.
    #
    # 150 memberi dua jalan untuk mengisi dua hari penuh, masing masing
    # sekitar dua jam, dengan margin lebar ke batas enam jam. Untuk mengisi
    # sekali jalan, isi 288 di kotak isian workflow_dispatch.
    #
    # YANG TERBARU DIKERJAKAN DULU, sebab bingkai hari ini lebih berguna
    # daripada bingkai kemarin lusa. Gelungnya sendiri tetap urut maju
    # supaya pasangan penjejak gerak selalu sudah ada waktu dibutuhkan.
    JATAH = int(os.environ.get("HIMAWARI_JATAH") or 150)

    def _kurang(nama):
        f = KELUARAN / nama
        return not f.exists() or f.stat().st_size < 1000

    # PANAHNYA IKUT DIHITUNG, bukan cuma gambarnya. Kalau yang dilihat cuma
    # nama_bingkai, slot yang gambarnya sudah jadi tapi panahnya belum tidak
    # akan pernah masuk jatah, jadi tidak pernah dikerjakan, dan panah itu
    # hilang selamanya tanpa ada yang memberi tahu. Gampang terjadi sebab
    # penjejakan gerak butuh slot pasangan 30 menit sebelumnya, dan pasangan
    # itu bisa saja belum ada waktu gambarnya dibuat.
    perlu = [w for w in urut
             if gambar_ulang or _kurang(nama_bingkai(w)) or _kurang(nama_angin(w))]
    if len(perlu) > JATAH:
        pilih = set(sorted(perlu, reverse=True)[:JATAH])
        print("slot baru       : %d, dikerjakan %d yang termuda dulu, "
              "sisanya jalan berikutnya" % (len(perlu), JATAH))
    else:
        pilih = set(perlu)
        if perlu:
            print("slot baru       : %d, semuanya dikerjakan sekarang" % len(perlu))
        else:
            print("slot baru       : tidak ada, semua sudah tergambar")

    bingkai_jadi = []
    angin_jadi = []
    for w in urut:
        tag = w.strftime("%d %b %H:%M")
        berkas = KELUARAN / nama_bingkai(w)
        sudah = berkas.exists() and berkas.stat().st_size > 1000 and not gambar_ulang
        fa = KELUARAN / nama_angin(w)
        perlu_angin = gambar_ulang or not fa.exists() or fa.stat().st_size < 1000

        arr = None
        if sudah and not perlu_angin:
            bingkai_jadi.append(w)
            angin_jadi.append(w)
            continue
        # Di luar jatah jalan ini. Gambar lamanya, kalau ada, TETAP DIAKUI
        # sah supaya pembersih di bawah tidak menghapusnya.
        if w not in pilih:
            if sudah:
                bingkai_jadi.append(w)
                if not perlu_angin:
                    angin_jadi.append(w)
            continue

        sebab = "gambarnya belum ada" if not sudah else "kisinya dipakai menjejak gerak"
        arr, redam = kisi_slot(w, sebab)
        terisi = float(100.0 * np.isfinite(arr).mean()) if arr is not None else 0.0
        if arr is None or terisi < 40:
            print("    gagal atau cacat, terisi %.0f persen" % terisi)
            # GAMBAR LAMA TETAP DIAKUI SAH. Slot ini kadang diunduh ulang
            # cuma untuk menjejak gerak, padahal gambarnya sudah tersimpan.
            # Kalau unduhan ulang itu gagal lalu harinya ikut dicoret, nama
            # berkasnya hilang dari daftar sah dan pembersih di bawah
            # MENGHAPUS gambar yang sebenarnya masih baik.
            if sudah:
                bingkai_jadi.append(w)
            continue
        if not sudah:
            H.simpan(arr, berkas, redam)
            print("    jadi, %.0f persen terisi, %.0f KB"
                  % (terisi, berkas.stat().st_size / 1024))
        bingkai_jadi.append(w)
        if not perlu_angin:
            angin_jadi.append(w)
            continue

        # ---- GERAK AWAN, dijejak dari pasangan BERJARAK 30 MENIT ----
        # Pasangannya slot setengah jam SEBELUM patokan, bukan hari
        # sebelumnya. Dalam sehari awannya sudah berganti seluruhnya, jadi
        # menjejak antar hari tidak berarti apa apa.
        #
        # Medannya dicap jam patokan, sebab artinya "gerak awan sampai jam
        # ini", dan dengan begitu dia sejajar waktu dengan citra puncak
        # awan yang jadi alasnya di peta.
        lalu_w = w - dt.timedelta(minutes=H.JEDA_GERAK_MENIT)
        lalu_arr, _ = kisi_slot(lalu_w, "pasangan untuk menjejak gerak")
        if lalu_arr is None:
            print("    pasangan gerak tidak ada, %s tanpa panah" % tag)
            continue
        try:
            U, V, BT = G.medan(lalu_arr, arr, H.JEDA_GERAK_MENIT * 60.0)
            n = G.gambar(U, V, BT, fa)
            print("    gerak awan %s, %d panah, %.0f KB"
                  % (tag, n, fa.stat().st_size / 1024))
            angin_jadi.append(w)
        except Exception as e:
            print("    gerak awan %s gagal, %s" % (tag, e))

    # ---- CITRA WARNA ASLI, gelung dan unduhan sendiri ----
    # Sengaja dipisah dari gelung di atas. Kanal tampak itu 535 MB per slot
    # lawan 25 MB punya inframerah, dan kalau unduhannya putus di tengah,
    # yang hilang cukup gambarnya saja. Suhu kecerahan dan gerak awan sudah
    # terbit duluan dan tidak ikut gagal.
    tampak_jadi = []
    for w in sorted(H.slot_tampak()):
        ft = KELUARAN / nama_tampak(w)
        if ft.exists() and ft.stat().st_size > 1000 and not gambar_ulang:
            tampak_jadi.append(w)
            continue
        folder = SEMENTARA / ("tampak_" + w.strftime("%Y%m%d_%H%M"))
        try:
            alb, redam = {}, None
            for band in sorted(T.KANAL):
                jalur = T.unduh(w, band, folder)
                a, rd = T.kisi(jalur)
                if a is None:
                    raise RuntimeError("kanal %d kosong" % band)
                alb[band] = a
                redam = rd
            siang = T.gambar(alb, redam, w, ft)
            print("  citra tampak %s, siang %.0f%% domain, %.0f KB"
                  % (w.strftime("%d %b"), siang, ft.stat().st_size / 1024))
            tampak_jadi.append(w)
        except Exception as e:
            print("  citra tampak %s gagal, %s" % (w.strftime("%d %b"), e))
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    if not bingkai_jadi:
        print("tidak ada bingkai yang jadi, katalog tidak disentuh")
        return 0

    # bingkai lama dibuang, simpanan tidak boleh tumbuh tanpa batas
    sah = ({nama_bingkai(w) for w in bingkai_jadi} | {nama_angin(w) for w in angin_jadi}
           | {nama_tampak(w) for w in tampak_jadi})
    for f in KELUARAN.glob("*_preview.webp"):
        if f.name not in sah:
            f.unlink()
            print("  buang bingkai lama", f.name)

    # Katalognya BERDIRI SENDIRI, bentuknya sama dengan katalog model supaya
    # frontend tidak perlu jalur khusus. `bounds` itu pusat sel pertama dan
    # terakhir, `image_bounds` tepi gambarnya, dan bedanya setengah sel.
    kat = KELUARAN / "catalog.json"
    data = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "himawari9_ahi",
        "model_label": "Satelit Himawari-9",
        "versi_render": H.VERSI_RENDER,
        # Satelit tidak punya jam inisiasi, jadi yang dipakai jam PENGAMATAN
        # terbaru. Frontend memakainya untuk lencana Last update, dan untuk
        # pengamatan itu justru angka yang paling benar di situ.
        "run_time": max(bingkai_jadi).strftime("%Y-%m-%dT%H:%M:00Z"),
        "observasi": True,
        "sumber": "Himawari-9 AHI kanal 13, NOAA Open Data di AWS",
        "region": {
            "bounds": [H.LON0, H.LAT0, H.LON1, H.LAT1],
            "image_bounds": H.image_bounds(),
        },
        "layers": {
            H.LAYER: {
                "kind": "scalar",
                "level": "surface",
                "units": "\u00b0C",
                "frames": [
                    {"valid_time": w.strftime("%Y-%m-%dT%H:%M:00Z"),
                     "preview_image": nama_bingkai(w)}
                    for w in sorted(bingkai_jadi)
                ],
            },
        },
    }
    if tampak_jadi:
        data["layers"][T.LAYER] = {
            "kind": "scalar",
            "level": "surface",
            "units": "",
            "frames": [
                {"valid_time": w.strftime("%Y-%m-%dT%H:%M:00Z"),
                 "preview_image": nama_tampak(w)}
                for w in sorted(tampak_jadi)
            ],
        }
    if angin_jadi:
        data["layers"][G.LAYER] = {
            "kind": "scalar",
            "level": "surface",
            "units": "m/detik",
            "frames": [
                {"valid_time": w.strftime("%Y-%m-%dT%H:%M:00Z"),
                 "preview_image": nama_angin(w)}
                for w in sorted(angin_jadi)
            ],
        }
    kat.write_text(json.dumps(data, indent=1), "utf-8")
    print("katalog diperbarui, %d bingkai" % len(bingkai_jadi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
