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
    slot = H.slot_tersedia()
    if not slot:
        print("tidak ada slot yang terdaftar di ember, berhenti tanpa mengubah apa apa")
        return 0
    pakai = H.pilih_bingkai(slot)
    print("slot terbaru di ember:", slot[0].strftime("%Y-%m-%d %H:%M"), "UTC")
    print("bingkai yang dipakai :", ", ".join(w.strftime("%H:%M") for w in pakai))

    import numpy as np

    urut = sorted(pakai)
    # Gerak awan dijejak dari DUA citra berurutan, jadi satu slot kadang
    # perlu dibangun kisinya walau gambarnya sudah tersimpan. Yang perlu itu
    # slot yang medannya sendiri belum ada, dan slot SEBELUM slot semacam
    # itu. Biasanya cuma satu dua, sebab tiap jalan paling cuma ada satu
    # bingkai baru.
    perlu_kisi = set()
    for i in range(1, len(urut)):
        f = KELUARAN / nama_angin(urut[i])
        if gambar_ulang or not f.exists() or f.stat().st_size < 1000:
            perlu_kisi.add(urut[i - 1])
            perlu_kisi.add(urut[i])

    bingkai_jadi = []
    angin_jadi = []
    lalu_w, lalu_arr = None, None      # kisi slot sebelumnya, cuma SATU yang ditahan
    for idx, w in enumerate(urut):
        berkas = KELUARAN / nama_bingkai(w)
        sudah = berkas.exists() and berkas.stat().st_size > 1000 and not gambar_ulang
        arr = None
        if sudah and w not in perlu_kisi:
            print("  %s sudah ada, dilewati" % w.strftime("%H:%M"))
            bingkai_jadi.append(w)
        else:
            sebab = "gambarnya belum ada" if not sudah else "kisinya dipakai menjejak gerak"
            print("  %s mengunduh %d segmen, %s" % (w.strftime("%H:%M"), len(H.SEGMEN), sebab))
            folder = SEMENTARA / w.strftime("%Y%m%d_%H%M")
            jalur = H.unduh(w, folder)
            hasil = H.bingkai(jalur)
            arr, redam = hasil if hasil else (None, None)
            shutil.rmtree(folder, ignore_errors=True)
            if arr is None:
                print("    gagal, dilewati")
                lalu_w, lalu_arr = None, None
                continue
            terisi = float(100.0 * np.isfinite(arr).mean())
            if terisi < 40:
                print("    cuma %.0f persen terisi, dibuang sebagai cacat" % terisi)
                lalu_w, lalu_arr = None, None
                continue
            if not sudah:
                H.simpan(arr, berkas, redam)
                print("    jadi, %.0f persen terisi, %.0f KB"
                      % (terisi, berkas.stat().st_size / 1024))
            bingkai_jadi.append(w)

        # ---- GERAK AWAN, dijejak dari pasangan slot ini dan sebelumnya ----
        # Medannya dicap dengan jam slot BELAKANGAN, sebab artinya "gerak
        # awan sampai jam ini". Dengan begitu bingkai terbaru selalu punya
        # angin, dan yang tidak punya justru bingkai paling tua.
        fa = KELUARAN / nama_angin(w)
        if fa.exists() and fa.stat().st_size > 1000 and not gambar_ulang:
            angin_jadi.append(w)
        elif lalu_arr is not None and lalu_w is not None and arr is not None:
            detik = (w - lalu_w).total_seconds()
            try:
                U, V, BT = G.medan(lalu_arr, arr, detik)
                n = G.gambar(U, V, BT, fa)
                print("    gerak awan %s, %d panah, %.0f KB"
                      % (w.strftime("%H:%M"), n, fa.stat().st_size / 1024))
                angin_jadi.append(w)
            except Exception as e:
                print("    gerak awan %s gagal, %s" % (w.strftime("%H:%M"), e))
        lalu_w, lalu_arr = w, arr

    # ---- CITRA WARNA ASLI, gelung dan unduhan sendiri ----
    # Sengaja dipisah dari gelung di atas. Kanal tampak itu 535 MB per slot
    # lawan 25 MB punya inframerah, dan kalau unduhannya putus di tengah,
    # yang hilang cukup gambarnya saja. Suhu kecerahan dan gerak awan sudah
    # terbit duluan dan tidak ikut gagal.
    tampak_jadi = []
    for w in urut:
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
                  % (w.strftime("%H:%M"), siang, ft.stat().st_size / 1024))
            tampak_jadi.append(w)
        except Exception as e:
            print("  citra tampak %s gagal, %s" % (w.strftime("%H:%M"), e))
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
        "model": "himawari9_ahi_b13",
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
