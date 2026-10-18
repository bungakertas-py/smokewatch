# -*- coding: utf-8 -*-
"""Menjalankan sumber data SATELIT untuk Smokewatch, folder dan katalog
   sendiri di data/output/himawari/.

   Dua layer sekali jalan, asap dan debu, lalu titik panas. Empat kanal
   diunduh per bingkai, 7, 11, 13, dan 15, dan kanal 13 dipakai dua duanya
   jadi cuma diunduh sekali."""
from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from backend.pipeline import himawari as H

AKAR = Path(__file__).resolve().parents[1]
KELUARAN = Path(sys.argv[1]) if len(sys.argv) > 1 else AKAR / "data" / "output" / "himawari"
SEMENTARA = Path(sys.argv[2]) if len(sys.argv) > 2 else AKAR / "data" / "tmp_himawari"


def nama(layer: str, w: dt.datetime) -> str:
    return "%s_%s_preview.webp" % (layer, w.strftime("%Y%m%d_%H%M"))


def versi_lama() -> int:
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
        print("tidak ada slot di ember, berhenti tanpa mengubah apa apa")
        return 0
    pakai = H.pilih_bingkai(slot)
    print("slot terbaru:", slot[0].strftime("%Y-%m-%d %H:%M"), "UTC")
    print("bingkai dipakai:", ", ".join(w.strftime("%H:%M") for w in pakai))

    jadi = []
    for w in sorted(pakai):
        f_asap = KELUARAN / nama(H.LAYER_ASAP, w)
        f_api = KELUARAN / nama(H.LAYER_API, w)
        if f_asap.exists() and f_api.exists() and not gambar_ulang:
            print("  %s sudah ada, dilewati" % w.strftime("%H:%M"))
            jadi.append(w)
            continue
        print("  %s mengunduh empat kanal" % w.strftime("%H:%M"))
        folder = SEMENTARA / w.strftime("%Y%m%d_%H%M")
        suhu = {}
        for band in (7, 11, 13, 15):
            jalur = H.unduh_band(w, band, folder)
            arr, redam = H.suhu_kisi(jalur)
            if arr is None:
                break
            suhu[band] = arr
            suhu["redam"] = redam
        shutil.rmtree(folder, ignore_errors=True)
        if len(suhu) < 5:
            print("    kanal tidak lengkap, dilewati")
            continue
        terisi = float(100.0 * np.isfinite(suhu[13]).mean())
        if terisi < 40:
            print("    cuma %.0f persen terisi, dibuang sebagai cacat" % terisi)
            continue
        H.gambar_asap(suhu[11], suhu[13], suhu[15], suhu["redam"], f_asap)
        n_api = H.gambar_api(suhu[7], suhu[13], suhu["redam"], f_api, w)
        print("    jadi, %.0f persen terisi, asap %.0f KB, titik panas %d piksel, %.0f KB"
              % (terisi, f_asap.stat().st_size / 1024, n_api, f_api.stat().st_size / 1024))
        jadi.append(w)

    if not jadi:
        print("tidak ada bingkai yang jadi, katalog tidak disentuh")
        return 0

    sah = {nama(l, w) for w in jadi for l in (H.LAYER_ASAP, H.LAYER_API)}
    for f in KELUARAN.glob("*_preview.webp"):
        if f.name not in sah:
            f.unlink()
            print("  buang bingkai lama", f.name)

    def bingkai(layer):
        return [{"valid_time": w.strftime("%Y-%m-%dT%H:%M:00Z"),
                 "preview_image": nama(layer, w)} for w in sorted(jadi)]

    data = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "himawari9_ahi",
        "model_label": "Satelit Himawari-9",
        "run_time": max(jadi).strftime("%Y-%m-%dT%H:%M:00Z"),
        "observasi": True,
        "versi_render": H.VERSI_RENDER,
        "sumber": "Himawari-9 AHI, NOAA Open Data di AWS",
        "region": {"bounds": [H.LON0, H.LAT0, H.LON1, H.LAT1],
                   "image_bounds": H.image_bounds()},
        "layers": {
            H.LAYER_ASAP: {"kind": "scalar", "level": "surface", "units": "",
                           "frames": bingkai(H.LAYER_ASAP)},
            H.LAYER_API: {"kind": "scalar", "level": "surface", "units": "",
                          "frames": bingkai(H.LAYER_API)},
        },
    }
    (KELUARAN / "catalog.json").write_text(json.dumps(data, indent=1), "utf-8")
    print("katalog diperbarui, %d bingkai" % len(jadi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
