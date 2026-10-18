# -*- coding: utf-8 -*-
"""Pembaca Himawari Standard Data, numpy saja.

   Sengaja TIDAK memakai satpy. Yang kita perlukan cuma satu band inframerah
   dan satu proyeksi geostasioner yang rumusnya tertutup, sedangkan satpy
   menyeret pyresample, xarray, dan dask. Di GitAction itu tiga menit pasang
   dan ratusan megabita, untuk pekerjaan yang muat di satu berkas.

   Berkasnya berblok. Tiap blok dimulai nomor blok satu bita lalu panjang
   blok dua bita, jadi kepalanya bisa dilompati tanpa tahu isinya."""
import bz2, struct
import numpy as np

def _blok(buf):
    """Peta nomor blok ke (awal, panjang), plus tempat data dimulai."""
    o, peta = 0, {}
    for _ in range(11):
        nomor = buf[o]
        panjang = struct.unpack_from("<H", buf, o + 1)[0]
        peta[nomor] = (o, panjang)
        o += panjang
    return peta, o

def baca(jalur):
    """Memulangkan dict berisi hitungan mentah dan seluruh keterangan navigasi."""
    raw = open(jalur, "rb").read()
    buf = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
    peta, awal_data = _blok(buf)

    o2 = peta[2][0]
    bit, kolom, baris = struct.unpack_from("<HHH", buf, o2 + 3)

    o3 = peta[3][0]
    sub_lon = struct.unpack_from("<d", buf, o3 + 3)[0]
    cfac, lfac = struct.unpack_from("<II", buf, o3 + 11)
    coff, loff = struct.unpack_from("<ff", buf, o3 + 19)
    # Blok 3 menyimpan JARI JARI, bukan kuadratnya. Sempat kubaca sebagai
    # kuadrat dan hasilnya seluruh domain menciut jadi sepita tipis di
    # khatulistiwa, sebab jari jari bumi jadi 79 km. Sudah diperiksa ke
    # berkasnya, urutannya Rs, Req, Rpol, lalu e2, lalu dua rasio.
    rs, req, rpol = struct.unpack_from("<ddd", buf, o3 + 27)
    e2, rasio_pol, rasio_eq = struct.unpack_from("<ddd", buf, o3 + 51)

    o5 = peta[5][0]
    band = struct.unpack_from("<H", buf, o5 + 3)[0]
    lambda_um = struct.unpack_from("<d", buf, o5 + 5)[0]
    cnt_error, cnt_luar = struct.unpack_from("<HH", buf, o5 + 15)
    gain, offset = struct.unpack_from("<dd", buf, o5 + 19)
    c0, c1, c2 = struct.unpack_from("<ddd", buf, o5 + 35)
    cahaya, planck, boltz = struct.unpack_from("<ddd", buf, o5 + 83)

    o7 = peta[7][0]
    seg_total, seg_nomor = struct.unpack_from("<BB", buf, o7 + 3)
    baris_awal = struct.unpack_from("<H", buf, o7 + 5)[0]

    data = np.frombuffer(buf, dtype="<u2", count=kolom * baris,
                         offset=awal_data).reshape(baris, kolom)
    return dict(hitungan=data, kolom=kolom, baris=baris, band=band,
                sub_lon=sub_lon, cfac=cfac, lfac=lfac, coff=coff, loff=loff,
                rs=rs, req=req, rpol=rpol, e2=e2, rasio_pol=rasio_pol,
                lambda_um=lambda_um, gain=gain, offset=offset,
                c0=c0, c1=c1, c2=c2, cahaya=cahaya, planck=planck, boltz=boltz,
                cnt_error=cnt_error, cnt_luar=cnt_luar,
                seg_total=seg_total, seg_nomor=seg_nomor, baris_awal=baris_awal)

def suhu_kecerahan(h):
    """Hitungan jadi suhu kecerahan dalam kelvin, lewat radiansi dan Planck."""
    c = h["hitungan"].astype(np.float64)
    buruk = (c == h["cnt_error"]) | (c == h["cnt_luar"])
    rad = h["gain"] * c + h["offset"]          # W / (m2 sr um)
    rad = np.where(rad > 0, rad, np.nan)
    lam = h["lambda_um"] * 1e-6                # meter
    L = rad * 1e6                              # W / (m2 sr m)
    hc_k = h["planck"] * h["cahaya"] / h["boltz"]
    te = (hc_k / lam) / np.log1p(2 * h["planck"] * h["cahaya"] ** 2 / (lam ** 5 * L))
    tb = h["c0"] + h["c1"] * te + h["c2"] * te * te
    tb[buruk] = np.nan
    return tb
