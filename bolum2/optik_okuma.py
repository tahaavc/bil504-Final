import os
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Eksik Python paketi var. Gerekli paketler: numpy, pandas ve "
        "OpenCV (cv2) veya Pillow (PIL). Ornek kurulum: "
        "pip install numpy pandas pillow"
    ) from exc

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


# ==========================================================
# 1. AYARLAR
# ==========================================================

SORU_SAYISI = 20
SECENEKLER = ["A", "B", "C", "D", "E"]

# Tum gorseller bu boyuta normalize edilir.
GENISLIK = 1000
YUKSEKLIK = 750

# Doldurulmus baloncuk icin koyuluk ve renk-nötrlük esikleri.
# Optik pembe/mor oldugu icin sadece koyu olmak yetmez; koyu ve gri/siyaha
# yakin pikselleri almak yanlis harf/cember okumalarini azaltir.
KOYU_ESIK = 140
RENK_FARK_ESIK = 75

# Dolu baloncuk blob filtreleri
MIN_BLOB_ALANI = 700
MAX_BLOB_ALANI = 6000
MIN_BLOB_BOYUTU = 20
MAX_BLOB_GENISLIK = 110
MAX_BLOB_YUKSEKLIK = 90

# Kumeleme / eslestirme toleranslari
SATIR_KUME_TOLERANSI = 28
SUTUN_KUME_TOLERANSI = 35
SATIR_ESLESME_TOLERANSI = 34
SUTUN_ESLESME_TOLERANSI = 46
BUBBLE_RADIUS = 24


# ==========================================================
# 2. GUVENLI OKUMA / KAYDETME
# ==========================================================

def _gorsel_kutuphanesi_kontrol():
    if cv2 is None and Image is None:
        raise ImportError(
            "Gorsel okumak icin OpenCV (cv2) veya Pillow (PIL) kurulu olmali."
        )


def guvenli_oku(dosya_yolu):
    _gorsel_kutuphanesi_kontrol()

    if cv2 is not None:
        veri = np.fromfile(dosya_yolu, dtype=np.uint8)
        img = cv2.imdecode(veri, cv2.IMREAD_COLOR)

        if img is None:
            return None

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with Image.open(dosya_yolu) as img:
        return np.array(img.convert("RGB"))


def guvenli_kaydet(dosya_yolu, img):
    _gorsel_kutuphanesi_kontrol()

    uzanti = os.path.splitext(dosya_yolu)[1] or ".png"
    kayit_yolu = dosya_yolu if os.path.splitext(dosya_yolu)[1] else dosya_yolu + uzanti

    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if cv2 is not None:
        kayit_img = img
        if img.ndim == 3:
            kayit_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        basarili, buffer = cv2.imencode(uzanti, kayit_img)

        if basarili:
            buffer.tofile(kayit_yolu)
            return True

        return False

    Image.fromarray(img).save(kayit_yolu)
    return True


# ==========================================================
# 3. GORUNTU OKUMA
# ==========================================================

def goruntu_oku(dosya_yolu):
    img = guvenli_oku(dosya_yolu)

    if img is None:
        raise FileNotFoundError(f"Goruntu okunamadi: {dosya_yolu}")

    if cv2 is not None:
        return cv2.resize(img, (GENISLIK, YUKSEKLIK), interpolation=cv2.INTER_AREA)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((GENISLIK, YUKSEKLIK), Image.Resampling.LANCZOS)
    return np.array(pil_img)


# ==========================================================
# 4. DEBUG CIZIM YARDIMCISI
# ==========================================================

class DebugCanvas:
    def __init__(self, img):
        self.cv2_aktif = cv2 is not None

        if self.cv2_aktif:
            self.img = img.copy()
            self.draw = None
        else:
            self.img = Image.fromarray(img)
            self.draw = ImageDraw.Draw(self.img)

    def circle(self, center, radius, color, width=2):
        x, y = int(round(center[0])), int(round(center[1]))

        if self.cv2_aktif:
            cv2.circle(self.img, (x, y), int(radius), color, int(width))
            return

        self.draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=width
        )

    def rectangle(self, p1, p2, color, width=2):
        p1 = (int(round(p1[0])), int(round(p1[1])))
        p2 = (int(round(p2[0])), int(round(p2[1])))

        if self.cv2_aktif:
            cv2.rectangle(self.img, p1, p2, color, int(width))
            return

        self.draw.rectangle((p1, p2), outline=color, width=width)

    def line(self, p1, p2, color, width=1):
        p1 = (int(round(p1[0])), int(round(p1[1])))
        p2 = (int(round(p2[0])), int(round(p2[1])))

        if self.cv2_aktif:
            cv2.line(self.img, p1, p2, color, int(width))
            return

        self.draw.line((p1, p2), fill=color, width=width)

    def text(self, text, pos, color, scale=0.45, width=1):
        pos = (int(round(pos[0])), int(round(pos[1])))

        if self.cv2_aktif:
            cv2.putText(
                self.img,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                int(width)
            )
            return

        self.draw.text(pos, text, fill=color)

    def sonuc(self):
        if self.cv2_aktif:
            return self.img

        return np.array(self.img)


# ==========================================================
# 5. DOLU BALONCUK BLOB'LARINI BULMA
# ==========================================================

def koyu_maske_olustur(img):
    arr = img.astype(np.int16)
    gri = arr.mean(axis=2)
    kanal_farki = arr.max(axis=2) - arr.min(axis=2)

    return (
        (gri < KOYU_ESIK) &
        (kanal_farki < RENK_FARK_ESIK) &
        (arr[:, :, 0] < 180) &
        (arr[:, :, 1] < 180) &
        (arr[:, :, 2] < 180)
    )


def _bilesenleri_cv2_ile_bul(mask):
    mask_u8 = mask.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8
    )

    bilesenler = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        bilesenler.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "cx": float(cx),
            "cy": float(cy)
        })

    return bilesenler


def _bilesenleri_python_ile_bul(mask):
    yukseklik, genislik = mask.shape
    gezildi = np.zeros(mask.shape, dtype=bool)
    bilesenler = []
    ys, xs = np.where(mask)
    komsular = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    )

    for bas_y, bas_x in zip(ys, xs):
        if gezildi[bas_y, bas_x] or not mask[bas_y, bas_x]:
            continue

        stack = [(int(bas_y), int(bas_x))]
        gezildi[bas_y, bas_x] = True
        alan = 0
        toplam_x = 0
        toplam_y = 0
        min_x = max_x = int(bas_x)
        min_y = max_y = int(bas_y)

        while stack:
            y, x = stack.pop()
            alan += 1
            toplam_x += x
            toplam_y += y
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            for dy, dx in komsular:
                ny = y + dy
                nx = x + dx

                if ny < 0 or nx < 0 or ny >= yukseklik or nx >= genislik:
                    continue

                if gezildi[ny, nx] or not mask[ny, nx]:
                    continue

                gezildi[ny, nx] = True
                stack.append((ny, nx))

        bilesenler.append({
            "x": min_x,
            "y": min_y,
            "w": max_x - min_x + 1,
            "h": max_y - min_y + 1,
            "area": alan,
            "cx": toplam_x / alan,
            "cy": toplam_y / alan
        })

    return bilesenler


def bilesenleri_bul(mask):
    if cv2 is not None:
        return _bilesenleri_cv2_ile_bul(mask)

    return _bilesenleri_python_ile_bul(mask)


def dolu_bloblari_bul(img):
    """
    Pembe form cizgilerini ve harfleri degil, koyu/gri doldurulmus isaretleri
    blob olarak bulur.
    """

    mask = koyu_maske_olustur(img)
    bilesenler = bilesenleri_bul(mask)
    bloblar = []

    for bilesen in bilesenler:
        x = bilesen["x"]
        y = bilesen["y"]
        w = bilesen["w"]
        h = bilesen["h"]
        area = bilesen["area"]
        cx = bilesen["cx"]
        cy = bilesen["cy"]

        if area < MIN_BLOB_ALANI or area > MAX_BLOB_ALANI:
            continue

        if w < MIN_BLOB_BOYUTU or h < MIN_BLOB_BOYUTU:
            continue

        if w > MAX_BLOB_GENISLIK or h > MAX_BLOB_YUKSEKLIK:
            continue

        oran = w / float(h)

        if oran < 0.55 or oran > 1.9:
            continue

        # Sol siyah hizalama seritlerini ve sayfa kenarlarini ele.
        if cx < 50 or cx > GENISLIK - 20:
            continue

        if cy < 50 or cy > YUKSEKLIK - 25:
            continue

        bloblar.append(bilesen)

    return bloblar, mask.astype(np.uint8) * 255


# ==========================================================
# 6. OPTIK DUZENINI GORSEL UZERINDEN CIKARMA
# ==========================================================

def _kume_bilgisi(degerler, tolerans):
    if not degerler:
        return []

    kumeler = []

    for deger in sorted(float(d) for d in degerler):
        if not kumeler:
            kumeler.append([deger])
            continue

        ortalama = sum(kumeler[-1]) / len(kumeler[-1])

        if abs(deger - ortalama) <= tolerans:
            kumeler[-1].append(deger)
        else:
            kumeler.append([deger])

    return [
        {
            "ortalama": sum(kume) / len(kume),
            "adet": len(kume),
            "degerler": kume
        }
        for kume in kumeler
    ]


def _on_satira_tamamla(satir_kumeleri):
    if not satir_kumeleri:
        return []

    if len(satir_kumeleri) > 10:
        satir_kumeleri = sorted(
            satir_kumeleri,
            key=lambda item: item["adet"],
            reverse=True
        )[:10]

    satirlar = sorted(item["ortalama"] for item in satir_kumeleri)

    if len(satirlar) == 10:
        return satirlar

    if len(satirlar) >= 2:
        return np.linspace(satirlar[0], satirlar[-1], 10).tolist()

    tek_satir = satirlar[0]
    tahmini_aralik = 55
    return [tek_satir + (i - 4.5) * tahmini_aralik for i in range(10)]


def _bes_sutuna_tamamla(sutun_kumeleri):
    if not sutun_kumeleri:
        return []

    if len(sutun_kumeleri) > 5:
        sutun_kumeleri = sorted(
            sutun_kumeleri,
            key=lambda item: item["adet"],
            reverse=True
        )[:5]

    gozlenen = sorted(item["ortalama"] for item in sutun_kumeleri)

    if len(gozlenen) == 5:
        return gozlenen

    if len(gozlenen) == 1:
        merkez = gozlenen[0]
        tahmini_aralik = 72
        return [merkez + (i - 2) * tahmini_aralik for i in range(5)]

    en_iyi_sutunlar = None
    en_iyi_skor = float("inf")

    for aralik in np.linspace(45, 90, 181):
        for gozlenen_x in gozlenen:
            for secenek_index in range(5):
                baslangic = gozlenen_x - secenek_index * aralik
                aday = [baslangic + i * aralik for i in range(5)]

                skor = 0
                for x in gozlenen:
                    skor += min((x - sutun_x) ** 2 for sutun_x in aday)

                # Gorsel disina tasan adaylari zayiflat.
                skor += max(0, 35 - aday[0]) ** 2 * 0.05
                skor += max(0, aday[-1] - (GENISLIK - 15)) ** 2 * 0.05

                if skor < en_iyi_skor:
                    en_iyi_skor = skor
                    en_iyi_sutunlar = aday

    return sorted(en_iyi_sutunlar)


def _bolme_x_hesapla(bloblar):
    xs = sorted(blob["cx"] for blob in bloblar)

    if len(xs) < 2:
        return GENISLIK / 2

    bosluklar = [
        (xs[i + 1] - xs[i], xs[i], xs[i + 1])
        for i in range(len(xs) - 1)
    ]

    en_buyuk_bosluk, sol, sag = max(bosluklar, key=lambda item: item[0])

    if en_buyuk_bosluk < 70:
        return GENISLIK / 2

    return (sol + sag) / 2


def optik_duzeni_cikar(bloblar):
    if not bloblar:
        return {
            "bolme_x": GENISLIK / 2,
            "satirlar": [],
            "sutunlar": {"sol": [], "sag": []}
        }

    bolme_x = _bolme_x_hesapla(bloblar)
    satir_kumeleri = _kume_bilgisi(
        [blob["cy"] for blob in bloblar],
        SATIR_KUME_TOLERANSI
    )
    satirlar = _on_satira_tamamla(satir_kumeleri)

    sol_bloblar = [blob for blob in bloblar if blob["cx"] < bolme_x]
    sag_bloblar = [blob for blob in bloblar if blob["cx"] >= bolme_x]

    sol_sutunlar = _bes_sutuna_tamamla(_kume_bilgisi(
        [blob["cx"] for blob in sol_bloblar],
        SUTUN_KUME_TOLERANSI
    ))
    sag_sutunlar = _bes_sutuna_tamamla(_kume_bilgisi(
        [blob["cx"] for blob in sag_bloblar],
        SUTUN_KUME_TOLERANSI
    ))

    return {
        "bolme_x": bolme_x,
        "satirlar": satirlar,
        "sutunlar": {
            "sol": sol_sutunlar,
            "sag": sag_sutunlar
        }
    }


def _en_yakin_index(degerler, hedef):
    if not degerler:
        return None, float("inf")

    mesafeler = [abs(deger - hedef) for deger in degerler]
    en_index = int(np.argmin(mesafeler))
    return en_index, mesafeler[en_index]


def _merkezler_duzenden_olustur(duzen):
    merkezler = {}
    satirlar = duzen["satirlar"]

    for satir_index, y in enumerate(satirlar[:10]):
        merkezler[satir_index + 1] = [
            (x, y) for x in duzen["sutunlar"]["sol"][:5]
        ]
        merkezler[satir_index + 11] = [
            (x, y) for x in duzen["sutunlar"]["sag"][:5]
        ]

    return merkezler


# ==========================================================
# 7. CEVAPLARI OKUMA
# ==========================================================

def cevaplari_blob_ile_oku(img, debug_kaydet=False):
    bloblar, mask = dolu_bloblari_bul(img)
    duzen = optik_duzeni_cikar(bloblar)
    merkezler = _merkezler_duzenden_olustur(duzen)

    cevaplar = ["BOŞ"] * SORU_SAYISI
    skorlar = {
        soru_no: [0, 0, 0, 0, 0]
        for soru_no in range(1, SORU_SAYISI + 1)
    }

    eslesen_bloblar = []
    canvas = DebugCanvas(img) if debug_kaydet else None

    if debug_kaydet:
        for soru_no in range(1, SORU_SAYISI + 1):
            for secenek_index, merkez in enumerate(merkezler.get(soru_no, [])):
                canvas.circle(merkez, BUBBLE_RADIUS, (0, 220, 0), 2)
                canvas.text(
                    SECENEKLER[secenek_index],
                    (merkez[0] - 7, merkez[1] + 5),
                    (0, 0, 255),
                    0.42,
                    1
                )

    for blob in bloblar:
        blok = "sol" if blob["cx"] < duzen["bolme_x"] else "sag"
        satir_index, satir_mesafe = _en_yakin_index(duzen["satirlar"], blob["cy"])
        sutun_index, sutun_mesafe = _en_yakin_index(
            duzen["sutunlar"][blok],
            blob["cx"]
        )

        if satir_index is None or sutun_index is None:
            continue

        if satir_mesafe > SATIR_ESLESME_TOLERANSI:
            continue

        if sutun_mesafe > SUTUN_ESLESME_TOLERANSI:
            continue

        soru_no = satir_index + 1 if blok == "sol" else satir_index + 11

        if soru_no < 1 or soru_no > SORU_SAYISI:
            continue

        skorlar[soru_no][sutun_index] += blob["area"]
        eslesen_bloblar.append((blob, soru_no, sutun_index))

        if debug_kaydet:
            merkez = merkezler[soru_no][sutun_index]
            canvas.rectangle(
                (blob["x"], blob["y"]),
                (blob["x"] + blob["w"], blob["y"] + blob["h"]),
                (255, 0, 0),
                2
            )
            canvas.line((blob["cx"], blob["cy"]), merkez, (255, 0, 255), 1)

    debug_verileri = []

    for soru_no in range(1, SORU_SAYISI + 1):
        soru_skorlari = skorlar[soru_no]
        en_yuksek = max(soru_skorlari)
        en_index = soru_skorlari.index(en_yuksek)

        if en_yuksek == 0:
            cevap = "BOŞ"
        else:
            cevap = SECENEKLER[en_index]

        cevaplar[soru_no - 1] = cevap

        debug_verileri.append({
            "Soru": soru_no,
            "A": soru_skorlari[0],
            "B": soru_skorlari[1],
            "C": soru_skorlari[2],
            "D": soru_skorlari[3],
            "E": soru_skorlari[4],
            "Okunan": cevap
        })

        if debug_kaydet:
            debug_cevap = "BOS" if cevap == "BOŞ" else cevap
            y = 20 + soru_no * 22
            canvas.text(f"{soru_no}: {debug_cevap}", (10, y), (255, 0, 0), 0.55, 1)
            skor_yazi = " ".join([
                f"{SECENEKLER[i]}:{soru_skorlari[i]}" for i in range(5)
            ])
            canvas.text(skor_yazi, (95, y), (255, 0, 0), 0.36, 1)

    debug_img = canvas.sonuc() if debug_kaydet else img

    return cevaplar, debug_img, mask, debug_verileri


def cevaplari_oku(dosya_yolu, debug_kaydet=False, debug_klasoru=None):
    img = goruntu_oku(dosya_yolu)

    cevaplar, debug_img, mask, debug_verileri = cevaplari_blob_ile_oku(
        img,
        debug_kaydet=debug_kaydet
    )

    if debug_kaydet:
        if debug_klasoru is None:
            zaman = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_klasoru = os.path.abspath(f"debug_ciktilari_{zaman}")

        os.makedirs(debug_klasoru, exist_ok=True)

        dosya_adi = os.path.splitext(os.path.basename(dosya_yolu))[0]
        debug_yolu = os.path.join(debug_klasoru, f"okunan_{dosya_adi}.png")
        mask_yolu = os.path.join(debug_klasoru, f"maske_{dosya_adi}.png")
        csv_yolu = os.path.join(debug_klasoru, f"blob_skorlari_{dosya_adi}.csv")

        kayit1 = guvenli_kaydet(debug_yolu, debug_img)
        kayit2 = guvenli_kaydet(mask_yolu, mask)

        pd.DataFrame(debug_verileri).to_csv(
            csv_yolu,
            index=False,
            encoding="utf-8-sig"
        )

        print("DEBUG KLASORU:", debug_klasoru)
        print("OKUNAN GORSEL:", kayit1, debug_yolu)
        print("MASKE GORSELI:", kayit2, mask_yolu)
        print("CSV:", csv_yolu)

    return cevaplar


# ==========================================================
# 8. SONUC HESAPLAMA
# ==========================================================

def sonuc_hesapla(ogrenci_cevaplari, cevap_anahtari):
    dogru = 0
    yanlis = 0
    bos = 0
    gecersiz = 0
    detaylar = []

    for i in range(SORU_SAYISI):
        ogr = ogrenci_cevaplari[i]
        dog = cevap_anahtari[i]

        if ogr == "BOŞ":
            bos += 1
            durum = "BOŞ"
        elif dog == "BOŞ":
            yanlis += 1
            durum = "ANAHTAR OKUMA HATASI"
        elif ogr == dog:
            dogru += 1
            durum = "DOĞRU"
        else:
            yanlis += 1
            durum = "YANLIŞ"

        detaylar.append({
            "Soru": i + 1,
            "Öğrenci Cevabı": ogr,
            "Doğru Cevap": dog,
            "Durum": durum
        })

    return dogru, yanlis, bos, gecersiz, detaylar


# ==========================================================
# 9. TUM OPTIKLERI DEGERLENDIRME
# ==========================================================

def optik_degerlendir(
    cevap_anahtari_yolu,
    ogrenci_klasoru,
    sonuc_excel_yolu="optik_sonuclar.xlsx"
):
    zaman = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_klasoru = os.path.abspath(f"debug_ciktilari_{zaman}")

    print("Cevap anahtari okunuyor...")

    cevap_anahtari = cevaplari_oku(
        cevap_anahtari_yolu,
        debug_kaydet=True,
        debug_klasoru=debug_klasoru
    )

    print("Cevap Anahtari:")
    print(cevap_anahtari)

    genel_sonuclar = []
    detayli_sonuclar = []

    for dosya_adi in os.listdir(ogrenci_klasoru):
        if dosya_adi.lower().endswith((".jpg", ".jpeg", ".png")):
            dosya_yolu = os.path.join(ogrenci_klasoru, dosya_adi)
            ogrenci_no = os.path.splitext(dosya_adi)[0]

            print(f"{ogrenci_no} numarali ogrenci okunuyor...")

            ogrenci_cevaplari = cevaplari_oku(
                dosya_yolu,
                debug_kaydet=True,
                debug_klasoru=debug_klasoru
            )

            print("Ogrenci Cevaplari:")
            print(ogrenci_cevaplari)

            dogru, yanlis, bos, gecersiz, detaylar = sonuc_hesapla(
                ogrenci_cevaplari,
                cevap_anahtari
            )

            genel_sonuclar.append({
                "Öğrenci No": ogrenci_no,
                "Dosya Adı": dosya_adi,
                "Doğru": dogru,
                "Yanlış": yanlis,
                "Boş": bos,
                "Geçersiz": gecersiz,
                "Toplam Soru": SORU_SAYISI
            })

            for detay in detaylar:
                detayli_sonuclar.append({
                    "Öğrenci No": ogrenci_no,
                    "Dosya Adı": dosya_adi,
                    "Soru": detay["Soru"],
                    "Öğrenci Cevabı": detay["Öğrenci Cevabı"],
                    "Doğru Cevap": detay["Doğru Cevap"],
                    "Durum": detay["Durum"]
                })

    genel_df = pd.DataFrame(genel_sonuclar)
    detay_df = pd.DataFrame(detayli_sonuclar)

    with pd.ExcelWriter(sonuc_excel_yolu) as writer:
        genel_df.to_excel(writer, sheet_name="Genel Sonuclar", index=False)
        detay_df.to_excel(writer, sheet_name="Detayli Sonuclar", index=False)

    print("Islem tamamlandi.")
    print(f"Excel dosyasi olusturuldu: {sonuc_excel_yolu}")

    return genel_df, detay_df


# ==========================================================
# 10. CALISTIRMA
# ==========================================================

if __name__ == "__main__":
    cevap_anahtari_yolu = "cevap_anahtari.jpeg"
    ogrenci_klasoru = "ogrenciler"
    sonuc_excel_yolu = "optik_sonuclar.xlsx"

    genel_sonuclar, detayli_sonuclar = optik_degerlendir(
        cevap_anahtari_yolu,
        ogrenci_klasoru,
        sonuc_excel_yolu
    )

    print(genel_sonuclar)
