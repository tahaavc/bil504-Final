# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# AYARLAR (Fasulye için optimize edilmiş)

MIN_MESAFE = 24      # px - iki tane merkezi arası min mesafe
DIST_ESIK = 0.3      # distance transform lokal maksima eşiği
MAX_YUKSEKLIK = 900  # px - işlem hızı için boyutlandırma

# YARDIMCI FONKSİYONLAR

def goruntu_goster_coklu(goruntuler, basliklar, satir=2, sutun=3,
                         baslik_genel=""):
    """Birden fazla görüntüyü ızgara halinde gösterir."""
    plt.figure(figsize=(16, 10))
    if baslik_genel:
        plt.suptitle(baslik_genel, fontsize=14, fontweight='bold')
    for i, (img, baslik) in enumerate(zip(goruntuler, basliklar)):
        plt.subplot(satir, sutun, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(baslik, fontsize=11)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 1. ÖN İŞLEME

def on_isleme(goruntu_yolu, max_yukseklik=MAX_YUKSEKLIK):
    """
    Görüntüyü okur, boyutlandırır ve binary mask üretir.
    """
    orijinal = cv2.imread(goruntu_yolu)
    if orijinal is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {goruntu_yolu}")

    h, w = orijinal.shape[:2]
    if h > max_yukseklik:
        oran = max_yukseklik / h
        orijinal = cv2.resize(orijinal, (int(w * oran), max_yukseklik))

    gri = cv2.cvtColor(orijinal, cv2.COLOR_BGR2GRAY)
    bulanik = cv2.medianBlur(gri, 5)

    _, binary = cv2.threshold(
        bulanik, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return orijinal, gri, binary

# 2. MORFOLOJİK TEMİZLİK

def morfolojik_temizlik(binary):
    """
    Binary maskedeki gürültüleri temizler.
    - Opening: küçük beyaz noktaları (gürültü) siler
    - Closing: tane içindeki olası siyah boşlukları doldurur
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    temiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    temiz = cv2.morphologyEx(temiz, cv2.MORPH_CLOSE, kernel, iterations=3)
    return temiz

# 3. YÖNTEM A: BASİT KONTUR SAYMA (Karşılaştırma için)

def basit_kontur_say(binary, orijinal, min_alan=200):
    """findContours ile basit sayım."""
    konturlar, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    gecerli = [k for k in konturlar if cv2.contourArea(k) > min_alan]

    sonuc = orijinal.copy()
    cv2.drawContours(sonuc, gecerli, -1, (0, 255, 0), 2)

    for i, k in enumerate(gecerli):
        M = cv2.moments(k)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(sonuc, str(i + 1), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(sonuc, f"Basit Yontem: {len(gecerli)} kume",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return len(gecerli), sonuc

# 4. YÖNTEM B: WATERSHED SEGMENTASYONU (Asıl Yöntem)

def watershed_say(binary, orijinal,
                  min_mesafe=MIN_MESAFE, dist_esik=DIST_ESIK):
    """Distance Transform + Lokal Maksima + Watershed."""
    # 1. Distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 2. Lokal maksima bul
    kernel = np.ones((min_mesafe, min_mesafe), np.uint8)
    dist_dilated = cv2.dilate(dist, kernel)
    lokal_max = (dist == dist_dilated) & (dist > dist_esik * dist.max())
    lokal_max = lokal_max.astype(np.uint8) * 255

    # 3. Her lokal maksimayı ayrı bir etikete dönüştür
    _, markerler = cv2.connectedComponents(lokal_max)

    # 4. Watershed için arka plan / ön plan / bilinmeyen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kesin_arka = cv2.dilate(binary, kernel, iterations=3)
    bilinmeyen = cv2.subtract(kesin_arka, lokal_max)

    markerler = markerler + 1
    markerler[bilinmeyen == 255] = 0

    # 5. Watershed uygula
    sonuc = orijinal.copy()
    markerler = cv2.watershed(sonuc, markerler)

    # 6. Sınırları kırmızıya boya
    sonuc[markerler == -1] = [0, 0, 255]

    # 7. Tane sayısı
    benzersiz_etiketler = np.unique(markerler)
    tane_sayisi = len(benzersiz_etiketler) - 2

    # 8. Numaralandır
    sayac = 0
    for etiket in benzersiz_etiketler:
        if etiket <= 1:
            continue
        sayac += 1
        mask = (markerler == etiket).astype(np.uint8) * 255
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(sonuc, str(sayac), (cx - 8, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    cv2.putText(sonuc, f"Watershed: {tane_sayisi} tane",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 0), 2)

    return tane_sayisi, sonuc, dist, lokal_max

# 5. ANA PIPELINE

def tane_say_pipeline(goruntu_yolu, kayit_klasoru="cikti_bolum3",
                      gercek_sayi=None):
    """Tüm pipeline'ı çalıştırır."""
    print("\n" + "=" * 65)
    print(f"BOLUM 3 - Tane Sayma Pipeline")
    print(f"Goruntu: {goruntu_yolu}")
    print("=" * 65)

    # 1. Ön işleme
    orijinal, gri, binary = on_isleme(goruntu_yolu)
    print(f"[1/5] On isleme tamamlandi ({orijinal.shape[1]}x{orijinal.shape[0]} px)")

    # 2. Morfolojik temizlik
    temiz = morfolojik_temizlik(binary)
    print(f"[2/5] Morfolojik temizlik tamamlandi")

    # 3. Basit kontur sayma
    basit_sayi, basit_gorsel = basit_kontur_say(temiz, orijinal)
    print(f"[3/5] Basit kontur yontemi   : {basit_sayi} kume bulundu")

    # 4. Watershed
    ws_sayi, ws_gorsel, dist, lokal_max = watershed_say(temiz, orijinal)
    print(f"[4/5] Watershed yontemi      : {ws_sayi} tane bulundu")

    # Doğruluk
    if gercek_sayi is not None:
        print()
        print(f"  Gercek tane sayisi       : {gercek_sayi}")
        print(f"  Watershed dogrulugu      : %{100 * (1 - abs(ws_sayi - gercek_sayi) / gercek_sayi):.1f}")
        print(f"  Basit yontem dogrulugu   : %{100 * (1 - abs(basit_sayi - gercek_sayi) / gercek_sayi):.1f}")

    # 5. Görselleştirme
    dist_gorsel = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Lokal maksima görseli — connectedComponentsWithStats ile garanti merkezler
    lokal_gorsel = orijinal.copy()
    _, _, _, merkezler = cv2.connectedComponentsWithStats(lokal_max)
    # 0. etiket arka plan, 1'den başlayarak her tane merkezi
    for i in range(1, len(merkezler)):
        cx, cy = int(merkezler[i][0]), int(merkezler[i][1])
        # 3 katmanlı görünür nokta: siyah dış, beyaz halka, pembe iç
        cv2.circle(lokal_gorsel, (cx, cy), 16, (0, 0, 0), -1)
        cv2.circle(lokal_gorsel, (cx, cy), 13, (255, 255, 255), -1)
        cv2.circle(lokal_gorsel, (cx, cy), 10, (255, 0, 255), -1)

    goruntu_goster_coklu(
        [orijinal, gri, binary, temiz, dist_gorsel, lokal_gorsel],
        ["1. Orijinal Goruntu",
         "2. Gri Tonlama",
         "3. Otsu Esikleme (Binary)",
         "4. Morfolojik Temizlik",
         "5. Distance Transform",
         f"6. Lokal Maksima ({ws_sayi} merkez)"],
        satir=2, sutun=3,
        baslik_genel="BOLUM 3: Pipeline Adimlari"
    )

    goruntu_goster_coklu(
        [basit_gorsel, ws_gorsel],
        [f"Basit Kontur: {basit_sayi} kume\n(Birbirine degen taneleri tek sayar)",
         f"Watershed: {ws_sayi} tane\n(Tum taneleri ayri tespit eder)"],
        satir=1, sutun=2,
        baslik_genel="Yontemlerin Karsilastirilmasi"
    )

    # 6. Kaydet
    print(f"[5/5] Cikti dosyalari kaydediliyor...")
    if not os.path.exists(kayit_klasoru):
        os.makedirs(kayit_klasoru)

    isim = os.path.splitext(os.path.basename(goruntu_yolu))[0]
    cv2.imwrite(f"{kayit_klasoru}/{isim}_01_orijinal.jpg", orijinal)
    cv2.imwrite(f"{kayit_klasoru}/{isim}_02_gri.jpg", gri)
    cv2.imwrite(f"{kayit_klasoru}/{isim}_03_binary.jpg", binary)
    cv2.imwrite(f"{kayit_klasoru}/{isim}_04_temiz.jpg", temiz)
    cv2.imwrite(f"{kayit_klasoru}/{isim}_05_distance.jpg", dist_gorsel)
    cv2.imwrite(f"{kayit_klasoru}/{isim}_06_lokal_max.jpg", lokal_gorsel)
    cv2.imwrite(f"{kayit_klasoru}/{isim}_07_basit_kontur.jpg", basit_gorsel)
    cv2.imwrite(f"{kayit_klasoru}/{isim}_08_watershed.jpg", ws_gorsel)

    print(f"      -> {kayit_klasoru}/ klasorune kaydedildi")
    print()
    print("=" * 65)
    print(f"SONUC: Goruntude TOPLAM {ws_sayi} TANE TESPIT EDILDI")
    print("=" * 65)
    print()

    return {
        'basit_sayi': basit_sayi,
        'watershed_sayi': ws_sayi,
        'gorsel': ws_gorsel
    }

# 6. ÇALIŞTIR

if __name__ == "__main__":
    goruntu_yolu = "fasulye.jpg"
    gercek_sayi = 40

    if os.path.exists(goruntu_yolu):
        sonuc = tane_say_pipeline(goruntu_yolu, gercek_sayi=gercek_sayi)
    else:
        print(f"HATA: '{goruntu_yolu}' bulunamadi.")
        print("Lutfen bir tane goruntusu ekleyin ve dosya yolunu guncelleyin.")
        
        
        
        
        