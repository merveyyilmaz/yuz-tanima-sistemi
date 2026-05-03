# 🎭 Bulut Tabanlı Yüz Tanıma Sistemi

> **Bulut Bilişim Proje Sunumu**  
> PaaS (Platform as a Service) Mimarisi ile Canlıya Alma

**Ekip:**
- Emircan ALKAN — 220106206014  
- Kaan BAYDERE — 220106109052  
- Yiğit KARABULUT — 220106206049  
- Merve YILMAZ — 220106109018  

---

## 📌 Proje Özeti

Yerel ortamda geliştirilen görüntü işleme ve yüz tanıma modelinin, **Streamlit Community Cloud** (PaaS) üzerinde ölçeklenebilir, donanımdan bağımsız ve herkes tarafından erişilebilir bir web uygulamasına dönüştürülmesidir.

---

## 🏗️ Mimari

```
Yerel Geliştirme (Python + OpenCV)
        │
        ▼
GitHub Deposu  ←── Sürekli İzleme
        │
        ▼
Streamlit Community Cloud (PaaS)
        │
   ┌────┴────┐
   │ Container│  ← Otomatik Build & Deploy
   │  (izole) │
   └────┬────┘
        │
        ▼
Web Uygulaması (HTTPS, herkese açık URL)
```

**Kullanılan PaaS:** Streamlit Community Cloud  
**Backend:** Python 3.11, face_recognition (dlib HOG modeli), OpenCV  
**Deployment:** GitHub → otomatik CI/CD  

---

## 🚀 Kurulum

### Yerel Çalıştırma

```bash
# 1. Depoyu klonla
git clone https://github.com/KULLANICI_ADI/yuz-tanima-sistemi.git
cd yuz-tanima-sistemi

# 2. Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Sistem bağımlılıklarını yükle (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev

# 4. Python paketlerini yükle
pip install -r requirements.txt

# 5. Uygulamayı başlat
streamlit run app.py
```

### Streamlit Community Cloud'a Deploy

1. Bu repoyu GitHub'a push edin
2. [share.streamlit.io](https://share.streamlit.io) adresine gidin
3. **New app** → GitHub reponuzu seçin → **`app.py`** → Deploy

> ✅ Streamlit Cloud `packages.txt` ve `requirements.txt` dosyalarını otomatik okuyarak ortamı hazırlar.

---

## 📁 Dosya Yapısı

```
yuz-tanima-sistemi/
│
├── app.py                    # Ana Streamlit uygulaması
├── requirements.txt          # Python bağımlılıkları
├── packages.txt              # Sistem bağımlılıkları (Streamlit Cloud)
├── README.md                 # Bu dosya
│
├── .streamlit/
│   └── config.toml           # Tema ve sunucu ayarları
│
└── known_faces/              # Önceden kayıtlı yüz fotoğrafları
    ├── ahmet_yilmaz.jpg      # → "Ahmet Yilmaz" olarak tanınır
    ├── ayse_kaya.jpg         # → "Ayse Kaya" olarak tanınır
    └── README.md
```

---

## 🎮 Kullanım Kılavuzu

### Kişi Kaydetme
1. Sol kenar çubuğunda **"Yeni Kişi Ekle"** bölümüne gidin
2. Ad Soyad girin
3. Net bir yüz fotoğrafı yükleyin
4. **"Kaydet"** butonuna tıklayın

### Fotoğraf Analizi
1. **"Fotoğraf Analizi"** sekmesine geçin
2. Bir veya birden fazla fotoğraf yükleyin
3. Sistem otomatik olarak analiz eder ve sonuçları gösterir
4. Sonucu JPEG olarak indirebilirsiniz

### Webcam ile Tespit
1. **"Webcam Tespiti"** sekmesine geçin
2. **"📸 Kameradan Görüntü Al"** butonuna tıklayın
3. Kamera izni verin, fotoğraf çekin
4. Anlık analiz sonucu görüntülenir

### Ayarlar
- **Eşleşme Toleransı:** 0.3 (çok katı) — 0.7 (çok toleranslı). Varsayılan: 0.5

---

## 🧠 Teknik Detaylar

| Bileşen | Teknoloji |
|---------|-----------|
| Yüz Tespiti | dlib HOG (Histogram of Oriented Gradients) |
| Yüz Kodlama | 128 boyutlu yüz vektörü |
| Karşılaştırma | Öklid mesafesi (face_distance) |
| Görüntü İşleme | OpenCV 4.9 |
| Web Arayüzü | Streamlit 1.35 |
| Bulut Platformu | Streamlit Community Cloud (PaaS) |

---

## 🔮 Gelecek Geliştirmeler

- [ ] Bulut nesne depolama (AWS S3 / GCS) ile fotoğraf arşivleme  
- [ ] Analiz sonuçlarının bulut veritabanında (Firebase / Supabase) loglanması  
- [ ] WebSocket tabanlı gerçek zamanlı video akışı  
- [ ] Çoklu kullanıcı desteği ve kimlik doğrulama  
- [ ] REST API endpoint'leri (FastAPI entegrasyonu)  

---

## ⚠️ Önemli Notlar

- `face_recognition` kütüphanesi `dlib` gerektirdiğinden ilk build süresi **5-10 dakika** sürebilir
- `packages.txt` dosyası Streamlit Cloud'da sistem bağımlılıklarını otomatik kurar
- Yüz tanıma HOG modeli kullanır; GPU gerekmez, CPU üzerinde çalışır
- GDPR/KVKK uyumluluğu için üretim ortamında veri saklama politikaları uygulanmalıdır

---

*Bulut Bilişim Dersi — 2024/2025 Akademik Yılı*
