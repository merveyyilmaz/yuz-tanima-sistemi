import streamlit as st
import cv2
import numpy as np
from PIL import Image
import face_recognition
import io
import os
import json
import time
from datetime import datetime
import base64

# ── Sayfa Yapılandırması ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bulut Tabanlı Yüz Tanıma Sistemi",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Stilleri ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Ana arka plan */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Başlık kutusu */
    .main-header {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(79, 70, 229, 0.4);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    /* Metric kartları */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-card h3 {
        color: #a5b4fc;
        font-size: 0.85rem;
        margin: 0 0 0.4rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    /* Sonuç kutuları */
    .result-box-success {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .result-box-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.5);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .result-box-info {
        background: rgba(79, 70, 229, 0.15);
        border: 1px solid rgba(79, 70, 229, 0.5);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }

    /* Log tablosu */
    .log-table {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.85rem;
        color: #a5b4fc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Genel metin rengi */
    .stMarkdown p, .stMarkdown li {
        color: rgba(255,255,255,0.85);
    }

    /* Butonlar */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.5);
    }

    /* Tab stili */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: rgba(255,255,255,0.6);
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Başlatma ────────────────────────────────────────────────────
if "known_faces" not in st.session_state:
    st.session_state.known_faces = {}       # {isim: encoding_listesi}
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []    # Tespit geçmişi
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "total_recognized" not in st.session_state:
    st.session_state.total_recognized = 0


# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────
def load_known_faces_from_folder():
    """known_faces/ klasöründeki yüzleri otomatik yükle"""
    folder = "known_faces"
    if not os.path.exists(folder):
        return
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            if name not in st.session_state.known_faces:
                img_path = os.path.join(folder, filename)
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    st.session_state.known_faces[name] = encs[0]


def process_image(image_np, tolerance=0.5):
    """
    Numpy görüntüsü alır, yüzleri tespit eder ve tanır.
    Döndürür: annotated_image (BGR numpy), results (list of dict)
    """
    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) if len(image_np.shape) == 3 else image_np
    
    # Yüz konumları ve encodinglerini bul
    face_locations = face_recognition.face_locations(rgb, model="hog")
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    results = []
    annotated = image_np.copy()

    known_names = list(st.session_state.known_faces.keys())
    known_encs = [st.session_state.known_faces[n] for n in known_names]

    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        name = "Bilinmeyen Kişi"
        confidence = 0.0
        color = (0, 165, 255)  # Turuncu = bilinmeyen

        if known_encs:
            distances = face_recognition.face_distance(known_encs, enc)
            best_idx = int(np.argmin(distances))
            best_dist = distances[best_idx]
            if best_dist < tolerance:
                name = known_names[best_idx]
                confidence = round((1 - best_dist) * 100, 1)
                color = (0, 200, 100)  # Yeşil = tanınan

        # Dikdörtgen çiz
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        # Etiket arka planı
        label = f"{name} ({confidence}%)" if confidence > 0 else name
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (left, bottom), (left + lw + 8, bottom + lh + 10), color, -1)
        cv2.putText(annotated, label, (left + 4, bottom + lh + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        results.append({
            "name": name,
            "confidence": confidence,
            "bbox": (top, right, bottom, left),
            "recognized": confidence > 0
        })

    return annotated, results


def log_detection(results, source):
    """Tespit sonuçlarını loga ekle"""
    ts = datetime.now().strftime("%H:%M:%S")
    for r in results:
        st.session_state.detection_log.append({
            "time": ts,
            "source": source,
            "name": r["name"],
            "confidence": r["confidence"],
            "recognized": r["recognized"]
        })
    st.session_state.total_detections += len(results)
    st.session_state.total_recognized += sum(1 for r in results if r["recognized"])
    # Son 50 kaydı tut
    if len(st.session_state.detection_log) > 50:
        st.session_state.detection_log = st.session_state.detection_log[-50:]


def pil_to_np(pil_img):
    return np.array(pil_img.convert("RGB"))


def np_to_pil(np_img):
    if np_img.shape[2] == 3:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np_img)


# ── Uygulama Başlangıcında Bilinen Yüzleri Yükle ─────────────────────────────
load_known_faces_from_folder()


# ── Başlık ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎭 Bulut Tabanlı Yüz Tanıma Sistemi</h1>
    <p>PaaS Mimarisi ile Geliştirilmiş · OpenCV + face_recognition · Streamlit Community Cloud</p>
</div>
""", unsafe_allow_html=True)

# ── Metrikler ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <h3>📚 Kayıtlı Kişi</h3>
        <p>{len(st.session_state.known_faces)}</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <h3>🔍 Toplam Tespit</h3>
        <p>{st.session_state.total_detections}</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <h3>✅ Tanınan</h3>
        <p>{st.session_state.total_recognized}</p>
    </div>""", unsafe_allow_html=True)
with col4:
    acc = (st.session_state.total_recognized / st.session_state.total_detections * 100
           if st.session_state.total_detections > 0 else 0)
    st.markdown(f"""<div class="metric-card">
        <h3>🎯 Tanıma Oranı</h3>
        <p>{acc:.0f}%</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")
    st.markdown("---")

    tolerance = st.slider(
        "🎯 Eşleşme Toleransı",
        min_value=0.3, max_value=0.7, value=0.5, step=0.05,
        help="Düşük = daha katı eşleşme, Yüksek = daha toleranslı"
    )

    st.markdown("---")
    st.markdown("### 📖 Kişi Veritabanı")

    if st.session_state.known_faces:
        for name in st.session_state.known_faces:
            col_n, col_d = st.columns([3, 1])
            with col_n:
                st.markdown(f"👤 **{name}**")
            with col_d:
                if st.button("🗑️", key=f"del_{name}", help=f"{name} sil"):
                    del st.session_state.known_faces[name]
                    st.rerun()
    else:
        st.info("Henüz kayıtlı yüz yok.\nAşağıdan kişi ekleyin.")

    st.markdown("---")
    st.markdown("### ➕ Yeni Kişi Ekle")
    new_name = st.text_input("Ad Soyad", placeholder="Ahmet Yılmaz")
    new_photo = st.file_uploader("Fotoğraf Yükle", type=["jpg", "jpeg", "png"],
                                  key="reg_photo")

    if st.button("💾 Kaydet", use_container_width=True):
        if new_name and new_photo:
            img_pil = Image.open(new_photo)
            img_np = pil_to_np(img_pil)
            encs = face_recognition.face_encodings(img_np)
            if encs:
                st.session_state.known_faces[new_name.strip()] = encs[0]
                st.success(f"✅ {new_name} kaydedildi!")
                st.rerun()
            else:
                st.error("❌ Fotoğrafta yüz bulunamadı.")
        else:
            st.warning("Ad ve fotoğraf giriniz.")

    st.markdown("---")
    st.markdown("### 📊 Sistem Bilgisi")
    st.markdown(f"""
    <div style='font-size:0.8rem; color: rgba(255,255,255,0.6);'>
    🌐 <b>Platform:</b> PaaS (Streamlit Cloud)<br>
    🐍 <b>Backend:</b> Python + OpenCV<br>
    🤖 <b>Model:</b> dlib HOG + face_recognition<br>
    ☁️ <b>Mimari:</b> Bulut Tabanlı
    </div>
    """, unsafe_allow_html=True)


# ── Ana İçerik Sekmeleri ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📸 Fotoğraf Analizi",
    "📷 Webcam Tespiti",
    "📋 Tespit Geçmişi"
])


# ─── TAB 1: Fotoğraf Analizi ──────────────────────────────────────────────────
with tab1:
    st.markdown("### 📸 Fotoğraf Yükleyerek Yüz Analizi")
    st.markdown("Bir veya birden fazla fotoğraf yükleyin, sistem yüzleri otomatik tespit edip tanıyacaktır.")

    uploaded_files = st.file_uploader(
        "Fotoğraf(lar) seçin",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="analyze_photos"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown(f"---\n#### 🖼️ `{uploaded_file.name}`")
            
            img_pil = Image.open(uploaded_file)
            img_np = pil_to_np(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            col_orig, col_result = st.columns(2)

            with col_orig:
                st.markdown("**Orijinal Görüntü**")
                st.image(img_pil, use_column_width=True)

            with st.spinner("🔍 Yüzler analiz ediliyor..."):
                start = time.time()
                annotated_bgr, results = process_image(img_bgr, tolerance)
                elapsed = time.time() - start

            with col_result:
                st.markdown("**Analiz Sonucu**")
                result_pil = np_to_pil(annotated_bgr)
                st.image(result_pil, use_column_width=True)

            # Sonuç özeti
            if results:
                log_detection(results, f"Fotoğraf: {uploaded_file.name}")
                st.markdown(f"⏱️ İşlem süresi: **{elapsed:.2f}s** | Bulunan yüz: **{len(results)}**")

                for i, r in enumerate(results, 1):
                    if r["recognized"]:
                        st.markdown(f"""<div class="result-box-success">
                            ✅ <b>Yüz #{i}</b>: {r['name']} — Güven: <b>{r['confidence']}%</b>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="result-box-warning">
                            ⚠️ <b>Yüz #{i}</b>: Bilinmeyen Kişi (veritabanında kayıt yok)
                        </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="result-box-info">
                    ℹ️ Bu görüntüde hiç yüz tespit edilemedi.
                </div>""", unsafe_allow_html=True)

            # İndir butonu
            buf = io.BytesIO()
            result_pil.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="⬇️ Sonucu İndir",
                data=buf.getvalue(),
                file_name=f"analiz_{uploaded_file.name}",
                mime="image/jpeg"
            )
    else:
        st.markdown("""<div class="result-box-info">
            📂 Lütfen sol taraftaki alandan fotoğraf yükleyin.<br>
            Birden fazla fotoğraf aynı anda işlenebilir.
        </div>""", unsafe_allow_html=True)


# ─── TAB 2: Webcam ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📷 Webcam ile Gerçek Zamanlı Yüz Tespiti")
    st.markdown("Kameranızdan anlık görüntü alarak yüz tespiti ve tanıma yapın.")

    st.info("""
    **Nasıl çalışır?**  
    Streamlit Cloud'da doğrudan webcam akışı teknik kısıtlamalar nedeniyle desteklenmez.  
    Bunun yerine **anlık fotoğraf** çekerek analiz yapabilirsiniz.  
    Bu, PaaS mimarisinin sunucu-taraflı işlem modeliyle tam uyumludur.
    """)

    cam_image = st.camera_input("📸 Kameradan Görüntü Al")

    if cam_image:
        img_pil = Image.open(cam_image)
        img_np = pil_to_np(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with st.spinner("🔍 Analiz ediliyor..."):
            start = time.time()
            annotated_bgr, results = process_image(img_bgr, tolerance)
            elapsed = time.time() - start

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Orijinal**")
            st.image(img_pil, use_column_width=True)
        with col_r:
            st.markdown("**Analiz Sonucu**")
            result_pil = np_to_pil(annotated_bgr)
            st.image(result_pil, use_column_width=True)

        st.markdown(f"⏱️ **İşlem süresi:** {elapsed:.2f}s | **Bulunan yüz:** {len(results)}")

        if results:
            log_detection(results, "Webcam")
            for i, r in enumerate(results, 1):
                if r["recognized"]:
                    st.markdown(f"""<div class="result-box-success">
                        ✅ <b>Yüz #{i}</b>: {r['name']} — Güven: <b>{r['confidence']}%</b>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="result-box-warning">
                        ⚠️ <b>Yüz #{i}</b>: Bilinmeyen Kişi
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="result-box-info">
                ℹ️ Görüntüde yüz tespit edilemedi. Kameraya bakın ve tekrar deneyin.
            </div>""", unsafe_allow_html=True)

        # Yeniden çek
        st.button("🔄 Yeni Görüntü Al", on_click=lambda: None)


# ─── TAB 3: Log / Geçmiş ──────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📋 Tespit Geçmişi")

    if st.session_state.detection_log:
        col_clear, col_export = st.columns([1, 5])
        with col_clear:
            if st.button("🗑️ Temizle"):
                st.session_state.detection_log = []
                st.session_state.total_detections = 0
                st.session_state.total_recognized = 0
                st.rerun()
        with col_export:
            log_json = json.dumps(st.session_state.detection_log, ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ JSON İndir",
                data=log_json,
                file_name="tespit_gecmisi.json",
                mime="application/json"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Tablo başlığı
        header_cols = st.columns([1, 2, 3, 2, 1])
        headers = ["Zaman", "Kaynak", "Kişi", "Güven", "Durum"]
        for hc, h in zip(header_cols, headers):
            hc.markdown(f"**{h}**")

        st.markdown("---")

        for log in reversed(st.session_state.detection_log):
            row = st.columns([1, 2, 3, 2, 1])
            row[0].markdown(f"`{log['time']}`")
            row[1].markdown(log["source"])
            row[2].markdown(f"**{log['name']}**")
            row[3].markdown(f"{log['confidence']}%" if log["confidence"] > 0 else "—")
            row[4].markdown("✅" if log["recognized"] else "⚠️")
    else:
        st.markdown("""<div class="result-box-info">
            📭 Henüz tespit kaydı bulunmuyor.<br>
            Fotoğraf analizi veya webcam tespiti yapıldıkça kayıtlar burada görünür.
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color: rgba(255,255,255,0.3); font-size:0.8rem;'>
    Bulut Bilişim Proje · PaaS Mimarisi ile Bulut Tabanlı Yüz Tanıma Sistemi<br>
    Emircan ALKAN · Kaan BAYDERE · Yiğit KARABULUT · Merve YILMAZ
</div>
""", unsafe_allow_html=True)
