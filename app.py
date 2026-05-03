import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import json
import time
from datetime import datetime

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
    /* Aydınlık arka plan */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff, #e8f0fe, #f5f0ff);
    }

    /* Başlık kutusu */
    .main-header {
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.25);
    }
    .main-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p  { color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; }

    /* Metrik kartlar */
    .metric-card {
        background: white;
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(37,99,235,0.08);
    }
    .metric-card h3 { color: #2563eb; font-size: 0.85rem; margin: 0 0 0.4rem 0; text-transform: uppercase; }
    .metric-card p  { color: #1e293b; font-size: 2rem; font-weight: 700; margin: 0; }

    /* Sonuç kutuları */
    .result-box-success {
        background: #f0fdf4; border: 1px solid #86efac;
        border-radius: 12px; padding: 1rem 1.5rem; margin: 0.5rem 0; color: #166534;
    }
    .result-box-warning {
        background: #fffbeb; border: 1px solid #fcd34d;
        border-radius: 12px; padding: 1rem 1.5rem; margin: 0.5rem 0; color: #92400e;
    }
    .result-box-info {
        background: #eff6ff; border: 1px solid #bfdbfe;
        border-radius: 12px; padding: 1rem 1.5rem; margin: 0.5rem 0; color: #1e40af;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f8faff !important;
        border-right: 1px solid #dbeafe;
    }

    /* Genel metin */
    .stMarkdown p, .stMarkdown li { color: #1e293b; }

    /* Butonlar */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
        box-shadow: 0 2px 8px rgba(37,99,235,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    }

    /* Sekmeler */
    .stTabs [data-baseweb="tab-list"] {
        background: #e0eaff; border-radius: 10px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] { color: #3b5fc0; border-radius: 8px; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2563eb, #0ea5e9) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── DeepFace Lazy Import ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Yüz tanıma modeli yükleniyor...")
def load_deepface():
    from deepface import DeepFace
    return DeepFace

# ── Session State ─────────────────────────────────────────────────────────────
if "known_faces" not in st.session_state:
    st.session_state.known_faces = {}       # {isim: PIL.Image}
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "total_recognized" not in st.session_state:
    st.session_state.total_recognized = 0

# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────
def pil_to_np(pil_img):
    return np.array(pil_img.convert("RGB"))

def np_to_pil(np_img_rgb):
    return Image.fromarray(np_img_rgb)

def detect_and_recognize(image_np, tolerance=0.4):
    """
    DeepFace ile yüz tespiti ve tanıma.
    Döndürür: annotated_image (RGB numpy), results (list of dict)
    """
    DeepFace = load_deepface()
    results = []
    annotated = image_np.copy()

    # 1) Yüz tespiti
    try:
        faces = DeepFace.extract_faces(
            img_path=image_np,
            detector_backend="opencv",
            enforce_detection=False
        )
    except Exception:
        faces = []

    for face_obj in faces:
        region = face_obj.get("facial_area", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)
        confidence_det = face_obj.get("confidence", 0)

        if w < 20 or h < 20 or confidence_det < 0.5:
            continue

        name = "Bilinmeyen Kişi"
        match_confidence = 0.0
        color = (255, 140, 0)  # Turuncu = bilinmeyen

        # 2) Kayıtlı yüzlerle karşılaştır
        if st.session_state.known_faces:
            face_crop = image_np[y:y+h, x:x+w]
            if face_crop.size > 0:
                best_score = 0.0
                best_name = None
                for kname, kimg_pil in st.session_state.known_faces.items():
                    try:
                        kimg_np = pil_to_np(kimg_pil)
                        result = DeepFace.verify(
                            img1_path=face_crop,
                            img2_path=kimg_np,
                            model_name="Facenet",
                            detector_backend="skip",
                            enforce_detection=False
                        )
                        dist = result.get("distance", 1.0)
                        score = max(0.0, (1.0 - dist) * 100)
                        if result.get("verified", False) and score > best_score:
                            best_score = score
                            best_name = kname
                    except Exception:
                        continue

                if best_name and best_score > (1 - tolerance) * 100:
                    name = best_name
                    match_confidence = round(best_score, 1)
                    color = (0, 200, 100)  # Yeşil = tanınan

        # Dikdörtgen çiz
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        label = f"{name} ({match_confidence}%)" if match_confidence > 0 else name
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x, y+h), (x+lw+8, y+h+lh+10), color, -1)
        cv2.putText(annotated, label, (x+4, y+h+lh+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        results.append({
            "name": name,
            "confidence": match_confidence,
            "recognized": match_confidence > 0
        })

    return annotated, results


def log_detection(results, source):
    ts = datetime.now().strftime("%H:%M:%S")
    for r in results:
        st.session_state.detection_log.append({
            "time": ts, "source": source,
            "name": r["name"], "confidence": r["confidence"],
            "recognized": r["recognized"]
        })
    st.session_state.total_detections += len(results)
    st.session_state.total_recognized += sum(1 for r in results if r["recognized"])
    if len(st.session_state.detection_log) > 50:
        st.session_state.detection_log = st.session_state.detection_log[-50:]


# ── Başlık ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎭 Bulut Tabanlı Yüz Tanıma Sistemi</h1>
    <p>PaaS Mimarisi ile Geliştirilmiş · DeepFace + OpenCV · Streamlit Community Cloud</p>
</div>
""", unsafe_allow_html=True)

# ── Metrikler ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card"><h3>📚 Kayıtlı Kişi</h3>
        <p>{len(st.session_state.known_faces)}</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card"><h3>🔍 Toplam Tespit</h3>
        <p>{st.session_state.total_detections}</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card"><h3>✅ Tanınan</h3>
        <p>{st.session_state.total_recognized}</p></div>""", unsafe_allow_html=True)
with col4:
    acc = (st.session_state.total_recognized / st.session_state.total_detections * 100
           if st.session_state.total_detections > 0 else 0)
    st.markdown(f"""<div class="metric-card"><h3>🎯 Tanıma Oranı</h3>
        <p>{acc:.0f}%</p></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")
    st.markdown("---")

    tolerance = st.slider(
        "🎯 Eşleşme Toleransı",
        min_value=0.3, max_value=0.7, value=0.4, step=0.05,
        help="Düşük = daha katı eşleşme"
    )

    st.markdown("---")
    st.markdown("### 📖 Kişi Veritabanı")

    if st.session_state.known_faces:
        for kname in list(st.session_state.known_faces.keys()):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"👤 **{kname}**")
            with c2:
                if st.button("🗑️", key=f"del_{kname}"):
                    del st.session_state.known_faces[kname]
                    st.rerun()
    else:
        st.info("Henüz kayıtlı yüz yok.\nAşağıdan kişi ekleyin.")

    st.markdown("---")
    st.markdown("### ➕ Yeni Kişi Ekle")
    new_name = st.text_input("Ad Soyad", placeholder="Ahmet Yılmaz")
    new_photo = st.file_uploader("Fotoğraf Yükle", type=["jpg","jpeg","png"], key="reg_photo")

    if st.button("💾 Kaydet", use_container_width=True):
        if new_name and new_photo:
            pil_img = Image.open(new_photo).convert("RGB")
            # Yüz var mı kontrol et
            DeepFace = load_deepface()
            try:
                faces = DeepFace.extract_faces(
                    img_path=pil_to_np(pil_img),
                    detector_backend="opencv",
                    enforce_detection=True
                )
                if faces:
                    st.session_state.known_faces[new_name.strip()] = pil_img
                    st.success(f"✅ {new_name} kaydedildi!")
                    st.rerun()
                else:
                    st.error("❌ Fotoğrafta yüz bulunamadı.")
            except Exception:
                st.error("❌ Fotoğrafta yüz bulunamadı.")
        else:
            st.warning("Ad ve fotoğraf giriniz.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color: #64748b;'>
    🌐 <b>Platform:</b> PaaS (Streamlit Cloud)<br>
    🤖 <b>Model:</b> DeepFace · Facenet<br>
    🐍 <b>Backend:</b> Python + OpenCV<br>
    ☁️ <b>Mimari:</b> Bulut Tabanlı
    </div>
    """, unsafe_allow_html=True)

# ── Sekmeler ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸 Fotoğraf Analizi", "📷 Webcam Tespiti", "📋 Tespit Geçmişi"])

# ─── TAB 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📸 Fotoğraf Yükleyerek Yüz Analizi")

    uploaded_files = st.file_uploader(
        "Fotoğraf(lar) seçin",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        key="analyze_photos"
    )

    if uploaded_files:
        for uf in uploaded_files:
            st.markdown(f"---\n#### 🖼️ `{uf.name}`")
            img_pil = Image.open(uf).convert("RGB")
            img_np  = pil_to_np(img_pil)

            col_orig, col_res = st.columns(2)
            with col_orig:
                st.markdown("**Orijinal**")
                st.image(img_pil, use_column_width=True)

            with st.spinner("🔍 Analiz ediliyor..."):
                start = time.time()
                annotated_np, results = detect_and_recognize(img_np, tolerance)
                elapsed = time.time() - start

            with col_res:
                st.markdown("**Analiz Sonucu**")
                st.image(annotated_np, use_column_width=True)

            if results:
                log_detection(results, f"Fotoğraf: {uf.name}")
                st.markdown(f"⏱️ İşlem süresi: **{elapsed:.2f}s** | Bulunan yüz: **{len(results)}**")
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
                    ℹ️ Bu görüntüde yüz tespit edilemedi.
                </div>""", unsafe_allow_html=True)

            buf = io.BytesIO()
            np_to_pil(annotated_np).save(buf, format="JPEG", quality=95)
            st.download_button("⬇️ Sonucu İndir", buf.getvalue(),
                               file_name=f"analiz_{uf.name}", mime="image/jpeg")
    else:
        st.markdown("""<div class="result-box-info">
            📂 Analiz etmek istediğiniz fotoğrafı yükleyin.<br>
            Birden fazla fotoğraf aynı anda işlenebilir.
        </div>""", unsafe_allow_html=True)

# ─── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📷 Webcam ile Yüz Tespiti")
    st.info("""
    **Nasıl çalışır?**  
    Kameradan anlık fotoğraf çekip analiz edebilirsiniz.  
    Bu, PaaS mimarisinin sunucu-taraflı işlem modeliyle tam uyumludur.
    """)

    cam_img = st.camera_input("📸 Kameradan Görüntü Al")

    if cam_img:
        img_pil = Image.open(cam_img).convert("RGB")
        img_np  = pil_to_np(img_pil)

        with st.spinner("🔍 Analiz ediliyor..."):
            start = time.time()
            annotated_np, results = detect_and_recognize(img_np, tolerance)
            elapsed = time.time() - start

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Orijinal**")
            st.image(img_pil, use_column_width=True)
        with c2:
            st.markdown("**Analiz Sonucu**")
            st.image(annotated_np, use_column_width=True)

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
                ℹ️ Yüz tespit edilemedi. Kameraya bakın ve tekrar deneyin.
            </div>""", unsafe_allow_html=True)

# ─── TAB 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📋 Tespit Geçmişi")

    if st.session_state.detection_log:
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("🗑️ Temizle"):
                st.session_state.detection_log = []
                st.session_state.total_detections = 0
                st.session_state.total_recognized = 0
                st.rerun()
        with c2:
            log_json = json.dumps(st.session_state.detection_log, ensure_ascii=False, indent=2)
            st.download_button("⬇️ JSON İndir", log_json,
                               file_name="tespit_gecmisi.json", mime="application/json")

        st.markdown("<br>", unsafe_allow_html=True)
        hcols = st.columns([1, 2, 3, 2, 1])
        for hc, h in zip(hcols, ["Zaman","Kaynak","Kişi","Güven","Durum"]):
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
            📭 Henüz tespit kaydı yok. Fotoğraf analizi yapınca burada görünür.
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color: #94a3b8; font-size:0.8rem;'>
    Bulut Bilişim Proje · PaaS Mimarisi ile Bulut Tabanlı Yüz Tanıma Sistemi<br>
    Emircan ALKAN · Kaan BAYDERE · Yiğit KARABULUT · Merve YILMAZ
</div>
""", unsafe_allow_html=True)
