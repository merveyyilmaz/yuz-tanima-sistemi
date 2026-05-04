import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

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
    .stApp {
        background: linear-gradient(135deg, #f0f4ff, #e8f0fe, #f5f0ff);
    }
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
    section[data-testid="stSidebar"] {
        background: #f8faff !important;
        border-right: 1px solid #dbeafe;
    }
    .stMarkdown p, .stMarkdown li { color: #1e293b; }
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
        box-shadow: 0 2px 8px rgba(37,99,235,0.2);
    }
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


# ── OpenCV Haar Cascade Yükle ─────────────────────────────────────────────────
@st.cache_resource
def load_face_detector():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return face_cascade


def extract_face_vector(img_gray, x, y, w, h, size=64):
    """Yüz bölgesini sabit boyuta getir ve düzleştir → basit özellik vektörü"""
    face_roi = img_gray[y:y+h, x:x+w]
    if face_roi.size == 0:
        return None
    face_resized = cv2.resize(face_roi, (size, size))
    # Histogram eşitleme → aydınlatma farkına karşı dayanıklı
    face_eq = cv2.equalizeHist(face_resized)
    vector = face_eq.flatten().astype(np.float32)
    # Normalize et
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


# ── Session State ─────────────────────────────────────────────────────────────
if "known_faces" not in st.session_state:
    st.session_state.known_faces = {}       # {isim: [vektör listesi]}
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "total_recognized" not in st.session_state:
    st.session_state.total_recognized = 0


# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────
def pil_to_gray(pil_img):
    return np.array(pil_img.convert("L"))

def pil_to_np(pil_img):
    return np.array(pil_img.convert("RGB"))


def detect_and_recognize(image_pil, threshold=0.75):
    """
    PIL görüntüsü alır, yüzleri tespit eder ve tanır.
    Döndürür: annotated_image (RGB numpy), results (list of dict)
    """
    face_cascade = load_face_detector()
    img_np  = pil_to_np(image_pil)
    img_gray = pil_to_gray(image_pil)
    annotated = img_np.copy()

    # Yüz tespiti
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    results = []

    if len(faces) == 0:
        return annotated, results

    for (x, y, w, h) in faces:
        name = "Bilinmeyen Kişi"
        confidence = 0.0
        color = (255, 140, 0)  # BGR turuncu

        # Özellik vektörü çıkar
        vec = extract_face_vector(img_gray, x, y, w, h)

        if vec is not None and st.session_state.known_faces:
            best_score = 0.0
            best_name = None

            for kname, kvectors in st.session_state.known_faces.items():
                for kv in kvectors:
                    score = float(cosine_similarity([vec], [kv])[0][0])
                    if score > best_score:
                        best_score = score
                        best_name = kname

            if best_score >= threshold:
                name = best_name
                confidence = round(best_score * 100, 1)
                color = (0, 200, 100)  # BGR yeşil

        # Dikdörtgen çiz
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        label = f"{name} ({confidence}%)" if confidence > 0 else name
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x, y+h), (x+lw+8, y+h+lh+10), color, -1)
        cv2.putText(annotated, label, (x+4, y+h+lh+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        results.append({
            "name": name,
            "confidence": confidence,
            "recognized": confidence > 0
        })

    return annotated, results


def register_face(pil_img):
    """Kayıt fotoğrafından yüz vektörü çıkar"""
    face_cascade = load_face_detector()
    img_gray = pil_to_gray(pil_img)

    faces = face_cascade.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    # En büyük yüzü al
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    return extract_face_vector(img_gray, x, y, w, h)


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
    <p>PaaS Mimarisi ile Geliştirilmiş · OpenCV Haar Cascade · Streamlit Community Cloud</p>
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

    threshold = st.slider(
        "🎯 Tanıma Eşiği",
        min_value=0.60, max_value=0.90, value=0.85, step=0.01,
        help="Yüksek = daha katı eşleşme"
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
    new_photos = st.file_uploader(
        "Fotoğraf Yükle (1-3 adet)",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        key="reg_photo"
    )

    if st.button("💾 Kaydet", use_container_width=True):
        if new_name and new_photos:
            vectors = []
            for photo in new_photos:
                pil_img = Image.open(photo).convert("RGB")
                vec = register_face(pil_img)
                if vec is not None:
                    vectors.append(vec)

            if vectors:
                name_key = new_name.strip()
                if name_key not in st.session_state.known_faces:
                    st.session_state.known_faces[name_key] = vectors
                else:
                    st.session_state.known_faces[name_key].extend(vectors)
                st.success(f"✅ {new_name} kaydedildi! ({len(vectors)} yüz)")
                st.rerun()
            else:
                st.error("❌ Fotoğraflarda yüz bulunamadı. Net ve yakın bir fotoğraf deneyin.")
        else:
            st.warning("Ad ve fotoğraf giriniz.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color: #64748b;'>
    🌐 <b>Platform:</b> PaaS (Streamlit Cloud)<br>
    🤖 <b>Model:</b> OpenCV Haar Cascade<br>
    🐍 <b>Backend:</b> Python + scikit-learn<br>
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

            col_orig, col_res = st.columns(2)
            with col_orig:
                st.markdown("**Orijinal**")
                st.image(img_pil, use_column_width=True)

            with st.spinner("🔍 Analiz ediliyor..."):
                start = time.time()
                annotated_np, results = detect_and_recognize(img_pil, threshold)
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
                    ℹ️ Bu görüntüde yüz tespit edilemedi. Daha net ve yakın bir fotoğraf deneyin.
                </div>""", unsafe_allow_html=True)

            buf = io.BytesIO()
            Image.fromarray(annotated_np).save(buf, format="JPEG", quality=95)
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
    Kameradan anlık fotoğraf çekip yüz tespiti ve tanıma yapabilirsiniz.
    """)

    cam_img = st.camera_input("📸 Kameradan Görüntü Al")

    if cam_img:
        img_pil = Image.open(cam_img).convert("RGB")

        with st.spinner("🔍 Analiz ediliyor..."):
            start = time.time()
            annotated_np, results = detect_and_recognize(img_pil, threshold)
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
