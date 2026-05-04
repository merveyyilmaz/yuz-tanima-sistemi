import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
import time
from datetime import datetime
from scipy.spatial.distance import cosine

# ── Sayfa Yapılandırması ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bulut Tabanlı Yüz Tanıma Sistemi",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f0f4ff, #e8f0fe, #f5f0ff); }
    .main-header {
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        padding: 2rem; border-radius: 16px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(37,99,235,0.25);
    }
    .main-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p  { color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; }
    .metric-card {
        background: white; border: 1px solid #dbeafe; border-radius: 12px;
        padding: 1.2rem; text-align: center; box-shadow: 0 2px 8px rgba(37,99,235,0.08);
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
        background: #f8faff !important; border-right: 1px solid #dbeafe;
    }
    .stMarkdown p, .stMarkdown li { color: #1e293b; }
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
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


# ── Dedektör ve LBP ──────────────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def extract_lbp_features(gray_face, size=128):
    """
    LBP (Local Binary Pattern) tabanlı güçlü özellik vektörü.
    Aydınlatma değişimlerine ve küçük pozisyon farklarına dayanıklı.
    """
    resized = cv2.resize(gray_face, (size, size))
    eq      = cv2.equalizeHist(resized)

    # LBP hesapla
    lbp = np.zeros_like(eq)
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            center = eq[i, j]
            code = 0
            code |= (1 << 7) if eq[i-1, j-1] >= center else 0
            code |= (1 << 6) if eq[i-1, j  ] >= center else 0
            code |= (1 << 5) if eq[i-1, j+1] >= center else 0
            code |= (1 << 4) if eq[i,   j+1] >= center else 0
            code |= (1 << 3) if eq[i+1, j+1] >= center else 0
            code |= (1 << 2) if eq[i+1, j  ] >= center else 0
            code |= (1 << 1) if eq[i+1, j-1] >= center else 0
            code |= (1 << 0) if eq[i,   j-1] >= center else 0
            lbp[i, j] = code

    # Grid bazlı histogram (8x8 = 64 bölge)
    grid = 8
    cell = size // grid
    hist_all = []
    for r in range(grid):
        for c in range(grid):
            cell_lbp = lbp[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
            hist, _ = np.histogram(cell_lbp, bins=32, range=(0, 256))
            hist_all.extend(hist)

    vec = np.array(hist_all, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def get_face_vector(pil_img):
    """PIL görüntüsünden yüz tespiti yap ve LBP vektörü döndür"""
    detector = load_detector()
    gray = np.array(pil_img.convert("L"))

    faces = detector.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=6, minSize=(60, 60)
    )
    if len(faces) == 0:
        # Daha az katı ayarla tekrar dene
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    face_gray = gray[y:y+h, x:x+w]
    vec = extract_lbp_features(face_gray)
    return vec, (x, y, w, h)


def compare_faces(vec, known_vectors):
    """
    Kayıtlı tüm vektörlerle karşılaştır, en iyi skoru döndür.
    Cosine distance kullanır (1 = aynı, 0 = tamamen farklı).
    """
    scores = []
    for kv in known_vectors:
        sim = 1.0 - cosine(vec, kv)
        scores.append(sim)
    return max(scores), np.mean(scores)


def detect_and_recognize(pil_img, threshold=0.78):
    detector = load_detector()
    img_np   = np.array(pil_img.convert("RGB"))
    gray     = np.array(pil_img.convert("L"))
    annotated = img_np.copy()

    faces = detector.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=6, minSize=(60, 60)
    )
    if len(faces) == 0:
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )

    results = []
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        vec = extract_lbp_features(face_gray)

        name = "Bilinmeyen Kişi"
        confidence = 0.0
        color = (255, 140, 0)

        if st.session_state.known_faces:
            best_name  = None
            best_score = 0.0

            for kname, kvecs in st.session_state.known_faces.items():
                top_score, avg_score = compare_faces(vec, kvecs)
                # Hem en iyi hem ortalama skoru değerlendir
                combined = 0.7 * top_score + 0.3 * avg_score
                if combined > best_score:
                    best_score = combined
                    best_name  = kname

            if best_score >= threshold:
                name = best_name
                confidence = round(best_score * 100, 1)
                color = (0, 200, 100)

        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        label = f"{name} ({confidence}%)" if confidence > 0 else name
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x, y+h), (x+lw+8, y+h+lh+10), color, -1)
        cv2.putText(annotated, label, (x+4, y+h+lh+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        results.append({"name": name, "confidence": confidence, "recognized": confidence > 0})

    return annotated, results


def log_detection(results, source):
    ts = datetime.now().strftime("%H:%M:%S")
    for r in results:
        st.session_state.detection_log.append({
            "time": ts, "source": source,
            "name": str(r["name"]),
            "confidence": float(r["confidence"]),
            "recognized": bool(r["recognized"])
        })
    st.session_state.total_detections += len(results)
    st.session_state.total_recognized += sum(1 for r in results if r["recognized"])
    if len(st.session_state.detection_log) > 50:
        st.session_state.detection_log = st.session_state.detection_log[-50:]


# ── Session State ─────────────────────────────────────────────────────────────
if "known_faces" not in st.session_state:
    st.session_state.known_faces = {}
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "total_recognized" not in st.session_state:
    st.session_state.total_recognized = 0


# ── Başlık ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎭 Bulut Tabanlı Yüz Tanıma Sistemi</h1>
    <p>PaaS Mimarisi ile Geliştirilmiş · LBP + OpenCV · Streamlit Community Cloud</p>
</div>
""", unsafe_allow_html=True)

# ── Metrikler ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><h3>📚 Kayıtlı Kişi</h3>
        <p>{len(st.session_state.known_faces)}</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><h3>🔍 Toplam Tespit</h3>
        <p>{st.session_state.total_detections}</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><h3>✅ Tanınan</h3>
        <p>{st.session_state.total_recognized}</p></div>""", unsafe_allow_html=True)
with c4:
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
        min_value=0.65, max_value=0.95, value=0.78, step=0.01,
        help="Yüksek = daha katı, yanlış tanıma azalır"
    )

    st.markdown("---")
    st.markdown("### 📖 Kişi Veritabanı")
    if st.session_state.known_faces:
        for kname in list(st.session_state.known_faces.keys()):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                n_vecs = len(st.session_state.known_faces[kname])
                st.markdown(f"👤 **{kname}** `({n_vecs} foto)`")
            with col_b:
                if st.button("🗑️", key=f"del_{kname}"):
                    del st.session_state.known_faces[kname]
                    st.rerun()
    else:
        st.info("Henüz kayıtlı yüz yok.")

    st.markdown("---")
    st.markdown("### ➕ Yeni Kişi Ekle")
    st.caption("💡 Daha iyi tanıma için 3-5 farklı fotoğraf yükleyin")
    new_name   = st.text_input("Ad Soyad", placeholder="Ahmet Yılmaz")
    new_photos = st.file_uploader(
        "Fotoğraf(lar) Yükle",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        key="reg_photo"
    )

    if st.button("💾 Kaydet", use_container_width=True):
        if new_name and new_photos:
            vectors = []
            for photo in new_photos:
                pil_img = Image.open(photo).convert("RGB")
                vec, _ = get_face_vector(pil_img)
                if vec is not None:
                    vectors.append(vec)

            if vectors:
                key = new_name.strip()
                if key in st.session_state.known_faces:
                    st.session_state.known_faces[key].extend(vectors)
                else:
                    st.session_state.known_faces[key] = vectors
                st.success(f"✅ {new_name} kaydedildi! ({len(vectors)} yüz vektörü)")
                st.rerun()
            else:
                st.error("❌ Yüz bulunamadı. Net, yakın ve iyi aydınlatılmış fotoğraf deneyin.")
        else:
            st.warning("Ad ve fotoğraf giriniz.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color: #64748b;'>
    🌐 <b>Platform:</b> PaaS (Streamlit Cloud)<br>
    🤖 <b>Model:</b> LBP + Cosine Distance<br>
    🐍 <b>Backend:</b> Python + OpenCV<br>
    ☁️ <b>Mimari:</b> Bulut Tabanlı
    </div>
    """, unsafe_allow_html=True)


# ── Sekmeler ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸 Fotoğraf Analizi", "📷 Webcam Tespiti", "📋 Tespit Geçmişi"])

# ─── TAB 1 ───────────────────────────────────────────────────────────────────
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
                    ℹ️ Yüz tespit edilemedi. Daha net ve yakın bir fotoğraf deneyin.
                </div>""", unsafe_allow_html=True)

            buf = io.BytesIO()
            Image.fromarray(annotated_np).save(buf, format="JPEG", quality=95)
            st.download_button("⬇️ Sonucu İndir", buf.getvalue(),
                               file_name=f"analiz_{uf.name}", mime="image/jpeg")
    else:
        st.markdown("""<div class="result-box-info">
            📂 Analiz etmek istediğiniz fotoğrafı yükleyin.
        </div>""", unsafe_allow_html=True)

# ─── TAB 2 ───────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📷 Webcam ile Yüz Tespiti")
    st.info("Kameradan anlık fotoğraf çekip yüz tespiti ve tanıma yapabilirsiniz.")

    cam_img = st.camera_input("📸 Kameradan Görüntü Al")
    if cam_img:
        img_pil = Image.open(cam_img).convert("RGB")

        with st.spinner("🔍 Analiz ediliyor..."):
            start = time.time()
            annotated_np, results = detect_and_recognize(img_pil, threshold)
            elapsed = time.time() - start

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Orijinal**")
            st.image(img_pil, use_column_width=True)
        with col_b:
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

# ─── TAB 3 ───────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📋 Tespit Geçmişi")

    if st.session_state.detection_log:
        col_a, col_b = st.columns([1, 5])
        with col_a:
            if st.button("🗑️ Temizle"):
                st.session_state.detection_log = []
                st.session_state.total_detections = 0
                st.session_state.total_recognized = 0
                st.rerun()
        with col_b:
            log_json = json.dumps(st.session_state.detection_log, ensure_ascii=False, indent=2)
            st.download_button("⬇️ JSON İndir", log_json,
                               file_name="tespit_gecmisi.json", mime="application/json")

        st.markdown("<br>", unsafe_allow_html=True)
        hcols = st.columns([1, 2, 3, 2, 1])
        for hc, h in zip(hcols, ["Zaman","Kaynak","Kişi","Güven","Durum"]):
            hc.markdown(f"**{h}**")
        st.markdown("---")
        for entry in reversed(st.session_state.detection_log):
            row = st.columns([1, 2, 3, 2, 1])
            row[0].markdown(f"`{entry['time']}`")
            row[1].markdown(entry["source"])
            row[2].markdown(f"**{entry['name']}**")
            row[3].markdown(f"{entry['confidence']}%" if entry["confidence"] > 0 else "—")
            row[4].markdown("✅" if entry["recognized"] else "⚠️")
    else:
        st.markdown("""<div class="result-box-info">
            📭 Henüz tespit kaydı yok.
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color: #94a3b8; font-size:0.8rem;'>
    Bulut Bilişim Proje · PaaS Mimarisi ile Bulut Tabanlı Yüz Tanıma Sistemi<br>
    Emircan ALKAN · Kaan BAYDERE · Yiğit KARABULUT · Merve YILMAZ
</div>
""", unsafe_allow_html=True)
