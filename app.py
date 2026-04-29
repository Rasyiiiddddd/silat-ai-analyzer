# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from analyzer import analisis_foto, analisis_video

# ── KONFIGURASI HALAMAN ──────────────────────────────────────
st.set_page_config(
    page_title="SilatAI — Pose Analyzer",
    page_icon="🥋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS KUSTOM (tema hitam-oranye) ───────────────────────────
st.markdown("""
<style>
    /* Background utama */
    .stApp { background-color: #0f0f0f; color: #eeeeee; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #2a2a2a;
    }

    /* Tombol */
    .stButton > button {
        background-color: #E8541A;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #c0441a;
        color: white;
    }

    /* Upload box */
    .stFileUploader {
        background-color: #141414;
        border: 1.5px dashed #2a2a2a;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Metric card */
    [data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem;
    }
    [data-testid="stMetric"] label {
        color: #555555 !important;
        font-size: 11px !important;
        letter-spacing: 1px !important;
    }
    [data-testid="stMetricValue"] {
        color: #eeeeee !important;
        font-size: 22px !important;
    }
    [data-testid="stMetricDelta"] {
        color: #E8541A !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #E8541A;
    }

    /* Divider */
    hr { border-color: #2a2a2a; }

    /* Tab */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #141414;
        border-radius: 8px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #555;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E8541A !important;
        color: white !important;
    }

    /* Header brand */
    .brand-header {
        padding: 1rem 0;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 1rem;
    }
    .brand-title {
        color: #E8541A;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 2px;
        margin: 0;
    }
    .brand-sub {
        color: #555;
        font-size: 10px;
        letter-spacing: 4px;
        margin: 0;
    }

    /* Card SWOT */
    .swot-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        height: 100%;
    }
    .swot-title {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .swot-item {
        color: #aaa;
        font-size: 13px;
        margin: 0.2rem 0;
    }

    /* Badge status */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .badge-selesai { background: #0F6E56; color: white; }
    .badge-proses  { background: #854F0B; color: white; }
    .badge-error   { background: #A32D2D; color: white; }

    /* Ringkasan Gemini */
    .ringkasan-box {
        background: #141414;
        border-left: 3px solid #E8541A;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        color: #ccc;
        font-size: 14px;
        line-height: 1.7;
        margin: 0.5rem 0;
    }

    /* Saran item */
    .saran-item {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        color: #ccc;
        font-size: 13px;
        margin: 0.3rem 0;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <p class="brand-title">SILATAI</p>
        <p class="brand-sub">POSE ANALYZER</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Mode Analisis**")
    mode = st.radio(
        label="mode",
        options=["Foto", "Video"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("<span style='color:#555;font-size:12px'>Tentang Aplikasi</span>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#444;font-size:12px;line-height:1.6'>
    SilatAI mendeteksi pose tubuh pesilat menggunakan MediaPipe AI dan memberikan analisis gerakan secara otomatis.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='color:#333;font-size:11px'>
    MediaPipe · OpenCV · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("## Analisis Pose")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)


# ── FUNGSI TAMPILKAN HASIL ───────────────────────────────────
def tampil_hasil(hasil):
    if hasil is None:
        st.error("Pose tidak terdeteksi. Pastikan tubuh penuh terlihat jelas.")
        return

    st.markdown(
        f'<span class="badge badge-selesai">SELESAI</span> &nbsp; '
        f'Teknik dominan: <b style="color:#E8541A">{hasil["dominan"]}</b>'
        + (f' &nbsp;·&nbsp; {hasil["total_frame"]} frame diproses'
           if hasil.get("mode") == "video" else ""),
        unsafe_allow_html=True,
    )

    st.divider()

    # Preview gambar hasil
    img_rgb = cv2.cvtColor(hasil["image_bgr"], cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Hasil deteksi kerangka tubuh", use_column_width=True)

    st.divider()

    # Kartu sudut sendi
    st.markdown("#### Sudut Sendi")
    c1, c2, c3, c4 = st.columns(4)
    subs = [
        "PUKULAN LURUS" if hasil["siku_kanan"]  > 140 else "TANGKISAN",
        "PUKULAN LURUS" if hasil["siku_kiri"]   > 140 else "TANGKISAN",
        "KUDA-KUDA STABIL" if hasil["lutut_kanan"] < 150 else "POSISI TINGGI",
        "KUDA-KUDA STABIL" if hasil["lutut_kiri"]  < 150 else "POSISI TINGGI",
    ]
    with c1: st.metric("SIKU KANAN",  f'{hasil["siku_kanan"]:.1f}°',  subs[0])
    with c2: st.metric("SIKU KIRI",   f'{hasil["siku_kiri"]:.1f}°',   subs[1])
    with c3: st.metric("LUTUT KANAN", f'{hasil["lutut_kanan"]:.1f}°', subs[2])
    with c4: st.metric("LUTUT KIRI",  f'{hasil["lutut_kiri"]:.1f}°',  subs[3])

    st.divider()

    # Grafik kecenderungan serangan
    st.markdown("#### Kecenderungan Serangan")
    skor = hasil["skor_serangan"]
    total = sum(skor.values()) or 1
    for nama, val in skor.items():
        pct = val / total
        col_label, col_bar, col_pct = st.columns([2, 7, 1])
        with col_label:
            st.markdown(
                f"<div style='color:#eee;font-size:13px;padding-top:6px'>{nama}</div>",
                unsafe_allow_html=True)
        with col_bar:
            st.progress(pct)
        with col_pct:
            st.markdown(
                f"<div style='color:#eee;font-size:13px;padding-top:6px'>{round(pct*100)}%</div>",
                unsafe_allow_html=True)

    st.divider()

    # Tab: SWOT + Saran + Ringkasan Gemini
    tab1, tab2, tab3 = st.tabs(["SWOT Report", "Saran Latihan", "Analisis Pelatih"])

    with tab1:
        swot = hasil.get("swot", {})
        warna = {
            "Strengths":     "#2ECC71",
            "Weaknesses":    "#E74C3C",
            "Opportunities": "#3498DB",
            "Threats":       "#F39C12",
        }
        col_s, col_w = st.columns(2)
        col_o, col_t = st.columns(2)
        cols = {"Strengths": col_s, "Weaknesses": col_w,
                "Opportunities": col_o, "Threats": col_t}
        for key, col in cols.items():
            with col:
                poin = swot.get(key, [])
                isi  = "".join(
                    f'<div class="swot-item">• {p}</div>' for p in poin
                ) if poin else '<div class="swot-item">—</div>'
                st.markdown(f"""
                <div class="swot-card">
                    <div class="swot-title" style="color:{warna[key]}">{key.upper()}</div>
                    {isi}
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        saran = hasil.get("saran", [])
        if saran:
            for s in saran:
                st.markdown(
                    f'<div class="saran-item">✅ &nbsp; {s}</div>',
                    unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#555;font-size:13px">Hubungkan Gemini API untuk saran latihan personal.</div>',
                unsafe_allow_html=True)

    with tab3:
        ringkasan = hasil.get("ringkasan", "")
        if ringkasan:
            st.markdown(
                f'<div class="ringkasan-box">{ringkasan}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#555;font-size:13px">Hubungkan Gemini API untuk analisis pelatih.</div>',
                unsafe_allow_html=True)


# ── MODE FOTO ────────────────────────────────────────────────
if mode == "Foto":
    st.markdown("##### Upload Foto Pesilat")
    uploaded = st.file_uploader(
        label="foto",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded.name)[1]
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Menganalisis pose..."):
            hasil = analisis_foto(tmp_path)

        os.unlink(tmp_path)
        tampil_hasil(hasil)


# ── MODE VIDEO ───────────────────────────────────────────────
elif mode == "Video":
    st.markdown("##### Upload Video Pesilat")
    uploaded = st.file_uploader(
        label="video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded.name)[1]
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        progress_bar = st.progress(0, text="Memproses video...")
        preview_slot = st.empty()
        frame_counter = [0]

        def update_frame(frame):
            frame_counter[0] += 1
            if frame_counter[0] % 10 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_slot.image(rgb, use_column_width=True)

        hasil_container = [None]

        def selesai(hasil):
            hasil_container[0] = hasil

        from analyzer import analisis_video
        analisis_video(
            video_path=tmp_path,
            callback_frame=update_frame,
            callback_selesai=selesai,
        )

        progress_bar.progress(1.0, text="Selesai!")
        os.unlink(tmp_path)
        preview_slot.empty()

        tampil_hasil(hasil_container[0])
    