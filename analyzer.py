# analyzer.py
import cv2
import mediapipe as mp
import numpy as np

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
LM         = mp_pose.PoseLandmark

# ── FUNGSI UMUM ──────────────────────────────────────────────
def hitung_sudut(lm, a, b, c, h, w):
    def p(i): return np.array([lm[i].x * w, lm[i].y * h])
    A, B, C = p(a), p(b), p(c)
    cos = np.dot(A-B, C-B) / (np.linalg.norm(A-B)*np.linalg.norm(C-B) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def klasifikasi_serangan(lm, h, w):
    skor = {"Pukulan": 0, "Tendangan": 0, "Sapuan": 0, "Tangkisan": 0}
    sk   = hitung_sudut(lm, LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST,  h, w)
    ski  = hitung_sudut(lm, LM.LEFT_SHOULDER,  LM.LEFT_ELBOW,  LM.LEFT_WRIST,   h, w)
    slk  = hitung_sudut(lm, LM.RIGHT_HIP,      LM.RIGHT_KNEE,  LM.RIGHT_ANKLE,  h, w)
    slki = hitung_sudut(lm, LM.LEFT_HIP,       LM.LEFT_KNEE,   LM.LEFT_ANKLE,   h, w)
    if sk   > 140: skor["Pukulan"]   += 2
    if ski  > 140: skor["Pukulan"]   += 2
    if sk   < 90:  skor["Tangkisan"] += 1
    if ski  < 90:  skor["Tangkisan"] += 1
    if slk  > 150 and lm[LM.RIGHT_ANKLE].y < lm[LM.RIGHT_HIP].y:
        skor["Tendangan"] += 3
    if slki > 150 and lm[LM.LEFT_ANKLE].y  < lm[LM.LEFT_HIP].y:
        skor["Tendangan"] += 3
    if slk  < 120 or slki < 120:
        skor["Sapuan"] += 2
    return skor, sk, ski, slk, slki

def buat_swot_fallback(dominan, sk, ski, slk, slki):
    """SWOT sederhana sebagai cadangan jika Gemini tidak tersedia."""
    swot = {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}
    swot["Strengths"].append(f"Teknik {dominan.lower()} terdeteksi dominan")
    if min(slk, slki) < 150:
        swot["Strengths"].append("Kuda-kuda stabil (lutut ditekuk)")
    if max(sk, ski) > 160 and min(sk, ski) > 160:
        swot["Weaknesses"].append("Kedua siku terlalu terbuka")
    if slk > 170 and slki > 170:
        swot["Weaknesses"].append("Kuda-kuda terlalu tinggi")
    if not swot["Weaknesses"]:
        swot["Weaknesses"].append("Perlu lebih banyak data untuk analisis mendalam")
    swot["Opportunities"].append("Latih variasi serangan non-dominan")
    swot["Opportunities"].append("Gabungkan teknik untuk combo")
    swot["Threats"].append("Pola serangan mudah dibaca lawan")
    swot["Threats"].append("Kurang fleksibilitas gerakan")
    return swot

def gambar_kerangka(img, landmarks):
    titik = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=3, circle_radius=4)
    garis = mp_drawing.DrawingSpec(color=(232, 84, 26),  thickness=2)
    mp_drawing.draw_landmarks(img, landmarks, mp_pose.POSE_CONNECTIONS, titik, garis)

def overlay_sudut(img, sk, ski, slk, slki):
    info = [
        (f"Siku Kanan : {sk:.1f}",   (10, 30)),
        (f"Siku Kiri  : {ski:.1f}",  (10, 55)),
        (f"Lutut Kanan: {slk:.1f}",  (10, 80)),
        (f"Lutut Kiri : {slki:.1f}", (10, 105)),
    ]
    ov = img.copy()
    cv2.rectangle(ov, (5, 5), (230, 118), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    for teks, pos in info:
        cv2.putText(img, teks, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 200, 255), 1, cv2.LINE_AA)

def _paket_hasil(mode, img, sk, ski, slk, slki, skor, dominan,
                 total_frame=0, gunakan_gemini=True):
    """Bungkus hasil analisis + panggil Gemini jika tersedia."""
    swot  = buat_swot_fallback(dominan, sk, ski, slk, slki)
    saran = []
    ringkasan = ""

    if gunakan_gemini:
        try:
            from gemini_analyzer import buat_analisis_gemini
            gem = buat_analisis_gemini(
                mode=mode,
                dominan=dominan,
                skor_serangan=skor,
                siku_kanan=sk,
                siku_kiri=ski,
                lutut_kanan=slk,
                lutut_kiri=slki,
                total_frame=total_frame,
            )
            if gem["sukses"]:
                swot      = gem["swot"]
                saran     = gem["saran"]
                ringkasan = gem["ringkasan"]
        except Exception:
            pass  # Tetap pakai fallback jika import gagal

    return {
        "mode":          mode,
        "image_bgr":     img,
        "siku_kanan":    sk,
        "siku_kiri":     ski,
        "lutut_kanan":   slk,
        "lutut_kiri":    slki,
        "skor_serangan": skor,
        "dominan":       dominan,
        "swot":          swot,
        "saran":         saran,
        "ringkasan":     ringkasan,
        "total_frame":   total_frame,
    }

# ── MODE 1: ANALISIS FOTO ────────────────────────────────────
def analisis_foto(image_path):
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2,
                        min_detection_confidence=0.5)
    img = cv2.imread(image_path)
    if img is None:
        pose.close()
        return None

    # Resize agar tidak berat (max 720p)
    h, w = img.shape[:2]
    if h > 720:
        skala = 720 / h
        img   = cv2.resize(img, (int(w * skala), int(h * skala)))

    h, w  = img.shape[:2]
    hasil = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pose.close()
    if not hasil.pose_landmarks:
        return None

    lm = hasil.pose_landmarks.landmark
    skor, sk, ski, slk, slki = klasifikasi_serangan(lm, h, w)
    dominan = max(skor, key=skor.get)
    gambar_kerangka(img, hasil.pose_landmarks)
    overlay_sudut(img, sk, ski, slk, slki)
    return _paket_hasil("foto", img, sk, ski, slk, slki, skor, dominan)

# ── MODE 2: ANALISIS VIDEO ───────────────────────────────────
def analisis_video(video_path, callback_frame=None, callback_selesai=None,
                   stop_flag=None):
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pose.close()
        if callback_selesai:
            callback_selesai(None)
        return

    total_skor   = {"Pukulan": 0, "Tendangan": 0, "Sapuan": 0, "Tangkisan": 0}
    sum_sk = sum_ski = sum_slk = sum_slki = 0.0
    frame_count  = 0
    frame_proses = 0
    last_frame   = None

    while True:
        if stop_flag and stop_flag[0]:
            break
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize frame agar ringan (max 480p)
        h, w = frame.shape[:2]
        if w > 854:
            skala = 854 / w
            frame = cv2.resize(frame, (854, int(h * skala)))

        # Proses setiap 5 frame
        if frame_count % 5 != 0:
            if callback_frame:
                callback_frame(frame.copy())
            continue

        h, w  = frame.shape[:2]
        hasil = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if hasil.pose_landmarks:
            lm = hasil.pose_landmarks.landmark
            skor, sk, ski, slk, slki = klasifikasi_serangan(lm, h, w)
            for k in total_skor:
                total_skor[k] += skor[k]
            sum_sk   += sk;  sum_ski  += ski
            sum_slk  += slk; sum_slki += slki
            frame_proses += 1
            gambar_kerangka(frame, hasil.pose_landmarks)
            overlay_sudut(frame, sk, ski, slk, slki)
            last_frame = frame.copy()

        if callback_frame:
            callback_frame(frame.copy())

    cap.release()
    pose.close()

    if frame_proses == 0:
        if callback_selesai:
            callback_selesai(None)
        return

    sk_r   = sum_sk   / frame_proses
    ski_r  = sum_ski  / frame_proses
    slk_r  = sum_slk  / frame_proses
    slki_r = sum_slki / frame_proses
    dominan = max(total_skor, key=total_skor.get)

    hasil_akhir = _paket_hasil(
        "video", last_frame, sk_r, ski_r, slk_r, slki_r,
        total_skor, dominan, total_frame=frame_proses
    )
    if callback_selesai:
        callback_selesai(hasil_akhir)

# ── MODE 3: KAMERA REAL-TIME ─────────────────────────────────
def analisis_kamera(callback_frame=None, callback_data=None, stop_flag=None):
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pose.close()
        return

    while True:
        if stop_flag and stop_flag[0]:
            break
        ret, frame = cap.read()
        if not ret:
            break
        h, w  = frame.shape[:2]
        hasil = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if hasil.pose_landmarks:
            lm = hasil.pose_landmarks.landmark
            skor, sk, ski, slk, slki = klasifikasi_serangan(lm, h, w)
            gambar_kerangka(frame, hasil.pose_landmarks)
            overlay_sudut(frame, sk, ski, slk, slki)
            if callback_data:
                callback_data(skor, sk, ski, slk, slki)
        if callback_frame:
            callback_frame(frame.copy())

    cap.release()
    pose.close()