# gemini_analyzer.py
from google import genai
from google.genai import types
import json

# ── KONFIGURASI ──────────────────────────────────────────────
API_KEY = "AIzaSyDvzufTbP22PPzE_ZBQiniUV2fUuKNHc18"

client = genai.Client(api_key=API_KEY)


def buat_analisis_gemini(
    mode: str,
    dominan: str,
    skor_serangan: dict,
    siku_kanan: float,
    siku_kiri: float,
    lutut_kanan: float,
    lutut_kiri: float,
    total_frame: int = 0,
) -> dict:
    total = sum(skor_serangan.values()) or 1
    pct   = {k: round(v / total * 100) for k, v in skor_serangan.items()}
    info_mode = f"video ({total_frame} frame terdeteksi)" if mode == "video" else mode

    prompt = f"""
Kamu adalah pelatih pencak silat profesional berpengalaman yang menganalisis gerakan atlet.

Data hasil deteksi pose AI dari {info_mode}:

SUDUT SENDI:
- Siku Kanan : {siku_kanan:.1f}°
- Siku Kiri  : {siku_kiri:.1f}°
- Lutut Kanan: {lutut_kanan:.1f}°
- Lutut Kiri : {lutut_kiri:.1f}°

KECENDERUNGAN SERANGAN:
- Pukulan   : {pct['Pukulan']}%
- Tendangan : {pct['Tendangan']}%
- Sapuan    : {pct['Sapuan']}%
- Tangkisan : {pct['Tangkisan']}%
- Teknik dominan: {dominan}

Berikan analisis dalam format JSON berikut (HANYA JSON, tanpa teks lain):
{{
  "swot": {{
    "Strengths": ["poin 1", "poin 2", "poin 3"],
    "Weaknesses": ["poin 1", "poin 2"],
    "Opportunities": ["poin 1", "poin 2"],
    "Threats": ["poin 1", "poin 2"]
  }},
  "saran": ["saran latihan 1", "saran latihan 2", "saran latihan 3"],
  "ringkasan": "Paragraf singkat 2-3 kalimat dari sudut pandang pelatih tentang kemampuan atlet ini."
}}

Gunakan bahasa Indonesia. Analisis harus spesifik berdasarkan data sudut sendi dan kecenderungan serangan di atas.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        teks = response.text.strip()

        # Bersihkan markdown code block jika ada
        if "```" in teks:
            teks = teks.split("```")[1]
            if teks.startswith("json"):
                teks = teks[4:]
        teks = teks.strip()

        data = json.loads(teks)
        return {
            "sukses":    True,
            "swot":      data.get("swot", {}),
            "saran":     data.get("saran", []),
            "ringkasan": data.get("ringkasan", ""),
        }

    except Exception as e:
        return {
            "sukses":    False,
            "error":     str(e),
            "swot": {
                "Strengths":     [f"Teknik {dominan.lower()} terdeteksi dominan"],
                "Weaknesses":    ["Koneksi ke Gemini gagal — analisis terbatas"],
                "Opportunities": ["Coba analisis ulang dengan koneksi internet stabil"],
                "Threats":       ["Data tidak lengkap untuk analisis mendalam"],
            },
            "saran":     ["Pastikan API key Gemini sudah benar", "Cek koneksi internet"],
            "ringkasan": f"Teknik dominan terdeteksi: {dominan}. Hubungkan ke Gemini untuk analisis lebih lengkap.",
        }