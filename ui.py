# ui.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import threading
import os
import cv2
from analyzer import analisis_foto, analisis_video, analisis_kamera

ORANGE  = "#E8541A"
BG_DARK = "#0f0f0f"
BG_MID  = "#141414"
BG_CARD = "#1a1a1a"
BORDER  = "#2a2a2a"
TEXT    = "#eeeeee"
MUTED   = "#555555"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class SilatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SilatAI — Pose Analyzer")
        self.geometry("1100x720")
        self.configure(fg_color=BG_DARK)
        self.resizable(True, True)
        self._hasil      = None
        self._stop_flag  = [False]
        self._mode_aktif = None
        self._build_ui()

    def _build_ui(self):
        sidebar = ctk.CTkFrame(self, fg_color="#0a0a0a", corner_radius=0,
                               border_width=1, border_color=BORDER)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        sidebar.configure(width=200)

        ctk.CTkLabel(sidebar, text="SILATAI", font=("Segoe UI", 16, "bold"),
                     text_color=ORANGE).pack(pady=(24, 2), padx=20, anchor="w")
        ctk.CTkLabel(sidebar, text="POSE ANALYZER", font=("Segoe UI", 9),
                     text_color=MUTED).pack(padx=20, anchor="w")
        ctk.CTkFrame(sidebar, height=1, fg_color=BORDER).pack(fill="x", pady=16)

        for item in ["Analisis Pose", "SWOT Report", "Grafik Serangan", "Riwayat"]:
            ctk.CTkButton(
                sidebar, text=item, anchor="w",
                fg_color="transparent", hover_color=BG_CARD,
                text_color=MUTED, font=("Segoe UI", 13),
                corner_radius=0, height=38
            ).pack(fill="x")

        main = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        main.pack(side="left", fill="both", expand=True)

        top = ctk.CTkFrame(main, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(20, 0))
        ctk.CTkLabel(top, text="Analisis Pose", font=("Segoe UI", 18, "bold"),
                     text_color=TEXT).pack(side="left")
        self.badge = ctk.CTkLabel(top, text=" SIAP ", font=("Segoe UI", 10, "bold"),
                                  text_color="white", fg_color="#444", corner_radius=10)
        self.badge.pack(side="right", padx=4)

        self.scroll = ctk.CTkScrollableFrame(main, fg_color="transparent",
                                             scrollbar_button_color=BORDER)
        self.scroll.pack(fill="both", expand=True, padx=24, pady=16)

        self._build_mode_selector()
        self._build_upload_zone()
        self._build_preview()
        self._build_cards()
        self._build_grafik()
        self._build_ringkasan()
        self._build_swot()
        self._build_saran()
        self._build_tombol()

    # ── MODE SELECTOR ────────────────────────────────────────
    def _build_mode_selector(self):
        frame = ctk.CTkFrame(self.scroll, fg_color=BG_CARD, corner_radius=8,
                             border_width=1, border_color=BORDER)
        frame.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(frame, text="Pilih Mode Analisis", font=("Segoe UI", 11),
                     text_color=MUTED).pack(padx=16, pady=(12, 8), anchor="w")
        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 12))
        self.btn_foto   = self._buat_mode_btn(btn_row, "Foto",        self._set_mode_foto)
        self.btn_video  = self._buat_mode_btn(btn_row, "Video",       self._set_mode_video)
        self.btn_kamera = self._buat_mode_btn(btn_row, "Kamera Live", self._set_mode_kamera)
        self.btn_foto.pack(side="left", padx=(0, 8))
        self.btn_video.pack(side="left", padx=(0, 8))
        self.btn_kamera.pack(side="left")
        self._set_mode_foto()

    def _buat_mode_btn(self, parent, label, cmd):
        return ctk.CTkButton(
            parent, text=label, width=120, height=34,
            fg_color="transparent", border_width=1, border_color=BORDER,
            hover_color=BG_DARK, text_color=MUTED,
            font=("Segoe UI", 12), corner_radius=6, command=cmd
        )

    def _reset_mode_btns(self):
        for btn in [self.btn_foto, self.btn_video, self.btn_kamera]:
            btn.configure(fg_color="transparent", border_color=BORDER, text_color=MUTED)

    def _set_mode_foto(self):
        self._stop_kamera()
        self._reset_mode_btns()
        self.btn_foto.configure(fg_color=ORANGE, border_color=ORANGE, text_color="white")
        self._mode_aktif = "foto"
        if hasattr(self, "lbl_hint"):
            self.lbl_hint.configure(text="Klik untuk upload foto pesilat")
        if hasattr(self, "btn_aksi"):
            self.btn_aksi.configure(text="Pilih Foto", command=self._pilih_foto)
        if hasattr(self, "badge"):
            self.badge.configure(text=" FOTO ", fg_color="#444")

    def _set_mode_video(self):
        self._stop_kamera()
        self._reset_mode_btns()
        self.btn_video.configure(fg_color=ORANGE, border_color=ORANGE, text_color="white")
        self._mode_aktif = "video"
        if hasattr(self, "lbl_hint"):
            self.lbl_hint.configure(text="Klik untuk upload video pesilat")
        if hasattr(self, "btn_aksi"):
            self.btn_aksi.configure(text="Pilih Video", command=self._pilih_video)
        if hasattr(self, "badge"):
            self.badge.configure(text=" VIDEO ", fg_color="#444")

    def _set_mode_kamera(self):
        self._reset_mode_btns()
        self.btn_kamera.configure(fg_color=ORANGE, border_color=ORANGE, text_color="white")
        self._mode_aktif = "kamera"
        if hasattr(self, "lbl_hint"):
            self.lbl_hint.configure(text="Kamera webcam aktif saat tombol ditekan")
        if hasattr(self, "btn_aksi"):
            self.btn_aksi.configure(text="Mulai Kamera", command=self._mulai_kamera)
        if hasattr(self, "badge"):
            self.badge.configure(text=" LIVE ", fg_color=ORANGE)

    # ── UPLOAD ZONE ──────────────────────────────────────────
    def _build_upload_zone(self):
        zone = ctk.CTkFrame(self.scroll, fg_color=BG_MID, corner_radius=10,
                            border_width=1, border_color=BORDER)
        zone.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(zone, text="⬆", font=("Segoe UI", 28),
                     text_color=ORANGE).pack(pady=(22, 4))
        self.lbl_hint = ctk.CTkLabel(zone, text="Klik untuk upload foto pesilat",
                                     font=("Segoe UI", 13), text_color=MUTED)
        self.lbl_hint.pack()
        ctk.CTkLabel(zone, text="JPG · PNG · MP4 · AVI",
                     font=("Segoe UI", 10), text_color=BORDER).pack(pady=(2, 8))
        self.lbl_file = ctk.CTkLabel(zone, text="", font=("Segoe UI", 11),
                                     text_color=ORANGE)
        self.lbl_file.pack()
        self.btn_aksi = ctk.CTkButton(
            zone, text="Pilih Foto", fg_color=ORANGE, hover_color="#c0441a",
            text_color="white", font=("Segoe UI", 12, "bold"),
            corner_radius=6, height=34, command=self._pilih_foto
        )
        self.btn_aksi.pack(pady=(4, 16))

    # ── PREVIEW ──────────────────────────────────────────────
    def _build_preview(self):
        self.frame_preview = ctk.CTkFrame(self.scroll, fg_color=BG_CARD,
                                          corner_radius=8, border_width=1,
                                          border_color=BORDER)
        self.frame_preview.pack(fill="x", pady=(0, 14))
        self.lbl_preview = ctk.CTkLabel(self.frame_preview,
                                        text="Preview akan muncul di sini",
                                        font=("Segoe UI", 12), text_color=MUTED)
        self.lbl_preview.pack(pady=20)

    def _update_preview(self, frame_bgr):
        try:
            rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            skala = min(640 / w, 360 / h, 1.0)
            baru  = (int(w * skala), int(h * skala))
            pil   = Image.fromarray(rgb).resize(baru, Image.LANCZOS)
            img   = ctk.CTkImage(pil, size=baru)
            self.lbl_preview.configure(image=img, text="")
            self.lbl_preview.image = img
        except Exception:
            pass

    # ── KARTU METRIK ─────────────────────────────────────────
    def _build_cards(self):
        frame = ctk.CTkFrame(self.scroll, fg_color="transparent")
        frame.pack(fill="x", pady=(0, 14))
        frame.columnconfigure((0, 1, 2, 3), weight=1)
        self.lbl_vals = []
        self.lbl_subs = []
        for i, label in enumerate(["Siku Kanan", "Siku Kiri", "Lutut Kanan", "Lutut Kiri"]):
            card = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=8,
                                border_width=1, border_color=BORDER)
            card.grid(row=0, column=i, padx=(0 if i == 0 else 8, 0), sticky="nsew")
            ctk.CTkLabel(card, text=label.upper(), font=("Segoe UI", 9),
                         text_color=MUTED).pack(pady=(14, 2), padx=14, anchor="w")
            v = ctk.CTkLabel(card, text="—", font=("Segoe UI", 22, "bold"), text_color=TEXT)
            v.pack(padx=14, anchor="w")
            s = ctk.CTkLabel(card, text="", font=("Segoe UI", 9), text_color=ORANGE)
            s.pack(padx=14, anchor="w", pady=(0, 14))
            self.lbl_vals.append(v)
            self.lbl_subs.append(s)

    def _update_cards(self, sk, ski, slk, slki):
        vals = [sk, ski, slk, slki]
        subs = [
            "PUKULAN LURUS" if sk  > 140 else "TANGKISAN",
            "PUKULAN LURUS" if ski > 140 else "TANGKISAN",
            "KUDA-KUDA STABIL" if slk  < 150 else "POSISI TINGGI",
            "KUDA-KUDA STABIL" if slki < 150 else "POSISI TINGGI",
        ]
        for i in range(4):
            self.lbl_vals[i].configure(text=f"{vals[i]:.1f}°")
            self.lbl_subs[i].configure(text=subs[i])

    # ── GRAFIK SERANGAN ──────────────────────────────────────
    def _build_grafik(self):
        self.frame_grafik = ctk.CTkFrame(self.scroll, fg_color=BG_CARD,
                                         corner_radius=8, border_width=1,
                                         border_color=BORDER)
        self.frame_grafik.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(self.frame_grafik, text="KECENDERUNGAN SERANGAN",
                     font=("Segoe UI", 10),
                     text_color=MUTED).pack(padx=16, pady=(14, 8), anchor="w")
        self._render_bars({"Pukulan": 0, "Tendangan": 0, "Sapuan": 0, "Tangkisan": 0})

    def _render_bars(self, skor):
        for w in self.frame_grafik.winfo_children()[1:]:
            w.destroy()
        total = sum(skor.values()) or 1
        for nama, val in skor.items():
            pct = val / total
            row = ctk.CTkFrame(self.frame_grafik, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=3)
            ctk.CTkLabel(row, text=nama, font=("Segoe UI", 12),
                         text_color=TEXT, width=90, anchor="w").pack(side="left")
            bar = ctk.CTkProgressBar(row, height=6, fg_color=BG_DARK,
                                     progress_color=ORANGE, corner_radius=3)
            bar.set(pct)
            bar.pack(side="left", fill="x", expand=True, padx=8)
            ctk.CTkLabel(row, text=f"{round(pct * 100)}%", font=("Segoe UI", 11),
                         text_color=TEXT, width=36, anchor="e").pack(side="left")
        ctk.CTkFrame(self.frame_grafik, height=1, fg_color="transparent").pack(pady=6)

    # ── RINGKASAN GEMINI ─────────────────────────────────────
    def _build_ringkasan(self):
        frame = ctk.CTkFrame(self.scroll, fg_color=BG_CARD, corner_radius=8,
                             border_width=1, border_color=BORDER)
        frame.pack(fill="x", pady=(0, 14))
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 4))
        ctk.CTkLabel(header, text="ANALISIS PELATIH", font=("Segoe UI", 10),
                     text_color=MUTED).pack(side="left")
        ctk.CTkLabel(header, text=" Gemini AI ", font=("Segoe UI", 9),
                     text_color="white", fg_color="#1a73e8",
                     corner_radius=6).pack(side="right")
        self.lbl_ringkasan = ctk.CTkLabel(
            frame, text="Analisis AI akan muncul setelah proses selesai.",
            font=("Segoe UI", 12), text_color=MUTED,
            justify="left", wraplength=800
        )
        self.lbl_ringkasan.pack(padx=16, pady=(0, 14), anchor="w")

    # ── SWOT ─────────────────────────────────────────────────
    def _build_swot(self):
        frame = ctk.CTkFrame(self.scroll, fg_color="transparent")
        frame.pack(fill="x", pady=(0, 14))
        frame.columnconfigure((0, 1), weight=1)
        warna = {
            "Strengths":     "#2ECC71",
            "Weaknesses":    "#E74C3C",
            "Opportunities": "#3498DB",
            "Threats":       "#F39C12",
        }
        self.swot_labels = {}
        for i, (key, w) in enumerate(warna.items()):
            card = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=8,
                                border_width=1, border_color=BORDER)
            card.grid(row=i // 2, column=i % 2,
                      padx=(0 if i % 2 == 0 else 8, 0),
                      pady=(0 if i < 2 else 8, 0), sticky="nsew")
            ctk.CTkLabel(card, text=key.upper(), font=("Segoe UI", 10, "bold"),
                         text_color=w).pack(padx=14, pady=(12, 4), anchor="w")
            lbl = ctk.CTkLabel(card, text="—", font=("Segoe UI", 11),
                               text_color=MUTED, justify="left", wraplength=220)
            lbl.pack(padx=14, pady=(0, 14), anchor="w")
            self.swot_labels[key] = lbl

    def _update_swot(self, swot):
        for key, lbl in self.swot_labels.items():
            poin = swot.get(key, [])
            if isinstance(poin, list):
                lbl.configure(text="\n".join(f"• {p}" for p in poin))
            else:
                lbl.configure(text=str(poin))

    # ── SARAN LATIHAN ────────────────────────────────────────
    def _build_saran(self):
        self.frame_saran = ctk.CTkFrame(self.scroll, fg_color=BG_CARD,
                                        corner_radius=8, border_width=1,
                                        border_color=BORDER)
        self.frame_saran.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(self.frame_saran, text="SARAN LATIHAN",
                     font=("Segoe UI", 10), text_color=MUTED).pack(
                         padx=16, pady=(14, 8), anchor="w")
        self.lbl_saran = ctk.CTkLabel(
            self.frame_saran,
            text="Saran latihan akan muncul setelah analisis selesai.",
            font=("Segoe UI", 12), text_color=MUTED,
            justify="left", wraplength=800
        )
        self.lbl_saran.pack(padx=16, pady=(0, 14), anchor="w")

    def _update_saran(self, saran):
        if saran:
            teks = "\n".join(f"✅  {s}" for s in saran)
        else:
            teks = "—"
        self.lbl_saran.configure(text=teks, text_color=TEXT)

    # ── TOMBOL BAWAH ─────────────────────────────────────────
    def _build_tombol(self):
        row = ctk.CTkFrame(self.scroll, fg_color="transparent")
        row.pack(fill="x", pady=(0, 20))
        ctk.CTkButton(
            row, text="Reset", fg_color="transparent", border_width=1,
            border_color=BORDER, hover_color=BG_CARD, text_color=MUTED,
            font=("Segoe UI", 12), corner_radius=6, height=36,
            command=self._reset
        ).pack(side="left", padx=(0, 8))
        self.btn_stop = ctk.CTkButton(
            row, text="Stop Kamera", fg_color="transparent", border_width=1,
            border_color="#E74C3C", hover_color=BG_CARD, text_color="#E74C3C",
            font=("Segoe UI", 12), corner_radius=6, height=36,
            command=self._stop_kamera, state="disabled"
        )
        self.btn_stop.pack(side="left")
        self.lbl_status = ctk.CTkLabel(row, text="", font=("Segoe UI", 11),
                                       text_color=ORANGE)
        self.lbl_status.pack(side="right")

    # ── LOGIKA FOTO ──────────────────────────────────────────
    def _pilih_foto(self):
        path = filedialog.askopenfilename(
            title="Pilih foto pesilat",
            filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        self.lbl_file.configure(text=os.path.basename(path))
        self.lbl_status.configure(text="Menganalisis foto...")
        self.lbl_ringkasan.configure(text="Meminta analisis ke Gemini AI...", text_color=MUTED)
        threading.Thread(target=self._proses_foto, args=(path,), daemon=True).start()

    def _proses_foto(self, path):
        hasil = analisis_foto(path)
        self.after(0, self._tampil_hasil, hasil)

    # ── LOGIKA VIDEO ─────────────────────────────────────────
    def _pilih_video(self):
        path = filedialog.askopenfilename(
            title="Pilih video pesilat",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return
        self.lbl_file.configure(text=os.path.basename(path))
        self.lbl_status.configure(text="Memproses video...")
        self.badge.configure(text=" PROSES ", fg_color="#854F0B")
        self.lbl_ringkasan.configure(
            text="Menunggu video selesai diproses...", text_color=MUTED)
        self._stop_flag = [False]

        def selesai(hasil):
            self.after(0, self._tampil_hasil, hasil)

        def update_frame(frame):
            self.after(0, self._update_preview, frame)

        threading.Thread(
            target=analisis_video,
            kwargs=dict(
                video_path=path,
                callback_frame=update_frame,
                callback_selesai=selesai,
                stop_flag=self._stop_flag,
            ),
            daemon=True,
        ).start()

    # ── LOGIKA KAMERA ────────────────────────────────────────
    def _mulai_kamera(self):
        self._stop_flag = [False]
        self.btn_stop.configure(state="normal")
        self.badge.configure(text=" LIVE ", fg_color=ORANGE)
        self.lbl_status.configure(text="Kamera aktif — deteksi berjalan")
        threading.Thread(
            target=analisis_kamera,
            kwargs=dict(
                callback_frame=lambda f: self.after(0, self._update_preview, f),
                callback_data=lambda skor, sk, ski, slk, slki: self.after(
                    0, self._update_cards, sk, ski, slk, slki
                ),
                stop_flag=self._stop_flag,
            ),
            daemon=True,
        ).start()

    def _stop_kamera(self):
        self._stop_flag[0] = True
        if hasattr(self, "btn_stop"):
            self.btn_stop.configure(state="disabled")
        if hasattr(self, "badge"):
            self.badge.configure(text=" SIAP ", fg_color="#444")
        if hasattr(self, "lbl_status"):
            self.lbl_status.configure(text="Kamera dihentikan")

    # ── TAMPIL HASIL ─────────────────────────────────────────
    def _tampil_hasil(self, hasil):
        if hasil is None:
            messagebox.showerror(
                "Gagal",
                "Pose tidak terdeteksi.\nPastikan tubuh penuh terlihat jelas."
            )
            self.lbl_status.configure(text="Gagal deteksi")
            self.badge.configure(text=" ERROR ", fg_color="#A32D2D")
            return

        self._hasil = hasil
        self._update_preview(hasil["image_bgr"])
        self._update_cards(
            hasil["siku_kanan"], hasil["siku_kiri"],
            hasil["lutut_kanan"], hasil["lutut_kiri"]
        )
        self._render_bars(hasil["skor_serangan"])
        self._update_swot(hasil["swot"])
        self._update_saran(hasil.get("saran", []))

        ringkasan = hasil.get("ringkasan", "")
        if ringkasan:
            self.lbl_ringkasan.configure(text=ringkasan, text_color=TEXT)
        else:
            self.lbl_ringkasan.configure(
                text="Tambahkan API key Gemini di gemini_analyzer.py untuk analisis AI.",
                text_color=MUTED
            )

        extra = f"  •  {hasil['total_frame']} frame" if hasil.get("mode") == "video" else ""
        self.lbl_status.configure(
            text=f"Selesai  •  Dominan: {hasil['dominan']}{extra}"
        )
        self.badge.configure(text=" SELESAI ", fg_color="#0F6E56")

    # ── RESET ────────────────────────────────────────────────
    def _reset(self):
        self._stop_kamera()
        self._hasil = None
        self.lbl_file.configure(text="")
        self.lbl_status.configure(text="")
        self.lbl_preview.configure(image=None, text="Preview akan muncul di sini")
        self.lbl_preview.image = None
        for v, s in zip(self.lbl_vals, self.lbl_subs):
            v.configure(text="—")
            s.configure(text="")
        self._render_bars({"Pukulan": 0, "Tendangan": 0, "Sapuan": 0, "Tangkisan": 0})
        for lbl in self.swot_labels.values():
            lbl.configure(text="—")
        self.lbl_ringkasan.configure(
            text="Analisis AI akan muncul setelah proses selesai.", text_color=MUTED)
        self.lbl_saran.configure(
            text="Saran latihan akan muncul setelah analisis selesai.", text_color=MUTED)
        self.badge.configure(text=" SIAP ", fg_color="#444")