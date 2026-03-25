import cv2
import os
import csv
import json
import subprocess
import threading
import datetime
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import welch, find_peaks
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  GIT
# ─────────────────────────────────────────────
def _find_git() -> str:
    import shutil
    g = shutil.which("git")
    if g: return g
    for c in [r"C:\Program Files\Git\cmd\git.exe",
              r"C:\Program Files\Git\bin\git.exe"]:
        if os.path.isfile(c): return c
    return "git"

def git_auto_commit(exp_dir, message):
    log = []
    try:
        git  = _find_git()
        root = os.path.dirname(os.path.abspath(__file__))
        def run(args):
            full = [git] + args
            log.append("$ " + " ".join(full))
            r = subprocess.run(full, cwd=root, capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            if r.stdout.strip(): log.append("  " + r.stdout.strip())
            if r.stderr.strip(): log.append("  " + r.stderr.strip())
            return r
        run(["init"])
        run(["config", "user.email", "masa-resorte@lab"])
        run(["config", "user.name",  "Masa Resorte"])
        gi = os.path.join(root, ".gitignore")
        if not os.path.exists(gi):
            open(gi,"w").write("__pycache__/\n*.pyc\n")
            run(["add", ".gitignore"])
        run(["add", "-A"])
        cr = run(["commit", "-m", message])
        out = (cr.stdout + cr.stderr).lower()
        if cr.returncode != 0 and "nothing to commit" not in out:
            return False, "\n".join(log)
        log.append("COMMIT OK")
        rr = run(["remote", "-v"])
        if not rr.stdout.strip():
            log.append("Sin remote — push omitido"); return True, "\n".join(log)
        br = run(["rev-parse","--abbrev-ref","HEAD"])
        branch = br.stdout.strip() or "main"
        pr = run(["push","origin",branch])
        if pr.returncode != 0:
            run(["push","--set-upstream","origin",branch])
        log.append("PUSH OK")
        return True, "\n".join(log)
    except Exception as e:
        log.append(str(e)); return False, "\n".join(log)

def next_experiment_folder(base):
    os.makedirs(base, exist_ok=True)
    nums = []
    for d in os.listdir(base):
        if d.startswith("experimento_"):
            try: nums.append(int(d.split("_")[1]))
            except: pass
    n = max(nums)+1 if nums else 1
    return os.path.join(base, f"experimento_{n:03d}")


# ─────────────────────────────────────────────
#  ANÁLISIS MATEMÁTICO
# ─────────────────────────────────────────────
def sinusoide_amortiguada(t, A, omega, phi, gamma, offset):
    return A * np.exp(-gamma*t) * np.cos(omega*t + phi) + offset

def smooth(arr, w=5):
    return np.convolve(arr, np.ones(w)/w, mode='same')

def detectar_marcadores(frame, lower, upper, kernel_sz=7, n=3):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    k    = np.ones((kernel_sz,kernel_sz), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:n]:
        M = cv2.moments(c)
        if M["m00"] > 0:
            pts.append((M["m10"]/M["m00"], M["m01"]/M["m00"]))
    return pts, mask

def asignar_triangulo(pts):
    pts = sorted(pts, key=lambda p: p[1])
    top  = pts[0]
    base = sorted(pts[1:], key=lambda p: p[0])
    return np.array(top), np.array(base[0]), np.array(base[1])

def img_to_world(px_pt, M):
    p = np.array([px_pt[0], px_pt[1], 1.0])
    r = M @ p
    return r[0], r[1]

def calcular_resultados(t_raw, y_raw, masa_total_kg):
    """
    masa_total_kg = masa_principal + masa_estimulo
    Ambas oscilan juntas, por lo que la masa efectiva del sistema es su suma.
    k = masa_total * omega0²
    c = 2 * masa_total * gamma
    """
    y_eq   = np.mean(y_raw)
    y_data = y_raw - y_eq
    y_s    = smooth(y_data, 5)
    vy     = smooth(np.gradient(y_s, t_raw), 5)
    ay     = smooth(np.gradient(vy,  t_raw), 5)

    # Estimación omega
    y_c    = y_data - np.mean(y_data)
    cruces = np.where(np.diff(np.sign(y_c)))[0]
    if len(cruces) >= 2:
        T_est = 2 * np.mean(np.diff(t_raw[cruces]))
        omega_est = 2*np.pi/T_est if T_est > 0 else 5.0
    else:
        omega_est = 5.0

    A0 = (np.max(y_data)-np.min(y_data))/2
    p0 = [A0, omega_est, 0.0, 0.05, 0.0]
    bounds = ([0,0.5,-np.pi,0,-0.5],[2,50,np.pi,5,0.5])

    try:
        popt,pcov = curve_fit(sinusoide_amortiguada, t_raw, y_data,
                              p0=p0, bounds=bounds, maxfev=20000)
        A_fit, omega_fit, phi_fit, gamma_fit, off_fit = popt
        perr   = np.sqrt(np.diag(pcov))
        T_fit  = 2*np.pi/omega_fit
        f_fit  = 1/T_fit
        omega0 = np.sqrt(omega_fit**2 + gamma_fit**2)
        # ── Usar masa total para derivar k y c ──────────────────────────
        k_fit  = masa_total_kg * omega0**2
        c_fit  = 2 * masa_total_kg * gamma_fit
        zeta   = gamma_fit / omega0
        ajuste_ok = True
    except:
        A_fit=A0; omega_fit=omega_est; phi_fit=0; gamma_fit=0; off_fit=0
        perr=np.zeros(5); T_fit=2*np.pi/omega_est; f_fit=1/T_fit
        omega0=omega_est
        k_fit=masa_total_kg*omega0**2; c_fit=0; zeta=0
        ajuste_ok=False

    disc = (2*zeta*omega0)**2 - 4*omega0**2
    if disc < 0:
        pr = -zeta*omega0; pi_ = omega0*np.sqrt(max(0,1-zeta**2))
        polos = [complex(pr, pi_), complex(pr,-pi_)]
    else:
        d = np.sqrt(disc)
        polos = [(-2*zeta*omega0+d)/2, (-2*zeta*omega0-d)/2]

    return {
        "y_data":   y_data, "y_s": y_s, "vy": vy, "ay": ay,
        "t_raw":    t_raw,
        "A_fit":    A_fit,  "omega_fit": omega_fit, "phi_fit": phi_fit,
        "gamma_fit":gamma_fit, "off_fit": off_fit, "perr": perr,
        "T_fit":    T_fit,  "f_fit": f_fit, "omega0": omega0,
        "k_fit":    k_fit,  "c_fit": c_fit, "zeta": zeta,
        "ajuste_ok":ajuste_ok, "polos": polos, "disc": disc,
        "masa_total_kg": masa_total_kg,
        # mantener alias para compatibilidad con figuras
        "masa_kg":  masa_total_kg,
    }


# ─────────────────────────────────────────────
#  GRÁFICAS
# ─────────────────────────────────────────────
def generate_fig1(r, path):
    t = r["t_raw"]; y = r["y_data"]; ys = r["y_s"]
    vy = r["vy"]; ay = r["ay"]
    fig,axs = plt.subplots(3,1,figsize=(12,10),sharex=True)
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axs:
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values(): sp.set_color("#333366")
        ax.tick_params(colors="#aaaacc",labelsize=8)
        ax.yaxis.label.set_color("#aaaacc"); ax.xaxis.label.set_color("#aaaacc")
        ax.title.set_color("#e0e0ff"); ax.grid(alpha=0.2)

    axs[0].scatter(t, y*100, s=4, alpha=0.4, color="#00BFFF", label="Datos")
    axs[0].plot(t, ys*100, lw=1.5, color="#00BFFF", label="Suavizado")
    if r["ajuste_ok"]:
        tf = np.linspace(t[0],t[-1],1000)
        ya = sinusoide_amortiguada(tf,r["A_fit"],r["omega_fit"],r["phi_fit"],r["gamma_fit"],r["off_fit"])
        axs[0].plot(tf, ya*100, lw=2, color="#00ffcc", linestyle="--",
                    label=f"Ajuste  T={r['T_fit']:.3f}s  k={r['k_fit']:.3f}N/m  ζ={r['zeta']:.4f}")
    axs[0].axhline(0,color="gray",lw=0.6,linestyle=":")
    axs[0].set_ylabel("y (cm)"); axs[0].legend(facecolor="#1a1a2e",labelcolor="#ccccff",fontsize=8)

    axs[1].plot(t, vy*100, lw=1.8, color="#7FFF00")
    axs[1].axhline(0,color="gray",lw=0.6,linestyle=":"); axs[1].set_ylabel("vy (cm/s)")

    axs[2].plot(t, ay*100, lw=1.8, color="#FF6347")
    axs[2].axhline(0,color="gray",lw=0.6,linestyle=":")
    axs[2].set_ylabel("ay (cm/s²)"); axs[2].set_xlabel("t (s)")

    fig.suptitle(f"Masa-Resorte — Cinemática  |  k={r['k_fit']:.3f} N/m  ω₀={r['omega0']:.3f} rad/s  ζ={r['zeta']:.4f}",
                 color="#fff",fontsize=12,fontweight="bold",y=0.99)
    plt.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)

def generate_fig2(r, path):
    y = r["y_data"]; vy = r["vy"]; t = r["t_raw"]
    fig,ax = plt.subplots(figsize=(7,7))
    fig.patch.set_facecolor("#0d0d0d"); ax.set_facecolor("#1a1a2e")
    for sp in ax.spines.values(): sp.set_color("#333366")
    ax.tick_params(colors="#aaaacc"); ax.grid(alpha=0.2)
    sc = ax.scatter(y*100, vy*100, c=t, cmap="plasma", s=10, alpha=0.8)
    plt.colorbar(sc,ax=ax,label="t (s)")
    ax.set_xlabel("Desplazamiento y (cm)"); ax.set_ylabel("Velocidad vy (cm/s)")
    ax.axhline(0,color="gray",lw=0.5,linestyle=":"); ax.axvline(0,color="gray",lw=0.5,linestyle=":")
    ax.set_title("Espacio de Fases (y, vy)",color="#e0e0ff")
    fig.suptitle("Masa-Resorte — Espacio de Fases",color="#fff",fontsize=12,fontweight="bold")
    plt.tight_layout()
    fig.savefig(path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)

def generate_fig3(r, path, fps):
    y = r["y_data"]; f_fit = r["f_fit"]
    fig,ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor("#0d0d0d"); ax.set_facecolor("#1a1a2e")
    for sp in ax.spines.values(): sp.set_color("#333366")
    ax.tick_params(colors="#aaaacc"); ax.grid(alpha=0.2,which="both")
    if len(y) > 64:
        f_psd,Pxx = welch(y, fs=fps, nperseg=min(256,len(y)//2))
        ax.semilogy(f_psd, Pxx, color="#DA70D6", lw=1.8)
        ax.axvline(f_fit,color="#00ffcc",lw=1.5,linestyle="--",label=f"f_ajuste={f_fit:.3f} Hz")
        peaks_idx,_ = find_peaks(Pxx, height=np.max(Pxx)*0.1)
        if len(peaks_idx)>0:
            f_p = f_psd[peaks_idx[np.argmax(Pxx[peaks_idx])]]
            ax.axvline(f_p,color="#ffdd00",lw=1.5,linestyle=":",label=f"f_pico={f_p:.3f} Hz")
        ax.legend(facecolor="#1a1a2e",labelcolor="#ccccff",fontsize=9)
    ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel("PSD [m²/Hz]")
    ax.set_title("Espectro de Potencia (PSD)",color="#e0e0ff")
    fig.suptitle("Masa-Resorte — PSD",color="#fff",fontsize=12,fontweight="bold")
    plt.tight_layout()
    fig.savefig(path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)

def generate_fig4(r, path):
    omega0=r["omega0"]; zeta=r["zeta"]; c=r["c_fit"]; k=r["k_fit"]; m=r["masa_total_kg"]
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,8),sharex=True)
    fig.patch.set_facecolor("#0d0d0d")
    for ax in [ax1,ax2]:
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values(): sp.set_color("#333366")
        ax.tick_params(colors="#aaaacc"); ax.grid(alpha=0.2,which="both")
    omega_r = np.logspace(-1,2,1000)
    H = 1.0/(m*(1j*omega_r)**2 + c*(1j*omega_r) + k)
    ax1.semilogx(omega_r,20*np.log10(np.abs(H)),color="#00BFFF",lw=2)
    ax1.axvline(omega0,color="#00ffcc",lw=1.5,linestyle="--",label=f"ω₀={omega0:.3f}")
    ax1.set_ylabel("|H(jω)| (dB)"); ax1.legend(facecolor="#1a1a2e",labelcolor="#ccccff",fontsize=9)
    ax1.set_title("Módulo",color="#e0e0ff")
    ax2.semilogx(omega_r,np.degrees(np.angle(H)),color="#FF6347",lw=2)
    ax2.axvline(omega0,color="#00ffcc",lw=1.5,linestyle="--",label=f"ω₀={omega0:.3f}")
    ax2.set_ylabel("∠H(jω) (°)"); ax2.set_xlabel("ω (rad/s)")
    ax2.legend(facecolor="#1a1a2e",labelcolor="#ccccff",fontsize=9)
    ax2.set_title("Fase",color="#e0e0ff")
    fig.suptitle(f"Masa-Resorte — Diagrama de Bode  H(s)=1/(ms²+cs+k)  m_total={m:.4f}kg",
                 color="#fff",fontsize=12,fontweight="bold")
    plt.tight_layout()
    fig.savefig(path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)

def generate_fig5(r, path):
    omega0=r["omega0"]; zeta=r["zeta"]; polos=r["polos"]; disc=r["disc"]
    fig,ax = plt.subplots(figsize=(7,7))
    fig.patch.set_facecolor("#0d0d0d"); ax.set_facecolor("#1a1a2e")
    for sp in ax.spines.values(): sp.set_color("#333366")
    ax.tick_params(colors="#aaaacc"); ax.grid(alpha=0.2)
    for p in polos:
        pr = p.real if np.iscomplex(p) else float(p)
        pi_ = p.imag if np.iscomplex(p) else 0.0
        ax.plot(pr, pi_, "rx", ms=16, mew=3)
        ax.annotate(f"  {pr:.3f}{'+' if pi_>=0 else ''}{pi_:.3f}j",
                    (pr, pi_), color="#ff8888", fontsize=8)
    ax.axhline(0,color="#555577",lw=0.8); ax.axvline(0,color="#555577",lw=0.8)
    th = np.linspace(0,2*np.pi,300)
    ax.plot(omega0*np.cos(th),omega0*np.sin(th),"--",color="#555577",lw=0.8,
            alpha=0.5,label=f"|s|=ω₀={omega0:.3f}")
    ax.set_xlabel("Re(s)",color="#aaaacc"); ax.set_ylabel("Im(s)",color="#aaaacc")
    ax.set_title(f"Polos (×)  ζ={zeta:.4f}  {'subamortiguado' if zeta<1 else 'sobreamortiguado'}",
                 color="#e0e0ff")
    ax.legend(facecolor="#1a1a2e",labelcolor="#ccccff",fontsize=8)
    ax.set_aspect("equal")
    fig.suptitle("Masa-Resorte — Diagrama de Polos y Ceros",color="#fff",fontsize=12,fontweight="bold")
    plt.tight_layout()
    fig.savefig(path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)

def save_evidence_image(frame_bgr, result, exp_name, out_path):
    TARGET_H = 900
    fh,fw = frame_bgr.shape[:2]
    sc = TARGET_H/fh
    frame_rs = cv2.resize(frame_bgr,(int(fw*sc),TARGET_H))
    fw2 = frame_rs.shape[1]
    frame_pil = Image.fromarray(cv2.cvtColor(frame_rs,cv2.COLOR_BGR2RGB))
    PANEL_W=440; BG=(13,13,35); ACC=(0,255,180); YEL=(255,220,0)
    RED=(255,80,100); SUB=(130,130,160); WHT=(210,210,255)
    r = result
    k_col = ACC if r.get("error_pct",99)<5 else RED
    panel = Image.new("RGB",(PANEL_W,TARGET_H),BG)
    draw  = ImageDraw.Draw(panel)
    try:
        from PIL import ImageFont
        try:
            fb=ImageFont.truetype("cour.ttf",46)
            fm=ImageFont.truetype("cour.ttf",24)
            fs=ImageFont.truetype("cour.ttf",19)
            ft=ImageFont.truetype("cour.ttf",15)
            ftt=ImageFont.truetype("cour.ttf",28)
        except: fb=fm=fs=ft=ftt=ImageFont.load_default()
    except: fb=fm=fs=ft=ftt=None
    def t(text,y,col,fnt,x=16): draw.text((x,y),str(text),fill=col,font=fnt)
    def sep(y): draw.line([(8,y),(PANEL_W-8,y)],fill=(50,50,100),width=1)
    y=16
    t("MASA-RESORTE",          y,ACC,ftt);  y+=38
    t("Sistema oscilante",     y,SUB,ft);   y+=24; sep(y); y+=12
    t("ARCHIVO",               y,SUB,ft);   y+=16
    t(r.get("video","")[:30],  y,WHT,fs);   y+=28; sep(y); y+=12
    # ── Masas ──────────────────────────────────────────────────────────
    t("MASA PRINCIPAL (kg)",   y,SUB,ft);   y+=16
    t(f"{r.get('masa_kg_principal', r.get('masa_kg',0)):.4f} kg", y,WHT,fm); y+=28
    t("MASA ESTÍMULO (kg)",    y,SUB,ft);   y+=16
    t(f"{r.get('masa_estimulo_kg',0):.4f} kg", y,WHT,fm); y+=28
    t("MASA TOTAL (kg)",       y,SUB,ft);   y+=16
    t(f"{r.get('masa_total_kg', r.get('masa_kg',0)):.4f} kg", y,YEL,fm); y+=34
    t("k (N/m)",               y,SUB,ft);   y+=16
    t(f"{r.get('k_fit',0):.4f}", y,ACC,fb); y+=54
    t("c (N·s/m)",             y,SUB,ft);   y+=16
    t(f"{r.get('c_fit',0):.5f}",y,YEL,fm); y+=34
    sep(y); y+=12
    t("ω₀ (rad/s)",            y,SUB,ft);   y+=16
    t(f"{r.get('omega0',0):.4f}",y,WHT,fm); y+=34
    t("T (s)",                 y,SUB,ft);   y+=16
    t(f"{r.get('T_fit',0):.4f}",y,WHT,fm); y+=34
    t("f (Hz)",                y,SUB,ft);   y+=16
    t(f"{r.get('f_fit',0):.4f}",y,WHT,fm); y+=34
    sep(y); y+=12
    t("ζ (amort.)",            y,SUB,ft);   y+=16
    col_z = ACC if r.get("zeta",1)<0.5 else YEL
    t(f"{r.get('zeta',0):.5f}",y,col_z,fm); y+=34
    t("Estado",                y,SUB,ft);   y+=16
    est = "subamortiguado" if r.get("zeta",1)<1 else "sobreamortiguado"
    t(est,                     y,col_z,fs); y+=28
    sep(y); y+=12
    t("EXPERIMENTO",           y,SUB,ft);   y+=16
    t(exp_name,                y,ACC,fs);   y+=26
    t("FECHA",                 y,SUB,ft);   y+=16
    ts=r.get("timestamp","")[:19].replace("T","  ")
    t(ts,                      y,SUB,ft)
    draw.line([(0,0),(0,TARGET_H)],fill=(0,200,150),width=3)
    out = Image.new("RGB",(fw2+PANEL_W,TARGET_H),BG)
    out.paste(frame_pil,(0,0)); out.paste(panel,(fw2,0))
    out.save(out_path,quality=95)


# ─────────────────────────────────────────────
#  APP PRINCIPAL
# ─────────────────────────────────────────────
class MasaResorteApp(tk.Tk):

    BG    = "#0d0d0d"; PANEL = "#121228"; CARD  = "#1a1a2e"
    ACCENT= "#00ffcc"; ACC2  = "#7b61ff"; TEXT  = "#e0e0ff"
    SUB   = "#888899"; RED   = "#ff4466"; YELLOW= "#ffdd00"
    BLUE  = "#66aaff"; PURP  = "#cc88ff"
    ORANGE= "#ffaa44"   # color para masa estímulo

    STEPS = ["video","frames","marcadores","masa","calibracion","tracking","resultados"]

    def __init__(self):
        super().__init__()
        self.title("🌀  Analizador Masa-Resorte")
        self.configure(bg=self.BG)
        self.geometry("1400x900"); self.state("zoomed")

        # Video
        self.cap=None; self.video_path=""; self.total_frames=0
        self.fps=30.0; self.current_frame=0
        self.frame_inicio=None; self.frame_fin=None
        self._video_rotation=0; self._slider_updating=False
        self.playing=False; self._display_scale=1.0; self._display_offset=(0,0)

        # Calibración marcadores
        self.ref_lower=None; self.ref_upper=None
        self.roi_marcador=None

        # Calibración masa
        self.masa_lower=None; self.masa_upper=None
        self.roi_masa_cal=None

        # Triángulo físico
        self.base_real_m=0.0; self.altura_real_m=0.0
        self.M_aff=None; self.dst_pts=None
        self.puntos_triangulo=[]

        # ── Masas ──────────────────────────────────────────────────────
        self.masa_kg          = 0.0   # masa principal (objeto que oscila)
        self.masa_estimulo_kg = 0.0   # masa del estímulo (cuelga de la principal)
        self.masa_total_kg    = 0.0   # suma efectiva usada en los cálculos

        # Tracking
        self.datos=[]; self.trail=[]
        self._tracking_running=False
        self.result={}

        # Modos canvas
        self.canvas_mode="none"  # none | roi_marc | roi_masa | triangulo
        self.roi_start=None; self.roi_rect_canvas=None

        # Experimento
        self.base_results=os.path.join(os.path.dirname(os.path.abspath(__file__)),"resultados")
        self.exp_dir=None
        self.step="video"

        self._build_ui()
        self._new_experiment()

    # ══════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════
    def _build_ui(self):
        self.columnconfigure(0,weight=0); self.columnconfigure(1,weight=1)
        self.columnconfigure(2,weight=0); self.rowconfigure(0,weight=1)
        self._build_sidebar(); self._build_canvas_area(); self._build_right_panel()

    # ── Sidebar con scroll ───────────────────
    def _build_sidebar(self):
        outer = tk.Frame(self, bg=self.PANEL, width=300)
        outer.grid(row=0, column=0, sticky="ns")
        outer.grid_propagate(False)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        _sb_canvas = tk.Canvas(outer, bg=self.PANEL, highlightthickness=0, width=280)
        _sb_canvas.grid(row=0, column=0, sticky="nsew")

        _vsb = ttk.Scrollbar(outer, orient="vertical", command=_sb_canvas.yview)
        _vsb.grid(row=0, column=1, sticky="ns")
        _sb_canvas.configure(yscrollcommand=_vsb.set)

        sb = tk.Frame(_sb_canvas, bg=self.PANEL)
        _win_id = _sb_canvas.create_window((0, 0), window=sb, anchor="nw")

        def _on_sb_configure(e):
            _sb_canvas.configure(scrollregion=_sb_canvas.bbox("all"))

        def _on_canvas_resize(e):
            _sb_canvas.itemconfig(_win_id, width=e.width)

        sb.bind("<Configure>", _on_sb_configure)
        _sb_canvas.bind("<Configure>", _on_canvas_resize)

        def _on_mousewheel(e):
            _sb_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        _sb_canvas.bind("<Enter>", lambda e: _sb_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        _sb_canvas.bind("<Leave>", lambda e: _sb_canvas.unbind_all("<MouseWheel>"))

        sb.columnconfigure(0, weight=1)

        tk.Label(sb,text="🌀",font=("Courier New",28),bg=self.PANEL,fg=self.ACCENT).grid(row=0,column=0,pady=(16,0))
        tk.Label(sb,text="MASA-RESORTE",font=("Courier New",10,"bold"),bg=self.PANEL,fg=self.TEXT).grid(row=1,column=0)
        tk.Label(sb,text="Analizador cinemático",font=("Courier New",7),bg=self.PANEL,fg=self.SUB).grid(row=2,column=0,pady=(0,8))

        ttk.Separator(sb,orient="horizontal").grid(row=3,column=0,sticky="ew",padx=14)

        tk.Label(sb,text="EXPERIMENTO ACTIVO",font=("Courier New",6,"bold"),
                 bg=self.PANEL,fg=self.SUB).grid(row=4,column=0,sticky="w",padx=16,pady=(10,1))
        self.lbl_exp=tk.Label(sb,text="—",font=("Courier New",9,"bold"),
                              bg=self.PANEL,fg=self.ACCENT,wraplength=260,anchor="w")
        self.lbl_exp.grid(row=5,column=0,sticky="ew",padx=16,pady=(0,4))
        self._sbtn(sb,6,"＋  Nuevo Experimento",self._new_experiment,self.CARD,self.TEXT)

        ttk.Separator(sb,orient="horizontal").grid(row=7,column=0,sticky="ew",padx=14,pady=5)

        tk.Label(sb,text="FLUJO DE TRABAJO",font=("Courier New",6,"bold"),
                 bg=self.PANEL,fg=self.SUB).grid(row=8,column=0,sticky="w",padx=16,pady=(2,3))
        self.step_labels={}
        step_names={"video":"1. Cargar Video","frames":"2. Marcar Frames",
                    "marcadores":"3. ROI Marcadores","masa":"4. ROI Masa",
                    "calibracion":"5. Detectar + Dimensiones","tracking":"6. Tracking",
                    "resultados":"7. Resultados"}
        for i,(k,name) in enumerate(step_names.items()):
            lbl=tk.Label(sb,text=f"  {name}",font=("Courier New",7),
                         bg=self.PANEL,fg=self.SUB,anchor="w")
            lbl.grid(row=9+i,column=0,sticky="ew",padx=14,pady=1)
            self.step_labels[k]=lbl

        ttk.Separator(sb,orient="horizontal").grid(row=16,column=0,sticky="ew",padx=14,pady=5)

        # Video
        tk.Label(sb,text="VIDEO",font=("Courier New",6,"bold"),bg=self.PANEL,fg=self.SUB).grid(row=17,column=0,sticky="w",padx=16,pady=(2,1))
        self._sbtn(sb,18,"📂  Cargar Video",self._load_video,self.CARD,self.TEXT)
        self.lbl_video=tk.Label(sb,text="Sin video",font=("Courier New",7),
                                bg=self.PANEL,fg=self.SUB,wraplength=260,anchor="w")
        self.lbl_video.grid(row=19,column=0,sticky="ew",padx=16)

        ttk.Separator(sb,orient="horizontal").grid(row=20,column=0,sticky="ew",padx=14,pady=4)

        # Nav
        tk.Label(sb,text="NAVEGACIÓN",font=("Courier New",6,"bold"),bg=self.PANEL,fg=self.SUB).grid(row=21,column=0,sticky="w",padx=16,pady=(2,2))
        nf=tk.Frame(sb,bg=self.PANEL); nf.grid(row=22,column=0,sticky="ew",padx=14)
        nf.columnconfigure((0,1,2),weight=1)
        for col,(txt,cmd,fg) in enumerate([("◀◀",self._prev_frame,self.TEXT),
                                            ("⏯",self._toggle_play,self.ACCENT),
                                            ("▶▶",self._next_frame,self.TEXT)]):
            tk.Button(nf,text=txt,command=cmd,bg=self.CARD,fg=fg,relief="flat",
                      font=("Courier New",12,"bold"),cursor="hand2",
                      activebackground=self.ACC2,activeforeground="#fff",
                      bd=0,pady=8).grid(row=0,column=col,sticky="ew",padx=2)

        ttk.Separator(sb,orient="horizontal").grid(row=23,column=0,sticky="ew",padx=14,pady=4)

        # Frames
        tk.Label(sb,text="FRAMES",font=("Courier New",6,"bold"),bg=self.PANEL,fg=self.SUB).grid(row=24,column=0,sticky="w",padx=16,pady=(2,2))
        mf=tk.Frame(sb,bg=self.PANEL); mf.grid(row=25,column=0,sticky="ew",padx=14)
        mf.columnconfigure((0,1),weight=1)
        tk.Button(mf,text="① INICIO",command=self._set_inicio,bg="#002244",fg=self.BLUE,
                  relief="flat",font=("Courier New",9,"bold"),cursor="hand2",
                  activebackground="#0044aa",activeforeground="#fff",bd=0,pady=9
                  ).grid(row=0,column=0,sticky="ew",padx=(0,2))
        tk.Button(mf,text="② FIN",command=self._set_fin,bg="#220044",fg=self.PURP,
                  relief="flat",font=("Courier New",9,"bold"),cursor="hand2",
                  activebackground="#6600aa",activeforeground="#fff",bd=0,pady=9
                  ).grid(row=0,column=1,sticky="ew",padx=(2,0))

        ttk.Separator(sb,orient="horizontal").grid(row=26,column=0,sticky="ew",padx=14,pady=4)

        # Calibración
        tk.Label(sb,text="CALIBRACIÓN",font=("Courier New",6,"bold"),bg=self.PANEL,fg=self.SUB).grid(row=27,column=0,sticky="w",padx=16,pady=(2,2))
        self._sbtn(sb,28,"🔺  ROI Marcadores (triángulo)",self._activate_roi_marc,"#002233",self.BLUE,bold=True)
        self._sbtn(sb,29,"⚫  ROI Masa (objeto)",         self._activate_roi_masa,"#220033",self.PURP,bold=True)
        self._sbtn(sb,30,"🔍  Detectar Triángulo",        self._detect_triangle,  "#002233",self.YELLOW,bold=True)

        ttk.Separator(sb,orient="horizontal").grid(row=31,column=0,sticky="ew",padx=14,pady=4)

        self._sbtn(sb,32,"▶  Ejecutar Tracking",self._run_tracking,"#003322",self.ACCENT,bold=True)
        self._sbtn(sb,33,"⏹  Cancelar Tracking",self._cancel_tracking,"#220000",self.RED)
        self._sbtn(sb,34,"↺  Resetear",          self._reset_all,    "#221100",self.YELLOW)

        ttk.Separator(sb,orient="horizontal").grid(row=35,column=0,sticky="ew",padx=14,pady=4)

        self._sbtn(sb,36,"📊  Ver Gráficas",     self._show_graficas,"#1a0033",self.ACC2,bold=True)
        self._sbtn(sb,38,"🔗  Configurar Remote",self._configure_remote,"#0a0a22","#6688ff")

        self.lbl_status=tk.Label(sb,text="Carga un video para comenzar",
                                 font=("Courier New",7),bg=self.PANEL,fg=self.SUB,
                                 wraplength=270,justify="left",anchor="w")
        self.lbl_status.grid(row=39,column=0,sticky="ew",padx=14,pady=(8,16))

    def _sbtn(self,parent,row,text,cmd,bg,fg,bold=False):
        tk.Button(parent,text=text,command=cmd,bg=bg,fg=fg,relief="flat",cursor="hand2",
                  font=("Courier New",9,"bold" if bold else ""),
                  activebackground=self.ACC2,activeforeground="#fff",
                  bd=0,pady=8).grid(row=row,column=0,sticky="ew",padx=14,pady=2)

    # ── Canvas central ────────────────────────
    def _build_canvas_area(self):
        mid=tk.Frame(self,bg=self.BG); mid.grid(row=0,column=1,sticky="nsew")
        mid.rowconfigure(0,weight=1); mid.rowconfigure(1,weight=0); mid.columnconfigure(0,weight=1)

        self.canvas=tk.Canvas(mid,bg="#000",highlightthickness=2,
                              highlightbackground=self.ACC2,cursor="crosshair")
        self.canvas.grid(row=0,column=0,sticky="nsew",padx=(10,4),pady=10)

        sf=tk.Frame(mid,bg=self.BG); sf.grid(row=1,column=0,sticky="ew",padx=10,pady=(0,8))
        sf.columnconfigure(1,weight=1)
        tk.Label(sf,text="0",font=("Courier New",7),bg=self.BG,fg=self.SUB).grid(row=0,column=0,padx=(0,4))
        self.slider=ttk.Scale(sf,from_=0,to=100,orient="horizontal",command=self._on_slider)
        self.slider.grid(row=0,column=1,sticky="ew")
        self.lbl_slider_end=tk.Label(sf,text="0",font=("Courier New",7),bg=self.BG,fg=self.SUB)
        self.lbl_slider_end.grid(row=0,column=2,padx=(4,0))

        self.canvas.bind("<ButtonPress-1>",  self._canvas_click)
        self.canvas.bind("<B1-Motion>",       self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.bind("<Left>",  lambda e: self._prev_frame())
        self.bind("<Right>", lambda e: self._next_frame())
        self.bind("<i>",     lambda e: self._set_inicio())
        self.bind("<f>",     lambda e: self._set_fin())
        self.bind("<space>", lambda e: self._toggle_play())
        self.bind("<Escape>",lambda e: self._deactivate_modes())

    # ── Panel derecho ─────────────────────────
    def _build_right_panel(self):
        ro=tk.Frame(self,bg=self.CARD,width=350)
        ro.grid(row=0,column=2,sticky="nsew",padx=(0,10),pady=10)
        ro.grid_propagate(False); ro.columnconfigure(0,weight=1); ro.rowconfigure(0,weight=1)

        rc=tk.Canvas(ro,bg=self.CARD,highlightthickness=0)
        rc.grid(row=0,column=0,sticky="nsew")
        vsb=ttk.Scrollbar(ro,orient="vertical",command=rc.yview)
        vsb.grid(row=0,column=1,sticky="ns"); rc.configure(yscrollcommand=vsb.set)

        panel=tk.Frame(rc,bg=self.CARD)
        rc.create_window((0,0),window=panel,anchor="nw")
        def _cfg(e): rc.configure(scrollregion=rc.bbox("all")); rc.itemconfig(1,width=rc.winfo_width())
        panel.bind("<Configure>",_cfg)
        rc.bind("<Configure>",lambda e: rc.itemconfig(1,width=e.width))

        BG=self.CARD; PAD=dict(fill="x",padx=12)
        def title(text,color=None):
            tk.Label(panel,text=text,font=("Courier New",8,"bold"),
                     bg=BG,fg=color or self.SUB,anchor="w").pack(**PAD,pady=(12,1))
        def val(attr,color,fs=14):
            lbl=tk.Label(panel,text="—",font=("Courier New",fs,"bold"),
                         bg=BG,fg=color,anchor="w")
            lbl.pack(**PAD,pady=(0,2)); setattr(self,attr,lbl)
        def sep():
            ttk.Separator(panel,orient="horizontal").pack(fill="x",padx=8,pady=5)

        tk.Label(panel,text="🌀  DATOS EN VIVO",font=("Courier New",11,"bold"),
                 bg=BG,fg=self.ACCENT,anchor="w").pack(**PAD,pady=(14,4))
        sep()

        title("ARCHIVO")
        self.rp_video=tk.Label(panel,text="—",font=("Courier New",9),
                               bg=BG,fg=self.TEXT,anchor="w",wraplength=320)
        self.rp_video.pack(**PAD,pady=(0,4)); sep()

        title("FRAME ACTUAL"); val("rp_frame",self.BLUE,14)
        title("INICIO / FIN",self.BLUE)
        self.rp_fi_ff=tk.Label(panel,text="— / —",font=("Courier New",13,"bold"),
                                bg=BG,fg=self.BLUE,anchor="w")
        self.rp_fi_ff.pack(**PAD,pady=(0,4)); sep()

        title("MODO CANVAS")
        self.rp_mode=tk.Label(panel,text="Normal",font=("Courier New",10,"bold"),
                               bg=BG,fg=self.SUB,anchor="w")
        self.rp_mode.pack(**PAD,pady=(0,4)); sep()

        # ── Entradas físicas ──────────────────────────────────────────
        title("MASA PRINCIPAL (kg)", self.YELLOW)
        self.entry_masa=tk.Entry(panel,bg="#0f0f22",fg=self.ACCENT,
                                 font=("Courier New",12,"bold"),relief="flat",
                                 insertbackground=self.ACCENT,justify="center")
        self.entry_masa.insert(0,"0.100"); self.entry_masa.pack(**PAD,pady=(0,2),ipady=5)
        tk.Label(panel,text="objeto que cuelga del resorte",
                 font=("Courier New",7),bg=BG,fg=self.SUB,anchor="w").pack(**PAD)

        title("MASA ESTÍMULO (kg)", self.ORANGE)
        self.entry_masa_estimulo=tk.Entry(panel,bg="#0f0f22",fg=self.ORANGE,
                                          font=("Courier New",12,"bold"),relief="flat",
                                          insertbackground=self.ORANGE,justify="center")
        self.entry_masa_estimulo.insert(0,"0.050")
        self.entry_masa_estimulo.pack(**PAD,pady=(0,2),ipady=5)
        tk.Label(panel,text="objeto colgado de la masa principal",
                 font=("Courier New",7),bg=BG,fg=self.SUB,anchor="w").pack(**PAD)

        # Label masa total (calculado automáticamente al detectar)
        title("MASA TOTAL EFECTIVA (kg)", self.ACCENT)
        self.lbl_masa_total=tk.Label(panel,text="0.150 kg  (0.100 + 0.050)",
                                      font=("Courier New",11,"bold"),
                                      bg=BG,fg=self.ACCENT,anchor="w")
        self.lbl_masa_total.pack(**PAD,pady=(0,4))

        # Actualizar el label en tiempo real mientras el usuario escribe
        self.entry_masa.bind("<KeyRelease>",         lambda e: self._update_masa_total_label())
        self.entry_masa_estimulo.bind("<KeyRelease>", lambda e: self._update_masa_total_label())

        title("BASE TRIÁNGULO (m)")
        self.entry_base=tk.Entry(panel,bg="#0f0f22",fg=self.PURP,
                                  font=("Courier New",11,"bold"),relief="flat",
                                  insertbackground=self.PURP,justify="center")
        self.entry_base.insert(0,"0.20"); self.entry_base.pack(**PAD,pady=(0,4),ipady=4)

        title("ALTURA TRIÁNGULO (m)")
        self.entry_altura=tk.Entry(panel,bg="#0f0f22",fg=self.PURP,
                                    font=("Courier New",11,"bold"),relief="flat",
                                    insertbackground=self.PURP,justify="center")
        self.entry_altura.insert(0,"0.15"); self.entry_altura.pack(**PAD,pady=(0,4),ipady=4)

        tk.Button(panel,text="🔍  Detectar triángulo + Aplicar",command=self._detect_and_apply,
                  bg="#220044",fg=self.YELLOW,relief="flat",
                  font=("Courier New",8,"bold"),cursor="hand2",
                  activebackground=self.ACC2,activeforeground="#fff",
                  bd=0,pady=7).pack(**PAD,pady=(4,4)); sep()

        title("PUNTOS TRIÁNGULO"); val("rp_tri",self.YELLOW,12)
        title("ESCALA"); val("rp_escala",self.SUB,11); sep()

        title("k  (N/m)",self.ACCENT)
        self.rp_k=tk.Label(panel,text="—",font=("Courier New",28,"bold"),
                            bg=BG,fg=self.ACCENT,anchor="w"); self.rp_k.pack(**PAD,pady=(0,2))

        title("c  (N·s/m)"); val("rp_c",self.YELLOW,14)
        title("ω₀ (rad/s)"); val("rp_omega0",self.TEXT,14)
        title("T  (s)");      val("rp_T",self.TEXT,13)
        title("f  (Hz)");     val("rp_f",self.TEXT,13)
        title("ζ  (amort.)"); val("rp_zeta",self.PURP,15)
        title("A  (m)");      val("rp_A",self.TEXT,12); sep()

        title("PUNTOS TRACKING"); val("rp_pts",self.PURP,13); sep()

        title("FUNCIÓN TRANSFERENCIA")
        self.rp_tf=tk.Label(panel,text="H(s) = 1/(ms²+cs+k)",
                             font=("Courier New",8),bg=BG,fg=self.SUB,
                             anchor="w",wraplength=330,justify="left")
        self.rp_tf.pack(**PAD,pady=(0,4)); sep()

        title("EXPERIMENTOS")
        tbl_f=tk.Frame(panel,bg=BG); tbl_f.pack(fill="x",padx=8,pady=(0,12))
        tbl_f.columnconfigure(0,weight=1)
        cols=("Exp","m_total(kg)","k(N/m)","ζ","T(s)")
        self.table=ttk.Treeview(tbl_f,columns=cols,show="headings",height=4)
        style=ttk.Style(); style.theme_use("clam")
        style.configure("Treeview",background="#0f0f22",foreground=self.TEXT,
                        fieldbackground="#0f0f22",font=("Courier New",7),rowheight=22)
        style.configure("Treeview.Heading",background=self.PANEL,foreground=self.ACCENT,
                        font=("Courier New",7,"bold"))
        style.map("Treeview",background=[("selected",self.ACC2)])
        cw={"Exp":60,"m_total(kg)":70,"k(N/m)":70,"ζ":60,"T(s)":55}
        for c in cols:
            self.table.heading(c,text=c); self.table.column(c,width=cw.get(c,60),anchor="center",stretch=True)
        sb2=ttk.Scrollbar(tbl_f,orient="vertical",command=self.table.yview)
        self.table.configure(yscrollcommand=sb2.set)
        sb2.grid(row=0,column=1,sticky="ns"); self.table.grid(row=0,column=0,sticky="ew")

    def _update_masa_total_label(self):
        """Actualiza el label de masa total en tiempo real mientras el usuario escribe."""
        try:
            mp = float(self.entry_masa.get())
            me = float(self.entry_masa_estimulo.get())
            total = mp + me
            self.lbl_masa_total.config(
                text=f"{total:.4f} kg  ({mp:.3f} + {me:.3f})",
                fg=self.ACCENT
            )
        except ValueError:
            self.lbl_masa_total.config(text="— (valores inválidos)", fg=self.RED)

    # ══════════════════════════════════════════
    #  EXPERIMENTO
    # ══════════════════════════════════════════
    def _new_experiment(self):
        self.exp_dir=next_experiment_folder(self.base_results)
        os.makedirs(self.exp_dir,exist_ok=True)
        self.lbl_exp.config(text=os.path.basename(self.exp_dir))
        self._status(f"Experimento: {os.path.basename(self.exp_dir)}")
        self._set_step("video")

    # ══════════════════════════════════════════
    #  VIDEO
    # ══════════════════════════════════════════
    def _load_video(self):
        path=filedialog.askopenfilename(
            filetypes=[("Videos","*.mp4 *.avi *.mov *.mkv *.MOV *.MP4"),("Todos","*.*")])
        if not path: return
        if self.cap: self.cap.release()
        self.cap=cv2.VideoCapture(path)
        if not self.cap.isOpened(): messagebox.showerror("Error","No se pudo abrir el video"); return
        self.video_path=path
        self.total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps=self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame=0; self.frame_inicio=None; self.frame_fin=None
        rot=int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
        self._video_rotation=rot
        self.slider.config(to=max(1,self.total_frames-1))
        self.lbl_slider_end.config(text=str(self.total_frames-1))
        self.lbl_video.config(text=os.path.basename(path))
        self.rp_video.config(text=os.path.basename(path))
        self._show_frame(); self._set_step("frames")
        self._status(f"Video · {self.total_frames} frames · {self.fps:.1f} fps")

    def _get_frame(self,n):
        if not self.cap: return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,n)
        ret,frame=self.cap.read()
        if not ret: return None
        rot=self._video_rotation
        if rot==90:  frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        elif rot==180: frame=cv2.rotate(frame,cv2.ROTATE_180)
        elif rot==270: frame=cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _show_frame(self,frame_bgr=None):
        if not self.cap: return
        if frame_bgr is None: frame_bgr=self._get_frame(self.current_frame)
        if frame_bgr is None: return
        self.update_idletasks()
        cw=max(self.canvas.winfo_width(),400); ch=max(self.canvas.winfo_height(),500)
        fh,fw=frame_bgr.shape[:2]
        scale=min(cw/fw,ch/fh)
        nw,nh=max(1,int(fw*scale)),max(1,int(fh*scale))
        self._display_scale=scale; self._display_offset=((cw-nw)//2,(ch-nh)//2)
        disp=cv2.resize(frame_bgr,(nw,nh))
        self._draw_overlay(disp)
        img=Image.fromarray(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB))
        self._tk_img=ImageTk.PhotoImage(img)
        ox,oy=self._display_offset
        self.canvas.delete("all")
        self.canvas.create_image(ox,oy,anchor="nw",image=self._tk_img)
        if self.roi_rect_canvas:
            x1,y1,x2,y2=self.roi_rect_canvas
            self.canvas.create_rectangle(x1,y1,x2,y2,outline=self.BLUE,width=2,dash=(4,3))
        if self.puntos_triangulo:
            cols=[(0,255,255),(255,100,0),(0,100,255)]
            labs=["TOP","BL","BR"]
            for i,(px,py) in enumerate(self.puntos_triangulo):
                sx=int(px*scale)+ox; sy=int(py*scale)+oy
                col=cols[i] if i<3 else (255,255,255)
                self.canvas.create_oval(sx-6,sy-6,sx+6,sy+6,fill=self._bgr_to_hex(col),outline="white",width=1)
                self.canvas.create_text(sx+12,sy,text=labs[i] if i<3 else str(i),
                                         fill=self._bgr_to_hex(col),font=("Courier New",8,"bold"))
            if len(self.puntos_triangulo)==3:
                pts=[(int(p[0]*scale)+ox,int(p[1]*scale)+oy) for p in self.puntos_triangulo]
                self.canvas.create_polygon(pts,outline=self.YELLOW,fill="",width=2)
        self.rp_frame.config(text=str(self.current_frame))
        fi=self.frame_inicio; ff=self.frame_fin
        self.rp_fi_ff.config(text=f"{fi if fi is not None else '—'} / {ff if ff is not None else '—'}")
        self._slider_updating=True
        try: self.slider.set(self.current_frame)
        except: pass
        self._slider_updating=False

    def _bgr_to_hex(self,bgr):
        b,g,r=bgr; return f"#{r:02x}{g:02x}{b:02x}"

    def _draw_overlay(self,frame):
        h,w=frame.shape[:2]
        if self.frame_inicio==self.current_frame:
            cv2.line(frame,(0,h//3),(w,h//3),(80,160,255),2)
            cv2.putText(frame,f"INICIO f{self.frame_inicio}",(8,h//3-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(80,160,255),2)
        if self.frame_fin==self.current_frame:
            cv2.line(frame,(0,2*h//3),(w,2*h//3),(200,100,255),2)
            cv2.putText(frame,f"FIN f{self.frame_fin}",(8,2*h//3-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,100,255),2)

    # ══════════════════════════════════════════
    #  CONTROLES NAVEGACIÓN
    # ══════════════════════════════════════════
    def _prev_frame(self):
        if self.cap: self.current_frame=max(0,self.current_frame-1); self._show_frame()
    def _next_frame(self):
        if self.cap: self.current_frame=min(self.total_frames-1,self.current_frame+1); self._show_frame()
    def _on_slider(self,val):
        if self.cap and not self._slider_updating:
            self.current_frame=int(float(val)); self._show_frame()
    def _toggle_play(self):
        if not self.cap: return
        self.playing=not self.playing
        if self.playing: self._play_loop()
    def _play_loop(self):
        if not self.playing: return
        self._next_frame()
        if self.current_frame>=self.total_frames-1: self.playing=False; return
        self.after(max(1,int(1000/self.fps)),self._play_loop)
    def _set_inicio(self):
        if self.cap: self.frame_inicio=self.current_frame; self._status(f"✅ Inicio → {self.frame_inicio}"); self._show_frame()
    def _set_fin(self):
        if self.cap: self.frame_fin=self.current_frame; self._status(f"✅ Fin → {self.frame_fin}"); self._show_frame()

    # ══════════════════════════════════════════
    #  MODOS CANVAS
    # ══════════════════════════════════════════
    def _activate_roi_marc(self):
        if not self.cap: self._status("⚠ Carga un video primero"); return
        self.canvas_mode="roi_marc"; self.roi_start=None; self.roi_rect_canvas=None
        self.rp_mode.config(text="ROI Marcadores — arrastra",fg=self.BLUE)
        self._status("🔺 Arrastra sobre UN marcador del triángulo")

    def _activate_roi_masa(self):
        if not self.cap: self._status("⚠ Carga un video primero"); return
        self.canvas_mode="roi_masa"; self.roi_start=None; self.roi_rect_canvas=None
        self.rp_mode.config(text="ROI Masa — arrastra",fg=self.PURP)
        self._status("⚫ Arrastra sobre la masa del resorte")

    def _detect_triangle(self):
        if not self.cap:
            self._status("⚠ Carga un video primero"); return
        if self.ref_lower is None:
            self._status("⚠ Primero calibra el color de los marcadores (ROI Marcadores)"); return

        frame = self._get_frame(self.frame_inicio if self.frame_inicio is not None else self.current_frame)
        if frame is None: return

        pts, mask = detectar_marcadores(frame, self.ref_lower, self.ref_upper)

        if len(pts) < 3:
            self._status(f"⚠ Solo se detectaron {len(pts)} marcadores — ajusta el ROI de color o la iluminación")
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self._show_frame(frame_bgr=mask_rgb)
            return

        try:
            P_top, P_bl, P_br = asignar_triangulo(pts[:3])
            self.puntos_triangulo = [tuple(P_top), tuple(P_bl), tuple(P_br)]
        except Exception as e:
            self._status(f"⚠ Error asignando triángulo: {e}"); return

        frame_vis = frame.copy()
        cols_bgr  = [(0,255,255),(255,100,0),(0,100,255)]
        labs      = ["TOP","BL","BR"]
        for i, (px, py) in enumerate(self.puntos_triangulo):
            cv2.circle(frame_vis,(int(px),int(py)),12,cols_bgr[i],-1)
            cv2.circle(frame_vis,(int(px),int(py)),14,(255,255,255),2)
            cv2.putText(frame_vis,labs[i],(int(px)+16,int(py)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,cols_bgr[i],2,cv2.LINE_AA)
        tri = np.array(self.puntos_triangulo, np.int32).reshape(-1,1,2)
        cv2.polylines(frame_vis,[tri],True,(0,220,180),2)

        ov = frame_vis.copy()
        cv2.rectangle(ov,(0,0),(frame_vis.shape[1],38),(8,8,28),-1)
        cv2.addWeighted(ov,0.72,frame_vis,0.28,0,frame_vis)
        cv2.putText(frame_vis,f"TRIANGULO DETECTADO — TOP·BL·BR  ({len(pts)} blobs encontrados)",
                    (8,26),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,180),1,cv2.LINE_AA)

        self._show_frame(frame_bgr=frame_vis)
        self.rp_tri.config(text="TOP · BL · BR ✓ (auto)")
        self._status(f"✅ Triángulo detectado — ingresa dimensiones y presiona Aplicar")
        self._set_step("calibracion")

    def _deactivate_modes(self):
        self.canvas_mode="none"; self.rp_mode.config(text="Normal",fg=self.SUB)

    def _canvas_to_video(self,cx,cy):
        ox,oy=self._display_offset; sc=self._display_scale
        return (cx-ox)/sc, (cy-oy)/sc

    def _canvas_click(self,event):
        if self.canvas_mode in ("roi_marc","roi_masa"):
            self.roi_start=(event.x,event.y)

    def _canvas_drag(self,event):
        if self.canvas_mode in ("roi_marc","roi_masa") and self.roi_start:
            x0,y0=self.roi_start
            self.roi_rect_canvas=(min(x0,event.x),min(y0,event.y),
                                   max(x0,event.x),max(y0,event.y))
            self._show_frame()

    def _canvas_release(self,event):
        if self.canvas_mode not in ("roi_marc","roi_masa"): return
        if not self.roi_start: return
        x0,y0=self.roi_start; x1,y1=event.x,event.y
        vx0,vy0=self._canvas_to_video(min(x0,x1),min(y0,y1))
        vx1,vy1=self._canvas_to_video(max(x0,x1),max(y0,y1))
        rw,rh=int(vx1-vx0),int(vy1-vy0)
        if rw>5 and rh>5:
            mode=self.canvas_mode
            frame=self._get_frame(self.frame_inicio if self.frame_inicio is not None else 0)
            if frame is not None:
                crop=frame[int(vy0):int(vy0+rh),int(vx0):int(vx0+rw)]
                if crop.size>0:
                    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                    hm=np.mean(hsv[:,:,0]); sm=np.mean(hsv[:,:,1]); vm=np.mean(hsv[:,:,2])
                    mh=15; ms=max(50,sm*0.45); mv=max(50,vm*0.45)
                    lo=np.array([max(0,hm-mh),max(0,sm-ms),max(0,vm-mv)])
                    hi=np.array([min(179,hm+mh),min(255,sm+ms),min(255,vm+mv)])
                    if mode=="roi_marc":
                        self.ref_lower=lo; self.ref_upper=hi
                        self._status(f"✅ Marcadores calibrados — H≈{hm:.0f} · detectando triángulo...")
                        self.rp_mode.config(text="Marcadores OK",fg=self.BLUE)
                        self._set_step("masa")
                        self.after(100, self._detect_triangle)
                    else:
                        self.masa_lower=lo; self.masa_upper=hi
                        self._status(f"✅ Masa calibrada — H≈{hm:.0f}")
                        self.rp_mode.config(text="Masa OK",fg=self.PURP)
                        self._set_step("calibracion")
        self.roi_rect_canvas=None; self.roi_start=None
        self.canvas_mode="none"

    def _detect_and_apply(self):
        self._detect_triangle()
        if len(self.puntos_triangulo) == 3:
            self._apply_dimensions()

    # ══════════════════════════════════════════
    #  DIMENSIONES Y HOMOGRAFÍA
    # ══════════════════════════════════════════
    def _apply_dimensions(self):
        if len(self.puntos_triangulo) < 3:
            self._status("⚠ Detecta primero el triángulo (botón 🔍)"); return
        try:
            self.base_real_m      = float(self.entry_base.get())
            self.altura_real_m    = float(self.entry_altura.get())
            self.masa_kg          = float(self.entry_masa.get())
            self.masa_estimulo_kg = float(self.entry_masa_estimulo.get())
            # ── Masa total efectiva: ambas oscilan juntas ────────────────
            self.masa_total_kg    = self.masa_kg + self.masa_estimulo_kg
        except ValueError:
            self._status("⚠ Valores numéricos inválidos"); return

        # Actualizar label con el desglose
        self.lbl_masa_total.config(
            text=f"{self.masa_total_kg:.4f} kg  ({self.masa_kg:.3f} + {self.masa_estimulo_kg:.3f})",
            fg=self.ACCENT
        )

        P_top,P_bl,P_br=asignar_triangulo(self.puntos_triangulo)
        half=self.base_real_m/2
        self.dst_pts=np.array([[0.0,0.0],[-half,self.altura_real_m],[half,self.altura_real_m]],dtype=np.float32)
        src=np.array([P_top,P_bl,P_br],dtype=np.float32)
        self.M_aff=cv2.getAffineTransform(src,self.dst_pts)

        d_px=np.linalg.norm(P_br-P_bl)
        escala=self.base_real_m/d_px if d_px>0 else 0
        self.rp_escala.config(text=f"≈{escala:.5f} m/px")
        self.rp_tri.config(text="TOP · BL · BR ✓")

        self._status(
            f"✅ Homografía aplicada · "
            f"m_principal={self.masa_kg:.3f}kg · "
            f"m_estímulo={self.masa_estimulo_kg:.3f}kg · "
            f"m_total={self.masa_total_kg:.3f}kg"
        )
        self._set_step("tracking")

    # ══════════════════════════════════════════
    #  TRACKING
    # ══════════════════════════════════════════
    def _run_tracking(self):
        errors=[]
        if self.frame_inicio is None or self.frame_fin is None: errors.append("marca frames de inicio y fin")
        if self.ref_lower is None: errors.append("calibra color de marcadores")
        if self.masa_lower is None: errors.append("calibra color de la masa")
        if self.M_aff is None: errors.append("aplica dimensiones del triángulo")
        if errors: messagebox.showwarning("Faltan pasos","Por favor:\n• "+"\n• ".join(errors)); return
        self._tracking_running=True; self.datos=[]; self.trail=[]
        self.rp_mode.config(text="TRACKING ▶",fg=self.ACCENT)
        self._status("⚙ Tracking en curso...")
        threading.Thread(target=self._tracking_worker,daemon=True).start()

    def _cancel_tracking(self):
        if self._tracking_running:
            self._tracking_running=False
            self.rp_mode.config(text="Cancelado",fg=self.RED)
            self._status("⏹ Tracking cancelado")

    def _tracking_worker(self):
        fi=self.frame_inicio; ff=self.frame_fin; fps=self.fps
        total=ff-fi+1; M=self.M_aff.copy()
        datos=[]; trail=[]

        for fn in range(fi,ff+1):
            if not self._tracking_running: break
            frame=self._get_frame(fn)
            if frame is None: break

            pts_f,_=detectar_marcadores(frame,self.ref_lower,self.ref_upper)
            if len(pts_f)>=3:
                try:
                    Pt,Pb1,Pb2=asignar_triangulo(pts_f[:3])
                    src_f=np.array([Pt,Pb1,Pb2],dtype=np.float32)
                    M_f=cv2.getAffineTransform(src_f,self.dst_pts)
                    if M_f is not None: M=M_f
                    hom_ok=True
                except: hom_ok=False
            else: hom_ok=False

            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(hsv,self.masa_lower,self.masa_upper)
            k=np.ones((7,7),np.uint8)
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
            mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE,k)
            cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            cx_px=cy_px=None
            if cnts:
                c=max(cnts,key=cv2.contourArea)
                Mm=cv2.moments(c)
                if Mm["m00"]>0:
                    cx_px=Mm["m10"]/Mm["m00"]; cy_px=Mm["m01"]/Mm["m00"]
                    x_m,y_m=img_to_world((cx_px,cy_px),M)
                    t=(fn-fi)/fps
                    datos.append((t,x_m,y_m))
                    trail.append((cx_px,cy_px))

            prog=int((fn-fi+1)/total*100)
            frame_disp=frame.copy()
            self._draw_tracking_frame(frame_disp,cx_px,cy_px,trail,pts_f,
                                       hom_ok,fn,fi,ff,
                                       datos[-1][2] if datos else 0,
                                       datos[-1][0] if datos else 0)
            self.after(0,lambda f=frame_disp,p=prog:self._update_canvas_tracking(f,p))
            import time; time.sleep(0.001)

        self.datos=datos; self.trail=trail
        self._tracking_running=False
        self.after(0,self._tracking_done)

    def _draw_tracking_frame(self,frame,cx,cy,trail,marcadores,hom_ok,fn,fi,ff,y_m,t):
        h,w=frame.shape[:2]; total=ff-fi+1; prog=(fn-fi+1)/total
        ov=frame.copy()
        cv2.rectangle(ov,(0,0),(w,42),(8,8,28),-1)
        cv2.addWeighted(ov,0.72,frame,0.28,0,frame)
        bw=int(w*prog)
        cv2.rectangle(frame,(0,38),(bw,42),(0,200,150),-1)
        cv2.rectangle(frame,(0,38),(w,42),(40,40,80),1)
        hom_str="HOM:OK" if hom_ok else "HOM:FB"
        hom_col=(0,255,200) if hom_ok else (180,50,255)
        cv2.putText(frame,f"TRACKING  f{fn}/{ff}  {int(prog*100)}%  {hom_str}",
                    (8,26),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,255,180),1,cv2.LINE_AA)
        cv2.putText(frame,f"t={t:.3f}s   y={y_m*100:.2f}cm",
                    (w-230,26),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,220,0),1,cv2.LINE_AA)
        for p in marcadores[:3]:
            cv2.circle(frame,(int(p[0]),int(p[1])),7,hom_col,-1)
        n=len(trail)
        for i in range(1,n):
            alpha=i/n; r=int(255*alpha); gc=int(200*(1-alpha*0.5))
            cv2.line(frame,(int(trail[i-1][0]),int(trail[i-1][1])),
                           (int(trail[i][0]),int(trail[i][1])),(0,gc,r),2,cv2.LINE_AA)
        if cx is not None:
            icx,icy=int(cx),int(cy)
            cv2.circle(frame,(icx,icy),11,(0,255,180),-1)
            cv2.circle(frame,(icx,icy),14,(255,255,255),1)
            cv2.line(frame,(icx-20,icy),(icx+20,icy),(255,255,255),1)
            cv2.line(frame,(icx,icy-20),(icx,icy+20),(255,255,255),1)
            cv2.putText(frame,f"y={y_m*100:.2f}cm",(icx+16,icy-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,255,180),1,cv2.LINE_AA)

    def _update_canvas_tracking(self,frame,prog):
        self._show_frame(frame_bgr=frame)
        self._status(f"⚙ Tracking {prog}%  —  {len(self.datos)} puntos")

    def _tracking_done(self):
        self._tracking_running=False; self.rp_mode.config(text="Normal",fg=self.SUB)
        if len(self.datos)<5:
            self._status("⚠ Datos insuficientes — revisa colores y marcadores"); return

        d=np.array(self.datos); t_raw=d[:,0]; y_raw=d[:,2]

        # ── Usar masa_total_kg en el cálculo ────────────────────────────
        r=calcular_resultados(t_raw, y_raw, self.masa_total_kg)

        self.result={**r,
            "video":            os.path.basename(self.video_path),
            "masa_kg_principal":self.masa_kg,
            "masa_estimulo_kg": self.masa_estimulo_kg,
            "masa_total_kg":    self.masa_total_kg,
            # alias para compatibilidad con figuras y evidencia
            "masa_kg":          self.masa_total_kg,
            "k_fit":            r["k_fit"],
            "c_fit":            r["c_fit"],
            "omega0":           r["omega0"],
            "T_fit":            r["T_fit"],
            "f_fit":            r["f_fit"],
            "zeta":             r["zeta"],
            "A_fit":            r["A_fit"],
            "gamma_fit":        r["gamma_fit"],
            "error_pct":        0,
            "n_puntos":         len(self.datos),
            "timestamp":        datetime.datetime.now().isoformat(),
            "experimento":      os.path.basename(self.exp_dir),
        }

        # Panel derecho
        self.rp_k.config(text=f"{r['k_fit']:.4f}", fg=self.ACCENT)
        self.rp_c.config(text=f"{r['c_fit']:.5f} N·s/m")
        self.rp_omega0.config(text=f"{r['omega0']:.4f} rad/s")
        self.rp_T.config(text=f"{r['T_fit']:.4f} s")
        self.rp_f.config(text=f"{r['f_fit']:.4f} Hz")
        z_col=self.ACCENT if r["zeta"]<0.5 else (self.YELLOW if r["zeta"]<1 else self.RED)
        self.rp_zeta.config(text=f"{r['zeta']:.5f}",fg=z_col)
        self.rp_A.config(text=f"{r['A_fit']*100:.3f} cm")
        self.rp_pts.config(text=str(len(self.datos)))

        # Función de transferencia — usa masa total
        o0=r["omega0"]; z=r["zeta"]; m=self.masa_total_kg
        tf_str=(f"H(s) = 1/({m:.4f}s² + {r['c_fit']:.5f}s + {r['k_fit']:.4f})\n"
                f"m={self.masa_kg:.3f}+{self.masa_estimulo_kg:.3f}={m:.4f}kg  "
                f"ω₀={o0:.4f}  ζ={z:.4f}")
        self.rp_tf.config(text=tf_str)

        self._save_results()
        self._set_step("resultados")
        self._status(
            f"✅ k={r['k_fit']:.4f} N/m  ω₀={r['omega0']:.4f} rad/s  "
            f"ζ={r['zeta']:.5f}  T={r['T_fit']:.4f}s  "
            f"m_total={self.masa_total_kg:.4f}kg"
        )

        frame_final=self._get_frame(self.frame_fin)
        if frame_final is not None and self.trail:
            self._draw_final_trail(frame_final,r)
            self._show_frame(frame_bgr=frame_final)

        self.table.insert("","end",values=(
            os.path.basename(self.exp_dir),
            f"{self.masa_total_kg:.3f}",
            f"{r['k_fit']:.4f}",
            f"{r['zeta']:.4f}",
            f"{r['T_fit']:.4f}"
        ))

        self._auto_git_commit()

    def _draw_final_trail(self,frame,r):
        h,w=frame.shape[:2]; n=len(self.trail)
        for i in range(1,n):
            alpha=i/n; rc=int(255*alpha); gc=int(200*(1-alpha*0.5))
            cv2.line(frame,(int(self.trail[i-1][0]),int(self.trail[i-1][1])),
                           (int(self.trail[i][0]),int(self.trail[i][1])),
                           (0,gc,rc),2,cv2.LINE_AA)
        if self.trail:
            cv2.circle(frame,(int(self.trail[0][0]),int(self.trail[0][1])),9,(80,160,255),-1)
            cv2.circle(frame,(int(self.trail[-1][0]),int(self.trail[-1][1])),9,(200,100,255),-1)
        ov=frame.copy()
        cv2.rectangle(ov,(0,h-46),(w,h),(8,8,28),-1)
        cv2.addWeighted(ov,0.8,frame,0.2,0,frame)
        cv2.putText(frame,
                    f"COMPLETO  k={r['k_fit']:.4f}N/m  ω₀={r['omega0']:.4f}rad/s  ζ={r['zeta']:.4f}  T={r['T_fit']:.4f}s",
                    (10,h-16),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,255,180),1,cv2.LINE_AA)

    def _save_results(self):
        r=self.result
        # JSON: excluir arrays numpy pero incluir todos los campos de masa
        json_safe = {k: v for k, v in r.items()
                     if not isinstance(v, (np.ndarray, list)) or k in ("polos",)}
        with open(os.path.join(self.exp_dir,"resultado.json"),"w") as f:
            json.dump(json_safe, f, indent=2, default=str)
        np.savetxt(os.path.join(self.exp_dir,"datos.csv"),
                   np.column_stack((r["t_raw"],r["y_data"])),
                   header="t(s),y_desplazamiento(m)",delimiter=",",fmt="%.6f",comments="")
        generate_fig1(r, os.path.join(self.exp_dir,"fig1_cinematica.png"))
        generate_fig2(r, os.path.join(self.exp_dir,"fig2_fases.png"))
        generate_fig3(r, os.path.join(self.exp_dir,"fig3_espectro.png"), self.fps)
        generate_fig4(r, os.path.join(self.exp_dir,"fig4_bode.png"))
        generate_fig5(r, os.path.join(self.exp_dir,"fig5_polos.png"))
        frame_ev=self._get_frame(self.frame_inicio)
        if frame_ev is not None:
            save_evidence_image(frame_ev,r,os.path.basename(self.exp_dir),
                                os.path.join(self.exp_dir,"evidencia.png"))

    # ══════════════════════════════════════════
    #  VER GRÁFICAS
    # ══════════════════════════════════════════
    def _show_graficas(self):
        if not self.result: messagebox.showinfo("Sin datos","Ejecuta el tracking primero"); return
        win=tk.Toplevel(self); win.title("Gráficas — Masa-Resorte")
        win.configure(bg=self.BG); win.geometry("900x650")
        nb=ttk.Notebook(win); nb.pack(fill="both",expand=True,padx=8,pady=8)
        figs=[
            ("Cinemática",   "fig1_cinematica.png"),
            ("Espacio Fases","fig2_fases.png"),
            ("PSD",          "fig3_espectro.png"),
            ("Bode",         "fig4_bode.png"),
            ("Polos",        "fig5_polos.png"),
        ]
        for name,fname in figs:
            path=os.path.join(self.exp_dir,fname)
            if not os.path.exists(path): continue
            frame=tk.Frame(nb,bg=self.BG); nb.add(frame,text=name)
            img=Image.open(path); img.thumbnail((880,600))
            tk_img=ImageTk.PhotoImage(img)
            lbl=tk.Label(frame,image=tk_img,bg=self.BG)
            lbl.image=tk_img; lbl.pack(pady=6)

    # ══════════════════════════════════════════
    #  RESET
    # ══════════════════════════════════════════
    def _reset_all(self):
        if not messagebox.askyesno("Resetear","¿Resetear toda la calibración y datos?"): return
        self.frame_inicio=None; self.frame_fin=None
        self.ref_lower=None; self.ref_upper=None
        self.masa_lower=None; self.masa_upper=None
        self.puntos_triangulo=[]; self.M_aff=None
        self.datos=[]; self.trail=[]; self.result={}
        self.roi_rect_canvas=None
        self.masa_kg=0.0; self.masa_estimulo_kg=0.0; self.masa_total_kg=0.0
        for attr in ["rp_k","rp_c","rp_omega0","rp_T","rp_f","rp_zeta","rp_A","rp_pts","rp_tri","rp_escala"]:
            getattr(self,attr).config(text="—",fg=self.SUB)
        self.rp_k.config(fg=self.ACCENT)
        self.lbl_masa_total.config(text="— (resetear)", fg=self.SUB)
        self._show_frame(); self._set_step("frames"); self._status("↺ Reseteado")

    # ══════════════════════════════════════════
    #  GIT
    # ══════════════════════════════════════════
    def _auto_git_commit(self):
        if not self.result: return
        r = self.result
        msg = (f"[{os.path.basename(self.exp_dir)}] "
               f"k={r.get('k_fit',0):.4f}N/m "
               f"omega0={r.get('omega0',0):.4f}rad/s "
               f"zeta={r.get('zeta',0):.4f} "
               f"T={r.get('T_fit',0):.4f}s "
               f"m_total={r.get('masa_total_kg',0):.4f}kg")
        def _c():
            ok, log = git_auto_commit(self.exp_dir, msg)
            suffix = "  |  ☁ commit OK" if ok else "  |  ☁ commit local (sin remote)"
            self.after(0, lambda s=suffix: self.lbl_status.config(
                text=self.lbl_status.cget("text") + s))
        threading.Thread(target=_c, daemon=True).start()

    def _do_git_commit(self):
        if not self.result: self._status("⚠ Sin resultados"); return
        r=self.result
        msg=(f"[{os.path.basename(self.exp_dir)}] "
             f"k={r.get('k_fit',0):.4f}N/m "
             f"omega0={r.get('omega0',0):.4f}rad/s "
             f"zeta={r.get('zeta',0):.4f} "
             f"T={r.get('T_fit',0):.4f}s")
        self._status("☁ Haciendo commit...")
        def _c():
            ok,log=git_auto_commit(self.exp_dir,msg)
            self.after(0,lambda: self._show_git_log(ok,log))
            self.after(0,lambda: self._status("✅ Commit OK" if ok else "❌ Commit falló"))
        threading.Thread(target=_c,daemon=True).start()

    def _show_git_log(self,ok,log_txt):
        win=tk.Toplevel(self); win.title("Git"); win.configure(bg=self.BG); win.geometry("700x380")
        tk.Label(win,text="Git Log",font=("Courier New",10,"bold"),
                 bg=self.BG,fg=self.ACCENT if ok else self.RED).pack(pady=(10,4))
        frm=tk.Frame(win,bg=self.BG); frm.pack(fill="both",expand=True,padx=10)
        txt=tk.Text(frm,bg="#0a0a1a",fg="#ccffcc" if ok else "#ffaaaa",
                    font=("Courier New",8),wrap="word",relief="flat")
        sb=ttk.Scrollbar(frm,command=txt.yview); txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right",fill="y"); txt.pack(fill="both",expand=True)
        txt.insert("1.0",log_txt); txt.config(state="disabled")
        tk.Button(win,text="Cerrar",command=win.destroy,bg=self.CARD,fg=self.TEXT,
                  relief="flat",font=("Courier New",9),padx=14,pady=6).pack(pady=8)

    def _configure_remote(self):
        git=_find_git(); root=os.path.dirname(os.path.abspath(__file__))
        r=subprocess.run([git,"remote","get-url","origin"],cwd=root,capture_output=True,text=True)
        current=r.stdout.strip() if r.returncode==0 else ""
        win=tk.Toplevel(self); win.title("Remote"); win.configure(bg=self.BG)
        win.geometry("540x200"); win.resizable(False,False)
        tk.Label(win,text="🔗  Remote Git (origin)",font=("Courier New",11,"bold"),
                 bg=self.BG,fg=self.ACCENT).pack(pady=(14,4))
        entry=tk.Entry(win,bg=self.CARD,fg=self.TEXT,insertbackground=self.ACCENT,
                       font=("Courier New",10),relief="flat",width=50)
        entry.insert(0,current); entry.pack(padx=20,pady=8,ipady=6)
        res=tk.Label(win,text="",font=("Courier New",8),bg=self.BG,fg=self.SUB); res.pack()
        def _apply():
            url=entry.get().strip()
            subprocess.run([git,"remote","remove","origin"],cwd=root,capture_output=True)
            r2=subprocess.run([git,"remote","add","origin",url],cwd=root,capture_output=True,text=True)
            res.config(text="✅ Configurado" if r2.returncode==0 else f"❌ {r2.stderr.strip()}")
        bf=tk.Frame(win,bg=self.BG); bf.pack(pady=8)
        tk.Button(bf,text="Aplicar",command=_apply,bg=self.ACCENT,fg="#000",
                  font=("Courier New",9,"bold"),relief="flat",padx=16,pady=6).pack(side="left",padx=4)
        tk.Button(bf,text="Cerrar",command=win.destroy,bg=self.CARD,fg=self.TEXT,
                  font=("Courier New",9),relief="flat",padx=16,pady=6).pack(side="left",padx=4)

    # ══════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════
    def _set_step(self,step):
        self.step=step
        idx=self.STEPS.index(step) if step in self.STEPS else -1
        names={"video":"1. Cargar Video","frames":"2. Marcar Frames",
               "marcadores":"3. ROI Marcadores","masa":"4. ROI Masa",
               "calibracion":"5. Detectar + Dimensiones","tracking":"6. Tracking",
               "resultados":"7. Resultados"}
        for i,(k,name) in enumerate(names.items()):
            si=self.STEPS.index(k)
            if si<idx:   col,pfx=self.ACCENT,"✓ "
            elif si==idx: col,pfx=self.YELLOW,"▶ "
            else:         col,pfx=self.SUB,"  "
            self.step_labels[k].config(text=f"  {pfx}{name}",fg=col)

    def _status(self,msg):
        self.lbl_status.config(text=msg); self.update_idletasks()

    def on_close(self):
        if self.cap: self.cap.release()
        self.destroy()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app=MasaResorteApp()
    app.protocol("WM_DELETE_WINDOW",app.on_close)
    app.mainloop()