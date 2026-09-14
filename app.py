"""
CURAVERSE — Breast Cancer Care Platform (prototype)
Single-file Flask app. UI is rendered directly from Python (no templates/ dir).

This is a DEMO / PROTOTYPE application:
- OTP login is simulated locally (no real SMS is sent).
- Image processing (mammogram + ultrasound) uses classic preprocessing +
  K-Means clustering purely for illustrative visual segmentation.
- Nothing in this app is a medical diagnosis. All clinical decisions must be
  made by a qualified clinician.
"""

from flask import Flask, request, redirect, session, render_template_string, send_from_directory
from markupsafe import escape
from pathlib import Path
import sqlite3
import uuid
import random

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


app = Flask(__name__)
app.secret_key = "curaverse-demo-secret"

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
DB = BASE / "patients.db"

DOCTOR_NAME = "Rupesh Kumar"
DOCTOR_ROLE = "Admin"
DEMO_OTP = "123456"


# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------

def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            symptoms TEXT,
            medical_history TEXT,
            family_history TEXT,
            examination TEXT,
            status TEXT,
            diagnosis TEXT,
            visit_date TEXT
        );

        CREATE TABLE IF NOT EXISTS medical_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            original TEXT,
            preprocessed TEXT,
            segmented TEXT,
            algorithm TEXT,
            inertia REAL,
            iterations INTEGER,
            stats TEXT
        );

        CREATE TABLE IF NOT EXISTS ultrasound_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            original TEXT,
            preprocessed TEXT,
            segmented TEXT,
            algorithm TEXT,
            inertia REAL,
            iterations INTEGER,
            stats TEXT,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            date TEXT,
            time TEXT,
            doctor TEXT,
            type TEXT,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            title TEXT,
            findings TEXT,
            recommendation TEXT
        );
        """
    )
    conn.commit()

    # Seed demo data once, so a fresh install visually matches the reference
    # design (12 patients / 3 appointments today / 5 reports / 2 pending).
    count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    if count == 0:
        demo_patients = [
            ("CV-1001", "Ananya Sharma", 32, "Female", "+91 90000 10001", "Active", "Follow-up advised", "05 Aug 2025"),
            ("CV-1002", "Priya Singh", 45, "Female", "+91 90000 10002", "Active", "Under observation", "04 Aug 2025"),
            ("CV-1003", "Neha Verma", 29, "Female", "+91 90000 10003", "Pending", "Report review pending", "03 Aug 2025"),
            ("CV-1004", "Sneha Patel", 52, "Female", "+91 90000 10004", "Active", "Routine screening", "02 Aug 2025"),
            ("CV-1005", "Kavita Rao", 38, "Female", "+91 90000 10005", "Completed", "Screening cleared", "29 Jul 2025"),
            ("CV-1006", "Meera Nair", 41, "Female", "+91 90000 10006", "Active", "Follow-up advised", "27 Jul 2025"),
            ("CV-1007", "Divya Iyer", 36, "Female", "+91 90000 10007", "Completed", "Screening cleared", "25 Jul 2025"),
            ("CV-1008", "Ritu Malhotra", 47, "Female", "+91 90000 10008", "Active", "Under observation", "22 Jul 2025"),
            ("CV-1009", "Pooja Desai", 55, "Female", "+91 90000 10009", "Pending", "Report review pending", "20 Jul 2025"),
            ("CV-1010", "Shalini Menon", 33, "Female", "+91 90000 10010", "Active", "Routine screening", "18 Jul 2025"),
            ("CV-1011", "Anjali Gupta", 49, "Female", "+91 90000 10011", "Completed", "Screening cleared", "15 Jul 2025"),
            ("CV-1012", "Sunita Reddy", 44, "Female", "+91 90000 10012", "Active", "Follow-up advised", "12 Jul 2025"),
        ]
        conn.executemany(
            """INSERT INTO patients
               (patient_id,name,age,gender,phone,status,diagnosis,visit_date)
               VALUES (?,?,?,?,?,?,?,?)""",
            demo_patients,
        )

        conn.executemany(
            """INSERT INTO appointments(patient_id,date,time,doctor,type,status)
               VALUES (?,?,?,?,?,?)""",
            [
                (1, "2025-08-06", "10:00 AM", "Dr. " + DOCTOR_NAME, "Follow-up", "Scheduled"),
                (2, "2025-08-06", "02:30 PM", "Dr. " + DOCTOR_NAME, "Consultation", "Scheduled"),
                (3, "2025-08-06", "04:00 PM", "Dr. " + DOCTOR_NAME, "Report Review", "Scheduled"),
            ],
        )

        conn.executemany(
            """INSERT INTO reports(patient_id,title,findings,recommendation)
               VALUES (?,?,?,?)""",
            [
                (1, "Screening Report", "No abnormal mass detected on initial review.", "Routine follow-up in 6 months."),
                (2, "Diagnostic Report", "Area of interest flagged for closer review.", "Refer for ultrasound correlation."),
                (3, "Follow-up Report", "Comparable to prior study.", "Continue routine monitoring."),
                (4, "Screening Report", "Clear screening result.", "Annual screening recommended."),
                (5, "Consultation Report", "Discussed family history and risk factors.", "Genetic counselling referral offered."),
            ],
        )

    conn.commit()
    conn.close()


# ----------------------------------------------------------------------------
# Icons (small inline SVGs, feather-style)
# ----------------------------------------------------------------------------

ICONS = {
    "home": '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V20a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V9.5"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "user-plus": '<path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><line x1="19" y1="8" x2="19" y2="14"/><line x1="22" y1="11" x2="16" y2="11"/>',
    "file-text": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "clipboard": '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>',
    "radio": '<circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 0 1 0 8.49M7.76 16.25a6 6 0 0 1 0-8.49M19.07 4.93a10 10 0 0 1 0 14.14M4.93 19.07a10 10 0 0 1 0-14.14"/>',
    "file": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "bar-chart": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "log-out": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
    "moon": '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
    "chevron-right": '<polyline points="9 18 15 12 9 6"/>',
    "chevron-down": '<polyline points="6 9 12 15 18 9"/>',
    "menu": '<line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/>',
    "arrow-left": '<line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "phone": '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6A19.79 19.79 0 0 1 2.11 4.18 2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57 2.81.7A2 2 0 0 1 22 16.92z"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "waveform": '<path d="M2 12h2l1.5 5L9 5l3 14 2.5-11L16 12h6"/>',
}


def icon(name, size=18, cls=""):
    body = ICONS.get(name, "")
    return (
        f'<svg class="ic {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    )


AVATAR_COLORS = ["#6f55e8", "#d52b88", "#2aa9a1", "#e08e2b", "#3b7ddb", "#c0447a"]


def avatar(name, size=32):
    letter = (name or "P")[0].upper()
    color = AVATAR_COLORS[sum(ord(c) for c in (name or "P")) % len(AVATAR_COLORS)]
    fs = int(size * 0.42)
    return (
        f'<span class="avatar" style="width:{size}px;height:{size}px;'
        f'font-size:{fs}px;background:{color}22;color:{color}">{letter}</span>'
    )


# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box}
:root{
  --bg:#f5f6fc;--card:#ffffff;--text:#182039;--muted:#868da3;--line:#eaecf5;--nav:#ffffff;
  --active-bg:#eef0ff;--active-text:#5b4fe0;
  --p1:#6c5ce7;--p2:#a855f7;--p3:#ec4899;
  --shadow:0 6px 20px rgba(30,20,70,0.06);
}
html.dark{
  --bg:#0d0f1c;--card:#161a2c;--text:#f3f4ff;--muted:#9aa0ba;--line:#262b45;--nav:#12162a;
  --active-bg:linear-gradient(90deg,var(--p1),var(--p3));--active-text:#ffffff;
  --shadow:0 6px 24px rgba(0,0,0,0.35);
}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 "Segoe UI",Arial,sans-serif;transition:background .25s,color .25s}
a{text-decoration:none;color:inherit}
.ic{display:inline-block;vertical-align:middle;flex-shrink:0}

/* ---------- App shell ---------- */
.side{position:fixed;left:0;top:0;bottom:0;width:238px;background:var(--nav);border-right:1px solid var(--line);z-index:40;display:flex;flex-direction:column;transition:background .25s}
.brand{height:74px;min-height:74px;padding:0 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:11px}
.logo{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,var(--p1),var(--p2));display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:17px;box-shadow:0 4px 12px rgba(108,92,231,.35);animation:pulseLogo 3.5s ease-in-out infinite}
.brand b{font-size:17px;letter-spacing:.2px}
.brand small{display:block;color:var(--muted);margin-top:2px;font-size:10.5px}
.nav{padding:12px 10px;overflow-y:auto;flex:1}
.nav a{display:flex;align-items:center;gap:11px;padding:10px 12px;margin:2px 0;border-radius:9px;color:var(--muted);font-size:13px;font-weight:600;transition:all .18s;position:relative}
.nav a:hover{background:var(--bg);color:var(--text)}
.nav a.active{background:var(--active-bg);color:var(--active-text)}
html.dark .nav a.active{color:#fff}
.badge-new{background:linear-gradient(90deg,var(--p1),var(--p3));color:#fff;padding:2px 6px;border-radius:8px;font-size:9px;font-weight:800;margin-left:auto}

.main{margin-left:238px;min-height:100vh}
.top{position:sticky;top:0;height:72px;background:var(--card);border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 26px;gap:15px;z-index:30;transition:background .25s}
.hbtn{display:none;background:none;border:0;color:var(--text);cursor:pointer}
.searchform{flex:1;max-width:380px}
.search{display:flex;align-items:center;gap:9px;width:100%;border:1px solid var(--line);background:var(--bg);border-radius:10px;padding:9px 13px;color:var(--muted)}
.search input{border:0;background:transparent;color:var(--text);width:100%;font:inherit;padding:0;margin:0}
.search input:focus{outline:none}
.topright{display:flex;align-items:center;gap:16px}

/* theme toggle pill */
.tgl{position:relative;width:52px;height:28px;border-radius:16px;background:#e7e9f5;border:1px solid var(--line);cursor:pointer;display:flex;align-items:center;padding:0 6px;justify-content:space-between;transition:background .25s}
html.dark .tgl{background:#232849}
.tgl .knob{position:absolute;top:2px;left:2px;width:22px;height:22px;border-radius:50%;background:linear-gradient(135deg,var(--p1),var(--p2));transition:transform .25s;box-shadow:0 2px 6px rgba(0,0,0,.25);display:flex;align-items:center;justify-content:center;color:#fff}
html.dark .tgl .knob{transform:translateX(24px)}
.tgl svg{width:13px;height:13px;color:var(--muted)}

.doc{display:flex;gap:9px;align-items:center;cursor:pointer}
.avatar{border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-weight:800}
.docname{line-height:1.25}
.docname b{font-size:13px}
.docname span{display:block;color:var(--muted);font-size:11px}

.content{padding:24px 27px 50px;max-width:1400px;animation:fadeUp .4s ease both}
h1{margin:0 0 4px;font-size:24px;font-weight:800}
.muted{color:var(--muted);font-size:12px}
.head{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;flex-wrap:wrap;gap:10px}

/* ---------- Hero ---------- */
.hero{background:linear-gradient(115deg,#eef0ff,#fdeef8);border:1px solid #e7ddf2;border-radius:16px;padding:26px 30px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;gap:24px;overflow:hidden;position:relative;animation:fadeUp .5s ease both}
html.dark .hero{background:linear-gradient(115deg,#241f3f,#33223f);border-color:#3a3054}
.hero .tag{color:var(--p3);font-weight:800;font-size:12px;letter-spacing:.4px}
.hero h2{margin:6px 0 8px;font-size:22px;font-weight:800}
.hero p{margin:0;color:var(--muted);max-width:480px}
.hero-art{width:150px;flex-shrink:0;animation:floaty 5s ease-in-out infinite}

/* ---------- Stat cards ---------- */
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:17px;box-shadow:var(--shadow);animation:fadeUp .5s ease both}
.cards .card:nth-child(1){animation-delay:.05s}.cards .card:nth-child(2){animation-delay:.1s}
.cards .card:nth-child(3){animation-delay:.15s}.cards .card:nth-child(4){animation-delay:.2s}
.stat-top{display:flex;justify-content:space-between;align-items:flex-start}
.stat-label{font-size:10.5px;color:var(--muted);font-weight:800;letter-spacing:.3px}
.stat-icon{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.stat-icon.blue{background:#e8f0ff;color:#3b7ddb}.stat-icon.purple{background:#f1ecff;color:#8b5cf6}
.stat-icon.green{background:#e6faf0;color:#22b573}.stat-icon.orange{background:#fff1e2;color:#f2994a}
html.dark .stat-icon.blue{background:#173257}html.dark .stat-icon.purple{background:#2c2149}
html.dark .stat-icon.green{background:#123a2b}html.dark .stat-icon.orange{background:#402c14}
.num{font-size:28px;font-weight:800;margin-top:10px}
.trend{font-size:11.5px;color:#35a66b;margin-top:4px}

/* ---------- Panels / lists ---------- */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:15px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:18px;margin-top:16px;box-shadow:var(--shadow);animation:fadeUp .5s ease both}
.panel h2{font-size:15.5px;margin:0 0 12px;font-weight:800}
.row{display:flex;align-items:center;justify-content:space-between;padding:11px 4px;border-bottom:1px solid var(--line);transition:background .15s;border-radius:8px}
.row:last-child{border-bottom:0}
.row:hover{background:var(--bg)}
.rleft{display:flex;align-items:center;gap:10px}
.rname{font-weight:700;font-size:13px}
.rsub{color:var(--muted);font-size:11.5px}
.rtime{font-weight:800;font-size:13px;width:72px;flex-shrink:0}

table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:11px 8px;border-bottom:1px solid var(--line);font-size:12.5px}
th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.3px}
tr:hover td{background:var(--bg)}

.btn{display:inline-flex;align-items:center;gap:7px;background:linear-gradient(90deg,var(--p1),var(--p3));color:#fff;border:0;border-radius:9px;padding:10px 16px;font-weight:700;cursor:pointer;font-size:13px;transition:transform .15s,box-shadow .15s;box-shadow:0 4px 14px rgba(108,92,231,.28)}
.btn:hover{transform:translateY(-1px);box-shadow:0 8px 20px rgba(108,92,231,.38)}
.secondary{background:var(--card);color:var(--text);border:1px solid var(--line);box-shadow:none}
.secondary:hover{background:var(--bg);box-shadow:none}
.pill{padding:3px 10px;border-radius:20px;font-size:10.5px;font-weight:800;display:inline-block}
.pill.active,.pill.scheduled{background:#e6faf0;color:#1c9c62}
.pill.pending{background:#fff1e2;color:#c9740f}
.pill.completed,.pill.cancelled{background:#eef0f7;color:#5c6785}
html.dark .pill.active,html.dark .pill.scheduled{background:#123a2b}
html.dark .pill.pending{background:#402c14}
html.dark .pill.completed,html.dark .pill.cancelled{background:#232849}
.new{background:linear-gradient(90deg,var(--p1),var(--p3));color:#fff;padding:2px 7px;border-radius:9px;font-size:9px;margin-left:6px}
.note{padding:11px 13px;background:#fff0f7;border-left:3px solid var(--p3);font-size:11.5px;border-radius:0 8px 8px 0;display:flex;gap:8px;align-items:flex-start}
html.dark .note{background:#2a1c30}
.note.warn{background:#fff4e0;border-left-color:#e08e2b}
html.dark .note.warn{background:#3a2c14}
.note.err{background:#ffe3e3;border-left-color:#c0392b}
html.dark .note.err{background:#3a1c1c}

input,select,textarea{width:100%;padding:10px 12px;border:1px solid var(--line);background:var(--bg);color:var(--text);border-radius:9px;margin:5px 0 13px;font:inherit;transition:border-color .15s}
input:focus,select:focus,textarea:focus{outline:none;border-color:var(--p1)}
label{font-size:12px;font-weight:700;color:var(--muted)}
footer.appfoot{display:flex;justify-content:space-between;color:var(--muted);font-size:11.5px;padding:18px 4px 0;flex-wrap:wrap;gap:6px}

/* ---------- Image / ultrasound processing ---------- */
.steps{display:flex;padding:20px;gap:0}
.step{display:flex;align-items:center;gap:11px;flex:1;position:relative}
.step:not(:last-child):after{content:"";position:absolute;top:19px;left:calc(50% + 30px);right:calc(-50% + 20px);height:2px;background:var(--line)}
.circle{width:38px;height:38px;border-radius:50%;background:var(--bg);border:2px solid var(--line);display:flex;align-items:center;justify-content:center;font-weight:800;color:var(--muted);flex-shrink:0;z-index:1}
.step.done .circle{background:linear-gradient(135deg,var(--p1),var(--p3));border-color:transparent;color:#fff}
.step small{display:block;color:var(--muted);font-size:10.5px}
.ip{display:grid;grid-template-columns:1fr 1fr 1fr 1.1fr;gap:14px;padding:0 15px 15px}
.box{border:1px solid var(--line);border-radius:11px;padding:12px;text-align:center;background:var(--bg)}
.box h3{font-size:12px;margin:0 0 10px}
.box img{width:100%;border-radius:8px;display:block;background:#000}
.box .ph{height:180px;border-radius:8px;border:1.5px dashed var(--line);display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px}
.meta{margin-top:8px;font-size:10.5px;color:var(--muted)}
.pie{width:120px;height:120px;border-radius:50%;margin:10px auto}
.legend{display:flex;justify-content:space-around;text-align:center;margin-top:8px}
.legend span{display:block}
.legend .sw{width:9px;height:9px;border-radius:3px;display:inline-block;margin-right:4px}
.upload-row{display:flex;gap:10px;align-items:center;padding:16px;flex-wrap:wrap}
.upload-row input[type=file]{flex:1;min-width:220px;margin:0}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}

/* ---------- Auth (phone / OTP) ---------- */
.auth{min-height:100vh;display:flex;background:var(--bg)}
.auth-left{flex:1;background:linear-gradient(160deg,#efe8ff,#ffeaf3);display:flex;flex-direction:column;justify-content:space-between;padding:44px 50px;position:relative;overflow:hidden}
html.dark .auth-left{background:linear-gradient(160deg,#181231,#231627)}
.auth-right{width:480px;flex-shrink:0;display:flex;align-items:center;justify-content:center;padding:30px;background:var(--bg)}
.auth-logo{display:flex;align-items:center;gap:11px}
.auth-logo b{font-size:20px}
.auth-logo small{display:block;color:var(--muted);font-size:11.5px}
.auth-art{flex:1;display:flex;align-items:center;justify-content:center;animation:floaty 6s ease-in-out infinite}
.auth-art svg{width:min(320px,80%);height:auto}
.auth-bottom h3{font-size:23px;margin:0 0 6px;font-weight:800;background:linear-gradient(90deg,var(--p1),var(--p3));-webkit-background-clip:text;background-clip:text;color:transparent}
.auth-bottom p{margin:0;color:var(--muted)}
.authcard{width:100%;max-width:380px;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:34px 30px;box-shadow:var(--shadow);text-align:center;animation:fadeUp .45s ease both}
.authcard h1{font-size:20px;margin:2px 0 6px}
.authcard p.sub{color:var(--muted);margin:0 0 22px;font-size:13px}
.authcard p.sub b{color:var(--text)}
.phonerow{display:flex;gap:8px}
.phonerow .cc{width:70px;flex-shrink:0;text-align:center}
.otpboxes{display:flex;gap:8px;justify-content:center;margin:6px 0 16px}
.otpboxes input{width:44px;height:52px;text-align:center;font-size:20px;font-weight:800;margin:0;padding:0;border-radius:10px}
.otpboxes input:focus{border-color:var(--p1);box-shadow:0 0 0 3px rgba(108,92,231,.15)}
.otpboxes.shake{animation:shake .4s}
.timerow{display:flex;justify-content:center;align-items:center;gap:8px;color:var(--muted);font-size:12.5px;margin-bottom:18px}
.timerow b{color:var(--text)}
.resend{color:var(--p1);font-weight:700;cursor:pointer}
.resend.off{color:var(--muted);cursor:not-allowed;pointer-events:none}
.backlink{display:inline-flex;align-items:center;gap:5px;margin-top:16px;color:var(--muted);font-size:12.5px;cursor:pointer}
.backlink:hover{color:var(--text)}
.demo-note{margin-top:16px;text-align:left}

/* ---------- Loading overlay ---------- */
.loadov{position:fixed;inset:0;background:rgba(10,10,20,.55);display:none;align-items:center;justify-content:center;z-index:999;flex-direction:column;gap:14px;color:#fff}
.loadov.show{display:flex}
.spinner{width:46px;height:46px;border-radius:50%;border:4px solid rgba(255,255,255,.25);border-top-color:#fff;animation:spin .8s linear infinite}

/* ---------- Animations ---------- */
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}
@keyframes pulseLogo{0%,100%{box-shadow:0 4px 12px rgba(108,92,231,.35)}50%{box-shadow:0 4px 20px rgba(236,72,153,.5)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-8px)}40%{transform:translateX(8px)}60%{transform:translateX(-6px)}80%{transform:translateX(6px)}}

@media(max-width:1000px){.cards{grid-template-columns:1fr 1fr}.ip{grid-template-columns:1fr 1fr}.auth-right{width:420px}}
@media(max-width:860px){.auth-left{display:none}.auth-right{width:100%}}
@media(max-width:680px){
  .side{transform:translateX(-100%);transition:transform .25s;width:238px}
  .side.open{transform:translateX(0)}
  .main{margin-left:0}
  .hbtn{display:block}
  .searchform{display:none}
  .content{padding:16px 14px}
  .cards,.grid2,.ip{grid-template-columns:1fr}
  .hero-art{display:none}
  .docname{display:none}
}
"""

THEME_SCRIPT = """
(function(){ if(localStorage.getItem("curaverse-theme")==="dark") document.documentElement.classList.add("dark"); })();
function toggleTheme(){
  var d = document.documentElement.classList.toggle("dark");
  localStorage.setItem("curaverse-theme", d ? "dark" : "light");
}
function toggleSidebar(){ document.querySelector(".side").classList.toggle("open"); }
function showLoader(msg){
  var ov = document.getElementById("loadov");
  if(!ov) return;
  if(msg) document.getElementById("loadmsg").textContent = msg;
  ov.classList.add("show");
}
function countUp(){
  document.querySelectorAll(".num[data-count]").forEach(function(el){
    var target = parseFloat(el.getAttribute("data-count")) || 0;
    var isFloat = target % 1 !== 0;
    var start = 0, dur = 800, t0 = null;
    function step(ts){
      if(!t0) t0 = ts;
      var p = Math.min((ts - t0) / dur, 1);
      var val = start + (target - start) * (1 - Math.pow(1 - p, 3));
      el.textContent = isFloat ? val.toFixed(1) : Math.round(val);
      if(p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
}
document.addEventListener("DOMContentLoaded", countUp);
"""


def ribbon_illustration():
    """Small original line-art illustration (woman silhouette + ribbon + leaves)."""
    return """
    <svg viewBox="0 0 260 300" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="rg1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#ec4899"/><stop offset="1" stop-color="#a855f7"/>
        </linearGradient>
      </defs>
      <circle cx="130" cy="150" r="118" fill="url(#rg1)" opacity="0.08"/>
      <path d="M95 60c-5 20 0 38 14 48-16 8-26 24-26 46 0 34 22 70 22 96" fill="none" stroke="#6c5ce7" stroke-width="2.5" opacity="0.55"/>
      <circle cx="112" cy="70" r="26" fill="none" stroke="#6c5ce7" stroke-width="2.5" opacity="0.55"/>
      <path d="M60 200c22 18 40 30 40 30M60 230c26 8 46 4 46 4" fill="none" stroke="#22b573" stroke-width="2.5" opacity="0.5"/>
      <path d="M132 150c-2-22 -34-30 -34-52 0-14 12-22 24-16 10 5 12 18 10 30"
            fill="none" stroke="url(#rg1)" stroke-width="11" stroke-linecap="round"/>
      <path d="M148 150c2-22 34-30 34-52 0-14-12-22-24-16-10 5-12 18-10 30"
            fill="none" stroke="url(#rg1)" stroke-width="11" stroke-linecap="round"/>
      <path d="M140 150c0 20 -10 40 -2 56M140 150c0 20 10 40 2 56"
            fill="none" stroke="url(#rg1)" stroke-width="9" stroke-linecap="round"/>
    </svg>
    """


# ----------------------------------------------------------------------------
# Shell / layout
# ----------------------------------------------------------------------------

NAV = [
    ("home", "Dashboard", "/", False),
    ("users", "Patients", "/patients", False),
    ("user-plus", "Add Patient", "/add", False),
    ("file-text", "Medical History", "/medical-history", False),
    ("activity", "Symptoms", "/symptoms", False),
    ("clipboard", "Clinical Examination", "/clinical-examination", False),
    ("image", "Image Processing", "/image-processing", False),
    ("radio", "Ultrasound", "/ultrasound", True),
    ("file", "Reports", "/reports", False),
    ("calendar", "Appointments", "/appointments", False),
    ("bar-chart", "Analytics", "/analytics", False),
    ("settings", "Settings", "/settings", False),
    ("log-out", "Logout", "/logout", False),
]


def shell(body, active):
    nav_html = "".join(
        f'<a class="{"active" if label == active else ""}" href="{url}">'
        f'{icon(ic, 18)}<span>{label}</span>'
        f'{"<span class=\'badge-new\'>New</span>" if is_new else ""}</a>'
        for ic, label, url, is_new in NAV
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Curaverse</title><style>{CSS}</style></head>
    <body>
    <div class="side">
        <div class="brand">
            <span class="logo">C</span>
            <div><b>Curaverse</b><small>Breast Cancer Care</small></div>
        </div>
        <nav class="nav">{nav_html}</nav>
    </div>
    <main class="main">
        <header class="top">
            <div style="display:flex;align-items:center;gap:14px;flex:1">
                <button class="hbtn" onclick="toggleSidebar()">{icon('menu', 22)}</button>
                <form class="searchform" action="/patients" method="get">
                    <div class="search">{icon('search', 16)}<input name="q" placeholder="Search patients, reports, or anything..."></div>
                </form>
            </div>
            <div class="topright">
                <div class="tgl" onclick="toggleTheme()"><span class="knob">{icon('moon', 12)}</span>{icon('sun', 13)}</div>
                <div class="doc">
                    {avatar(DOCTOR_NAME, 38)}
                    <div class="docname"><b>{DOCTOR_NAME}</b><span>{DOCTOR_ROLE}</span></div>
                    {icon('chevron-down', 15)}
                </div>
            </div>
        </header>
        <section class="content">{body}</section>
    </main>
    <div class="loadov" id="loadov"><div class="spinner"></div><div id="loadmsg">Processing…</div></div>
    <script>{THEME_SCRIPT}</script>
    </body></html>
    """


def page(body, active):
    return render_template_string(shell(body, active))


# ----------------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------------

@app.before_request
def auth_gate():
    public = {"/login", "/verify", "/send-otp", "/verify-otp", "/resend-otp", "/logout"}
    if request.path in public or request.path.startswith("/uploads/"):
        return None
    if not session.get("ok"):
        return redirect("/login")
    return None


def auth_page(card_html, extra_script=""):
    return render_template_string(f"""<!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Curaverse</title><style>{CSS}</style></head>
    <body>
    <script>{THEME_SCRIPT}</script>
    <div class="auth">
      <div class="auth-left">
        <div class="auth-logo"><span class="logo">C</span><div><b>Curaverse</b><small>Your Health, Our Priority</small></div></div>
        <div class="auth-art">{ribbon_illustration()}</div>
        <div class="auth-bottom"><h3>Early Detection Saves Lives</h3><p>Together for a healthier tomorrow</p></div>
      </div>
      <div class="auth-right">{card_html}</div>
    </div>
    <script>{extra_script}</script>
    </body></html>""")


@app.get("/login")
def login():
    card = """
    <div class="authcard">
      <div class="logo" style="margin:0 auto 14px">C</div>
      <h1>Login to Curaverse</h1>
      <p class="sub">Enter your mobile number to receive a one-time password.</p>
      <form method="post" action="/send-otp">
        <label>Mobile Number</label>
        <div class="phonerow">
          <input class="cc" value="+91" disabled>
          <input name="phone" placeholder="98765 43210" maxlength="10" pattern="[0-9]{10}" inputmode="numeric" required>
        </div>
        <button class="btn" style="width:100%;justify-content:center;margin-top:6px">Send OTP</button>
      </form>
      <div class="note demo-note">""" + icon("alert", 14) + """<span>Demo mode — no SMS is actually sent. Any 10-digit number works.</span></div>
    </div>
    """
    return auth_page(card)


@app.post("/send-otp")
def send_otp():
    phone = "".join(ch for ch in request.form.get("phone", "") if ch.isdigit())[-10:]
    if not phone:
        return redirect("/login")
    session["phone"] = phone
    session["otp"] = "".join(random.choices("0123456789", k=6))
    return redirect("/verify")


@app.get("/verify")
def verify_page():
    phone = session.get("phone")
    if not phone:
        return redirect("/login")
    card = f"""
    <div class="authcard">
      <div class="logo" style="margin:0 auto 14px">C</div>
      <h1>Verify Your Phone Number</h1>
      <p class="sub">We have sent a 6-digit OTP to<br><b>+91 {phone[:5]} {phone[5:]}</b></p>
      <form method="post" action="/verify-otp" id="otpform">
        <input type="hidden" name="otp" id="otpFull">
        <div class="otpboxes" id="otpboxes">
          <input maxlength="1" inputmode="numeric" autofocus>
          <input maxlength="1" inputmode="numeric">
          <input maxlength="1" inputmode="numeric">
          <input maxlength="1" inputmode="numeric">
          <input maxlength="1" inputmode="numeric">
          <input maxlength="1" inputmode="numeric">
        </div>
        <div class="timerow">{icon('clock', 14)}<span id="timer">00:58</span>
          <span class="resend off" id="resendBtn" onclick="resendOtp()">Resend OTP</span>
        </div>
        <button class="btn" style="width:100%;justify-content:center" onclick="return fillOtp()">Verify OTP</button>
      </form>
      <a class="backlink" href="/login">{icon('arrow-left', 13)} Change Number</a>
      <div class="note demo-note">{icon('alert', 14)}<span>Demo OTP: <b>{DEMO_OTP}</b> also always works.</span></div>
    </div>
    """
    script = """
    var boxes = document.querySelectorAll('#otpboxes input');
    boxes.forEach(function(b, i){
      b.addEventListener('input', function(){
        b.value = b.value.replace(/[^0-9]/g,'');
        if(b.value && boxes[i+1]) boxes[i+1].focus();
      });
      b.addEventListener('keydown', function(e){
        if(e.key === 'Backspace' && !b.value && boxes[i-1]) boxes[i-1].focus();
      });
    });
    function fillOtp(){
      var code = Array.from(boxes).map(function(b){return b.value;}).join('');
      if(code.length !== 6){
        document.getElementById('otpboxes').classList.add('shake');
        setTimeout(function(){document.getElementById('otpboxes').classList.remove('shake');}, 400);
        return false;
      }
      document.getElementById('otpFull').value = code;
      showLoader('Verifying OTP…');
      return true;
    }
    var t = 58;
    var timerEl = document.getElementById('timer');
    var resendEl = document.getElementById('resendBtn');
    var iv = setInterval(function(){
      t--;
      if(t <= 0){
        clearInterval(iv);
        timerEl.textContent = '00:00';
        resendEl.classList.remove('off');
      } else {
        timerEl.textContent = '00:' + (t < 10 ? '0' + t : t);
      }
    }, 1000);
    function resendOtp(){
      if(resendEl.classList.contains('off')) return;
      fetch('/resend-otp', {method:'POST'}).then(function(){
        boxes.forEach(function(b){ b.value = ''; });
        boxes[0].focus();
        t = 58;
        resendEl.classList.add('off');
        timerEl.textContent = '00:58';
        iv = setInterval(function(){
          t--;
          if(t <= 0){ clearInterval(iv); timerEl.textContent='00:00'; resendEl.classList.remove('off'); }
          else { timerEl.textContent = '00:' + (t < 10 ? '0' + t : t); }
        }, 1000);
      });
    }
    """
    return auth_page(card, script)


@app.post("/resend-otp")
def resend_otp():
    if session.get("phone"):
        session["otp"] = "".join(random.choices("0123456789", k=6))
    return ("", 204)


@app.post("/verify-otp")
def verify_otp():
    entered = request.form.get("otp", "")
    if entered and (entered == session.get("otp") or entered == DEMO_OTP):
        session["ok"] = True
        return redirect("/")
    session["ok"] = False
    return redirect("/verify")


@app.get("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------

def status_pill(status):
    s = (status or "").lower()
    return f'<span class="pill {s}">{status or "—"}</span>'


@app.get("/")
def dashboard():
    conn = db()
    patients = conn.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    recent = conn.execute("SELECT * FROM patients ORDER BY id ASC LIMIT 4").fetchall()
    reports_count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    appts = conn.execute(
        """SELECT a.*, p.name FROM appointments a JOIN patients p ON p.id=a.patient_id
           ORDER BY a.id LIMIT 3"""
    ).fetchall()
    conn.close()

    pending_count = sum((p["status"] or "").lower() == "pending" for p in patients)

    recent_rows = "".join(
        f'''<div class="row"><div class="rleft">{avatar(p["name"], 34)}
            <div><div class="rname">{p["name"]}</div><div class="rsub">Age {p["age"] or "—"}</div></div></div>
            <div class="rleft"><span class="muted">{p["visit_date"] or ""}</span>{icon('chevron-right',15)}</div></div>'''
        for p in recent
    ) or '<p class="muted">No patient records yet.</p>'

    appt_rows = "".join(
        f'''<div class="row"><div class="rleft"><span class="rtime">{a["time"]}</span>
            <div><div class="rname">{a["name"]}</div><div class="rsub">{a["type"]}</div></div></div>
            {icon('chevron-right',15)}</div>'''
        for a in appts
    ) or '<p class="muted">No appointments scheduled.</p>'

    body = f"""
    <div class="head">
      <div><h1>Welcome back,<br>{DOCTOR_NAME}</h1></div>
      <a class="btn" href="/add">{icon('user-plus',15)} Add Patient</a>
    </div>
    <div class="hero">
      <div><div class="tag">Better Care &nbsp;•&nbsp; Better Tomorrow</div>
      <p>Curaverse helps you manage patient data, track medical history and use AI-powered analysis for early detection and better care.</p></div>
      <div class="hero-art">{ribbon_illustration()}</div>
    </div>
    <div class="cards">
      <div class="card"><div class="stat-top"><span class="stat-label">TOTAL PATIENTS</span><span class="stat-icon blue">{icon('users',16)}</span></div><div class="num" data-count="{len(patients)}">0</div><div class="trend">Patient records</div></div>
      <div class="card"><div class="stat-top"><span class="stat-label">TODAY'S APPOINTMENTS</span><span class="stat-icon purple">{icon('calendar',16)}</span></div><div class="num" data-count="{len(appts)}">0</div><div class="trend">Scheduled today</div></div>
      <div class="card"><div class="stat-top"><span class="stat-label">REPORTS GENERATED</span><span class="stat-icon green">{icon('file',16)}</span></div><div class="num" data-count="{reports_count}">0</div><div class="trend">Clinical reports</div></div>
      <div class="card"><div class="stat-top"><span class="stat-label">PENDING FOLLOW-UPS</span><span class="stat-icon orange">{icon('clock',16)}</span></div><div class="num" data-count="{pending_count}">0</div><div class="trend">Needs attention</div></div>
    </div>
    <div class="grid2">
      <div class="panel"><h2>Recent Patients</h2>{recent_rows}<a class="muted" href="/patients" style="display:block;text-align:center;margin-top:10px">View All</a></div>
      <div class="panel"><h2>Upcoming Appointments</h2>{appt_rows}<a class="muted" href="/appointments" style="display:block;text-align:center;margin-top:10px">View All</a></div>
    </div>
    <footer class="appfoot"><span>© 2025 Curaverse. All rights reserved.</span><span>Caring Today • Healthier Tomorrow</span></footer>
    """
    return page(body, "Dashboard")


# ----------------------------------------------------------------------------
# Patients
# ----------------------------------------------------------------------------

@app.get("/patients")
def patients_view():
    q = request.args.get("q", "").strip()
    conn = db()
    if q:
        like = f"%{q}%"
        items = conn.execute(
            "SELECT * FROM patients WHERE name LIKE ? OR patient_id LIKE ? ORDER BY id DESC",
            (like, like),
        ).fetchall()
    else:
        items = conn.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    conn.close()
    rows = "".join(
        f'''<tr><td>{p["patient_id"]}</td><td style="display:flex;align-items:center;gap:8px">{avatar(p["name"],26)}{p["name"]}</td>
            <td>{p["age"] or "—"}</td><td>{p["gender"] or "—"}</td><td>{p["diagnosis"] or "—"}</td>
            <td>{status_pill(p["status"])}</td>
            <td><a class="btn secondary" href="/patient/{p["id"]}">Open</a></td></tr>'''
        for p in items
    ) or '<tr><td colspan="7" class="muted">No matching patients.</td></tr>'
    body = f"""
    <div class="head">
      <div><h1>Patients</h1><span class="muted">Dashboard › Patients</span></div>
      <a class="btn" href="/add">{icon('user-plus',15)} Add Patient</a>
    </div>
    <div class="panel">
      <form method="get"><div class="search" style="max-width:340px">{icon('search',16)}<input name="q" value="{escape(q)}" placeholder="Search patient name or ID"></div></form>
    </div>
    <div class="panel">
      <table><tr><th>ID</th><th>Name</th><th>Age</th><th>Gender</th><th>Case</th><th>Status</th><th></th></tr>{rows}</table>
    </div>
    """
    return page(body, "Patients")


@app.route("/add", methods=["GET", "POST"])
def add_patient():
    if request.method == "POST":
        f = request.form
        patient_id = "CV-" + str(uuid.uuid4().int % 9000 + 1000)
        conn = db()
        conn.execute(
            """INSERT INTO patients
               (patient_id,name,age,gender,phone,symptoms,medical_history,
                family_history,examination,status,diagnosis,visit_date)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,date('now'))""",
            (
                patient_id, f.get("name", ""), f.get("age") or None,
                f.get("gender", ""), f.get("phone", ""), f.get("symptoms", ""),
                f.get("medical_history", ""), f.get("family_history", ""),
                f.get("examination", ""), f.get("status", "Active"),
                f.get("diagnosis", ""),
            ),
        )
        conn.commit()
        pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return redirect(f"/patient/{pid}")

    body = f"""
    <div class="head"><div><h1>Add Patient</h1><span class="muted">Dashboard › Add Patient</span></div></div>
    <form method="post">
      <div class="panel">
        <h2>Patient Information</h2>
        <div class="grid2">
          <div><label>Full Name</label><input name="name" required></div>
          <div><label>Age</label><input name="age" type="number" min="0"></div>
          <div><label>Gender</label><select name="gender"><option>Female</option><option>Male</option><option>Other</option></select></div>
          <div><label>Phone</label><input name="phone"></div>
          <div><label>Case / Diagnosis</label><input name="diagnosis"></div>
          <div><label>Status</label><select name="status"><option>Active</option><option>Pending</option><option>Completed</option></select></div>
        </div>
      </div>
      <div class="panel">
        <h2>Case Details</h2>
        <div class="grid2">
          <div><label>Symptoms</label><textarea name="symptoms"></textarea></div>
          <div><label>Medical History</label><textarea name="medical_history"></textarea></div>
          <div><label>Family History</label><textarea name="family_history"></textarea></div>
          <div><label>Clinical Examination</label><textarea name="examination"></textarea></div>
        </div>
        <button class="btn">{icon('check',15)} Save Patient</button>
      </div>
    </form>
    """
    return page(body, "Add Patient")


@app.get("/patient/<int:pid>")
def patient_details(pid):
    conn = db()
    p = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    conn.close()
    if not p:
        return redirect("/patients")

    sections = "".join(
        f'<div class="panel"><h2>{label}</h2><p class="muted">{p[key] or "No record"}</p></div>'
        for label, key in [
            ("Medical History", "medical_history"),
            ("Family History", "family_history"),
            ("Symptoms", "symptoms"),
            ("Clinical Examination", "examination"),
        ]
    )

    body = f"""
    <div class="head">
      <div><h1>Patient Details</h1><span class="muted">Dashboard › Patients › {p["patient_id"]}</span></div>
      <div style="display:flex;gap:8px">
        <a class="btn secondary" href="/image-processing/{pid}">{icon('image',15)} Image Processing</a>
        <a class="btn" href="/ultrasound/{pid}">{icon('radio',15)} Ultrasound</a>
      </div>
    </div>
    <div class="panel"><div class="grid2">
      <p><b>Patient ID</b><br><span class="muted">{p["patient_id"]}</span></p>
      <p><b>Name</b><br><span class="muted">{p["name"]}</span></p>
      <p><b>Age / Gender</b><br><span class="muted">{p["age"] or "—"} / {p["gender"] or "—"}</span></p>
      <p><b>Phone</b><br><span class="muted">{p["phone"] or "—"}</span></p>
    </div></div>
    <div class="grid2">{sections}</div>
    """
    return page(body, "Patients")


def simple(title, key, active):
    conn = db()
    patients = conn.execute("SELECT * FROM patients ORDER BY id").fetchall()
    p = patients[0] if patients else None
    value = (p[key] or "") if p else ""
    conn.close()
    body = f"""
    <div class="head"><div><h1>{title}</h1><span class="muted">Dashboard › {title}</span></div></div>
    <div class="panel">
      <h2>{p["name"] if p else "Patient"}</h2>
      <textarea rows="6" placeholder="Enter {title} notes...">{value}</textarea>
      <button class="btn">{icon('check',15)} Save {title}</button>
    </div>
    """
    return page(body, active)


@app.get("/medical-history")
def medical_history():
    return simple("Medical History", "medical_history", "Medical History")


@app.get("/symptoms")
def symptoms():
    return simple("Symptoms", "symptoms", "Symptoms")


@app.get("/clinical-examination")
def clinical_examination():
    return simple("Clinical Examination", "examination", "Clinical Examination")


# ----------------------------------------------------------------------------
# Image processing pipeline (shared by mammogram + ultrasound)
# ----------------------------------------------------------------------------

MAX_DIM = 900

MODE_CONFIG = {
    "mammogram": {
        "labels": ["Low Density", "Medium Density", "High Density"],
        "colors": [(85, 18, 130), (55, 183, 122), (242, 198, 38)],
        "denoise": 3,
    },
    "ultrasound": {
        "labels": ["Hypoechoic (dark)", "Isoechoic (mid)", "Hyperechoic (bright)"],
        "colors": [(30, 60, 120), (34, 181, 155), (235, 210, 60)],
        "denoise": 5,  # heavier speckle-noise reduction typical for ultrasound
    },
}


def run_pipeline(path, name, mode="mammogram"):
    cfg = MODE_CONFIG[mode]
    original = Image.open(path)
    original = ImageOps.exif_transpose(original)
    if max(original.size) > MAX_DIM:
        original.thumbnail((MAX_DIM, MAX_DIM))
    original.convert("RGB").save(path)

    gray = original.convert("L")
    pre = ImageOps.autocontrast(gray)
    pre = ImageEnhance.Contrast(pre).enhance(1.15)
    pre = pre.filter(ImageFilter.MedianFilter(cfg["denoise"]))

    pre_name = "pre_" + name
    pre.save(UPLOADS / pre_name)

    arr = np.asarray(pre, dtype=np.float32)
    flat = arr.reshape(-1, 1)

    labels = None
    inertia = 0.0
    iterations = 1

    if KMeans is not None:
        try:
            km = KMeans(n_clusters=3, n_init=10, random_state=42)
            km.fit(flat)
            labels = km.labels_
            inertia = float(km.inertia_)
            iterations = int(km.n_iter_)
            centers = km.cluster_centers_.ravel()
            order = np.argsort(centers)
            remap = np.empty_like(order)
            remap[order] = np.arange(3)
            labels = remap[labels]
        except Exception:
            labels = None

    if labels is None:
        labels = np.digitize(flat.ravel(), [85, 170])
        inertia = 0.0
        iterations = 1

    mask = labels.reshape(arr.shape)
    seg = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    stats = []
    total = mask.size
    for i, color in enumerate(cfg["colors"]):
        sel = mask == i
        seg[sel] = color
        vals = arr[sel]
        pct = 100.0 * sel.sum() / total if total else 0.0
        stats.append({
            "label": cfg["labels"][i],
            "pct": round(pct, 1),
            "mean": round(float(vals.mean()), 1) if vals.size else 0.0,
            "min": int(vals.min()) if vals.size else 0,
            "max": int(vals.max()) if vals.size else 0,
            "color": "rgb(%d,%d,%d)" % color,
        })

    seg_name = "seg_" + name
    Image.fromarray(seg).save(UPLOADS / seg_name)

    return pre_name, seg_name, inertia, iterations, stats


def render_pipeline_ui(title, subtitle, patients, patient, last, upload_url, back_url, mode, error_html=""):
    import json as _json

    original_html = (
        f'<img src="/uploads/{last["original"]}">' if last
        else f'<div class="ph">{icon("upload",26)}<br>Upload Image</div>'
    )
    pre_html = (
        f'<img src="/uploads/{last["preprocessed"]}">' if last
        else '<div class="ph">Waiting</div>'
    )
    seg_html = (
        f'<img src="/uploads/{last["segmented"]}">' if last
        else '<div class="ph">Waiting</div>'
    )

    stats = _json.loads(last["stats"]) if (last and last["stats"]) else None
    cfg = MODE_CONFIG[mode]

    if stats:
        conic_parts = []
        acc = 0
        for s in stats:
            start = acc
            acc += s["pct"]
            conic_parts.append(f'{s["color"]} {start}% {acc}%')
        pie_style = "background:conic-gradient(" + ",".join(conic_parts) + ")"
        legend = "".join(
            f'<span><span class="sw" style="background:{s["color"]}"></span>{s["label"]}<br><b>{s["pct"]}%</b></span>'
            for s in stats
        )
        stat_rows = "".join(
            f'<tr><td><span class="dot" style="background:{s["color"]}"></span>{s["label"]}</td>'
            f'<td>{s["mean"]}</td><td>{s["min"]}</td><td>{s["max"]}</td></tr>'
            for s in stats
        )
        inertia_val = f'{last["inertia"]:.2f}'
        iterations_val = last["iterations"]
        analysis_note = ""
        if mode == "ultrasound":
            brightest = max(stats, key=lambda s: s["mean"])
            analysis_note = (
                f'<div class="note warn" style="margin-top:12px">{icon("alert",14)}'
                f'<span><b>{brightest["label"]}</b> covers {brightest["pct"]}% of the scan area. '
                "This is an automated visual segmentation only — always confirm with a radiologist.</span></div>"
            )
    else:
        pie_style = "background:var(--line)"
        legend = '<span class="muted">Run an analysis to see the cluster breakdown.</span>'
        stat_rows = '<tr><td colspan="4" class="muted">No data yet</td></tr>'
        inertia_val = "—"
        iterations_val = "—"
        analysis_note = ""

    step_done = "done" if last else ""
    patient_options = "".join(
        f'<option value="{p["id"]}" {"selected" if patient and p["id"] == patient["id"] else ""}>{p["patient_id"]} — {p["name"]}</option>'
        for p in patients
    )

    body = f"""
    <div class="head">
      <div><h1>{title}</h1><span class="muted">Dashboard › {title}</span></div>
    </div>
    {error_html}
    <div class="panel" style="padding:0">
      <div class="steps">
        <div class="step done"><div class="circle">{icon('check',16) if True else '1'}</div><div><b>Select Patient</b><small>{patient["name"] if patient else "—"}</small></div></div>
        <div class="step {step_done}"><div class="circle">2</div><div><b>Upload Image</b><small>{subtitle}</small></div></div>
        <div class="step {step_done}"><div class="circle">3</div><div><b>Preprocessing</b><small>Denoise, Normalize, Gray</small></div></div>
        <div class="step {step_done}"><div class="circle">4</div><div><b>K-Means Segmentation</b><small>K = 3 clusters</small></div></div>
      </div>

      <form method="get" style="padding:0 15px 15px" onchange="this.submit()">
        <label>Patient</label>
        <select name="__unused" onchange="window.location='{back_url}/'+this.value" style="display:none"></select>
        <select onchange="window.location='{back_url}/'+this.value">{patient_options}</select>
      </form>

      <div class="ip">
        <div class="box"><h3>Original Image</h3>{original_html}<div class="meta">{patient["name"] if patient else "-"}</div></div>
        <div class="box"><h3>Preprocessed</h3>{pre_html}<div class="meta">Grayscale • Denoised</div></div>
        <div class="box"><h3>K-Means Segmentation</h3>{seg_html}<div class="meta">{cfg["labels"][0]} • {cfg["labels"][1]} • {cfg["labels"][2]}</div></div>
        <div class="box" style="text-align:left">
          <h3 style="text-align:center">Analysis Summary</h3>
          <p class="muted" style="margin:6px 0">Algorithm<br><b style="color:var(--text)">K-Means Clustering</b></p><hr style="border-color:var(--line)">
          <p class="muted" style="margin:6px 0">Clusters (K)<br><b style="color:var(--text)">3</b></p>
          <p class="muted" style="margin:6px 0">Inertia<br><b style="color:var(--text)">{inertia_val}</b></p>
          <p class="muted" style="margin:6px 0">Iterations<br><b style="color:var(--text)">{iterations_val}</b></p>
          <div class="note">{icon('alert',14)}<span>Prototype analysis — clinical decision remains with clinician.</span></div>
        </div>
      </div>
      {analysis_note}

      <div class="grid2" style="padding:0 15px 15px;margin-top:0">
        <div class="panel" style="margin-top:0">
          <h3 style="margin:0 0 4px">Cluster Distribution</h3>
          <div class="pie" style="{pie_style}"></div>
          <div class="legend">{legend}</div>
        </div>
        <div class="panel" style="margin-top:0">
          <h3 style="margin:0 0 4px">Cluster Statistics</h3>
          <table><tr><th>Cluster</th><th>Mean</th><th>Min</th><th>Max</th></tr>{stat_rows}</table>
        </div>
      </div>

      <form method="post" action="{upload_url}" enctype="multipart/form-data" class="upload-row" onsubmit="showLoader('Running {title.lower()} pipeline…')">
        <input type="file" name="image" accept=".png,.jpg,.jpeg" required>
        <button class="btn">{icon('upload',15)} Run {title}</button>
      </form>
    </div>
    """
    return body


@app.route("/image-processing", defaults={"pid": None}, methods=["GET", "POST"])
@app.route("/image-processing/<int:pid>", methods=["GET", "POST"])
def image_processing(pid):
    conn = db()
    patients = conn.execute("SELECT * FROM patients ORDER BY id").fetchall()
    patient = next((x for x in patients if x["id"] == pid), patients[0] if patients else None)

    error_html = ""
    if request.method == "POST" and patient:
        file = request.files.get("image")
        if file and file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg"}:
                session["ip_error"] = f"'{file.filename}' is not a .png/.jpg/.jpeg file."
                return redirect(request.path)
            name = "img_" + uuid.uuid4().hex + ".png"
            save_path = UPLOADS / name
            file.save(save_path)
            try:
                import json as _json
                pre_name, seg_name, inertia, iterations, stats = run_pipeline(save_path, name, "mammogram")
            except Exception as e:
                save_path.unlink(missing_ok=True)
                conn.close()
                session["ip_error"] = f"Could not process this image: {e}"
                return redirect(request.path)
            conn.execute(
                """INSERT INTO medical_images
                   (patient_id,original,preprocessed,segmented,algorithm,inertia,iterations,stats)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (patient["id"], name, pre_name, seg_name, "K-Means Clustering",
                 inertia, iterations, _json.dumps(stats)),
            )
            conn.commit()
            conn.close()
            return redirect(f"/image-processing/{patient['id']}")

    last = None
    if patient:
        last = conn.execute(
            "SELECT * FROM medical_images WHERE patient_id=? ORDER BY id DESC LIMIT 1",
            (patient["id"],),
        ).fetchone()
    conn.close()

    ip_error = session.pop("ip_error", None)
    if ip_error:
        error_html = f'<div class="note err" style="margin-bottom:15px">{icon("alert",14)}<span>{escape(ip_error)}</span></div>'

    body = render_pipeline_ui(
        "Image Processing", "Mammogram", patients, patient, last,
        f"/image-processing/{patient['id']}" if patient else "/image-processing",
        "/image-processing", "mammogram", error_html,
    )
    return page(body, "Image Processing")


@app.route("/ultrasound", defaults={"pid": None}, methods=["GET", "POST"])
@app.route("/ultrasound/<int:pid>", methods=["GET", "POST"])
def ultrasound(pid):
    conn = db()
    patients = conn.execute("SELECT * FROM patients ORDER BY id").fetchall()
    patient = next((x for x in patients if x["id"] == pid), patients[0] if patients else None)

    error_html = ""
    if request.method == "POST" and patient:
        file = request.files.get("image")
        if file and file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg"}:
                session["us_error"] = f"'{file.filename}' is not a .png/.jpg/.jpeg file."
                return redirect(request.path)
            name = "us_" + uuid.uuid4().hex + ".png"
            save_path = UPLOADS / name
            file.save(save_path)
            try:
                import json as _json
                pre_name, seg_name, inertia, iterations, stats = run_pipeline(save_path, name, "ultrasound")
            except Exception as e:
                save_path.unlink(missing_ok=True)
                conn.close()
                session["us_error"] = f"Could not process this image: {e}"
                return redirect(request.path)
            conn.execute(
                """INSERT INTO ultrasound_images
                   (patient_id,original,preprocessed,segmented,algorithm,inertia,iterations,stats)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (patient["id"], name, pre_name, seg_name, "K-Means Clustering",
                 inertia, iterations, _json.dumps(stats)),
            )
            conn.commit()
            conn.close()
            return redirect(f"/ultrasound/{patient['id']}")

    last = None
    if patient:
        last = conn.execute(
            "SELECT * FROM ultrasound_images WHERE patient_id=? ORDER BY id DESC LIMIT 1",
            (patient["id"],),
        ).fetchone()
    conn.close()

    us_error = session.pop("us_error", None)
    if us_error:
        error_html = f'<div class="note err" style="margin-bottom:15px">{icon("alert",14)}<span>{escape(us_error)}</span></div>'

    body = render_pipeline_ui(
        "Ultrasound", "Breast Ultrasound Scan", patients, patient, last,
        f"/ultrasound/{patient['id']}" if patient else "/ultrasound",
        "/ultrasound", "ultrasound", error_html,
    )
    return page(body, "Ultrasound")


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------

@app.route("/reports", methods=["GET", "POST"])
def reports():
    conn = db()
    patients = conn.execute("SELECT * FROM patients ORDER BY id").fetchall()

    if request.method == "POST" and patients:
        f = request.form
        conn.execute(
            "INSERT INTO reports(patient_id,title,findings,recommendation) VALUES(?,?,?,?)",
            (f.get("patient_id") or patients[0]["id"], f.get("title", "Clinical Report"),
             f.get("findings", ""), f.get("recommendation", "")),
        )
        conn.commit()

    rows = conn.execute(
        """SELECT r.*, p.name, p.patient_id FROM reports r
           JOIN patients p ON p.id=r.patient_id ORDER BY r.id DESC"""
    ).fetchall()
    conn.close()

    patient_options = "".join(
        f'<option value="{p["id"]}">{p["patient_id"]} — {p["name"]}</option>' for p in patients
    )
    report_rows = "".join(
        f'<tr><td>{r["patient_id"]}</td><td>{r["name"]}</td><td>{r["title"]}</td><td>{r["findings"] or "—"}</td></tr>'
        for r in rows
    ) or '<tr><td colspan="4" class="muted">No reports yet.</td></tr>'

    body = f"""
    <div class="head"><div><h1>Reports</h1><span class="muted">Dashboard › Reports</span></div></div>
    <div class="panel">
      <h2>Create Report</h2>
      <form method="post">
        <label>Patient</label><select name="patient_id">{patient_options}</select>
        <label>Report Title</label><input name="title" value="Clinical Report">
        <label>Findings</label><textarea name="findings"></textarea>
        <label>Recommendation</label><textarea name="recommendation"></textarea>
        <button class="btn">{icon('check',15)} Save Report</button>
      </form>
    </div>
    <div class="panel"><h2>Saved Reports</h2>
      <table><tr><th>ID</th><th>Patient</th><th>Title</th><th>Findings</th></tr>{report_rows}</table>
    </div>
    """
    return page(body, "Reports")


# ----------------------------------------------------------------------------
# Appointments
# ----------------------------------------------------------------------------

@app.route("/appointments", methods=["GET", "POST"])
def appointments():
    conn = db()
    patients = conn.execute("SELECT * FROM patients ORDER BY id").fetchall()

    if request.method == "POST" and patients:
        f = request.form
        conn.execute(
            "INSERT INTO appointments(patient_id,date,time,doctor,type,status) VALUES(?,?,?,?,?,?)",
            (f.get("patient_id") or patients[0]["id"], f.get("date", ""),
             f.get("time", ""), f.get("doctor", "Dr. " + DOCTOR_NAME),
             f.get("type", "Consultation"), f.get("status", "Scheduled")),
        )
        conn.commit()

    rows = conn.execute(
        """SELECT a.*, p.name, p.patient_id FROM appointments a
           JOIN patients p ON p.id=a.patient_id ORDER BY a.id DESC"""
    ).fetchall()
    conn.close()

    options = "".join(
        f'<option value="{p["id"]}">{p["patient_id"]} — {p["name"]}</option>' for p in patients
    )
    table_rows = "".join(
        f'<tr><td>{a["patient_id"]}</td><td>{a["name"]}</td><td>{a["date"] or "—"}</td>'
        f'<td>{a["time"] or "—"}</td><td>{a["doctor"] or "—"}</td><td>{status_pill(a["status"])}</td></tr>'
        for a in rows
    ) or '<tr><td colspan="6" class="muted">No appointments yet.</td></tr>'

    body = f"""
    <div class="head"><div><h1>Appointments</h1><span class="muted">Dashboard › Appointments</span></div></div>
    <div class="panel">
      <h2>Schedule Appointment</h2>
      <form method="post">
        <label>Patient</label><select name="patient_id">{options}</select>
        <div class="grid2">
          <div><label>Date</label><input name="date" type="date"></div>
          <div><label>Time</label><input name="time" type="time"></div>
          <div><label>Doctor</label><input name="doctor" value="Dr. {DOCTOR_NAME}"></div>
          <div><label>Type</label><input name="type" value="Consultation"></div>
        </div>
        <label>Status</label><select name="status"><option>Scheduled</option><option>Completed</option><option>Cancelled</option></select>
        <button class="btn">{icon('check',15)} Save Appointment</button>
      </form>
    </div>
    <div class="panel"><h2>Appointments</h2>
      <table><tr><th>ID</th><th>Patient</th><th>Date</th><th>Time</th><th>Doctor</th><th>Status</th></tr>{table_rows}</table>
    </div>
    """
    return page(body, "Appointments")


# ----------------------------------------------------------------------------
# Analytics / Settings
# ----------------------------------------------------------------------------

@app.get("/analytics")
def analytics():
    conn = db()
    patients_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    images_count = conn.execute("SELECT COUNT(*) FROM medical_images").fetchone()[0]
    us_count = conn.execute("SELECT COUNT(*) FROM ultrasound_images").fetchone()[0]
    reports_count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    appointments_count = conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
    conn.close()

    body = f"""
    <div class="head"><div><h1>Analytics</h1><span class="muted">Dashboard › Analytics</span></div></div>
    <div class="cards">
      <div class="card"><div class="stat-top"><span class="stat-label">PATIENTS</span><span class="stat-icon blue">{icon('users',16)}</span></div><div class="num" data-count="{patients_count}">0</div></div>
      <div class="card"><div class="stat-top"><span class="stat-label">MAMMOGRAMS PROCESSED</span><span class="stat-icon purple">{icon('image',16)}</span></div><div class="num" data-count="{images_count}">0</div></div>
      <div class="card"><div class="stat-top"><span class="stat-label">ULTRASOUNDS PROCESSED</span><span class="stat-icon green">{icon('radio',16)}</span></div><div class="num" data-count="{us_count}">0</div></div>
      <div class="card"><div class="stat-top"><span class="stat-label">APPOINTMENTS</span><span class="stat-icon orange">{icon('calendar',16)}</span></div><div class="num" data-count="{appointments_count}">0</div></div>
    </div>
    <div class="panel" style="margin-top:16px">
      <h2>Curaverse Imaging Pipeline</h2>
      <p class="muted">Preprocessing → K-Means segmentation (K=3) → visual cluster analysis, shared by both the Mammogram (Image Processing) and Ultrasound modules.</p>
      <div class="note">{icon('alert',14)}<span>This is a prototype/demo workflow and is not a medical diagnosis. Reports generated: {reports_count}.</span></div>
    </div>
    """
    return page(body, "Analytics")


@app.get("/settings")
def settings():
    body = f"""
    <div class="head"><div><h1>Settings</h1><span class="muted">Dashboard › Settings</span></div></div>
    <div class="grid2">
      <div class="panel">
        <h2>Profile</h2>
        <label>Name</label><input value="{DOCTOR_NAME}">
        <label>Role</label><input value="{DOCTOR_ROLE}">
        <button class="btn">{icon('check',15)} Save Profile</button>
      </div>
      <div class="panel">
        <h2>Application</h2>
        <p>Theme: <b>Curaverse Indigo + Pink</b> (light / dark toggle in the top bar)</p>
        <p>Authentication: <b>Mobile OTP Login</b></p>
        <p>Image Processing: <b>K-Means, K = 3</b> (Mammogram &amp; Ultrasound)</p>
      </div>
    </div>
    """
    return page(body, "Settings")


@app.get("/uploads/<path:name>")
def uploads(name):
    return send_from_directory(UPLOADS, name)


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
