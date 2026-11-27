"""
score_analysis.py - secilen oran bandindaki maclarin SKOR ve IY/MS dagilimi.

upset_finder.py'deki select() ciktisi (sel) ile calisir. Arsivden:
  - her skorun (1-0, 2-0, 2-1, 1-2 ...) ve IY/MS sonucunun (1/2, 2/1, X/X ...)
    kac kez gerceklestigini,
  - Wilson %95 guven araligini,
  - buradan "adil oran" (1 / frekans) ve adil oran araligini hesaplar.
Bahis sitesinin verdigi oranlari girerseniz beklenen degeri (EV) de gosterir.

Streamlit uygulamanizda:
    from score_analysis import render_streamlit
    render_streamlit(sel)          # sel = bandina gore filtrelenmis DataFrame

Gerekli kolonlar: hg, ag, ftr, htr  (upset_finder.prepare() bunlari uretir;
kendi prepare()'inizi degistirdiyseniz 'htr': df["HTR"] satirini ekleyin).

NOT: football-data.co.uk'de dogru skor ve IY/MS ORANLARI yok; arsivden sadece
gerceklesme sikligi gelir. Bahis oranini siz girersiniz.
"""
import math

import numpy as np
import pandas as pd

HTFT_ORDER = ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"]
FOCUS_SCORES = ["1-0", "2-0", "2-1", "1-2", "0-0", "1-1"]
FOCUS_HTFT = ["1/2", "2/1", "X/X"]
MIN_HITS = 10  # bunun altindaki sonuclar "az veri" olarak isaretlenir


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def add_labels(sel):
    """'score' (ev-dep, orn. 2-1) ve 'htft' (orn. 2/1) kolonlarini ekler."""
    s = sel.copy()
    s["score"] = s["hg"].astype(int).astype(str) + "-" + s["ag"].astype(int).astype(str)
    code = {"H": "1", "D": "X", "A": "2"}
    h = s["htr"].map(code) if "htr" in s.columns else pd.Series(np.nan, index=s.index)
    f = s["ftr"].map(code)
    s["htft"] = np.where(h.notna() & f.notna(), h.astype(str) + "/" + f.astype(str), None)
    return s


def freq_table(s, col, order=None):
    """Her sonucun adedi, frekansi, guven araligi ve adil orani."""
    d = s.dropna(subset=[col])
    n = len(d)
    if n == 0:
        return pd.DataFrame()
    counts = d[col].value_counts()
    if order:
        counts = counts.reindex(order, fill_value=0)
    rows = []
    for key, c in counts.items():
        c = int(c)
        p = c / n if n else float("nan")
        lo, hi = wilson(c, n)
        rows.append(
            {
                "sonuc": key,
                "adet": c,
                "oran_%": p * 100,
                "p_alt_%": lo * 100,
                "p_ust_%": hi * 100,
                "adil_oran": (1 / p) if c else np.nan,
                "adil_min": (1 / hi) if hi and hi > 0 else np.nan,
                "adil_max": (1 / lo) if lo and lo > 0 else np.nan,
                "not": "az veri" if c < MIN_HITS else "",
            }
        )
    return pd.DataFrame(rows)


def add_value(tbl, odds_col="bahis_orani"):
    """Girilen bahis oranlarina gore beklenen deger (EV) hesaplar.

    EV_%          : frekans * oran - 1
    EV_temkinli_% : ayni hesap ama Wilson ALT siniri ile (kotumser senaryo)
    """
    t = tbl.copy()
    odd = pd.to_numeric(t[odds_col], errors="coerce")
    t["EV_%"] = (t["oran_%"] / 100 * odd - 1) * 100
    t["EV_temkinli_%"] = (t["p_alt_%"] / 100 * odd - 1) * 100
    return t


def render_streamlit(sel, key="scores"):
    """Streamlit'te skor + IY/MS sekmelerini cizer."""
    import streamlit as st

    s = add_labels(sel)
    n = len(s)
    st.subheader("Skor ve İY/MS dağılımı (arşiv)")
    st.caption(f"Seçilen oran bandındaki maç sayısı: {n}")
    if n == 0:
        st.info("Bu bantta maç yok.")
        return
    if n < 100:
        st.warning("Örneklem küçük (n<100): skor bazında adetler çok düşük kalır, "
                   "sonuçlar oynak olur. Sezon/lig ekleyin ya da bandı genişletin.")

    tab_score, tab_htft = st.tabs(["Skor (ev-dep)", "İY/MS"])

    for tab, col, order, focus, label in (
        (tab_score, "score", None, FOCUS_SCORES, "Skor"),
        (tab_htft, "htft", HTFT_ORDER, FOCUS_HTFT, "İY/MS"),
    ):
        with tab:
            tbl = freq_table(s, col, order)
            if tbl.empty:
                st.info("Bu sütun için veri yok (IY sonucu eksik olabilir).")
                continue
            only_focus = st.checkbox(
                "Sadece seçtiklerim: " + ", ".join(focus),
                value=False, key=f"{key}_{col}_focus",
            )
            show = tbl[tbl["sonuc"].isin(focus)] if only_focus else tbl
            show = show.assign(bahis_orani=np.nan)
            cols = ["sonuc", "adet", "oran_%", "p_alt_%", "p_ust_%",
                    "adil_oran", "adil_min", "adil_max", "not", "bahis_orani"]
            st.caption("Sağdaki 'bahis_orani' sütununa sitenin verdiği oranı yazın.")
            edited = st.data_editor(
                show[cols].round(2),
                disabled=[c for c in cols if c != "bahis_orani"],
                column_config={
                    "sonuc": label,
                    "oran_%": "Arşiv sıklığı %",
                    "p_alt_%": "Alt sınır %",
                    "p_ust_%": "Üst sınır %",
                    "adil_oran": "Adil oran",
                    "adil_min": "Adil min",
                    "adil_max": "Adil max",
                    "bahis_orani": st.column_config.NumberColumn(
                        "Bahis oranı", min_value=1.01, step=0.05),
                },
                hide_index=True,
                key=f"{key}_{col}_editor",
            )
            priced = edited.dropna(subset=["bahis_orani"])
            if len(priced):
                res = add_value(priced)[
                    ["sonuc", "adet", "adil_oran", "bahis_orani", "EV_%", "EV_temkinli_%", "not"]
                ].round(2)
                st.dataframe(res, hide_index=True)
                st.caption(
                    "EV_% > 0 ise arşiv frekansına göre oran 'adil'den yüksek. "
                    "Temkinli sütun güven aralığının alt sınırını kullanır; "
                    "pozitif çıkması için yüksek adet gerekir. Geçmiş sıklık "
                    "gelecek maçın garantisi değildir."
                )
