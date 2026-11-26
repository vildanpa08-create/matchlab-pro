#!/usr/bin/env python3
"""
upset_finder.py - "Bu oran yapisi daha once verilmis mi, ne olmus?"

football-data.co.uk'den (ulusal ligler) mac sonuclarini ve oranlarini indirir,
verdiginiz oran bandina uyan maclari bulur ve favorinin kac kez kazandigini /
berabere kaldigini / kaybettigini, piyasanin ima ettigi olasiliklarla
karsilastirarak gosterir. Surpriz maclarda sut / kirmizi kart farklarini da cikarir.

NOT: Veride Unibet yok. Varsayilan olarak Pinnacle kapanis (PSC), yoksa ortalama
kapanis (AvgC), yoksa Bet365 kapanis oranlari kullanilir. Kapanis oranlari
piyasanin en verimli tahmini sayildigi icin karsilastirma icin uygundur.
Sampiyonlar Ligi bu kaynakta yok; sadece ulusal ligler.

Kullanim:
  # Lille-Bayern benzeri: deplasman favorisi 1.25-1.45, ev sahibi 5.0-9.0
  python upset_finder.py

  # Birebir oran benzerligi (ev, beraberlik, deplasman), her biri +-%12
  python upset_finder.py --like 6.0 5.75 1.38 --tol 0.12

  # Baska ligler / sezonlar (E0=EPL, D1=Bundesliga, SP1=La Liga, I1=Serie A,
  # F1=Ligue 1, T1=Super Lig, N1=Hollanda, P1=Portekiz, B1=Belcika)
  python upset_finder.py --leagues E0 D1 SP1 --seasons 2223 2324 2425 2526

  # Internet kisitliysa CSV'leri elle indirip klasore koyun (E0_2526.csv gibi)
  python upset_finder.py --local-dir ./csv

Gereksinim: pip install pandas
"""
import argparse
import math
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
CACHE_DIR = Path("fd_cache")

# (kolon oneki, aciklama) - oncelik sirasina gore
BOOKS = [
    ("PSC", "Pinnacle kapanis"),
    ("AvgC", "Ortalama kapanis"),
    ("B365C", "Bet365 kapanis"),
    ("PS", "Pinnacle acilis"),
    ("Avg", "Ortalama"),
    ("BbAv", "Ortalama (eski format)"),
    ("B365", "Bet365"),
]

DEFAULT_LEAGUES = ["E0", "D1", "SP1", "I1", "F1", "T1", "N1", "P1"]


def default_seasons():
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2016, 2027)]


# ----------------------------------------------------------------- veri yukleme
def read_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def fetch(season, league, local_dir):
    name = f"{league}_{season}.csv"
    if local_dir:
        p = Path(local_dir) / name
        if p.exists():
            return read_csv(p)
    CACHE_DIR.mkdir(exist_ok=True)
    p = CACHE_DIR / name
    if not p.exists():
        url = BASE_URL.format(season=season, league=league)
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as r:
                p.write_bytes(r.read())
        except (HTTPError, URLError, TimeoutError) as e:
            print(f"  ! {name} yuklenemedi ({e})", file=sys.stderr)
            return None
    return read_csv(p)


def prepare(df, league, season, book):
    """Ham CSV'yi standart kolonlara cevirir. (DataFrame, oran kaynagi) doner."""
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    if any(c not in df.columns for c in need):
        return None, None

    chosen = None
    for pre, label in BOOKS:
        if book and pre != book:
            continue
        cols = [f"{pre}H", f"{pre}D", f"{pre}A"]
        if all(c in df.columns for c in cols):
            vals = df[cols].apply(pd.to_numeric, errors="coerce")
            if vals.notna().all(axis=1).sum() > 0:
                chosen = (pre, label)
                break
    if not chosen:
        return None, None
    pre, label = chosen

    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "league": league,
            "season": season,
            "book": pre,
            "home": df["HomeTeam"],
            "away": df["AwayTeam"],
            "hg": num("FTHG"),
            "ag": num("FTAG"),
            "ftr": df["FTR"],
            "oh": num(f"{pre}H"),
            "od": num(f"{pre}D"),
            "oa": num(f"{pre}A"),
            "hs": num("HS"),
            "as_": num("AS"),
            "hst": num("HST"),
            "ast": num("AST"),
            "hr": num("HR"),
            "ar": num("AR"),
        }
    )
    out = out.dropna(subset=["oh", "od", "oa", "hg", "ag"])
    out = out[out["ftr"].isin(["H", "D", "A"])]
    return out, label


# ------------------------------------------------------------------- hesaplama
def enrich(m):
    """Marj ayiklanmis olasiliklar, favori/underdog sutunlari ve favori sonucu."""
    m = m.copy()
    inv = 1.0 / m[["oh", "od", "oa"]]
    over = inv.sum(axis=1)
    m["margin"] = over - 1.0
    ph, pdr, pa = inv["oh"] / over, inv["od"] / over, inv["oa"] / over

    fav_away = m["oa"] < m["oh"]
    m["fav_side"] = np.where(fav_away, "A", "H")
    m["fav"] = np.where(fav_away, m["away"], m["home"])
    m["dog"] = np.where(fav_away, m["home"], m["away"])
    m["fav_odds"] = np.where(fav_away, m["oa"], m["oh"])
    m["dog_odds"] = np.where(fav_away, m["oh"], m["oa"])
    m["p_fav"] = np.where(fav_away, pa, ph)
    m["p_dog"] = np.where(fav_away, ph, pa)
    m["p_draw"] = pdr
    m["fav_res"] = np.where(
        m["ftr"] == "D", "D", np.where(m["ftr"] == m["fav_side"], "W", "L")
    )

    pick = lambda away_col, home_col: np.where(fav_away, m[away_col], m[home_col])
    m["fav_shots"], m["dog_shots"] = pick("as_", "hs"), pick("hs", "as_")
    m["fav_sot"], m["dog_sot"] = pick("ast", "hst"), pick("hst", "ast")
    m["fav_red"], m["dog_red"] = pick("ar", "hr"), pick("hr", "ar")
    return m


def select(m, a):
    if a.like:
        h, d, aw = a.like
        t = a.tol
        ok = (
            m["oh"].between(h * (1 - t), h * (1 + t))
            & m["od"].between(d * (1 - t), d * (1 + t))
            & m["oa"].between(aw * (1 - t), aw * (1 + t))
        )
        return m[ok]
    ok = m["fav_odds"].between(a.fav_min, a.fav_max) & m["dog_odds"].between(
        a.dog_min, a.dog_max
    )
    if a.venue == "away":
        ok &= m["fav_side"] == "A"
    elif a.venue == "home":
        ok &= m["fav_side"] == "H"
    return m[ok]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


# --------------------------------------------------------------------- rapor
def report(s, a):
    n = len(s)
    print("\n" + "=" * 64)
    if a.like:
        print(f"Aranan oranlar (ev/ber/dep): {a.like}  tolerans: +-{a.tol:.0%}")
    else:
        print(
            f"Favori ({a.venue}) {a.fav_min}-{a.fav_max}, rakip {a.dog_min}-{a.dog_max}"
        )
    print(f"Eslesen mac sayisi: {n}")
    if n == 0:
        print("Eslesen mac yok. Bantlari genisletin (--tol, --fav-max, --dog-min ...).")
        return
    if n < 100:
        print("UYARI: Orneklem kucuk (n<100), guven araligi genis. Sezon/lig ekleyin.")

    w = int((s["fav_res"] == "W").sum())
    d = int((s["fav_res"] == "D").sum())
    l = int((s["fav_res"] == "L").sum())
    lo, hi = wilson(l, n)
    exp_l = s["p_dog"].mean()

    print(
        f"Ortalama oranlar: favori {s['fav_odds'].mean():.2f} | "
        f"beraberlik {s['od'].mean():.2f} | rakip {s['dog_odds'].mean():.2f}"
    )
    print(f"\n{'':22}{'gerceklesen':>14}{'piyasa (marjsiz)':>20}")
    print(f"{'Favori kazandi':22}{w:>6} ({w / n:5.1%}){s['p_fav'].mean():>16.1%}")
    print(f"{'Beraberlik':22}{d:>6} ({d / n:5.1%}){s['p_draw'].mean():>16.1%}")
    print(f"{'Favori kaybetti':22}{l:>6} ({l / n:5.1%}){exp_l:>16.1%}")
    print(f"\nKayip orani %95 guven araligi: {lo:.1%} - {hi:.1%}")
    if lo <= exp_l <= hi:
        print("-> Piyasa tahmini bu aralikta: oranlar bu bantta iyi kalibre.")
    elif exp_l < lo:
        print("-> Piyasanin ima ettigi kayip orani aralik disinda ve DUSUK: "
              "bu bantta surprizler oranin soyledigi kadar nadir degil.")
    else:
        print("-> Piyasanin ima ettigi kayip orani aralik disinda ve YUKSEK: "
              "bu bantta favoriler oranin soyledigi kadar sik kaybetmiyor.")

    books = s.groupby("book").size().sort_values(ascending=False)
    print("Oran kaynagi: " + ", ".join(f"{b} ({c})" for b, c in books.items()))

    if s["league"].nunique() > 1:
        print("\nLig bazinda:")
        g = s.groupby("league").agg(
            mac=("fav_res", "size"),
            kayip=("fav_res", lambda x: int((x == "L").sum())),
        )
        g["kayip_orani"] = (g["kayip"] / g["mac"]).map("{:.1%}".format)
        print(g.to_string())

    # Surprizin anatomisi: favori sonucuna gore sut / kirmizi kart
    cols = ["fav_shots", "dog_shots", "fav_sot", "dog_sot", "fav_red", "dog_red"]
    if s[cols].notna().any().any():
        print("\nSurprizin anatomisi (favori sonucuna gore ortalamalar):")
        t = s.groupby("fav_res")[cols].mean().reindex(["W", "D", "L"]).round(2)
        t.insert(0, "mac", s.groupby("fav_res").size().reindex(["W", "D", "L"]))
        t.index = ["Fav kazandi", "Beraberlik", "Fav kaybetti"]
        t.columns = ["mac", "fav sut", "rakip sut", "fav isabetli", "rakip isabetli",
                     "fav kirmizi", "rakip kirmizi"]
        print(t.to_string())
        print("(Sutlar sadece verisi olan maclardan hesaplanir; kirmizi kart "
              "ortalamalari kucuk orneklemde oynak olur.)")

    ups = s[s["fav_res"] == "L"].sort_values("date", ascending=False)
    if len(ups):
        print(f"\nFavorinin kaybettigi maclar (en yeni {min(a.show, len(ups))}):")
        for _, r in ups.head(a.show).iterrows():
            dt = r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "?"
            print(
                f"  {dt} {r['league']:<4} {r['home']} {int(r['hg'])}-{int(r['ag'])} "
                f"{r['away']}   oran {r['oh']:.2f}/{r['od']:.2f}/{r['oa']:.2f}"
            )


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Oran bandi / surpriz analizi")
    ap.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    ap.add_argument("--seasons", nargs="+", default=default_seasons(),
                    help="ornek: 2324 2425 2526")
    ap.add_argument("--book", choices=[b for b, _ in BOOKS], default=None,
                    help="oran kaynagini sabitle (varsayilan: otomatik)")
    ap.add_argument("--venue", choices=["away", "home", "any"], default="away",
                    help="favori nerede oynuyor (varsayilan: away)")
    ap.add_argument("--fav-min", type=float, default=1.25)
    ap.add_argument("--fav-max", type=float, default=1.45)
    ap.add_argument("--dog-min", type=float, default=5.0)
    ap.add_argument("--dog-max", type=float, default=9.0)
    ap.add_argument("--like", nargs=3, type=float, metavar=("EV", "BER", "DEP"),
                    help="birebir oran benzerligi, ornek: --like 6.0 5.75 1.38")
    ap.add_argument("--tol", type=float, default=0.12,
                    help="--like icin goreli tolerans (0.12 = +-%%12)")
    ap.add_argument("--local-dir", default=None,
                    help="elle indirilmis CSV'ler (E0_2526.csv gibi)")
    ap.add_argument("--out", default="matched_matches.csv")
    ap.add_argument("--show", type=int, default=20)
    a = ap.parse_args()

    frames = []
    for lg in a.leagues:
        for se in a.seasons:
            raw = fetch(se, lg, a.local_dir)
            if raw is None:
                continue
            df, label = prepare(raw, lg, se, a.book)
            if df is None or df.empty:
                print(f"  - {lg} {se}: kullanilabilir oran/sonuc kolonu yok, atlandi",
                      file=sys.stderr)
                continue
            frames.append(df)
            print(f"  + {lg} {se}: {len(df)} mac ({label})")

    if not frames:
        print("\nHic veri yuklenemedi. Internet baglantinizi kontrol edin ya da "
              "CSV'leri indirip --local-dir ile verin.", file=sys.stderr)
        sys.exit(1)

    allm = enrich(pd.concat(frames, ignore_index=True))
    print(f"\nToplam {len(allm)} mac yuklendi.")
    sel = select(allm, a)
    report(sel, a)
    if len(sel):
        sel.to_csv(a.out, index=False)
        print(f"\nEslesen maclar kaydedildi: {a.out}")


if __name__ == "__main__":
    main()
