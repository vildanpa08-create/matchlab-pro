#!/usr/bin/env python3
"""
upset_finder.py - "Bu oran yapisi daha once verilmis mi, ne olmus?"

football-data.co.uk'den mac sonuclarini ve oranlarini indirir, verdiginiz oran
bandina uyan maclari bulur ve favorinin kac kez kazandigini / berabere kaldigini /
kaybettigini, piyasanin ima ettigi olasiliklarla karsilastirarak gosterir.

LIGLER (38):
  Ana ligler (22): Ingiltere (5 kademe), Iskocya (4), Almanya (2), Italya (2),
                   Ispanya (2), Fransa (2), Hollanda, Belcika, Portekiz, Turkiye, Yunanistan
  Ekstra ligler (16): Arjantin, Avusturya, Brezilya, Cin, Danimarka, Finlandiya,
                   Irlanda, Japonya, Meksika, Norvec, Polonya, Romanya, Rusya,
                   Isvec, Isvicre, ABD
  Liste icin:  python upset_finder.py --list-leagues
  Presetler:   top8 (varsayilan), big5, main (22 ana lig), extra (16), all (38)

NOT:
  * Veride Unibet yok. Varsayilan: Pinnacle kapanis (PSC), yoksa ortalama kapanis,
    yoksa Bet365 kapanis. Kapanis oranlari piyasanin en verimli tahmini sayilir.
  * Ekstra liglerde yari zaman (IY) ve sut istatistigi yok: IY/MS tablosu ve
    "surprizin anatomisi" bu liglerde bos kalir, skor tablosu calisir.
  * Ekstra ligler tek dosyada tum sezonlari icerir; --seasons yalnizca ilk
    sezonun baslangic yilini belirler (ornek 2223 -> 2022 ve sonrasi).
  * Sampiyonlar Ligi bu kaynakta yok; sadece ulusal ligler.

Kullanim:
  python upset_finder.py                                  # top8, son 11 sezon
  python upset_finder.py --leagues main                   # 22 ana lig
  python upset_finder.py --leagues all --scores           # 38 lig + skor tablolari
  python upset_finder.py --leagues E0 T1 BRA ARG          # tek tek secim
  python upset_finder.py --like 6.0 5.75 1.38 --tol 0.12  # birebir oran benzerligi
  python upset_finder.py --local-dir ./csv                # elle indirilmis CSV'ler
      (ana ligler: E0_2526.csv, ekstra ligler: ARG.csv)

Streamlit icinde:
  from upset_finder import ALL_LEAGUES, PRESETS, load_matches, select_band
  df = load_matches(["E0", "T1", "BRA"], seasons=["2425", "2526"])

Gereksinim: pip install pandas
"""
import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://football-data.co.uk/mmz4281/{season}/{league}.csv"
EXTRA_URL = "https://football-data.co.uk/new/{league}.csv"
CACHE_DIR = Path("fd_cache")
CACHE_MAX_AGE_H = 12  # guncel sezon ve ekstra ligler icin yeniden indirme suresi

MAIN_LEAGUES = {
    "E0": "Ingiltere Premier Lig", "E1": "Ingiltere Championship",
    "E2": "Ingiltere League One", "E3": "Ingiltere League Two",
    "EC": "Ingiltere National League",
    "SC0": "Iskocya Premiership", "SC1": "Iskocya Championship",
    "SC2": "Iskocya League One", "SC3": "Iskocya League Two",
    "D1": "Almanya Bundesliga", "D2": "Almanya 2. Bundesliga",
    "I1": "Italya Serie A", "I2": "Italya Serie B",
    "SP1": "Ispanya La Liga", "SP2": "Ispanya La Liga 2",
    "F1": "Fransa Ligue 1", "F2": "Fransa Ligue 2",
    "N1": "Hollanda Eredivisie", "B1": "Belcika Pro League",
    "P1": "Portekiz Primeira Liga", "T1": "Turkiye Super Lig",
    "G1": "Yunanistan Super League",
}
EXTRA_LEAGUES = {
    "ARG": "Arjantin", "AUT": "Avusturya", "BRA": "Brezilya", "CHN": "Cin",
    "DNK": "Danimarka", "FIN": "Finlandiya", "IRL": "Irlanda", "JPN": "Japonya",
    "MEX": "Meksika", "NOR": "Norvec", "POL": "Polonya", "ROU": "Romanya",
    "RUS": "Rusya", "SWE": "Isvec", "SWZ": "Isvicre", "USA": "ABD",
}
ALL_LEAGUES = {**MAIN_LEAGUES, **EXTRA_LEAGUES}
PRESETS = {
    "top8": ["E0", "D1", "SP1", "I1", "F1", "T1", "N1", "P1"],
    "big5": ["E0", "D1", "SP1", "I1", "F1"],
    "main": list(MAIN_LEAGUES),
    "extra": list(EXTRA_LEAGUES),
    "all": list(ALL_LEAGUES),
}

# (kolon oneki, aciklama) - oncelik sirasina gore
BOOKS = [
    ("PSC", "Pinnacle kapanis"),
    ("AvgC", "Ortalama kapanis"),
    ("B365C", "Bet365 kapanis"),
    ("PS", "Pinnacle acilis"),
    ("P", "Pinnacle (ekstra ligler)"),
    ("Avg", "Ortalama"),
    ("BbAv", "Ortalama (eski format)"),
    ("B365", "Bet365"),
]


def default_seasons():
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2016, 2027)]


def expand_leagues(items):
    """Preset adlarini ve lig kodlarini tek bir kod listesine cevirir."""
    out = []
    for it in items or ["top8"]:
        key = str(it).strip()
        if key.lower() in PRESETS:
            out.extend(PRESETS[key.lower()])
        elif key:
            out.append(key.upper())
    return list(dict.fromkeys(out))


def _recent_season_codes():
    t = date.today()
    y = t.year if t.month >= 7 else t.year - 1
    return {f"{yy % 100:02d}{(yy + 1) % 100:02d}" for yy in (y - 1, y)}


# ----------------------------------------------------------------- veri yukleme
def read_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def _download(url, path, max_age_hours=None):
    """Indirir. Taze onbellek varsa onu kullanir; indirme basarisizsa eski kopyaya duser."""
    if path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if max_age_hours is None or age_h < max_age_hours:
            return path
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as r:
            path.write_bytes(r.read())
    except (HTTPError, URLError, TimeoutError, OSError):
        if path.exists():
            return path
        raise
    return path


def fetch_main(season, league, local_dir):
    name = f"{league}_{season}.csv"
    if local_dir and (Path(local_dir) / name).exists():
        return read_csv(Path(local_dir) / name), None
    CACHE_DIR.mkdir(exist_ok=True)
    age = CACHE_MAX_AGE_H if season in _recent_season_codes() else None
    try:
        p = _download(BASE_URL.format(season=season, league=league), CACHE_DIR / name, age)
    except Exception as e:  # noqa: BLE001
        return None, f"indirilemedi ({e})"
    return read_csv(p), None


def fetch_extra(league, local_dir):
    name = f"{league}.csv"
    if local_dir and (Path(local_dir) / name).exists():
        return read_csv(Path(local_dir) / name), None
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        p = _download(EXTRA_URL.format(league=league), CACHE_DIR / name, CACHE_MAX_AGE_H)
    except Exception as e:  # noqa: BLE001
        return None, f"indirilemedi ({e})"
    return read_csv(p), None


def normalize_extra(df, min_year):
    """Ekstra lig dosyalarini ana lig kolon adlarina cevirir, min_year oncesini atar."""
    ren = {"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG", "Res": "FTR"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns and v not in df.columns})
    if "Date" in df.columns:
        yr = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.year
        df = df[yr >= min_year]
    return df


def prepare(df, league, season, book, season_col=None):
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
    season_val = df[season_col].astype(str) if season_col and season_col in df.columns else season
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "league": league,
            "season": season_val,
            "book": pre,
            "home": df["HomeTeam"],
            "away": df["AwayTeam"],
            "hg": num("FTHG"),
            "ag": num("FTAG"),
            "ftr": df["FTR"],
            "htr": df["HTR"] if "HTR" in df.columns else np.nan,
            "ht_h": num("HTHG"),
            "ht_a": num("HTAG"),
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


def load_matches(leagues=None, seasons=None, book=None, local_dir=None,
                 workers=6, progress=None, verbose=True):
    """Secili ligleri / sezonlari paralel indirir, birlestirir, enrich() uygular.

    leagues  : kod ve/veya preset listesi, orn. ["E0", "T1", "BRA"] veya ["all"]
    progress : istege bagli callback(done, total, etiket), orn. st.progress icin
    Donus    : enrich edilmis DataFrame (bos olabilir)
    """
    leagues = expand_leagues(leagues)
    seasons = list(seasons) if seasons else default_seasons()
    min_year = min(2000 + int(s[:2]) for s in seasons)

    jobs = []
    for lg in leagues:
        if lg in EXTRA_LEAGUES:
            jobs.append((lg, None))
        else:
            jobs.extend((lg, se) for se in seasons)

    def work(job):
        lg, se = job
        try:
            if se is None:
                raw, err = fetch_extra(lg, local_dir)
                if raw is None:
                    return job, None, None, err
                df, label = prepare(normalize_extra(raw, min_year), lg, "", book,
                                    season_col="Season")
            else:
                raw, err = fetch_main(se, lg, local_dir)
                if raw is None:
                    return job, None, None, err
                df, label = prepare(raw, lg, se, book)
            return job, df, label, None
        except Exception as e:  # noqa: BLE001
            return job, None, None, f"hata: {e}"

    frames, failed, done = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            job, df, label, err = fut.result()
            done += 1
            tag = f"{job[0]} {job[1] or '(tum sezonlar)'}"
            if df is None or df.empty:
                failed.append((tag, err or "kullanilabilir oran/sonuc kolonu yok"))
            else:
                frames.append(df)
            if progress:
                progress(done, len(jobs), tag)

    if verbose:
        print(f"  {len(frames)}/{len(jobs)} dosya yuklendi.")
        for tag, why in sorted(failed)[:15]:
            print(f"  - {tag}: {why}", file=sys.stderr)
        if len(failed) > 15:
            print(f"  ... ve {len(failed) - 15} dosya daha atlandi.", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    out = enrich(pd.concat(frames, ignore_index=True))
    return out.sort_values("date").reset_index(drop=True)


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


def select_band(m, like=None, tol=0.12, fav_min=1.25, fav_max=1.45,
                dog_min=5.0, dog_max=9.0, venue="away"):
    """Oran bandina uyan maclari secer (Streamlit tarafinda dogrudan kullanilabilir)."""
    if like:
        h, d, aw = like
        ok = (
            m["oh"].between(h * (1 - tol), h * (1 + tol))
            & m["od"].between(d * (1 - tol), d * (1 + tol))
            & m["oa"].between(aw * (1 - tol), aw * (1 + tol))
        )
        return m[ok]
    ok = m["fav_odds"].between(fav_min, fav_max) & m["dog_odds"].between(dog_min, dog_max)
    if venue == "away":
        ok &= m["fav_side"] == "A"
    elif venue == "home":
        ok &= m["fav_side"] == "H"
    return m[ok]


def select(m, a):
    return select_band(m, like=a.like, tol=a.tol, fav_min=a.fav_min, fav_max=a.fav_max,
                       dog_min=a.dog_min, dog_max=a.dog_max, venue=a.venue)


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
        print("\nLig bazinda (ligler arasi fark buyukse havuzlamaya dikkat):")
        g = s.groupby("league").agg(
            mac=("fav_res", "size"),
            kayip=("fav_res", lambda x: int((x == "L").sum())),
        )
        g["kayip_orani"] = (g["kayip"] / g["mac"]).map("{:.1%}".format)
        g.index = [f"{c} {ALL_LEAGUES.get(c, '')}".strip() for c in g.index]
        print(g.sort_values("mac", ascending=False).to_string())

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


def print_leagues():
    print("ANA LIGLER (sezon sezon dosya, IY + sut istatistigi var):")
    for c, n in MAIN_LEAGUES.items():
        print(f"  {c:<4} {n}")
    print("\nEKSTRA LIGLER (tek dosya, IY / sut yok):")
    for c, n in EXTRA_LEAGUES.items():
        print(f"  {c:<4} {n}")
    print("\nPresetler: " + ", ".join(f"{k} ({len(v)})" for k, v in PRESETS.items()))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Oran bandi / surpriz analizi")
    ap.add_argument("--leagues", nargs="+", default=["top8"],
                    help="lig kodlari ve/veya presetler: top8 big5 main extra all")
    ap.add_argument("--list-leagues", action="store_true", help="lig listesini yazdir")
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
                    help="elle indirilmis CSV'ler (E0_2526.csv, ARG.csv gibi)")
    ap.add_argument("--workers", type=int, default=6, help="paralel indirme sayisi")
    ap.add_argument("--out", default="matched_matches.csv")
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--scores", action="store_true",
                    help="skor ve IY/MS dagilimi + adil oranlari da yazdir")
    a = ap.parse_args()

    if a.list_leagues:
        print_leagues()
        return

    codes = expand_leagues(a.leagues)
    print(f"{len(codes)} lig, {len(a.seasons)} sezon hedefleniyor...")
    allm = load_matches(codes, a.seasons, a.book, a.local_dir, a.workers)
    if allm.empty:
        print("\nHic veri yuklenemedi. Internet baglantinizi kontrol edin ya da "
              "CSV'leri indirip --local-dir ile verin.", file=sys.stderr)
        sys.exit(1)

    print(f"\nToplam {len(allm)} mac yuklendi ({allm['league'].nunique()} lig).")
    sel = select(allm, a)
    report(sel, a)
    if a.scores and len(sel):
        from score_analysis import add_labels, freq_table, HTFT_ORDER

        lab = add_labels(sel)
        print("\nSkor dagilimi (ev-dep) ve arsiv adil oranlari:")
        print(freq_table(lab, "score").head(12).round(2).to_string(index=False))
        ht = freq_table(lab, "htft", HTFT_ORDER)
        print("\nIY/MS dagilimi (1=ev, X=ber, 2=dep):")
        print(ht.round(2).to_string(index=False) if len(ht)
              else "  (secilen ligler icin yari zaman verisi yok)")
    if len(sel):
        sel.to_csv(a.out, index=False)
        print(f"\nEslesen maclar kaydedildi: {a.out}")


if __name__ == "__main__":
    main()
