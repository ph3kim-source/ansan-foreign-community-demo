"""인벤토리 데이터셋(가상 또는 실데이터)을 화면용 지표로 집계한다.

원칙 (매뉴얼 · README):
- 통신·카드는 표본 추정치 → 비중·순위·증감률로만 보여준다. 절대값은 마스킹 판정에만 쓴다.
- 대민개방 X 데이터(DS-056·032·042·061)는 시민 화면에 금액·원값을 노출하지 않는다.
Streamlit에 의존하지 않으므로 테스트에서 그대로 불러 쓸 수 있다.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from config import (CACHE_DIR, CSV_ENCODING, DONG_NAME, DONGS, EVENTS, INDUSTRY_LABEL,
                    K_ANON_MIN, LATEST_MONTH, MASK_MIN_POP, NAT_BY_CODE, NAT_CODE,
                    RESIDENT_MIN_NIGHTS, SYNTHETIC_DIR)
from geo import utmk_to_wgs84

ALL_DONGS = [code for code, _, _ in DONGS]
ANSAN_SGG = {"41271", "41273"}


def _read(pattern: str, dtype: dict) -> pd.DataFrame:
    files = sorted(SYNTHETIC_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{SYNTHETIC_DIR / pattern} — 먼저 python src\\gen_synthetic.py 실행")
    return pd.concat([pd.read_csv(f, encoding=CSV_ENCODING, dtype=dtype) for f in files], ignore_index=True)


def _with_all_dongs(series: pd.Series) -> pd.DataFrame:
    df = series.reindex(ALL_DONGS, fill_value=0.0).rename("value").reset_index()
    df.columns = ["ADM_CD", "value"]
    df["ADM_NM"] = df["ADM_CD"].map(DONG_NAME)
    return df


def _share(df: pd.DataFrame, mask_min: float) -> pd.DataFrame:
    total = df["value"].sum()
    df["share"] = df["value"] / total if total > 0 else 0.0
    df["masked"] = df["value"] < mask_min
    return df


# ------------------------------------------------------------------ DS-028 사는 곳
@lru_cache(maxsize=1)
def ds028() -> pd.DataFrame:
    return _read("T37_AMD_STAY_F_POP_*_4127.csv",
                 {"ETL_YM": str, "SGG_CD": str, "ADM_CD": str, "CNTRY_NM": str})


def residence(month: str = LATEST_MONTH) -> pd.DataFrame:
    """거주 근사: 장기(1) & 연속숙박 LDGMT_DD ≥ RESIDENT_MIN_NIGHTS 인 POP 합."""
    df = ds028()
    sub = df[(df["ETL_YM"] == month) & (df["IN_FORN_DIV_NM"] == 1) & (df["LDGMT_DD"] >= RESIDENT_MIN_NIGHTS)]
    return sub.groupby(["ADM_CD", "CNTRY_NM"], as_index=False)["POP"].sum()


def residence_by_dong(nat: str, month: str = LATEST_MONTH) -> pd.DataFrame:
    res = residence(month)
    series = res[res["CNTRY_NM"] == nat].set_index("ADM_CD")["POP"]
    return _share(_with_all_dongs(series), MASK_MIN_POP)


def living_area_top(nat: str, n: int = 3, month: str = LATEST_MONTH) -> pd.DataFrame:
    df = residence_by_dong(nat, month)
    return df[~df["masked"]].sort_values("share", ascending=False).head(n)


def residence_trend(nats: list) -> pd.DataFrame:
    df = ds028()
    sub = df[(df["IN_FORN_DIV_NM"] == 1) & (df["LDGMT_DD"] >= RESIDENT_MIN_NIGHTS) & df["CNTRY_NM"].isin(nats)]
    return sub.groupby(["ETL_YM", "ADM_CD", "CNTRY_NM"], as_index=False)["POP"].sum()


# ------------------------------------------------------------------ DS-017 모이는 곳
@lru_cache(maxsize=1)
def ds017_profile() -> pd.DataFrame:
    """월 × 평일/주말 × 동 × 시간대 × 장단기 × 국적 일평균 POP (원본 → _cache)."""
    cache = CACHE_DIR / "ds017_profile.csv"
    raw_files = sorted(SYNTHETIC_DIR.glob("T38_AMD_INFLOW_TIME_F_POP_*_4127.csv"))
    # 공유판에는 원본(약 100MB) 대신 집계 캐시만 담는다 → 원본이 없으면 캐시를 그대로 쓴다
    if cache.exists() and (not raw_files or cache.stat().st_mtime > max(f.stat().st_mtime for f in raw_files)):
        return pd.read_csv(cache, dtype={"month": str, "ADM_CD": str})
    raw = _read("T38_AMD_INFLOW_TIME_F_POP_*_4127.csv", {"ETL_YMD": str, "SGG_CD": str, "ADM_CD": str})
    raw["month"] = raw["ETL_YMD"].str[:6]
    weekday = pd.to_datetime(raw["ETL_YMD"], format="%Y%m%d").dt.weekday
    raw["daytype"] = np.where(weekday >= 5, "주말", "평일")
    ndays = raw.groupby(["month", "daytype"])["ETL_YMD"].nunique().rename("ndays").reset_index()
    agg = raw.groupby(["month", "daytype", "ADM_CD", "TIMEZN_CD", "IN_FORN_DIV_NM", "CNTRY_NM"],
                      as_index=False)["POP"].sum().merge(ndays, on=["month", "daytype"])
    agg["POP"] = (agg["POP"] / agg["ndays"]).round(3)
    agg = agg.drop(columns="ndays")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(cache, index=False, encoding="utf-8-sig")
    return agg


def gathering(nat: str, daytype: str, hours, month: str = LATEST_MONTH, div: int = 1) -> pd.DataFrame:
    """해당 시간대 평균 POP의 동별 비중 (국적 안에서)."""
    prof = ds017_profile()
    sub = prof[(prof["month"] == month) & (prof["daytype"] == daytype) & (prof["CNTRY_NM"] == nat)
               & (prof["TIMEZN_CD"].isin(list(hours))) & (prof["IN_FORN_DIV_NM"] == div)]
    series = sub.groupby("ADM_CD")["POP"].sum() / max(len(list(hours)), 1)
    return _share(_with_all_dongs(series), MASK_MIN_POP)


def gathering_hourly(nats: list, daytype: str = "주말", month: str = LATEST_MONTH) -> pd.DataFrame:
    """대상 국적들의 (동 × 시간대) 점수 — 국적마다 비중으로 바꾼 뒤 평균 (큰 국적 쏠림 방지)."""
    prof = ds017_profile()
    sub = prof[(prof["month"] == month) & (prof["daytype"] == daytype) & prof["CNTRY_NM"].isin(nats)
               & (prof["IN_FORN_DIV_NM"] == 1)].copy()
    sub["share"] = sub["POP"] / sub.groupby(["CNTRY_NM", "TIMEZN_CD"])["POP"].transform("sum")
    score = sub.groupby(["ADM_CD", "TIMEZN_CD"], as_index=False)["share"].sum()
    score["share"] /= max(len(nats), 1)
    score["ADM_NM"] = score["ADM_CD"].map(DONG_NAME)
    return score


def event_recommend(nats: list, top: int = 5, hours=range(10, 22)) -> pd.DataFrame:
    """행사 기획 추천: 주말 10~21시 중 대상 국적이 가장 많이 모이는 (동, 시작 시각)."""
    score = gathering_hourly(nats)
    score = score[score["TIMEZN_CD"].isin(list(hours))]
    return score.sort_values("share", ascending=False).head(top).reset_index(drop=True)


# ------------------------------------------------------------------ DS-056 쓰는 곳 (비개방 → 비중·순위만)
@lru_cache(maxsize=1)
def ds056() -> pd.DataFrame:
    return _read("GGC_SALES_YM_AMD_*.csv", {"ta_ymd": str, "cty_rgn_no": str, "admi_cty_no": str,
                                             "card_tpbuz_cd": str, "gb": str, "nt": str})


def spending_by_dong(nat: str, month: str = LATEST_MONTH, industry: str = None) -> pd.DataFrame:
    df = ds056()
    sub = df[(df["ta_ymd"].str[:6] == month) & (df["nt"] == NAT_CODE[nat])]
    if industry:
        sub = sub[sub["card_tpbuz_cd"] == industry]
    series = sub.groupby("admi_cty_no")["cnt"].sum().astype(float)
    return _share(_with_all_dongs(series), K_ANON_MIN)


def spending_rank(nat: str, month: str = LATEST_MONTH, dong: str = None) -> pd.DataFrame:
    """국적별 업종 결제건수 비중·순위 (금액 비표시)."""
    df = ds056()
    sub = df[(df["ta_ymd"].str[:6] == month) & (df["nt"] == NAT_CODE[nat])]
    if dong:
        sub = sub[sub["admi_cty_no"] == dong]
    out = sub.groupby("card_tpbuz_cd", as_index=False)["cnt"].sum()
    if out.empty:
        return pd.DataFrame(columns=["순위", "업종", "결제건수 비중"])
    out["share"] = out["cnt"] / out["cnt"].sum()
    out = out.sort_values("share", ascending=False).reset_index(drop=True)
    out["순위"] = np.arange(1, len(out) + 1)
    out["업종"] = out["card_tpbuz_cd"].map(INDUSTRY_LABEL)
    out["결제건수 비중"] = (out["share"] * 100).round(1)
    return out[["순위", "업종", "결제건수 비중"]]


# ------------------------------------------------------------------ DS-076 / 077 전입·전출
@lru_cache(maxsize=1)
def ds076() -> pd.DataFrame:
    return _read("GYEONGGI_IN_STAT_*.csv", {"BF_YR_MON": str, "BF_SIDO": str, "BF_SIGNGU": str,
                                             "BS_YR_MON": str, "BS_SIGNGU": str, "BS_HCD": str})


@lru_cache(maxsize=1)
def ds077() -> pd.DataFrame:
    return _read("GYEONGGI_OUT_STAT_*.csv", {"BS_YR_MON": str, "BS_SIGNGU": str, "BS_HCD": str,
                                              "AF_YR_MON": str, "AF_SIDO": str, "AF_SIGNGU": str})


def new_arrivals(month: str = LATEST_MONTH) -> pd.DataFrame:
    """안산 밖·정보없음(99999)에서 온 외국인 전입(FOR_CNT) — K-10 미만은 비표시."""
    df = ds076()
    sub = df[(df["BS_YR_MON"] == month) & ~df["BF_SIGNGU"].isin(ANSAN_SGG)].copy()
    sub["ADM_CD"] = sub["BS_HCD"].str[:8]
    sub["abroad"] = np.where(sub["BF_SIGNGU"] == "99999", sub["FOR_CNT"], 0)
    agg = sub.groupby("ADM_CD").agg(value=("FOR_CNT", "sum"), abroad=("abroad", "sum"))
    out = agg.reindex(ALL_DONGS, fill_value=0).reset_index()
    out["ADM_NM"] = out["ADM_CD"].map(DONG_NAME)
    out["masked"] = out["value"] < K_ANON_MIN
    return out.sort_values("value", ascending=False).reset_index(drop=True)


def migration_by_month() -> pd.DataFrame:
    """동 × 월 외국인 전입·전출·순유입 (안산 내 이동 포함)."""
    din, dout = ds076().copy(), ds077().copy()
    din["ADM_CD"] = din["BS_HCD"].str[:8]
    dout["ADM_CD"] = dout["BS_HCD"].str[:8]
    a = din.groupby(["BS_YR_MON", "ADM_CD"])["FOR_CNT"].sum().rename("전입")
    b = dout.groupby(["AF_YR_MON", "ADM_CD"])["FOR_CNT"].sum().rename("전출")
    b.index = b.index.set_names(["BS_YR_MON", "ADM_CD"])
    out = pd.concat([a, b], axis=1).fillna(0).reset_index().rename(columns={"BS_YR_MON": "month"})
    out["순유입"] = out["전입"] - out["전출"]
    out["ADM_NM"] = out["ADM_CD"].map(DONG_NAME)
    return out


def arrivals_by_month() -> pd.DataFrame:
    """월별 안산 외부에서 온 외국인 전입 합계 (KPI 가입률 분모)."""
    df = ds076()
    sub = df[~df["BF_SIGNGU"].isin(ANSAN_SGG)]
    return sub.groupby("BS_YR_MON", as_index=False)["FOR_CNT"].sum().rename(columns={"BS_YR_MON": "month"})


# ------------------------------------------------------------------ DS-032 격자 (관리자 전용)
@lru_cache(maxsize=1)
def ds032() -> pd.DataFrame:
    df = _read("T1_GG_50CELL_FLOWPOP_*_4127.csv", {"ADMI_CD": str, "FORN_GB": str, "ETL_YMD": str})
    value_cols = [c for c in df.columns if c.startswith(("M_", "F_"))]
    df["POP"] = df[value_cols].sum(axis=1)
    lon, lat = utmk_to_wgs84(df["UTMK_X"].to_numpy(), df["UTMK_Y"].to_numpy())
    df["lon"], df["lat"] = lon, lat
    return df


def grid_cells(nat: str, ymd: str, time_cd: int, min_pop: float = 0.5) -> pd.DataFrame:
    df = ds032()
    sub = df[(df["FORN_GB"] == NAT_CODE[nat]) & (df["ETL_YMD"] == ymd) & (df["TIME_CD"] == time_cd)]
    return sub[sub["POP"] >= min_pop][["lon", "lat", "POP", "ADMI_NM"]].reset_index(drop=True)


def grid_nations() -> list:
    return [NAT_BY_CODE.get(c, c) for c in ds032()["FORN_GB"].unique()]


# ------------------------------------------------------------------ DS-042 / 061 행사 효과
@lru_cache(maxsize=1)
def ds042() -> pd.DataFrame:
    return _read("EVENT_VISIT_NATIONALITY_*.csv", {"CTY_CD": str, "ADMI_CD": str, "AREA_CD": str,
                                                    "FORN_GB": str, "ETL_YMD": str})


@lru_cache(maxsize=1)
def ds061() -> pd.DataFrame:
    return _read("EVENT_SALES_NATIONALITY_*.csv", {"CTY_CD": str, "ADMI_CD": str, "AREA_CD": str,
                                                    "card_tpbuz_cd": str, "NAT_CD": str, "ETL_YMD": str})


def _event(area_cd: str) -> dict:
    return next(e for e in EVENTS if e["area_cd"] == area_cd)


def event_visit_effect(area_cd: str) -> pd.DataFrame:
    """국적별 행사일 평균 ÷ 기준일(1·2주 전 같은 요일) 평균 − 1."""
    ev = _event(area_cd)
    df = ds042()
    sub = df[df["AREA_CD"] == area_cd].copy()
    sub["구분"] = np.where(sub["ETL_YMD"].isin(ev["days"]), "행사일", "기준일")
    piv = sub.pivot_table(index="FORN_GB", columns="구분", values="CNT", aggfunc="mean").fillna(0)
    piv = piv[(piv.get("기준일", 0) > 0)]
    piv["증가율(%)"] = ((piv["행사일"] / piv["기준일"] - 1) * 100).round(0)
    piv["국적"] = [("내국인" if code == "N" else NAT_BY_CODE.get(code, code)) for code in piv.index]
    piv["대상 국적"] = piv["국적"].isin(ev["target"])
    return piv.reset_index(drop=True)[["국적", "기준일", "행사일", "증가율(%)", "대상 국적"]]


def event_daily(area_cd: str) -> pd.DataFrame:
    df = ds042()
    sub = df[df["AREA_CD"] == area_cd].copy()
    sub["구분"] = np.where(sub["FORN_GB"] == "N", "내국인", "외국인")
    return sub.groupby(["ETL_YMD", "구분"], as_index=False)["CNT"].sum()


def event_sales_effect(area_cd: str) -> pd.DataFrame:
    """업종 × 내/외국인 결제 증가율 (금액은 증가율로만)."""
    ev = _event(area_cd)
    df = ds061()
    sub = df[df["AREA_CD"] == area_cd].copy()
    sub["구분"] = np.where(sub["ETL_YMD"].isin(ev["days"]), "행사일", "기준일")
    piv = sub.pivot_table(index=["card_tpbuz_cd", "NAT_CD"], columns="구분", values=["CNT", "AMT"],
                          aggfunc="mean").fillna(0)
    out = pd.DataFrame({
        "업종": [INDUSTRY_LABEL.get(i[0], i[0]) for i in piv.index],
        "구분": ["내국인" if i[1] == "N" else "외국인" for i in piv.index],
        "결제건수 증가율(%)": ((piv[("CNT", "행사일")] / piv[("CNT", "기준일")].replace(0, np.nan) - 1) * 100).round(0).to_numpy(),
        "결제금액 증가율(%)": ((piv[("AMT", "행사일")] / piv[("AMT", "기준일")].replace(0, np.nan) - 1) * 100).round(0).to_numpy(),
    })
    return out.sort_values(["구분", "결제건수 증가율(%)"], ascending=[False, False]).reset_index(drop=True)


def event_total_uplift(area_cd: str) -> float:
    eff = event_visit_effect(area_cd)
    foreign = eff[eff["국적"] != "내국인"]
    base, on = foreign["기준일"].sum(), foreign["행사일"].sum()
    return float((on / base - 1) * 100) if base > 0 else 0.0
