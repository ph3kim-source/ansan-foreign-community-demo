"""안산 이웃 (Ansan Neighbors) — 2안 v2 초안 프로토타입 (가상 데이터).

    cd 2안
    python src\\gen_synthetic.py     # 가상 데이터 (최초 1회)
    streamlit run src\\app.py

결정 근거: docs/기획_결정사항.md · 진행 상황/남은 일: docs/진행상황.md
"""
import datetime as dt
import tempfile
import uuid
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import community as cm
import metrics as mt
from config import (CARD_INDUSTRIES, DONG_NAME, DONGS, EVENTS, HOT_DAYS, HOT_MIN_LIKES, INDUSTRY_LABEL,
                    LATEST_MONTH, MASK_MIN_POP, MONTHS, NAT_FLAG, NAT_NAMES, OFFICIAL_LANGS, PERSONAS,
                    RESIDENT_MIN_NIGHTS, ROOM_SHORT, ROOMS, SHARE_MODE, SYNTHETIC_BADGE, TAGS, TIME_BANDS)
from geo import dong_centroids, load_dong_features

st.set_page_config(page_title="안산 이웃 · Ansan Neighbors", page_icon="🏘️", layout="wide")
st.markdown("""<style>
.syn{background:#FFF4D6;border:1px solid #E8B93A;color:#6B4E00;padding:7px 12px;border-radius:8px;font-size:.85rem;margin:0 0 10px}
.chip{display:inline-block;border-radius:999px;padding:1px 9px;font-size:.74rem;margin-right:4px;background:#E8EEFF;color:#2B3A8C}
.chip.hot{background:#FFE3E3;color:#9B1C1C}.chip.off{background:#DFF5E3;color:#17642B}.chip.warn{background:#FFF0D9;color:#8A4B00}
.meta{color:#6B7280;font-size:.8rem}
</style>""", unsafe_allow_html=True)

DONG_CODES = [c for c, _, _ in DONGS]
NATS = [n for n in NAT_NAMES if n != "기타"]


# ------------------------------------------------------------------ 공통
def badge():
    st.markdown(f"<div class='syn'>{SYNTHETIC_BADGE}</div>", unsafe_allow_html=True)


def source(text):
    st.caption(f"📎 출처: {text} · 가상 데이터")


def chip(text, kind=""):
    return f"<span class='chip {kind}'>{text}</span>"


def pct(x):
    return f"{x * 100:.1f}%"


def ago(ts):
    hours = int((dt.datetime.now() - dt.datetime.fromisoformat(ts)).total_seconds() // 3600)
    return "방금" if hours < 1 else f"{hours}시간 전" if hours < 24 else f"{hours // 24}일 전"


def act(fn, *args):
    """버튼 콜백 — 콜백은 새 스레드에서 돌므로 세션 DB를 다시 지정한 뒤 실행 (공유 시연판)."""
    cm.use_db(st.session_state.get("db_path"))
    fn(*args)


def user_room(user):
    return user["language"] if user["language"] in ROOMS else "etc"


def tr(d, lang):
    return d.get(lang) or d.get("ko", "")


def top_visible(df, n=3):
    return df[~df["masked"]].sort_values("share", ascending=False).head(n)


def dong_map(df, label, pins=None, height=480):
    rows = df.set_index("ADM_CD")
    vis = df[~df["masked"]]
    vmax = vis["share"].max() if len(vis) else 1.0
    feats = []
    for code, name, _, feat in load_dong_features():
        r = rows.loc[code] if code in rows.index else None
        if r is None or bool(r["masked"]):
            color, tip = [205, 205, 205, 110], f"<b>{name}</b><br/>{label}: 소수(마스킹)"
        else:
            t = (float(r["share"]) / vmax) ** 0.7 if vmax else 0
            color = [int(239 - 231 * t), int(246 - 177 * t), int(255 - 107 * t), 190]
            tip = f"<b>{name}</b><br/>{label}: {pct(r['share'])}"
        feats.append({"type": "Feature", "geometry": feat["geometry"], "properties": {"tip": tip, "color": color}})
    layers = [pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": feats}, stroked=True, filled=True,
                        get_fill_color="properties.color", get_line_color=[255, 255, 255, 220],
                        line_width_min_pixels=1, pickable=True)]
    if pins is not None and len(pins):
        layers.append(pdk.Layer("ScatterplotLayer", pins, get_position=["lon", "lat"], get_fill_color="color",
                                get_radius=70, radius_min_pixels=5, radius_max_pixels=14, stroked=True,
                                get_line_color=[255, 255, 255], line_width_min_pixels=1, pickable=True))
    st.pydeck_chart(pdk.Deck(map_style=None, layers=layers,
                             initial_view_state=pdk.ViewState(latitude=37.318, longitude=126.83, zoom=11.2),
                             tooltip={"html": "{tip}", "style": {"fontSize": "12px"}}), height=height)


def build_pins(selected, lang):
    rows, cent = [], dong_centroids()
    for p in cm.list_places():
        if p["source"] == "official" and "공공시설" in selected:
            rows.append({"lon": p["lon"], "lat": p["lat"], "color": [22, 163, 74, 230],
                         "tip": f"🏛️ <b>{p['name']}</b><br/>{DONG_NAME[p['dong']]} · 가상 위치"})
        elif p["source"] == "community" and "커뮤니티 제보" in selected:
            state = "검증됨" if p["verified"] else f"추천 {p['votes']}"
            rows.append({"lon": p["lon"], "lat": p["lat"],
                         "color": [234, 88, 12, 230] if p["verified"] else [251, 191, 36, 210],
                         "tip": f"📍 <b>{p['name']}</b> ({p['category']})<br/>{DONG_NAME[p['dong']]} · {state}"})
    if "행사" in selected:
        for e in cm.list_notices("event"):
            if e["dong"]:
                lon, lat = cent[e["dong"]]
                rows.append({"lon": lon + 0.002, "lat": lat, "color": [147, 51, 234, 230],
                             "tip": f"🎉 <b>{tr(e['titles'], lang)}</b><br/>{e['start_date']}"})
    if "모임·장터" in selected:
        posts = cm.list_posts(tags=["모임", "장터"], since_days=30, limit=500)
        for code in {p["dong"] for p in posts}:
            n_meet = sum(1 for p in posts if p["dong"] == code and p["tag"] == "모임")
            n_market = sum(1 for p in posts if p["dong"] == code and p["tag"] == "장터")
            lon, lat = cent[code]
            rows.append({"lon": lon, "lat": lat + 0.003, "color": [220, 38, 38, 220],
                         "tip": f"💬 <b>{DONG_NAME[code]}</b><br/>모임 {n_meet} · 장터 {n_market} (동 단위)"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 시민: 내 생활권
def page_home(user, lang):
    nat = user["nationality"]
    top = mt.living_area_top(nat, 3)
    key = f"area_{user['id']}"
    if key not in st.session_state:
        st.session_state[key] = list(top["ADM_CD"])
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(f"#### ① {NAT_FLAG.get(nat, '')} 내 국적 생활권 — 자동 설정")
        for col, (_, r) in zip(st.columns(3), top.iterrows()):
            col.metric(r["ADM_NM"], pct(r["share"]),
                       help=f"{nat} 거주자(장기·{RESIDENT_MIN_NIGHTS}박 이상) 중 이 동에 사는 비중")
        st.multiselect("내 생활권 동 (직접 바꿀 수 있어요)", DONG_CODES, key=key, format_func=DONG_NAME.get)
        source("DS-028 AMD_STAY_F_POP — 장기(1)·LDGMT_DD≥20 POP의 동별 비중")
    with c2:
        now = dt.datetime.now()
        daytype = "주말" if now.weekday() >= 5 else "평일"
        st.markdown(f"#### ② 지금 모이는 곳 ({daytype} {now.hour}시)")
        a, b = st.columns(2)
        a.markdown("**지금**")
        for i, r in enumerate(top_visible(mt.gathering(nat, daytype, [now.hour])).itertuples(), 1):
            a.markdown(f"{i}. {r.ADM_NM} · {pct(r.share)}")
        b.markdown("**이번 주말 오후**")
        for i, r in enumerate(top_visible(mt.gathering(nat, "주말", range(12, 18))).itertuples(), 1):
            b.markdown(f"{i}. {r.ADM_NM} · {pct(r.share)}")
        source("DS-017 AMD_INFLOW_TIME_F_POP — 2026-06 평일/주말·시간대별 일평균 POP의 동별 비중")
    st.divider()
    c3, c4 = st.columns([1, 1.2])
    with c3:
        st.markdown("#### ③ 🆕 새로 온 이웃이 많은 동 (2026-06)")
        na = mt.new_arrivals().head(8).copy()
        na["구분"] = np.where(na["ADM_CD"].isin(st.session_state[key]), "내 생활권", "기타")
        chart = alt.Chart(na[~na["masked"]]).mark_bar().encode(
            x=alt.X("value:Q", title="외국인 전입 (표본)"), y=alt.Y("ADM_NM:N", sort="-x", title=None),
            color=alt.Color("구분:N", scale=alt.Scale(domain=["내 생활권", "기타"], range=["#E8590C", "#94A3B8"]),
                            legend=None))
        st.altair_chart(chart, height=240)
        source("DS-076 GYEONGGI_IN_STAT — 안산 밖·정보없음(99999)에서 온 FOR_CNT, 10 미만 비표시 · "
               "신용정보 보유 '우량외국인'만 집계돼 과소추정")
    with c4:
        st.markdown("#### ④ 새로 온 이웃 Q&A — 내 생활권·내 언어방")
        posts = cm.list_posts(room=user_room(user), tags=["고민", "병원", "비자·행정", "집·생활"],
                              dongs=st.session_state[key] or None, since_days=30, limit=5)
        if not posts:
            st.caption("아직 글이 없어요. 커뮤니티 탭에서 첫 질문을 남겨보세요.")
        for p in posts:
            st.markdown(f"{chip(p['tag'])}{chip('📍 ' + DONG_NAME[p['dong']])} **{p['title']}** "
                        f"<span class='meta'>❤ {p['likes']} · 💬 {p['n_comments']}</span>", unsafe_allow_html=True)
        st.markdown("#### ⑤ 다가오는 행사")
        for e in [e for e in cm.list_notices("event") if e["start_date"] >= dt.date.today().isoformat()][:3]:
            mine = chip("내 국적 대상", "off") if nat in e["target"] else ""
            st.markdown(f"{mine}**{tr(e['titles'], lang)}** <span class='meta'>{e['start_date']} · "
                        f"{DONG_NAME.get(e['dong'], '')}</span>", unsafe_allow_html=True)


# ------------------------------------------------------------------ 시민: 생활권 지도
def page_map(user, lang):
    c = st.columns([1, 1.6, 1.4])
    nat_default = user["nationality"] or "중국"
    nat = c[0].selectbox("국적", NATS, index=NATS.index(nat_default), format_func=lambda n: f"{NAT_FLAG[n]} {n}",
                         key="map_nat")
    layer = c[1].radio("생활권 레이어", ["사는 곳", "모이는 곳", "쓰는 곳"], horizontal=True, key="map_layer")
    month = c[2].select_slider("기준월", MONTHS, value=LATEST_MONTH, key="map_month")
    if layer == "사는 곳":
        df, label = mt.residence_by_dong(nat, month), f"{nat} 거주 비중"
        src = "DS-028 AMD_STAY_F_POP (대민개방 O) — 장기·연속숙박 20박 이상 POP"
    elif layer == "모이는 곳":
        cc = st.columns(2)
        daytype = cc[0].radio("요일", ["평일", "주말"], index=1, horizontal=True, key="map_day")
        band = cc[1].radio("시간대", list(TIME_BANDS), index=2, horizontal=True, key="map_band")
        df, label = mt.gathering(nat, daytype, TIME_BANDS[band], month), f"{daytype} {band} 체류 비중"
        src = "DS-017 AMD_INFLOW_TIME_F_POP (대민개방 O) — 시간대별 일평균 POP"
    else:
        ind = st.selectbox("업종", ["전체"] + [x[0] for x in CARD_INDUSTRIES], key="map_ind",
                           format_func=lambda x: "전체" if x == "전체" else INDUSTRY_LABEL[x])
        df, label = mt.spending_by_dong(nat, month, None if ind == "전체" else ind), f"{nat} 결제건수 비중"
        st.markdown(chip("대민개방 X 데이터 — 금액 없이 비중·순위만 표시 · 공개 전 승인 필요", "warn"),
                    unsafe_allow_html=True)
        src = "DS-056 GGC_SALES_YM_AMD (대민개방 X) — nt=국적코드 결제건수(cnt) 비중"
    pins_sel = st.pills("지도 핀", ["공공시설", "커뮤니티 제보", "행사", "모임·장터"], selection_mode="multi",
                        default=["공공시설", "커뮤니티 제보"], key="map_pins")
    m1, m2 = st.columns([1.7, 1])
    with m1:
        dong_map(df, label, build_pins(pins_sel or [], lang))
        source(src)
    with m2:
        st.markdown(f"**{label} 순위**")
        rank = df.sort_values("share", ascending=False).head(10)
        st.dataframe(pd.DataFrame({"동": rank["ADM_NM"],
                                   "비중": ["소수" if m else pct(s) for s, m in zip(rank["share"], rank["masked"])]}),
                     hide_index=True)
        st.caption(f"마스킹: 값 {MASK_MIN_POP:g} 미만은 '소수'로만 표시 (개인 특정 방지)")
        if layer == "쓰는 곳":
            st.markdown("**업종 순위 (결제건수 비중 %)**")
            st.dataframe(mt.spending_rank(nat, month).head(8), hide_index=True)
    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        sel = st.selectbox("동네 살펴보기", DONG_CODES, index=DONG_CODES.index(user["dong"]),
                           format_func=DONG_NAME.get, key="map_dong")
        for p in cm.list_posts(dongs=[sel], limit=6):
            st.markdown(f"{chip(ROOM_SHORT[p['room']])}{chip(p['tag'])} {p['title']}", unsafe_allow_html=True)
        for p in [p for p in cm.list_places() if p["dong"] == sel]:
            st.markdown(f"📍 {p['name']} · {p['category']} · {'검증됨' if p['verified'] else '추천 ' + str(p['votes'])}")
    with d2:
        st.markdown("**📍 장소 제보** — 데이터가 못 잡는 가게·시설을 이웃이 채워요 (추천 5개 이상이면 검증됨)")
        with st.form("place_form", clear_on_submit=True):
            name = st.text_input("장소 이름")
            cat = st.selectbox("분류", ["음식", "식료품", "생활", "종교·모임", "기타"])
            d = st.selectbox("동", DONG_CODES, index=DONG_CODES.index(user["dong"]), format_func=DONG_NAME.get)
            if st.form_submit_button("제보하기") and name.strip():
                cm.add_place(user, name.strip(), cat, d)
                st.success("제보했어요! 이웃 추천을 받으면 '검증됨'이 붙어요.")
        for p in [p for p in cm.list_places() if p["source"] == "community" and not p["verified"]][:4]:
            st.button(f"👍 {p['name']} ({DONG_NAME[p['dong']]}) · 추천 {p['votes']}", key=f"vote_{p['id']}",
                      on_click=act, args=(cm.vote_place, p["id"], user["id"]))


# ------------------------------------------------------------------ 시민: 커뮤니티
def write_form(user, room):
    tag = st.selectbox("주제", TAGS, key=f"w_tag_{room}")
    if tag == "모임" and user["nationality"]:
        rec = top_visible(mt.gathering(user["nationality"], "주말", range(12, 18)))
        st.info("📊 모임 장소 추천 — 주말 오후 같은 국적 이웃이 많이 모이는 동: "
                + ", ".join(f"{r.ADM_NM}({pct(r.share)})" for r in rec.itertuples()) + "  · 출처 DS-017 (가상)")
    if tag == "장터":
        st.caption("🛒 물품만 거래 · 결제·보증 없이 연결만 · 방 양도 금지 · 신고 3건이면 자동 숨김")
    if tag == "사업장후기":
        st.caption("🏭 사업장명·법인명 금지(업종·동 단위). 임금체불은 ☎1350 고용노동부 상담으로 연결하세요.")
    with st.form(f"write_{room}", clear_on_submit=True):
        title = st.text_input("제목")
        body = st.text_area("내용", height=90)
        dong = st.selectbox("동네 태그 (지도와 연결)", DONG_CODES, index=DONG_CODES.index(user["dong"]),
                            format_func=DONG_NAME.get)
        extra = {}
        if tag == "장터":
            extra = {"price": int(st.number_input("가격(원)", min_value=0, step=1000)), "status": "판매중"}
        elif tag == "사업장후기":
            extra = {"industry": st.text_input("업종 (예: 금속가공)"),
                     "shift": st.selectbox("근무형태", ["주간", "야간", "주야교대"]),
                     "pay": st.slider("임금 지급", 1, 5, 3), "safety": st.slider("안전", 1, 5, 3),
                     "housing": st.slider("숙소", 1, 5, 3)}
        elif tag == "모임":
            extra = {"when": st.text_input("언제 (예: 토요일 15:00)"),
                     "capacity": int(st.number_input("정원", min_value=2, max_value=200, value=10))}
        if st.form_submit_button("익명으로 등록") and title.strip():
            try:
                cm.add_post(user, room, tag, title.strip(), body.strip(), dong, extra)
                st.success("등록했어요.")
            except (PermissionError, ValueError) as err:
                st.error(str(err))


def render_post(p, user, show_tr):
    with st.container(border=True):
        who = "🇰🇷 내국인 이웃" if p["is_korean"] else f"{NAT_FLAG.get(p['nationality'], '🌐')} {ROOM_SHORT.get(p['language'], '')}"
        hot = chip("HOT", "hot") if p["likes"] >= HOT_MIN_LIKES else ""
        st.markdown(f"{hot}{chip(p['tag'])}{chip('📍 ' + DONG_NAME.get(p['dong'], ''))}"
                    f"<span class='meta'>익명 · {who} · {ago(p['created_at'])}</span>", unsafe_allow_html=True)
        st.markdown(f"**{p['title']}**  \n{p['body']}")
        if show_tr and p["room"] not in ("ko",):
            st.markdown(f"<span class='meta'>🌐 번역: {p['body_ko'] or '(새 글 — 번역 API 연결 후 제공)'}</span>",
                        unsafe_allow_html=True)
        x = p["extra"]
        if p["tag"] == "장터" and x:
            st.markdown(chip(f"{x.get('price', 0):,}원") + chip(x.get("status", ""), "off"), unsafe_allow_html=True)
        if p["tag"] == "사업장후기" and x:
            st.markdown(f"<span class='meta'>업종 {x.get('industry', '')} · {x.get('shift', '')} · 임금 {'★' * x.get('pay', 0)}"
                        f" · 안전 {'★' * x.get('safety', 0)} · 숙소 {'★' * x.get('housing', 0)} · "
                        f"상담 ☎1350(고용노동부)</span>", unsafe_allow_html=True)
        if p["tag"] == "모임" and x:
            st.markdown(chip(f"🗓 {x.get('when', '')}") + chip(f"정원 {x.get('capacity', '')}"), unsafe_allow_html=True)
        cols = st.columns([1, 1, 1, 5])
        liked = cm.has_liked(p["id"], user["id"])
        cols[0].button(("❤️ " if liked else "🤍 ") + str(p["likes"]), key=f"like_{p['id']}",
                       on_click=act, args=(cm.toggle_like, p["id"], user["id"]))
        with cols[1].popover(f"💬 {p['n_comments']}"):
            for c in cm.list_comments(p["id"], p["author_id"]):
                st.markdown(f"**{c['alias']}** {c['body']}")
                if show_tr and c["body_ko"]:
                    st.caption(f"🌐 {c['body_ko']}")
            with st.form(f"c_{p['id']}", clear_on_submit=True):
                text = st.text_input("댓글 (익명)")
                if st.form_submit_button("등록") and text.strip():
                    cm.add_comment(user, p["id"], text.strip())
                    st.rerun()
        with cols[2].popover("🚩"):
            reason = st.selectbox("신고 사유", ["사기·브로커", "혐오 표현", "사업장 실명", "광고·스팸", "기타"],
                                  key=f"rr_{p['id']}")
            if st.button("신고하기", key=f"rp_{p['id']}"):
                hidden = cm.report_post(p["id"], user["id"], reason)
                st.toast("신고 접수" + (" — 누적 3건이라 글이 숨겨졌어요" if hidden else ""))


def page_community(user, lang):
    rooms = list(ROOMS)
    room = st.radio("언어방", rooms, index=rooms.index(user_room(user)), horizontal=True,
                    format_func=ROOM_SHORT.get, key="room")
    writable = cm.can_write(user, room)
    st.caption(ROOMS[room] + ("" if writable else " · 내국인은 교류방에서만 글을 쓸 수 있어요"))
    hot = cm.hot_posts()
    if hot:
        with st.expander(f"🔥 HOT — 최근 {HOT_DAYS}일 추천 {HOT_MIN_LIKES}개 이상 (모든 언어방)", expanded=True):
            for p in hot:
                st.markdown(f"{chip('HOT', 'hot')}{chip(ROOM_SHORT[p['room']])} **{p['title']}** · ❤ {p['likes']}",
                            unsafe_allow_html=True)
    if writable:
        with st.expander("✏️ 글쓰기"):
            write_form(user, room)
    tags = st.pills("주제", TAGS, selection_mode="multi", key="tags")
    show_tr = st.toggle("🌐 번역 보기", key="show_tr",
                        help="시드 글은 번역이 내장돼 있고, 새 글은 번역 API 연결 후 제공 (결정 R2-8)")
    for p in cm.list_posts(room=room, tags=tags or None, limit=40):
        render_post(p, user, show_tr)


# ------------------------------------------------------------------ 시민: 행사·정책 / Q&A
def page_events(user, lang):
    nat = user["nationality"]
    st.markdown("#### 🎉 다가오는 행사")
    for e in cm.list_notices("event"):
        if e["start_date"] < dt.date.today().isoformat():
            continue
        with st.container(border=True):
            mine = chip("내 국적 대상", "off") if nat and nat in e["target"] else ""
            st.markdown(f"{mine}{chip(e['start_date'])}{chip('📍 ' + DONG_NAME.get(e['dong'], ''))} "
                        f"**{tr(e['titles'], lang)}**  \n{tr(e['bodies'], lang)}", unsafe_allow_html=True)
    st.markdown("#### 📜 안산·경기 정책")
    topic = st.pills("주제", ["의료", "노동", "교육", "행정", "비자"], selection_mode="multi", key="pol_topic")
    for e in cm.list_notices("policy"):
        if topic and e["topic"] not in topic:
            continue
        with st.container(border=True):
            st.markdown(f"{chip(e['topic'])} **{tr(e['titles'], lang)}**  \n{tr(e['bodies'], lang)}",
                        unsafe_allow_html=True)
            if e["link"]:
                st.markdown(f"🔗 원문: {e['link']}")
    for e in cm.list_notices("notice"):
        st.info(f"📌 {tr(e['titles'], lang)} — {tr(e['bodies'], lang)}")


def page_qa(user, lang):
    st.markdown("#### ❓ 공식 Q&A — 안산시 담당자가 직접 답해요")
    with st.form("ask", clear_on_submit=True):
        title = st.text_input("질문 제목 (어떤 언어든 괜찮아요)")
        body = st.text_area("내용", height=80)
        if st.form_submit_button("질문하기") and title.strip():
            cm.add_question(user, title.strip(), body.strip())
            st.success("등록했어요. 답변이 달리면 여기서 확인할 수 있어요.")
    for q in cm.list_questions():
        with st.container(border=True):
            status = chip("답변 완료", "off") if q["answer"] else chip("답변 대기", "warn")
            st.markdown(f"{status}<span class='meta'>{ROOM_SHORT.get(q['language'], q['language'])} · "
                        f"{ago(q['created_at'])}</span>  \n**{q['title']}**  \n{q['body']}", unsafe_allow_html=True)
            if q["body_ko"]:
                st.caption(f"🌐 {q['body_ko']}")
            if q["answer"]:
                st.success(f"🏛️ 안산시: {q['answer']}")


# ------------------------------------------------------------------ 관리자
def page_admin_trends():
    st.markdown("#### 📊 국적별 생활권 동향 — DS-028 (2026-01~06)")
    nats = st.multiselect("국적", NATS, default=["중국", "우즈베키스탄", "러시아", "베트남", "카자흐스탄"], key="adm_nats")
    if not nats:
        return
    trend = mt.residence_trend(nats)
    city = trend.groupby(["ETL_YM", "CNTRY_NM"], as_index=False)["POP"].sum()
    base = city[city["ETL_YM"] == MONTHS[0]].set_index("CNTRY_NM")["POP"]
    city["지수(1월=100)"] = city["POP"] / city["CNTRY_NM"].map(base) * 100
    st.altair_chart(alt.Chart(city).mark_line(point=True).encode(
        x=alt.X("ETL_YM:N", title="월"), y=alt.Y("지수(1월=100):Q", scale=alt.Scale(zero=False)),
        color=alt.Color("CNTRY_NM:N", title="국적")), height=260)
    nat1 = st.selectbox("동별로 볼 국적", nats, key="adm_nat1")
    sub = trend[trend["CNTRY_NM"] == nat1].copy()
    sub["비중"] = sub["POP"] / sub.groupby("ETL_YM")["POP"].transform("sum")
    sub["동"] = sub["ADM_CD"].map(DONG_NAME)
    top6 = sub[sub["ETL_YM"] == LATEST_MONTH].nlargest(6, "비중")["동"]
    st.altair_chart(alt.Chart(sub[sub["동"].isin(top6)]).mark_line(point=True).encode(
        x=alt.X("ETL_YM:N", title="월"), y=alt.Y("비중:Q", axis=alt.Axis(format="%")), color="동:N"), height=260)
    source("DS-028 AMD_STAY_F_POP — 장기·20박 이상 POP (표본이라 지수·비중으로만 표시)")


def page_admin_grid():
    st.markdown("#### 🔥 50m 격자 히트맵 — 관리자 전용")
    st.markdown(chip("DS-032 GG_50CELL_FLOWPOP · 대민개방 X · 시민 화면은 행정동 단위만", "warn"), unsafe_allow_html=True)
    c = st.columns(3)
    nat = c[0].selectbox("국적", mt.grid_nations(), format_func=lambda n: f"{NAT_FLAG.get(n, '')} {n}", key="grid_nat")
    day = c[1].radio("대표일", ["20260610", "20260613"], horizontal=True, key="grid_day",
                     format_func=lambda d: "평일 6/10(수)" if d.endswith("10") else "주말 6/13(토)")
    hour = c[2].radio("시각", [14, 21], horizontal=True, format_func=lambda h: f"{h}시", key="grid_hour")
    cells = mt.grid_cells(nat, day, hour)
    outline = {"type": "FeatureCollection", "features": [f for *_, f in load_dong_features()]}
    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=pdk.ViewState(latitude=37.318, longitude=126.83, zoom=11.6),
                             layers=[pdk.Layer("GeoJsonLayer", outline, stroked=True, filled=False,
                                               get_line_color=[90, 90, 90, 160], line_width_min_pixels=1),
                                     pdk.Layer("HeatmapLayer", cells, get_position=["lon", "lat"], get_weight="POP",
                                               radius_pixels=28, aggregation="SUM")]), height=520)
    top = cells.groupby("ADMI_NM")["POP"].sum().sort_values(ascending=False).head(6)
    st.markdown("밀집 상위 동: " + " · ".join(f"**{k}** {v / top.sum() * 100:.0f}%" for k, v in top.items()))
    source(f"DS-032 — 2026-06 대표일 · 셀별 성·연령 컬럼 합 · 셀 {len(cells):,}개")


def page_admin_events():
    st.markdown("#### 🎪 행사 기획 추천 — DS-017 주말 시간대")
    targets = st.multiselect("대상 국적", NATS, default=["우즈베키스탄", "카자흐스탄", "러시아"], key="ev_targets")
    if targets:
        rec = mt.event_recommend(targets)
        st.dataframe(pd.DataFrame({"순위": range(1, len(rec) + 1), "동": rec["ADM_NM"],
                                   "시작 시각": rec["TIMEZN_CD"].astype(str) + "시",
                                   "대상 국적 평균 비중": (rec["share"] * 100).round(1)}), hide_index=True)
        heat = mt.gathering_hourly(targets)
        heat = heat[heat["TIMEZN_CD"].between(8, 23)]
        order = heat.groupby("ADM_NM")["share"].sum().sort_values(ascending=False).head(12).index.tolist()
        st.altair_chart(alt.Chart(heat[heat["ADM_NM"].isin(order)]).mark_rect().encode(
            x=alt.X("TIMEZN_CD:O", title="시각"), y=alt.Y("ADM_NM:N", sort=order, title=None),
            color=alt.Color("share:Q", title="비중", scale=alt.Scale(scheme="oranges"))), height=320)
        source("DS-017 AMD_INFLOW_TIME_F_POP — 2026-06 주말 일평균, 국적별 비중의 평균")
    st.divider()
    st.markdown("#### 📈 지난 행사 효과 — DS-042 · DS-061")
    st.markdown(chip("수시 제공 · 현재 미납품 → 시가 관심지역을 지정해 산출 요청 필요", "warn"), unsafe_allow_html=True)
    ev = st.selectbox("행사", EVENTS, key="ev_sel",
                      format_func=lambda e: f"{e['name']} · {e['dong']} · {e['days'][0][4:6]}/{e['days'][0][6:]}")
    st.metric("외국인 방문 증가율 (행사일 ÷ 1·2주 전 같은 요일)", f"{mt.event_total_uplift(ev['area_cd']):+.0f}%")
    eff = mt.event_visit_effect(ev["area_cd"])
    st.altair_chart(alt.Chart(eff).mark_bar().encode(
        x=alt.X("증가율(%):Q"), y=alt.Y("국적:N", sort="-x", title=None),
        color=alt.Color("대상 국적:N", scale=alt.Scale(domain=[True, False], range=["#9333EA", "#CBD5E1"]))), height=380)
    st.dataframe(mt.event_sales_effect(ev["area_cd"]), hide_index=True)
    source("DS-042 EVENT_VISIT_NATIONALITY · DS-061 EVENT_SALES_NATIONALITY — 금액은 증가율로만")


def page_admin_migration():
    st.markdown("#### 🧭 외국인 전입·정착 모니터링 — DS-076 · DS-077")
    st.markdown(chip("신용정보 보유 '우량외국인'만 집계 → 실제보다 적게 잡힘", "warn"), unsafe_allow_html=True)
    mig = mt.migration_by_month()
    month = st.select_slider("기준월", MONTHS, value=LATEST_MONTH, key="mig_month")
    cur = mig[mig["month"] == month].sort_values("순유입", ascending=False)
    st.altair_chart(alt.Chart(cur).mark_bar().encode(
        x=alt.X("순유입:Q"), y=alt.Y("ADM_NM:N", sort="-x", title=None),
        color=alt.condition("datum.순유입 > 0", alt.value("#2563EB"), alt.value("#DC2626"))), height=460)
    na = mt.new_arrivals(month)
    st.markdown("**정착 지원 우선 동** (해외·정보없음에서 온 외국인 전입 상위): "
                + " · ".join(f"{r.ADM_NM} {int(r.abroad)}" for r in na.nlargest(5, "abroad").itertuples()))
    source("DS-076 GYEONGGI_IN_STAT · DS-077 GYEONGGI_OUT_STAT — FOR_CNT")


def page_admin_ops(user):
    st.markdown("#### ❓ 공식 Q&A 답변 대기")
    pending = cm.list_questions(pending_only=True)
    if not pending:
        st.caption("대기 중인 질문이 없어요.")
    for q in pending:
        with st.container(border=True):
            st.markdown(f"**{q['title']}** <span class='meta'>{ROOM_SHORT.get(q['language'], '')} · {ago(q['created_at'])}</span>"
                        f"  \n{q['body']}", unsafe_allow_html=True)
            if q["body_ko"]:
                st.caption(f"🌐 {q['body_ko']}")
            with st.form(f"ans_{q['id']}", clear_on_submit=True):
                answer = st.text_area("답변", height=70)
                if st.form_submit_button("답변 등록") and answer.strip():
                    cm.answer_question(q["id"], user["id"], answer.strip())
                    st.rerun()
    st.markdown("#### 🚨 신고 처리 (언어방 리더 1차 검토 → 시 최종 판단)")
    for p in cm.reported_posts():
        with st.container(border=True):
            state = chip("숨김", "hot") if p["is_hidden"] else chip("노출 중")
            st.markdown(f"{state}{chip(ROOM_SHORT[p['room']])} **{p['title']}** · 신고 {p['n_reports']}건 "
                        f"({', '.join(sorted(set(p['reasons'])))})  \n{p['body']}", unsafe_allow_html=True)
            if p["body_ko"]:
                st.caption(f"🌐 {p['body_ko']}")
            b = st.columns(3)
            b[0].button("숨김 유지", key=f"h_{p['id']}", on_click=act, args=(cm.resolve_reports, p["id"], "hide"))
            b[1].button("복구", key=f"r_{p['id']}", on_click=act, args=(cm.resolve_reports, p["id"], "restore"))
            b[2].button("삭제", key=f"d_{p['id']}", on_click=act, args=(cm.resolve_reports, p["id"], "delete"))


def page_admin_kpi():
    st.markdown("#### 📈 성과지표 (KPI)")
    k = cm.kpi_snapshot()
    arrivals = mt.arrivals_by_month()
    members = pd.DataFrame(k["members_by_month"])
    members["month"] = members["ym"].str.replace("-", "")
    rate = arrivals.merge(members, on="month", how="left").fillna(0)
    rate["가입률(%)"] = (rate["n"] / rate["FOR_CNT"] * 100).round(1)
    latest_event = EVENTS[-1]
    c = st.columns(4)
    c[0].metric("전입 외국인 대비 가입률 (2026-06)", f"{rate['가입률(%)'].iloc[-1]:.1f}%")
    c[1].metric("최근 30일 활동 사용자", f"{k['active_30d']}명", help="글·댓글·추천 활동자 (시드 가상값)")
    c[2].metric(f"행사 방문 증가율 ({latest_event['name'][:8]}…)", f"{mt.event_total_uplift(latest_event['area_cd']):+.0f}%")
    answered = f"{k['n_answered']}/{k['n_questions']}"
    hours = f" · 평균 {k['avg_answer_hours']:.0f}시간" if k["avg_answer_hours"] else ""
    c[3].metric("공식 Q&A 응답", answered + hours)
    st.altair_chart(alt.Chart(rate).mark_bar().encode(x=alt.X("month:N", title="월"), y="가입률(%):Q"), height=220)
    source("가입률 = 회원 DB 월 신규가입(시드) ÷ DS-076 외국인 전입 FOR_CNT · 행사 = DS-042")


# ------------------------------------------------------------------ main
if SHARE_MODE and "db_path" not in st.session_state:
    st.session_state["db_path"] = str(Path(tempfile.gettempdir()) / f"ansan_neighbors_{uuid.uuid4().hex}.db")
cm.use_db(st.session_state.get("db_path"))
cm.init_db()
with st.sidebar:
    st.markdown("## 🏘️ 안산 이웃")
    st.caption("외국인 주민 커뮤니티 × 국적별 생활권 지도 — 2안 v2 초안")
    persona = st.selectbox("시연 사용자 (로그인 대신)", list(PERSONAS), format_func=lambda k: PERSONAS[k]["label"],
                           key="persona")
    user = cm.user_by_persona(persona)
    langs = list(OFFICIAL_LANGS)
    lang = st.selectbox("공식글 표시 언어", langs, format_func=OFFICIAL_LANGS.get, key=f"lang_{persona}",
                        index=langs.index(user["language"]) if user["language"] in langs else 0)
    st.divider()
    st.markdown("**사용 데이터셋 (모두 가상)**  \nDS-028 · DS-017 · DS-056 · DS-076/077 · DS-032 · DS-042/061")
    st.caption("근거: docs/기획_결정사항.md · 남은 일: docs/진행상황.md")
    if st.button("커뮤니티 DB 초기화"):
        cm.init_db(reset=True)
        st.rerun()

badge()
if SHARE_MODE:
    st.caption("공유 시연판 · 안산시 공식 서비스가 아닙니다 · 글·댓글·신고는 이 브라우저 세션에만 저장되고 "
               "새로고침하면 처음 상태로 돌아갑니다.")
if user["role"] == "official":
    tabs = st.tabs(["📊 생활권 동향", "🔥 격자 히트맵", "🎪 행사 기획·효과", "🧭 전입·정착", "🛠️ Q&A·신고", "📈 KPI"])
    with tabs[0]:
        page_admin_trends()
    with tabs[1]:
        page_admin_grid()
    with tabs[2]:
        page_admin_events()
    with tabs[3]:
        page_admin_migration()
    with tabs[4]:
        page_admin_ops(user)
    with tabs[5]:
        page_admin_kpi()
else:
    tabs = st.tabs(["🏠 내 생활권", "🗺️ 생활권 지도", "💬 커뮤니티", "🎉 행사·정책", "❓ 공식 Q&A"])
    with tabs[0]:
        page_home(user, lang)
    with tabs[1]:
        page_map(user, lang)
    with tabs[2]:
        page_community(user, lang)
    with tabs[3]:
        page_events(user, lang)
    with tabs[4]:
        page_qa(user, lang)
