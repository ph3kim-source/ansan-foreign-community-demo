"""커뮤니티(SQLite) — 언어방 익명 게시판·장터·사업장 후기·모임·장소 제보·공지·공식 Q&A.

결정 근거: docs/기획_결정사항.md R2·R3·R5·R6
- 익명 + 국적/언어 배지, 언어방 + 주제 태그
- 내국인은 교류방에서만 글쓰기
- 사업장 후기는 업종·동 단위 (사업장명 입력 차단)
- 장터는 물품만, 신고 REPORT_HIDE_THRESHOLD건이면 자동 숨김
"""
import datetime as dt
import json
import random
import re
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pandas as pd

from config import (ANCHOR_DONG_CSV, BUSINESS_NAME_PATTERNS, DB_PATH, DONG_CODE, HOT_DAYS,
                    HOT_MIN_LIKES, NAT_NAMES, NATIONS, PERSONAS, PLACE_VERIFY_UPVOTES,
                    REPORT_HIDE_THRESHOLD)
from geo import dong_centroids

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, persona_key TEXT UNIQUE, nationality TEXT,
  language TEXT, dong TEXT, role TEXT, is_korean INTEGER DEFAULT 0, joined_at TEXT);
CREATE TABLE IF NOT EXISTS posts(id INTEGER PRIMARY KEY, room TEXT, tag TEXT, title TEXT, body TEXT,
  body_ko TEXT, author_id INTEGER, dong TEXT, created_at TEXT, extra TEXT, is_hidden INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS comments(id INTEGER PRIMARY KEY, post_id INTEGER, author_id INTEGER,
  body TEXT, body_ko TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS likes(post_id INTEGER, user_id INTEGER, PRIMARY KEY(post_id, user_id));
CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY, post_id INTEGER, user_id INTEGER, reason TEXT,
  created_at TEXT, status TEXT DEFAULT 'open', UNIQUE(post_id, user_id));
CREATE TABLE IF NOT EXISTS places(id INTEGER PRIMARY KEY, name TEXT, category TEXT, dong TEXT, lon REAL,
  lat REAL, source TEXT, created_by INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS place_votes(place_id INTEGER, user_id INTEGER, PRIMARY KEY(place_id, user_id));
CREATE TABLE IF NOT EXISTS notices(id INTEGER PRIMARY KEY, kind TEXT, topic TEXT, titles TEXT, bodies TEXT,
  target TEXT, dong TEXT, start_date TEXT, link TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS questions(id INTEGER PRIMARY KEY, author_id INTEGER, language TEXT, title TEXT,
  body TEXT, body_ko TEXT, created_at TEXT, answer TEXT, answered_by INTEGER, answered_at TEXT);
"""


def _now() -> dt.datetime:
    return dt.datetime.now().replace(microsecond=0)


def _ts(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


_LOCAL = threading.local()


def use_db(path=None) -> None:
    """이 스크립트 실행(스레드)에서 쓸 DB 경로. None이면 기본 DB_PATH.

    공유 시연판은 방문자 세션마다 임시 DB를 쓰므로 app.py가 매 실행·콜백 시작 때 호출한다.
    """
    _LOCAL.path = Path(path) if path else None


def db_path() -> Path:
    return getattr(_LOCAL, "path", None) or DB_PATH


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(db_path())
    con.row_factory = sqlite3.Row
    return con


def _rows(sql: str, params=()) -> list:
    with closing(connect()) as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def _exec(sql: str, params=()) -> int:
    with closing(connect()) as con:
        cur = con.execute(sql, params)
        con.commit()
        return cur.lastrowid


def init_db(reset: bool = False) -> None:
    if reset and db_path().exists():
        db_path().unlink()
    with closing(connect()) as con:
        con.executescript(SCHEMA)
        empty = con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    if empty:
        _seed()


# ------------------------------------------------------------------ 규칙
def contains_business_name(text: str) -> bool:
    return any(re.search(p, text or "", flags=re.IGNORECASE) for p in BUSINESS_NAME_PATTERNS)


def can_write(user: dict, room: str) -> bool:
    return room == "exchange" or not user.get("is_korean")


# ------------------------------------------------------------------ 조회
def user_by_persona(key: str) -> dict:
    return _rows("SELECT * FROM users WHERE persona_key=?", (key,))[0]


def list_posts(room: str = None, tags=None, dongs=None, include_hidden: bool = False,
               since_days: int = None, limit: int = 60) -> list:
    sql = """SELECT p.*, u.nationality, u.language, u.is_korean,
             (SELECT COUNT(*) FROM likes l WHERE l.post_id=p.id) AS likes,
             (SELECT COUNT(*) FROM comments c WHERE c.post_id=p.id) AS n_comments,
             (SELECT COUNT(*) FROM reports r WHERE r.post_id=p.id AND r.status='open') AS n_reports
             FROM posts p JOIN users u ON u.id=p.author_id WHERE 1=1"""
    params = []
    if room:
        sql += " AND p.room=?"
        params.append(room)
    if tags:
        sql += " AND p.tag IN (%s)" % ",".join("?" * len(tags))
        params += list(tags)
    if dongs:
        sql += " AND p.dong IN (%s)" % ",".join("?" * len(dongs))
        params += list(dongs)
    if not include_hidden:
        sql += " AND p.is_hidden=0"
    if since_days:
        sql += " AND p.created_at >= ?"
        params.append(_ts(_now() - dt.timedelta(days=since_days)))
    sql += " ORDER BY p.created_at DESC LIMIT ?"
    params.append(limit)
    rows = _rows(sql, params)
    for r in rows:
        r["extra"] = json.loads(r["extra"]) if r["extra"] else {}
    return rows


def hot_posts(limit: int = 5) -> list:
    rows = list_posts(since_days=HOT_DAYS, limit=200)
    rows = [r for r in rows if r["likes"] >= HOT_MIN_LIKES]
    return sorted(rows, key=lambda r: (-r["likes"], r["created_at"]))[:limit]


def list_comments(post_id: int, post_author: int) -> list:
    rows = _rows("SELECT * FROM comments WHERE post_id=? ORDER BY created_at", (post_id,))
    anon = {}
    for r in rows:
        if r["author_id"] == post_author:
            r["alias"] = "익명(글쓴이)"
        else:
            anon.setdefault(r["author_id"], len(anon) + 1)
            r["alias"] = f"익명{anon[r['author_id']]}"
    return rows


def has_liked(post_id: int, user_id: int) -> bool:
    return bool(_rows("SELECT 1 FROM likes WHERE post_id=? AND user_id=?", (post_id, user_id)))


# ------------------------------------------------------------------ 쓰기
def add_post(user: dict, room: str, tag: str, title: str, body: str, dong: str, extra: dict = None) -> int:
    if not can_write(user, room):
        raise PermissionError("내국인은 교류방에서만 글을 쓸 수 있습니다.")
    if tag == "사업장후기" and (contains_business_name(title) or contains_business_name(body)):
        raise ValueError("사업장 후기에는 사업장명·법인명을 쓸 수 없습니다 (업종·동 단위로만).")
    return _exec("INSERT INTO posts(room, tag, title, body, body_ko, author_id, dong, created_at, extra)"
                 " VALUES(?,?,?,?,?,?,?,?,?)",
                 (room, tag, title, body, None, user["id"], dong, _ts(_now()),
                  json.dumps(extra or {}, ensure_ascii=False)))


def add_comment(user: dict, post_id: int, body: str) -> int:
    return _exec("INSERT INTO comments(post_id, author_id, body, body_ko, created_at) VALUES(?,?,?,?,?)",
                 (post_id, user["id"], body, None, _ts(_now())))


def toggle_like(post_id: int, user_id: int) -> None:
    if has_liked(post_id, user_id):
        _exec("DELETE FROM likes WHERE post_id=? AND user_id=?", (post_id, user_id))
    else:
        _exec("INSERT INTO likes(post_id, user_id) VALUES(?,?)", (post_id, user_id))


def report_post(post_id: int, user_id: int, reason: str) -> bool:
    """신고. 누적 REPORT_HIDE_THRESHOLD건이면 자동 숨김 → True."""
    with closing(connect()) as con:
        con.execute("INSERT OR IGNORE INTO reports(post_id, user_id, reason, created_at) VALUES(?,?,?,?)",
                    (post_id, user_id, reason, _ts(_now())))
        n = con.execute("SELECT COUNT(*) FROM reports WHERE post_id=? AND status='open'", (post_id,)).fetchone()[0]
        hidden = n >= REPORT_HIDE_THRESHOLD
        if hidden:
            con.execute("UPDATE posts SET is_hidden=1 WHERE id=?", (post_id,))
        con.commit()
    return hidden


def reported_posts() -> list:
    rows = list_posts(include_hidden=True, limit=500)
    rows = [r for r in rows if r["n_reports"] > 0]
    for r in rows:
        r["reasons"] = [x["reason"] for x in _rows(
            "SELECT reason FROM reports WHERE post_id=? AND status='open'", (r["id"],))]
    return rows


def resolve_reports(post_id: int, action: str) -> None:
    """action: 'hide'(숨김 유지) | 'restore'(복구) | 'delete'(삭제)."""
    with closing(connect()) as con:
        if action == "delete":
            con.execute("DELETE FROM posts WHERE id=?", (post_id,))
        else:
            con.execute("UPDATE posts SET is_hidden=? WHERE id=?", (1 if action == "hide" else 0, post_id))
        con.execute("UPDATE reports SET status='closed' WHERE post_id=?", (post_id,))
        con.commit()


# ------------------------------------------------------------------ 장소
def list_places() -> list:
    rows = _rows("""SELECT p.*, (SELECT COUNT(*) FROM place_votes v WHERE v.place_id=p.id) AS votes
                    FROM places p ORDER BY votes DESC""")
    for r in rows:
        r["verified"] = r["source"] == "official" or r["votes"] >= PLACE_VERIFY_UPVOTES
    return rows


def add_place(user: dict, name: str, category: str, dong: str) -> int:
    lon, lat = dong_centroids()[dong]
    rnd = random.Random(name)
    return _exec("INSERT INTO places(name, category, dong, lon, lat, source, created_by, created_at)"
                 " VALUES(?,?,?,?,?,?,?,?)",
                 (name, category, dong, lon + rnd.uniform(-0.002, 0.002), lat + rnd.uniform(-0.002, 0.002),
                  "community", user["id"], _ts(_now())))


def vote_place(place_id: int, user_id: int) -> None:
    _exec("INSERT OR IGNORE INTO place_votes(place_id, user_id) VALUES(?,?)", (place_id, user_id))


# ------------------------------------------------------------------ 공지·정책·행사
def list_notices(kind: str = None) -> list:
    rows = _rows("SELECT * FROM notices" + (" WHERE kind=?" if kind else "") + " ORDER BY start_date, id",
                 (kind,) if kind else ())
    for r in rows:
        r["titles"], r["bodies"] = json.loads(r["titles"]), json.loads(r["bodies"])
        r["target"] = json.loads(r["target"]) if r["target"] else []
    return rows


def add_notice(kind: str, topic: str, titles: dict, bodies: dict, target: list, dong: str,
               start_date: str, link: str = "") -> int:
    return _exec("INSERT INTO notices(kind, topic, titles, bodies, target, dong, start_date, link, created_at)"
                 " VALUES(?,?,?,?,?,?,?,?,?)",
                 (kind, topic, json.dumps(titles, ensure_ascii=False), json.dumps(bodies, ensure_ascii=False),
                  json.dumps(target, ensure_ascii=False), dong, start_date, link, _ts(_now())))


# ------------------------------------------------------------------ 공식 Q&A
def list_questions(pending_only: bool = False) -> list:
    return _rows("SELECT * FROM questions" + (" WHERE answer IS NULL" if pending_only else "")
                 + " ORDER BY created_at DESC")


def add_question(user: dict, title: str, body: str) -> int:
    return _exec("INSERT INTO questions(author_id, language, title, body, created_at) VALUES(?,?,?,?,?)",
                 (user["id"], user["language"], title, body, _ts(_now())))


def answer_question(qid: int, official_id: int, answer: str) -> None:
    _exec("UPDATE questions SET answer=?, answered_by=?, answered_at=? WHERE id=?",
          (answer, official_id, _ts(_now()), qid))


# ------------------------------------------------------------------ KPI
def kpi_snapshot() -> dict:
    since = _ts(_now() - dt.timedelta(days=30))
    active = _rows("""SELECT COUNT(DISTINCT a) AS n FROM (
                        SELECT author_id AS a FROM posts WHERE created_at>=?
                        UNION SELECT author_id FROM comments WHERE created_at>=?
                        UNION SELECT user_id FROM likes l JOIN posts p ON p.id=l.post_id WHERE p.created_at>=?)""",
                   (since, since, since))[0]["n"]
    posts_by_room = _rows("SELECT room, COUNT(*) AS n FROM posts WHERE created_at>=? GROUP BY room", (since,))
    qs = list_questions()
    answered = [q for q in qs if q["answer"]]
    hours = [(dt.datetime.fromisoformat(q["answered_at"]) - dt.datetime.fromisoformat(q["created_at"]))
             .total_seconds() / 3600 for q in answered]
    members = _rows("SELECT substr(joined_at,1,7) AS ym, COUNT(*) AS n FROM users WHERE is_korean=0 "
                    "AND role='resident' GROUP BY ym ORDER BY ym")
    return {"active_30d": active, "posts_by_room": posts_by_room, "members_by_month": members,
            "n_questions": len(qs), "n_answered": len(answered),
            "avg_answer_hours": (sum(hours) / len(hours)) if hours else None,
            "n_members": _rows("SELECT COUNT(*) AS n FROM users WHERE role='resident'")[0]["n"]}


# ------------------------------------------------------------------ 시드 (가상)
_NAT_W = {"중국": 34, "우즈베키스탄": 14, "러시아": 9, "카자흐스탄": 6, "베트남": 10, "인도네시아": 4,
          "네팔": 4, "필리핀": 3, "미얀마": 2, "몽골": 2, "우크라이나": 2, "캄보디아": 2, "방글라데시": 1,
          "키르기스스탄": 1, "태국": 1, "파키스탄": 1, "스리랑카": 1, "미국": 1, "일본": 1, "인도": 1}
_JOIN_PER_MONTH = {"2026-01": 25, "2026-02": 30, "2026-03": 46, "2026-04": 52, "2026-05": 49,
                   "2026-06": 51, "2026-07": 40, "2026-08": 38, "2026-09": 44, "2026-10": 41, "2026-11": 14}
_NAT_LANG = {n[0]: n[3] for n in NATIONS}

_POSTS = [
    # (room, tag, author_nat, dong, hours_ago, likes, title, body, body_ko, extra, comments[(body, ko)], reports)
    ("ru", "자유", "우즈베키스탄", "선부2동", 30, 8, "Где купить нормальную самсу?",
     "Ребята, где в Ансане продают настоящую самсу из тандыра? Скучаю по дому.",
     "안산에서 탄디르 화덕에 구운 진짜 삼사 파는 곳 어디예요? 고향 생각나네요.", {},
     [("В Сонбу-2 у рынка есть узбекская пекарня, очень вкусно.", "선부2동 시장 근처에 우즈벡 빵집 있어요, 정말 맛있어요."),
      ("Подтверждаю, по выходным очередь.", "인정, 주말엔 줄 서요.")], 0),
    ("ru", "고민", "우즈베키스탄", "선부2동", 20, 6, "Только приехал — как найти больницу?",
     "Я в Ансане 2 недели, работаю в Банволе. Куда идти к врачу, если плохо говорю по-корейски?",
     "안산 온 지 2주, 반월에서 일해요. 한국어를 잘 못하는데 병원은 어디로 가야 하나요?", {},
     [("Посмотри «Официальные вопросы» — там отвечали про бесплатный медосмотр.",
       "'공식 Q&A' 보세요 — 무료 건강검진 답변 있었어요."),
      ("В центре поддержки иностранцев есть переводчик на русский.", "외국인 지원기관에 러시아어 통역 있어요.")], 0),
    ("ru", "구인구직", "카자흐스탄", "선부1동", 50, 2, "Ищу работу: сварщик, 5 лет опыта",
     "Есть виза H-2. Ищу работу в Банволе или Сихва, можно ночные смены.",
     "H-2 비자 있어요. 반월·시화 쪽 일 찾아요, 야간 가능.", {}, [], 0),
    ("ru", "사업장후기", "우즈베키스탄", "초지동", 70, 9, "단원구 초지동 · 금속가공 · 주야교대",
     "Зарплату задерживают уже второй месяц. Общежитие — контейнер, зимой холодно.",
     "월급이 두 달째 밀리고 있어요. 기숙사는 컨테이너라 겨울에 추워요.",
     {"industry": "금속가공", "shift": "주야교대", "pay": 1, "safety": 2, "housing": 1},
     [("Обратись в центр консультаций для работников — помогают бесплатно.",
       "근로자 상담기관에 문의해 보세요 — 무료로 도와줘요.")], 0),
    ("ru", "장터", "러시아", "선부2동", 10, 1, "Продаю зимнюю куртку, 20 000 вон",
     "Размер L, носил один сезон. Встреча у станции Сонбу.", "L 사이즈, 한 시즌 입었어요. 선부동 직거래.",
     {"price": 20000, "status": "판매중"}, [], 0),
    ("ru", "모임", "우즈베키스탄", "선부2동", 5, 4, "Футбол в субботу в 15:00",
     "Собираем команду, играем каждую субботу. Пишите в комментарии!", "팀 모아요, 매주 토요일 경기. 댓글 주세요!",
     {"when": "토요일 15:00", "capacity": 14}, [("Я в деле!", "저 할게요!")], 0),
    ("ru", "비자·행정", "카자흐스탄", "본오1동", 90, 3, "Продление F-4: какие документы?",
     "Кто недавно продлевал F-4, какие документы просили?", "최근 F-4 연장하신 분, 어떤 서류 요구했어요?", {}, [], 0),
    ("ru", "비자·행정", "러시아", "원곡동", 3, 0, "Быстрая виза E-7, гарантия 100%",
     "Сделаем визу за 2 недели, оплата вперёд. Пишите в личку.",
     "2주 만에 비자 해드립니다, 선불. 개인 메시지 주세요.", {}, [], 3),
    ("vi", "모임", "베트남", "고잔동", 8, 5, "Họp mặt cộng đồng Việt Nam cuối tuần này",
     "Chủ nhật này 14h ở công viên gần Gojan, mọi người mang món ăn quê nhà nhé!",
     "이번 주 일요일 14시 고잔동 근처 공원, 고향 음식 가져오세요!", {"when": "일요일 14:00", "capacity": 30},
     [("Mình sẽ mang chả giò!", "저는 짜조 가져갈게요!")], 0),
    ("vi", "장터", "베트남", "고잔동", 26, 2, "Bán xe đẩy em bé, còn mới 90%",
     "Giá 50.000 won, ai cần nhắn mình nhé.", "5만 원, 필요하신 분 연락 주세요.",
     {"price": 50000, "status": "예약중"}, [], 0),
    ("vi", "고민", "베트남", "고잔동", 40, 7, "Con sắp vào lớp 1, cần chuẩn bị gì?",
     "Bé nhà mình năm sau vào tiểu học. Các mẹ đã chuẩn bị giấy tờ gì vậy?",
     "아이가 내년에 초등학교 들어가요. 엄마들은 어떤 서류 준비했어요?", {},
     [("Thành phố có thông báo hỗ trợ nhập học, xem mục Chính sách nhé.", "시에서 입학 지원 공지 올렸어요, 정책 탭 보세요.")], 0),
    ("vi", "병원", "베트남", "원곡동", 60, 4, "Bệnh viện nào có phiên dịch tiếng Việt?",
     "Mình đang mang thai, muốn tìm phòng khám có người nói tiếng Việt.",
     "임신 중인데 베트남어 하는 사람이 있는 병원 찾아요.", {}, [], 0),
    ("vi", "자유", "베트남", "원곡동", 100, 6, "Quán phở ngon ở Wongok",
     "Quán phở gần chợ Wongok nước dùng rất ngon, giá 9.000 won.", "원곡동 시장 근처 쌀국수집 국물 맛있어요, 9천 원.", {}, [], 0),
    ("ko", "자유", "중국", "원곡동", 15, 11, "원곡동 주말 시장 요즘 사람 많네요",
     "토요일 오후에 가니 발 디딜 틈이 없더라고요. 다들 어디서 오시는 건지.", None, {},
     [("지도에서 '모이는 곳' 보니까 토요일 오후엔 원곡동이 1등이던데요 ㅋㅋ", None), ("주차는 포기하세요", None)], 0),
    ("ko", "장터", "중국", "원곡동", 12, 3, "귀국 전에 전기밥솥·이불 정리합니다",
     "12월에 들어가요. 밥솥 2만, 이불 1만. 원곡동 직거래만 해요.", None, {"price": 20000, "status": "판매중"}, [], 0),
    ("ko", "구인구직", "중국", "원곡동", 33, 1, "식당 주방보조 구합니다 (원곡동, 주 5일)",
     "중국어 가능하면 우대해요. 근무 10시~20시.", None, {}, [], 0),
    ("ko", "비자·행정", "중국", "백운동", 80, 5, "F-4에서 F-5 전환 조건 아시는 분",
     "소득 요건이 어떻게 되는지 궁금합니다.", None, {},
     [("출입국 공식 안내는 정책 탭의 하이코리아 링크 참고하세요.", None)], 0),
    ("ko", "사업장후기", "중국", "원곡동", 120, 4, "단원구 원곡동 · 식품제조 · 주간",
     "급여 날짜는 정확히 지켜요. 다만 여름엔 작업장이 많이 더워요.", None,
     {"industry": "식품제조", "shift": "주간", "pay": 5, "safety": 3, "housing": 3}, [], 0),
    ("zh", "자유", "중국", "원곡동", 22, 5, "安山哪里能买到正宗火锅底料？",
     "最近特别想吃火锅，求推荐中国超市。", "요즘 훠궈가 너무 먹고 싶어요, 중국 마트 추천해 주세요.", {},
     [("元谷洞有好几家，周末人很多。", "원곡동에 여러 곳 있어요, 주말엔 사람 많아요.")], 0),
    ("zh", "구인구직", "중국", "신길동", 45, 1, "找工作，会开叉车", "有叉车证，希望在半月工业园区工作。",
     "지게차 자격증 있어요, 반월공단 희망해요.", {}, [], 0),
    ("zh", "비자·행정", "중국", "사동", 66, 3, "毕业后D-10签证怎么申请？",
     "在汉阳大学读研，明年毕业，想留在韩国找工作。", "한양대 대학원생인데 내년에 졸업해요, 한국에서 구직하고 싶어요.", {}, [], 0),
    ("zh", "장터", "중국", "사동", 7, 0, "转让二手自行车 3万韩元", "九成新，四洞附近自取。",
     "거의 새것, 사동 근처에서 가져가세요.", {"price": 30000, "status": "판매중"}, [], 0),
    ("en", "병원", "필리핀", "중앙동", 18, 3, "Any English-speaking dentist in Ansan?",
     "Chipped a tooth yesterday. Looking for a dentist who speaks English near Jungang-dong.",
     "어제 이가 깨졌어요. 중앙동 근처 영어 되는 치과 찾아요.", {}, [], 0),
    ("en", "모임", "파키스탄", "호수동", 28, 6, "Cricket on Sunday — players wanted",
     "We play every Sunday morning near Hosu park. All levels welcome!", "매주 일요일 오전 호수공원 근처에서 해요. 실력 무관!",
     {"when": "일요일 09:00", "capacity": 22}, [("Count me in!", "저도 껴주세요!")], 0),
    ("en", "집·생활", "미국", "고잔동", 55, 4, "How do I sort recycling here?",
     "Got a warning note on my door about trash. Is there a guide in English?",
     "문에 쓰레기 분리배출 경고문이 붙었어요. 영어 안내 있나요?", {}, [], 0),
    ("etc", "자유", "네팔", "원곡동", 35, 3, "नेपाली खाना पाइने पसल कहाँ छ?",
     "वनगोक-दोङ नजिक नेपाली मसला किन्न पाइन्छ?", "원곡동 근처에서 네팔 향신료 살 수 있나요?", {}, [], 0),
    ("etc", "고민", "인도네시아", "원곡동", 14, 5, "Masjid terdekat dari Wongok?",
     "Saya baru pindah ke Ansan. Ada yang tahu tempat sholat Jumat dekat Wongok?",
     "안산에 막 이사 왔어요. 원곡동 근처 금요예배 장소 아시는 분?", {},
     [("Ada, dekat stasiun Ansan. Nanti saya kirim lokasinya.", "있어요, 안산역 근처예요. 위치 보내드릴게요.")], 0),
    ("etc", "자유", "몽골", "사동", 75, 2, "Монгол хүнсний дэлгүүр хаана байдаг вэ?",
     "Садон орчимд монгол хоол, хүнс зардаг газар байна уу?", "사동 근처에 몽골 음식·식료품 파는 곳 있나요?", {}, [], 0),
    ("exchange", "모임", None, "선부1동", 16, 5, "한국어 ↔ 러시아어 언어교환 하실 분",
     "선부동 사는 한국인이에요. 러시아어 배우고 싶어요. 카페에서 주 1회 어때요?", None,
     {"when": "평일 저녁", "capacity": 4},
     [("Я хочу! Мне нужен корейский для работы.", "저요! 일 때문에 한국어가 필요해요.")], 0),
    ("exchange", "모임", "베트남", "고잔동", 46, 4, "다문화 가정 이웃 모임 – 아이들 같이 놀아요",
     "토요일 오전 고잔동 놀이터에서 만나요. 베트남·한국 엄마들 모두 환영해요.", None,
     {"when": "토요일 10:00", "capacity": 20}, [], 0),
    ("exchange", "자유", "미국", "와동", 60, 9, "Thanks to my Korean neighbors!",
     "Moved last week and my neighbors helped carry everything. 감사합니다!",
     "지난주 이사했는데 이웃분들이 짐 옮기는 걸 도와주셨어요. 감사합니다!", {}, [], 0),
]

_PLACES = [
    ("외국인주민 지원기관 (예시)", "공공시설", "원곡동", "official", 0),
    ("다문화가족 지원기관 (예시)", "공공시설", "고잔동", "official", 0),
    ("보건소 (예시)", "공공시설", "사동", "official", 0),
    ("보건소 (예시)", "공공시설", "고잔동", "official", 0),
    ("출입국 민원 안내 (예시)", "공공시설", "중앙동", "official", 0),
    ("외국인 근로자 상담 (예시)", "공공시설", "원곡동", "official", 0),
    ("우즈벡 빵집", "음식", "선부2동", "community", 8),
    ("러시아 식료품점", "식료품", "선부2동", "community", 5),
    ("베트남 쌀국수집", "음식", "원곡동", "community", 12),
    ("중국 식자재 마트", "식료품", "원곡동", "community", 9),
    ("할랄 정육점", "식료품", "원곡동", "community", 6),
    ("네팔 향신료 가게", "식료품", "원곡동", "community", 3),
    ("몽골 식료품점", "식료품", "사동", "community", 2),
    ("해외송금 창구", "생활", "원곡동", "community", 7),
    ("러시아어 가능 휴대폰 매장", "생활", "선부1동", "community", 4),
]


def _t(ko, en, zh, ru, vi):
    return {"ko": ko, "en": en, "zh": zh, "ru": ru, "vi": vi}


_NOTICES = [
    ("policy", "의료", _t("외국인 주민 무료 건강검진 (가상 공지)", "Free health check-up for foreign residents (sample)",
                       "外国居民免费体检（示例）", "Бесплатный медосмотр для иностранных жителей (пример)",
                       "Khám sức khỏe miễn phí cho cư dân nước ngoài (mẫu)"),
     _t("매월 둘째·넷째 일요일, 외국인주민 지원기관에서 무료 건강검진을 받을 수 있습니다. 러시아어·베트남어·중국어 통역 지원.",
        "Free check-ups on the 2nd and 4th Sunday of each month at the foreign resident support center. Interpreters available.",
        "每月第二、第四个星期日可在外国居民支援机构免费体检，提供中文、俄语、越南语翻译。",
        "Во 2-е и 4-е воскресенье месяца — бесплатный медосмотр в центре поддержки иностранцев. Есть переводчики.",
        "Chủ nhật tuần 2 và tuần 4 hằng tháng khám miễn phí tại trung tâm hỗ trợ cư dân nước ngoài. Có phiên dịch."),
     [], "원곡동", "2026-11-08", ""),
    ("policy", "노동", _t("임금체불·산재 무료 상담 안내 (가상 공지)", "Free consultation on unpaid wages and work injuries (sample)",
                       "拖欠工资·工伤免费咨询（示例）", "Бесплатная консультация: невыплата зарплаты и травмы (пример)",
                       "Tư vấn miễn phí về nợ lương và tai nạn lao động (mẫu)"),
     _t("임금체불·산재 상담은 외국인 근로자 상담기관과 고용노동부 고객상담센터(☎1350)에서 무료로 받을 수 있습니다. 사업장 후기 게시판에서 바로 연결됩니다.",
        "Get free help at the migrant worker counseling center or the Ministry of Employment and Labor call center (1350).",
        "可在外国劳动者咨询机构或雇佣劳动部咨询中心（1350）免费咨询。",
        "Бесплатная помощь — в центре консультаций для работников или по номеру Министерства труда 1350.",
        "Được tư vấn miễn phí tại trung tâm tư vấn lao động nước ngoài hoặc tổng đài Bộ Lao động 1350."),
     [], "원곡동", "2026-11-01", ""),
    ("policy", "교육", _t("다문화가정 자녀 초등 입학 준비 지원 (가상 공지)",
                       "School-entry support for children of multicultural families (sample)",
                       "多文化家庭子女入学准备支援（示例）",
                       "Помощь при поступлении в школу детям из многокультурных семей (пример)",
                       "Hỗ trợ chuẩn bị nhập học cho con em gia đình đa văn hóa (mẫu)"),
     _t("2027학년도 초등 입학 예정 자녀가 있는 다문화가정에 입학 안내서(5개 언어)와 준비물 꾸러미를 드립니다.",
        "Families with children entering elementary school in 2027 receive a 5-language guide and a supply kit.",
        "向2027学年入学的多文化家庭子女提供5种语言入学指南和学习用品。",
        "Семьям с детьми, идущими в школу в 2027 году, — памятка на 5 языках и набор принадлежностей.",
        "Gia đình có con vào tiểu học năm 2027 được nhận sách hướng dẫn 5 thứ tiếng và bộ đồ dùng học tập."),
     ["베트남", "중국", "필리핀"], "고잔동", "2026-11-05", ""),
    ("policy", "교육", _t("외국인 주민 한국어 교실 수강생 모집 (가상 공지)", "Korean classes for foreign residents — enrollment open (sample)",
                       "外国居民韩语班招生（示例）", "Набор на курсы корейского языка для иностранцев (пример)",
                       "Tuyển học viên lớp tiếng Hàn cho cư dân nước ngoài (mẫu)"),
     _t("평일 저녁·주말반 운영, 수강료 무료. 언어방 리더에게 문의하거나 공식 Q&A에 질문하세요.",
        "Weekday evening and weekend classes, free of charge.", "开设平日晚班和周末班，免费。",
        "Вечерние и выходные группы, бесплатно.", "Lớp tối các ngày trong tuần và cuối tuần, miễn phí."),
     [], "중앙동", "2026-11-03", ""),
    ("policy", "행정", _t("새로 온 외국인 주민 정착 안내 꾸러미 (가상 공지)", "Welcome kit for newly arrived foreign residents (sample)",
                       "新迁入外国居民定居指南（示例）", "Пакет для новых жителей-иностранцев (пример)",
                       "Bộ hướng dẫn định cư cho cư dân nước ngoài mới đến (mẫu)"),
     _t("전입 후 체류지 변경 신고, 건강보험, 쓰레기 배출, 병원 이용법을 5개 언어로 정리했습니다. 행정 접점(주민센터·지원기관) QR로도 받을 수 있어요.",
        "Address change report, health insurance, trash rules and clinics — in 5 languages. Also via QR at community centers.",
        "整理了迁入后滞留地变更申报、健康保险、垃圾分类、就医方法（5种语言）。",
        "Смена адреса, медстраховка, мусор, больницы — на 5 языках. Также по QR-коду в центрах.",
        "Khai báo đổi nơi cư trú, bảo hiểm y tế, phân loại rác, khám bệnh — bằng 5 thứ tiếng."),
     [], "원곡동", "2026-10-28", ""),
    ("policy", "비자", _t("체류기간 연장·체류자격 변경 — 하이코리아 원문 (중앙정부)",
                       "Extension of stay / change of status — see HiKorea (central gov.)",
                       "延长滞留期间·变更滞留资格 — 请见HiKorea（中央政府）",
                       "Продление пребывания и смена статуса — HiKorea (центральное правительство)",
                       "Gia hạn lưu trú, đổi tư cách lưu trú — xem HiKorea (chính phủ trung ương)"),
     _t("중앙정부 정책은 원문 링크만 제공합니다. 외국인종합안내센터 ☎1345.", "Central-government policies are linked only. Call 1345.",
        "中央政府政策仅提供原文链接。外国人综合咨询中心 1345。", "Политика центрального правительства — только ссылка. Справки: 1345.",
        "Chính sách trung ương chỉ cung cấp liên kết. Tổng đài 1345."),
     [], "", "2026-10-20", "https://www.hikorea.go.kr"),
    ("event", "행사", _t("김장 나눔 다문화 한마당 (가상)", "Kimjang sharing festival (sample)", "腌泡菜分享多文化活动（示例）",
                       "Праздник кимчхи для всех жителей (пример)", "Ngày hội chia sẻ kim chi đa văn hóa (mẫu)"),
     _t("함께 김장하고 나눠요. 장소: 원곡동 일대.", "Make and share kimchi together.", "一起腌泡菜、一起分享。",
        "Готовим и делим кимчхи вместе.", "Cùng làm và chia sẻ kim chi."),
     [], "원곡동", "2026-11-21", ""),
    ("event", "행사", _t("외국인 주민 취업 박람회 (가상)", "Job fair for foreign residents (sample)", "外国居民招聘会（示例）",
                       "Ярмарка вакансий для иностранных жителей (пример)", "Hội chợ việc làm cho cư dân nước ngoài (mẫu)"),
     _t("반월·시화 산단 기업 참여 예정.", "Companies from Banwol-Sihwa industrial complex.", "半月·始华工业园区企业参加。",
        "Участвуют предприятия промзоны Банвол-Сихва.", "Có doanh nghiệp khu công nghiệp Banwol-Sihwa tham gia."),
     [], "초지동", "2026-11-28", ""),
    ("event", "행사", _t("베트남 가족 송년 모임 (가상)", "Vietnamese families' year-end gathering (sample)", "越南家庭年终聚会（示例）",
                       "Новогодняя встреча вьетнамских семей (пример)", "Tiệc tất niên gia đình Việt Nam (mẫu)"),
     _t("가족 단위 참가 환영.", "Families welcome.", "欢迎家庭参加。", "Приглашаются семьи.", "Chào đón các gia đình."),
     ["베트남"], "고잔동", "2026-12-12", ""),
    ("event", "행사", _t("러시아어권 주민 새해 바자회 (가상)", "New Year bazaar for Russian-speaking residents (sample)",
                       "俄语圈居民新年义卖会（示例）", "Новогодний базар для русскоязычных жителей (пример)",
                       "Hội chợ năm mới cho cư dân nói tiếng Nga (mẫu)"),
     _t("고려인·중앙아시아 주민 누구나.", "Open to all Russian-speaking neighbors.", "欢迎所有俄语圈居民。",
        "Для всех русскоязычных соседей.", "Dành cho mọi cư dân nói tiếng Nga."),
     ["우즈베키스탄", "러시아", "카자흐스탄", "키르기스스탄"], "선부2동", "2026-12-19", ""),
    ("notice", "규칙", _t("커뮤니티 규칙 — 사업장명 금지·물품만 거래·혐오 표현 금지",
                       "Community rules — no company names in reviews, goods only in market, no hate speech",
                       "社区规则——评价禁止写公司名、跳蚤市场仅限物品、禁止仇恨言论",
                       "Правила: без названий компаний в отзывах, на барахолке только вещи, без языка вражды",
                       "Quy định: không ghi tên công ty, chợ chỉ bán đồ vật, không phát ngôn thù ghét"),
     _t("신고 3건이 쌓이면 글이 자동으로 숨겨지고, 시와 언어방 리더가 검토합니다.",
        "Posts with 3 reports are hidden automatically and reviewed.", "累计3次举报的帖子将自动隐藏并接受审核。",
        "Пост с 3 жалобами скрывается и проверяется.", "Bài có 3 báo cáo sẽ tự ẩn và được xem xét."),
     [], "", "2026-10-01", ""),
]

_QUESTIONS = [
    ("우즈베키스탄", "ru", "Где пройти бесплатный медосмотр?",
     "Я работаю 6 дней в неделю. Можно ли пройти медосмотр в воскресенье?", "주 6일 일해요. 일요일에 검진 받을 수 있나요?",
     96, 5, "네. 매월 둘째·넷째 일요일 외국인주민 지원기관에서 무료 검진이 있습니다(러시아어 통역). 정책 탭 '무료 건강검진' 공지를 확인해 주세요. (가상 답변)"),
    ("베트남", "vi", "Đăng ký lớp tiếng Hàn ở đâu?", "Mình muốn học tiếng Hàn buổi tối.",
     "한국어 교실 어디서 신청해요? 저녁반 원해요.", 70, 20, "평일 저녁반이 있습니다. 정책 탭 '한국어 교실' 공지의 신청 방법을 참고하세요. (가상 답변)"),
    ("중국", "ko", "전입 후 외국인등록 주소 변경은 어디서 하나요?", "원곡동으로 이사 왔습니다.", None,
     50, 9, "체류지 변경 신고는 주민센터 또는 하이코리아에서 할 수 있습니다. 기한은 하이코리아 원문을 확인해 주세요. (가상 답변)"),
    ("중국", "zh", "安山市有中文翻译服务吗？", "去医院的时候需要翻译。", "병원 갈 때 통역이 필요해요. 안산시에 중국어 통역 서비스 있나요?",
     2, None, None),
    ("파키스탄", "en", "How can I report unpaid wages without my boss knowing?",
     "I am afraid of losing my job.", "사장 모르게 임금체불을 신고하려면 어떻게 하나요? 일자리를 잃을까 걱정돼요.", 26, None, None),
]


def _seed() -> None:
    rnd = random.Random(20261111)
    now = _now()
    dong = pd.read_csv(ANCHOR_DONG_CSV)
    dong_w = dong[dong["ym"] == "2026-06"].set_index("dong")["foreign_resident"]
    dong_names, dong_weights = list(dong_w.index), list(dong_w.values)
    nats, nat_w = list(_NAT_W), list(_NAT_W.values())
    with closing(connect()) as con:
        for key, p in PERSONAS.items():
            con.execute("INSERT INTO users(persona_key, nationality, language, dong, role, is_korean, joined_at)"
                        " VALUES(?,?,?,?,?,?,?)", (key, p["nationality"], p["language"], p["dong"], p["role"],
                                                   p["is_korean"], p["joined"] + " 09:00:00"))
        for ym, n in _JOIN_PER_MONTH.items():
            for _ in range(n):
                nat = rnd.choices(nats, nat_w)[0]
                lang = ("ko" if rnd.random() < 0.7 else "zh") if nat == "중국" else _NAT_LANG[nat]
                d = DONG_CODE[rnd.choices(dong_names, dong_weights)[0]]
                day = rnd.randint(1, 28 if ym != "2026-11" else max(1, now.day - 1))
                con.execute("INSERT INTO users(nationality, language, dong, role, is_korean, joined_at)"
                            " VALUES(?,?,?,?,?,?)", (nat, lang, d, "resident", 0, f"{ym}-{day:02d} 12:00:00"))
        for _ in range(12):
            con.execute("INSERT INTO users(nationality, language, dong, role, is_korean, joined_at)"
                        " VALUES(?,?,?,?,?,?)", (None, "ko", DONG_CODE[rnd.choice(dong_names)], "resident", 1,
                                                 "2026-08-01 12:00:00"))
        users = [dict(r) for r in con.execute("SELECT * FROM users WHERE persona_key IS NULL").fetchall()]
        by_nat = {}
        for u in users:
            by_nat.setdefault(u["nationality"], []).append(u["id"])
        voters = [u["id"] for u in users]

        def author_for(nat):
            pool = by_nat.get(nat) or by_nat.get(None)
            return rnd.choice(pool)

        for room, tag, nat, dname, hrs, likes, title, body, body_ko, extra, comments, n_rep in _POSTS:
            created = now - dt.timedelta(hours=hrs)
            author = author_for(nat)
            cur = con.execute("INSERT INTO posts(room, tag, title, body, body_ko, author_id, dong, created_at, extra,"
                              " is_hidden) VALUES(?,?,?,?,?,?,?,?,?,?)",
                              (room, tag, title, body, body_ko, author, DONG_CODE[dname], _ts(created),
                               json.dumps(extra, ensure_ascii=False), 1 if n_rep >= REPORT_HIDE_THRESHOLD else 0))
            pid = cur.lastrowid
            for uid in rnd.sample(voters, likes):
                con.execute("INSERT OR IGNORE INTO likes(post_id, user_id) VALUES(?,?)", (pid, uid))
            for i, (c_body, c_ko) in enumerate(comments):
                con.execute("INSERT INTO comments(post_id, author_id, body, body_ko, created_at) VALUES(?,?,?,?,?)",
                            (pid, author_for(nat), c_body, c_ko, _ts(created + dt.timedelta(minutes=40 * (i + 1)))))
            for uid in rnd.sample(voters, n_rep):
                con.execute("INSERT INTO reports(post_id, user_id, reason, created_at) VALUES(?,?,?,?)",
                            (pid, uid, "비자 브로커·선불 요구", _ts(created + dt.timedelta(minutes=30))))
        centroids = dong_centroids()
        for name, cat, dname, source, votes in _PLACES:
            lon, lat = centroids[DONG_CODE[dname]]
            cur = con.execute("INSERT INTO places(name, category, dong, lon, lat, source, created_by, created_at)"
                              " VALUES(?,?,?,?,?,?,?,?)",
                              (name, cat, DONG_CODE[dname], lon + rnd.uniform(-0.003, 0.003),
                               lat + rnd.uniform(-0.003, 0.003), source, None, _ts(now - dt.timedelta(days=20))))
            for uid in rnd.sample(voters, votes):
                con.execute("INSERT OR IGNORE INTO place_votes(place_id, user_id) VALUES(?,?)", (cur.lastrowid, uid))
        for kind, topic, titles, bodies, target, dname, start, link in _NOTICES:
            con.execute("INSERT INTO notices(kind, topic, titles, bodies, target, dong, start_date, link, created_at)"
                        " VALUES(?,?,?,?,?,?,?,?,?)",
                        (kind, topic, json.dumps(titles, ensure_ascii=False), json.dumps(bodies, ensure_ascii=False),
                         json.dumps(target, ensure_ascii=False), DONG_CODE.get(dname, ""), start, link,
                         _ts(now - dt.timedelta(days=10))))
        official = con.execute("SELECT id FROM users WHERE persona_key='official'").fetchone()[0]
        for nat, lang, title, body, body_ko, hrs, ans_after, answer in _QUESTIONS:
            created = now - dt.timedelta(hours=hrs)
            answered_at = _ts(created + dt.timedelta(hours=ans_after)) if ans_after else None
            con.execute("INSERT INTO questions(author_id, language, title, body, body_ko, created_at, answer, answered_by,"
                        " answered_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (author_for(nat), lang, title, body, body_ko, _ts(created), answer,
                         official if answer else None, answered_at))
        con.commit()


if __name__ == "__main__":
    init_db(reset=True)
    print("seeded", db_path(), len(list_posts(include_hidden=True, limit=999)), "posts")
