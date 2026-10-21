"""
quiz_app.py — 春考汽修 · AI 自适应刷题（v3）

本版新增：
1. 错题本：答错自动入库，按题干去重，再答对一次自动移出；支持「一键重练错题」
2. 学情分析：分章节统计做题数/正确率，标出薄弱知识点，可生成 AI 诊断报告
3. 每轮 10 题（QUESTIONS_PER_ROUND 可调），AI 变式题每轮上限 3 道，防止题量失控
4. 提示功能：做题时可向 AI 要一个「不剧透答案」的启发式提示
5. 抽题优先抽没做过的题，避免每轮都是同样的几道

保留并沿用：
- qtype 题型字段自动迁移、单选/多选自动区分
- 答案归一化判分、AI 返回值校验、每题独立 widget key
- AI 失败不死循环、DB 连接上下文管理、WAL

依赖：streamlit / pandas / openai / syllabus.py
"""

import json
import os
import random
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

from syllabus import SYLLABUS_DICT, get_syllabus

# ============================== 基础配置 ==============================
st.set_page_config(page_title="春考AI自适应题库", page_icon="🧠", layout="centered")

DB_PATH = Path(__file__).resolve().parent / "tiku.db"
API_BASE = "https://api.deepseek.com"
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

QUESTIONS_PER_ROUND = 10      # 每轮题量（3 → 10）
MAX_VARIANTS_PER_ROUND = 3    # 每轮 AI 变式题上限
WEAK_ACCURACY = 0.6           # 正确率低于 60% 视为薄弱
WEAK_MIN_ATTEMPTS = 2         # 至少做 2 题才纳入薄弱判定
DEFAULT_STUDENT = "同学"

RANDOM_CHAPTER = "随机抽题"
AI_SOURCE = "AI-错题衍生"
LETTERS = "ABCDEF"
MULTI_ALIASES = ("multi", "multiple", "多选", "多选题")
PREFIX_RE = re.compile(r"^\s*[A-Fa-f]\s*[.、,，:：)）]\s*")

PAGES = ["📝 刷题", "📕 错题本", "📊 学情分析"]


# ============================== 通用工具 ==============================
def normalize_answer(ans) -> str:
    """'bcd' / 'B,C,D' / ['B','C'] / 'DCB' 统一成 'BCD'。"""
    if isinstance(ans, (list, tuple, set)):
        ans = "".join(str(a) for a in ans)
    return "".join(sorted(set(re.findall(r"[A-F]", str(ans).upper()))))


def guess_qtype(answer) -> str:
    return "multi" if len(normalize_answer(answer)) > 1 else "single"


def parse_options(raw) -> list:
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, list) or not data:
        raise ValueError("options 字段不是非空列表")
    return [str(x) for x in data]


def is_multi(q: dict) -> bool:
    qt = str(q.get("qtype") or "").strip().lower()
    if qt in MULTI_ALIASES:
        return True
    return len(normalize_answer(q["answer"])) > 1


def choice_letters(q: dict, picked) -> str:
    picked = [picked] if isinstance(picked, str) else list(picked)
    idx = [q["options"].index(p) for p in picked if p in q["options"]]
    return "".join(sorted(chr(65 + i) for i in idx))


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ============================== 数据库 ==============================
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def ensure_db():
    """建表 + 老库补 qtype 列并回填 + 新建 attempts 表 + WAL。"""
    with db() as conn:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT, stem TEXT NOT NULL, options TEXT NOT NULL,
                answer TEXT NOT NULL, explanation TEXT, difficulty TEXT,
                topic TEXT, qtype TEXT
            )
            """
        )

        cols = {row[1] for row in conn.execute("PRAGMA table_info(questions)")}
        if "qtype" not in cols:
            try:
                conn.execute("ALTER TABLE questions ADD COLUMN qtype TEXT")
            except sqlite3.OperationalError:
                pass

        todo = conn.execute(
            "SELECT id, answer FROM questions WHERE qtype IS NULL OR qtype = ''"
        ).fetchall()
        for rid, ans in todo:
            conn.execute("UPDATE questions SET qtype = ? WHERE id = ?", (guess_qtype(ans), rid))

        # 答题流水：错题本和学情分析都基于这张表
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT NOT NULL,
                qid INTEGER,
                stem TEXT NOT NULL,
                options TEXT,
                answer TEXT NOT NULL,
                explanation TEXT,
                topic TEXT,
                qtype TEXT,
                chosen TEXT,
                is_correct INTEGER NOT NULL,
                source TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_student ON attempts(student)")
    return True


# ============================== 题目读取 ==============================
def row_to_question(r):
    keys = r.keys()
    try:
        options = parse_options(r["options"])
    except Exception:
        return None
    answer = normalize_answer(r["answer"])
    if not answer or any(c not in LETTERS[: len(options)] for c in answer):
        return None
    return {
        "id": r["id"],
        "stem": r["stem"],
        "options": options,
        "answer": answer,
        "explanation": r["explanation"] or "",
        "topic": r["topic"] or "未分类",
        "qtype": (r["qtype"] if "qtype" in keys else None),
        "_uid": uuid.uuid4().hex[:8],
    }


def _query_questions(where_sql: str, params: list, limit: int):
    sql = f"SELECT * FROM questions {where_sql} ORDER BY RANDOM() LIMIT ?"
    with db() as conn:
        return conn.execute(sql, params + [limit]).fetchall()


def load_questions(chapter: str, student: str, limit: int = QUESTIONS_PER_ROUND) -> list:
    """优先抽这位学生还没做过的题；不够时退回全库随机（去重）。"""
    conds, params = [], []
    if chapter != RANDOM_CHAPTER:
        conds.append("topic = ?")
        params.append(chapter)

    unseen = " AND ".join(
        conds + ["id NOT IN (SELECT qid FROM attempts WHERE student = ? AND qid IS NOT NULL)"]
    )
    try:
        rows = _query_questions("WHERE " + unseen, params + [student], limit)
    except sqlite3.Error as e:
        st.error(f"读取题库失败：{e}")
        st.stop()

    qs, stems = [], set()
    for r in rows:
        q = row_to_question(r)
        if q and q["stem"] not in stems:
            qs.append(q)
            stems.add(q["stem"])

    if len(qs) < limit:
        where_sql = ("WHERE " + " AND ".join(conds)) if conds else ""
        for r in _query_questions(where_sql, params, limit * 3):
            q = row_to_question(r)
            if q and q["stem"] not in stems:
                qs.append(q)
                stems.add(q["stem"])
            if len(qs) >= limit:
                break
    return qs[:limit]


def load_questions_by_topics(topics: list, student: str, limit: int = QUESTIONS_PER_ROUND) -> list:
    topics = [t for t in topics if t]
    if not topics:
        return []
    ph = ",".join("?" * len(topics))
    where = (
        f"WHERE topic IN ({ph}) "
        "AND id NOT IN (SELECT qid FROM attempts WHERE student = ? AND qid IS NOT NULL)"
    )
    try:
        rows = _query_questions(where, topics + [student], limit)
    except sqlite3.Error as e:
        st.error(f"读取题库失败：{e}")
        return []
    qs, stems = [], set()
    for r in rows:
        q = row_to_question(r)
        if q and q["stem"] not in stems:
            qs.append(q)
            stems.add(q["stem"])
    return qs[:limit]


# ============================== 答题记录 / 学情数据 ==============================
def save_attempt(student: str, q: dict, chosen: str, correct: str, ok: bool, source: str) -> None:
    with db() as conn:
        conn.execute(
            """INSERT INTO attempts
               (student, qid, stem, options, answer, explanation, topic, qtype,
                chosen, is_correct, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                student, q.get("id"), q["stem"],
                json.dumps(q["options"], ensure_ascii=False),
                correct, q.get("explanation", ""), q.get("topic", "未分类"),
                q.get("qtype") or guess_qtype(correct),
                chosen, 1 if ok else 0, source, now_str(),
            ),
        )


def load_wrong_book(student: str, limit: int = 300) -> list:
    """每题只看最近一次作答：最近一次还错着，才算错题。"""
    with db() as conn:
        rows = conn.execute(
            """
            SELECT a.* FROM attempts a
            JOIN (
                SELECT stem, MAX(id) AS mid FROM attempts
                WHERE student = ? GROUP BY stem
            ) m ON a.id = m.mid
            WHERE a.is_correct = 0
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (student, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def load_stats(student: str):
    with db() as conn:
        rows = conn.execute(
            """
            SELECT COALESCE(topic, '未分类') AS topic,
                   COUNT(*) AS total,
                   COALESCE(SUM(is_correct), 0) AS correct
            FROM attempts WHERE student = ?
            GROUP BY COALESCE(topic, '未分类')
            ORDER BY total DESC
            """,
            (student,),
        ).fetchall()
        overall = conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(is_correct), 0) AS correct "
            "FROM attempts WHERE student = ?",
            (student,),
        ).fetchone()
    return [dict(r) for r in rows], dict(overall)


def build_wrong_round(student: str, limit: int = QUESTIONS_PER_ROUND) -> list:
    items = load_wrong_book(student, limit=200)
    if not items:
        return []
    random.shuffle(items)
    qs = []
    for r in items[: limit * 2]:
        try:
            options = parse_options(r["options"])
        except Exception:
            continue
        answer = normalize_answer(r["answer"])
        if not answer or any(c not in LETTERS[: len(options)] for c in answer):
            continue
        qs.append(
            {
                "id": r["qid"],
                "stem": r["stem"],
                "options": options,
                "answer": answer,
                "explanation": r["explanation"] or "",
                "topic": r["topic"] or "未分类",
                "qtype": r["qtype"],
                "_uid": uuid.uuid4().hex[:8],
                "_ai": r["source"] == "ai",
            }
        )
        if len(qs) >= limit:
            break
    return qs


# ============================== AI 能力 ==============================
@st.cache_resource(show_spinner=False)
def get_client() -> OpenAI:
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        st.error("❌ 未找到 DEEPSEEK_API_KEY，请在 .streamlit/secrets.toml 或环境变量中配置。")
        st.stop()
    return OpenAI(api_key=api_key, base_url=API_BASE, timeout=60, max_retries=1)


def fetch_examples(topic: str, exclude_stem: str, limit: int = 2):
    sql = """SELECT stem, options, explanation FROM questions
             WHERE stem != ? AND COALESCE(subject, '') != ?"""
    params = [exclude_stem, AI_SOURCE]
    if topic:
        sql += " AND topic = ?"
        params.append(topic)
    sql += " ORDER BY RANDOM() LIMIT ?"
    params.append(limit)
    with db() as conn:
        return conn.execute(sql, params).fetchall()


def build_prompt(wrong_q: dict) -> str:
    topic = wrong_q["topic"]
    base_topic = topic[3:] if topic.startswith("AI-") else topic

    try:
        syllabus_req = get_syllabus(base_topic) if base_topic else ""
    except Exception:
        syllabus_req = ""
    if not syllabus_req:
        syllabus_req = "严格按照山东省春考车辆维修类考试大纲出题，不得超纲。"

    type_req = (
        "多项选择题（正确答案为 2~4 个字母）"
        if is_multi(wrong_q)
        else "单项选择题（正确答案只有 1 个字母）"
    )

    rows = fetch_examples(base_topic, wrong_q["stem"])
    if not rows:
        rows = fetch_examples("", wrong_q["stem"])
    blocks = []
    for stem, options, explanation in rows:
        try:
            opts = "；".join(parse_options(options))
        except Exception:
            opts = str(options)
        blocks.append(f"题干：{stem}\n选项：{opts}\n解析：{(explanation or '')[:200]}")
    example_str = "\n\n".join(f"【风格参考真题】\n{b}" for b in blocks) or "（暂无参考题，请按春考真题风格命制）"

    return f"""你是一位山东省春考车辆维修类专业的资深命题专家。请严格遵守以下【考试大纲】要求，绝对不能超纲。

【章节】{base_topic}
【考纲要求】{syllabus_req}

【命题风格参考】
{example_str}

【学生做错的题目】
题干：{wrong_q['stem']}
选项：{'；'.join(wrong_q['options'])}
正确答案：{wrong_q['answer']}

请先分析该错题考查的核心知识点，再模仿参考题的设问方式，在考纲范围内生成 1 道全新的{type_req}。
要求：
1. 恰好 4 个选项，格式为 "A. 内容"、"B. 内容"、"C. 内容"、"D. 内容"；
2. answer 只写大写字母，多选按字母顺序拼接（如 "BC"），不得出现选项以外的字母；
3. 必须输出严格的 JSON 对象，不要输出任何其他文字：
{{"stem": "题干", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "...", "explanation": "...", "difficulty": "medium"}}
"""


def parse_ai_question(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("AI 返回的不是 JSON 对象")
    stem = str(raw.get("stem", "")).strip()
    if not stem:
        raise ValueError("AI 返回缺少题干")

    options = raw.get("options")
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            raise ValueError("options 不是合法列表")
    if not isinstance(options, list) or not 2 <= len(options) <= 6:
        raise ValueError("options 必须是 2~6 项的列表")

    options = [
        f"{chr(65 + i)}. {PREFIX_RE.sub('', str(o).strip()) or str(o).strip()}"
        for i, o in enumerate(options)
    ]

    answer = normalize_answer(raw.get("answer", ""))
    if not answer:
        raise ValueError("AI 返回缺少 answer")
    if any(c not in LETTERS[: len(options)] for c in answer):
        raise ValueError(f"answer 超出选项范围: {raw.get('answer')}")

    return {
        "stem": stem,
        "options": options,
        "answer": answer,
        "explanation": str(raw.get("explanation", "")).strip() or "AI 未提供解析。",
        "difficulty": str(raw.get("difficulty", "medium")),
    }


def save_question(q: dict, topic: str) -> None:
    with db() as conn:
        if conn.execute("SELECT 1 FROM questions WHERE stem = ? LIMIT 1", (q["stem"],)).fetchone():
            return
        conn.execute(
            """INSERT INTO questions
               (subject, stem, options, answer, explanation, difficulty, topic, qtype)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                AI_SOURCE, q["stem"],
                json.dumps(q["options"], ensure_ascii=False),
                q["answer"], q["explanation"], q["difficulty"],
                topic, guess_qtype(q["answer"]),
            ),
        )


def generate_variant(wrong_q: dict) -> dict:
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": build_prompt(wrong_q)}],
        temperature=0.7,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    new_q = parse_ai_question(json.loads(resp.choices[0].message.content))
    new_q["topic"] = wrong_q["topic"]
    new_q["qtype"] = guess_qtype(new_q["answer"])
    new_q["_uid"] = uuid.uuid4().hex[:8]
    new_q["_ai"] = True
    return new_q


def generate_hint(q: dict) -> str:
    prompt = f"""你是山东省春考车辆维修类专业的辅导老师。学生正在做下面这道题，暂时没有思路：

题干：{q['stem']}
选项：{'；'.join(q['options'])}

请给一个「启发式提示」，帮他自己想出来：
- 点出这道题考查的关键概念、结构原理或判断依据
- 如果适合，给一个排除思路（比如「先看气门与凸轮轴的相对位置」）
- 150 字以内，口语化，像老师当面点拨
- 绝对不要说破正确答案，也不要逐个点评选项
只输出提示正文。"""
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def generate_diagnosis(student: str, stats: list, wrong_items: list) -> str:
    lines = "\n".join(
        f"- {s['topic']}：{s['total']} 题，正确率 {s['correct'] / s['total']:.0%}"
        for s in stats if s["total"]
    )
    wrong = "\n".join(f"- [{w.get('topic', '')}] {w.get('stem', '')[:60]}" for w in wrong_items[:10]) or "（暂无）"

    prompt = f"""你是一位山东省春考车辆维修类专业的资深教师，正在给一位学生做学情诊断。

【学生】{student}
【分章节答题数据】
{lines}

【最近的错题】
{wrong}

请用 Markdown 输出，控制在 350 字以内：

### 最薄弱的 2~3 个知识点
逐条说明「这块考什么、他为什么容易错」。

### 针对性建议
每个薄弱点给 1 条具体可执行的做法，不要写「多做题」「认真复习」这类空话。

### 接下来 3 天的练习顺序
编号列表，写清每天练哪个章节、重点看什么。

只输出正文，不要复述上面的原始数据。"""
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


# ============================== 状态流转 ==============================
def start_round(chapter: str, student: str, mode: str = "chapter", questions=None) -> None:
    if questions is None:
        questions = load_questions(chapter, student, QUESTIONS_PER_ROUND)
    st.session_state.queue = questions
    st.session_state.chapter = chapter
    st.session_state.mode = mode
    st.session_state.score = 0
    st.session_state.phase = "answering" if questions else "empty"
    st.session_state.result = None
    st.session_state.ai_done = False
    st.session_state.ai_error = None
    st.session_state.variants_used = 0
    st.session_state.party = False
    if mode == "chapter":
        st.session_state.last_chapter = chapter


def next_question() -> None:
    st.session_state.queue.pop(0)
    st.session_state.phase = "answering" if st.session_state.queue else "finished"
    st.session_state.result = None
    st.session_state.ai_done = False
    st.session_state.ai_error = None
    st.rerun()


# ============================== 刷题页 ==============================
def render_empty(student: str) -> None:
    st.warning(f"⚠️ 题库里暂时没有【{st.session_state.get('chapter', '')}】分类的可用题目。")
    st.caption("可能原因：该章节还没录题，或 options / answer 字段格式不正确。")
    if st.button("重新加载", type="primary", use_container_width=True):
        start_round(st.session_state.get("last_chapter", RANDOM_CHAPTER), student, "chapter")
        st.rerun()


def render_finished(student: str) -> None:
    if not st.session_state.get("party"):
        st.balloons()
        st.session_state.party = True

    st.header("🎉 本轮完成！")
    st.write(f"你的得分：**{st.session_state.score}**")
    st.caption(f"模式：{st.session_state.get('chapter', '')}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("再来一轮", type="primary", use_container_width=True):
            start_round(st.session_state.get("last_chapter", RANDOM_CHAPTER), student, "chapter")
            st.rerun()
    with c2:
        if st.button("看看错题本 📕", use_container_width=True):
            st.session_state._pending_page = "📕 错题本"
            st.rerun()


def render_feedback(student: str, q: dict) -> None:
    res = st.session_state.get("result") or {}

    if res.get("ok"):
        st.success("✅ 回答正确！")
    else:
        st.error(f"❌ 回答错误！你的选择：{res.get('chosen') or '—'}，正确答案：{res.get('correct')}")

    if q.get("explanation"):
        st.info(f"**解析**\n\n{q['explanation']}")

    if res.get("ok"):
        if st.button("下一题 ➡️", type="primary", use_container_width=True):
            next_question()
        return

    # AI 变式题：每轮有额度上限，用完就直接下一题
    if st.session_state.get("variants_used", 0) >= MAX_VARIANTS_PER_ROUND:
        st.caption("本轮 AI 变式题额度已用完，这道题已记入错题本。")
        if st.button("下一题 ➡️", type="primary", use_container_width=True):
            next_question()
        return

    if not st.session_state.ai_done and not st.session_state.ai_error:
        with st.spinner("🤖 AI 导师正在结合考纲，为你定制同考点变式题..."):
            try:
                variant = generate_variant(q)
                save_question(variant, topic=q["topic"])
                st.session_state.queue.insert(1, variant)
                st.session_state.variants_used += 1
                st.session_state.ai_done = True
            except Exception as e:
                st.session_state.ai_error = f"{type(e).__name__}: {e}"[:200]
        st.rerun()

    if st.session_state.ai_done:
        st.success("🎯 已按你的错题考点生成一道变式题，就排在下一道！")
        if st.button("挑战 AI 定制变式题 ➡️", type="primary", use_container_width=True):
            next_question()
    else:
        st.warning(f"🤖 AI 出题失败：{st.session_state.ai_error}")
        if st.button("下一题 ➡️", type="primary", use_container_width=True):
            next_question()


def render_question(student: str) -> None:
    q = st.session_state.queue[0]
    editable = st.session_state.phase == "answering"
    multi = is_multi(q)

    if "hints" not in st.session_state:
        st.session_state.hints = {}

    st.write(f"**当前得分：{st.session_state.score}** ｜ 待做题目：{len(st.session_state.queue)} 道")
    st.divider()
    st.subheader(q["stem"])
    st.caption(f"所属大纲章节：{q['topic']} ｜ 题型：{'多选题' if multi else '单选题'}")

    key = f"choice_{q['_uid']}"
    if multi:
        st.info("📝 提示：本题为多项选择题")
        picked = st.multiselect("请选择你的答案（可多选）：", q["options"], key=key, disabled=not editable)
    else:
        picked = st.radio("请选择你的答案：", q["options"], index=None, key=key, disabled=not editable)

    if not editable:
        render_feedback(student, q)
        return

    hint = st.session_state.hints.get(q["_uid"])
    if hint:
        st.info(f"💡 {hint}")

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("💡 给个提示", use_container_width=True):
            if not hint:
                with st.spinner("AI 正在想怎么提示你..."):
                    try:
                        st.session_state.hints[q["_uid"]] = generate_hint(q)
                    except Exception as e:
                        st.warning(f"提示生成失败：{e}")
                st.rerun()
    with c2:
        submitted = st.button("提交答案", type="primary", use_container_width=True)

    if submitted:
        if not picked:
            st.warning("请先选择一个选项！")
            return
        correct = normalize_answer(q["answer"])
        chosen = choice_letters(q, picked)
        ok = chosen == correct
        save_attempt(student, q, chosen, correct, ok, "ai" if q.get("_ai") else "bank")
        if ok:
            st.session_state.score += 1
        st.session_state.result = {"chosen": chosen, "correct": correct, "ok": ok}
        st.session_state.phase = "graded"
        st.rerun()


def render_quiz_page(student: str) -> None:
    mode = st.session_state.get("mode", "chapter")

    with st.sidebar:
        st.header("⚙️ 刷题设置")
        if mode != "chapter":
            st.info("🎯 当前模式：" + ("错题重练" if mode == "wrong" else "薄弱章节强化"))
            if st.button("退出，回到章节练习", use_container_width=True):
                st.session_state.mode = "chapter"
                st.session_state.sel_chapter = st.session_state.get("last_chapter", RANDOM_CHAPTER)
                st.rerun()
        else:
            selected = st.selectbox(
                "选择练习的知识点章节：",
                [RANDOM_CHAPTER] + list(SYLLABUS_DICT.keys()),
                key="sel_chapter",
            )
            if st.session_state.get("chapter") != selected:
                start_round(selected, student, "chapter")
            if st.button("🔄 重新开始本轮", use_container_width=True):
                start_round(selected, student, "chapter")
                st.rerun()
            st.caption(f"本轮共 {QUESTIONS_PER_ROUND} 题，答错自动进错题本。")

    phase = st.session_state.get("phase", "empty")
    if phase == "empty":
        render_empty(student)
    elif phase == "finished":
        render_finished(student)
    else:
        render_question(student)


# ============================== 错题本 ==============================
def render_wrong_book(student: str) -> None:
    st.header("📕 错题本")
    items = load_wrong_book(student)
    if not items:
        st.info("错题本还是空的。做完一轮练习，答错的题会自动收集到这里。")
        return

    topic_count = {}
    for r in items:
        topic_count[r.get("topic") or "未分类"] = topic_count.get(r.get("topic") or "未分类", 0) + 1
    summary = " ｜ ".join(f"{k} {v} 道" for k, v in sorted(topic_count.items(), key=lambda x: -x[1]))
    st.caption(f"{student} 目前有 {len(items)} 道错题（同一题再答对一次就会移出）")
    st.caption(f"分布：{summary}")

    if st.button("🎯 抽 10 道错题重练", type="primary", use_container_width=True):
        qs = build_wrong_round(student, QUESTIONS_PER_ROUND)
        if qs:
            start_round("错题重练", student, "wrong", qs)
            st.session_state._pending_page = "📝 刷题"
            st.rerun()
        else:
            st.warning("错题数据解析失败，暂时无法重练。")

    st.divider()
    for r in items:
        try:
            opts = parse_options(r["options"]) if r["options"] else []
        except Exception:
            opts = []
        with st.expander(f"[{r.get('topic') or '未分类'}] {r['stem'][:38]}..."):
            st.write(f"**题干**：{r['stem']}")
            for o in opts:
                st.write(f"- {o}")
            st.write(f"**你的答案**：{r.get('chosen') or '—'} ｜ **正确答案**：{normalize_answer(r['answer'])}")
            if r.get("explanation"):
                st.write(f"**解析**：{r['explanation']}")
            st.caption(f"答错时间：{r.get('created_at', '')}")


# ============================== 学情分析 ==============================
def render_analytics(student: str) -> None:
    st.header("📊 学情分析")
    stats, overall = load_stats(student)
    total = overall["total"]
    if not total:
        st.info("还没有答题记录。先去「📝 刷题」做一轮，这里就会有数据了。")
        return

    acc = overall["correct"] / total
    weak = [
        s for s in stats
        if s["total"] >= WEAK_MIN_ATTEMPTS and s["correct"] / s["total"] < WEAK_ACCURACY
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总答题数", total)
    c2.metric("正确率", f"{acc:.0%}")
    c3.metric("薄弱章节", len(weak))
    c4.metric("覆盖章节", len(stats))

    st.subheader("分章节正确率")
    df = pd.DataFrame(
        [{"章节": s["topic"], "正确率": round(s["correct"] / s["total"], 3)} for s in stats]
    )
    st.bar_chart(df.set_index("章节"), height=300)

    for s in stats:
        a = s["correct"] / s["total"]
        if s["total"] >= WEAK_MIN_ATTEMPTS and a < WEAK_ACCURACY:
            badge = "🔴 薄弱"
        elif a >= 0.8:
            badge = "🟢 掌握"
        else:
            badge = "🟡 一般"
        st.write(f"**{s['topic']}** ｜ {s['correct']}/{s['total']} ｜ {a:.0%} ｜ {badge}")
        st.progress(max(0.0, min(a, 1.0)))

    st.divider()

    if weak:
        st.subheader("需要重点补的知识点")
        for s in sorted(weak, key=lambda x: x["correct"] / x["total"]):
            st.write(f"- **{s['topic']}**：正确率 {s['correct'] / s['total']:.0%}（{s['correct']}/{s['total']}）")
    else:
        st.success("目前没有明显薄弱的章节，继续保持～")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🤖 让 AI 分析我的薄弱点", type="primary", use_container_width=True):
            with st.spinner("AI 老师正在看你的答题记录..."):
                try:
                    text = generate_diagnosis(student, stats, load_wrong_book(student, 10))
                    st.session_state.diag = {"student": student, "text": text}
                except Exception as e:
                    st.error(f"分析失败：{type(e).__name__}: {e}")
            st.rerun()
    with c2:
        if weak and st.button("🎯 针对薄弱章节练 10 题", use_container_width=True):
            qs = load_questions_by_topics([w["topic"] for w in weak[:3]], student, QUESTIONS_PER_ROUND)
            if qs:
                start_round("薄弱强化", student, "weak", qs)
                st.session_state._pending_page = "📝 刷题"
                st.rerun()
            else:
                st.warning("这些章节暂时没有可抽的题目（可能都做过了）。")

    diag = st.session_state.get("diag")
    if diag and diag.get("student") == student:
        st.subheader("AI 学情诊断")
        st.markdown(diag["text"])


# ============================== 入口 ==============================
def main() -> None:
    ensure_db()

    st.title("🧠 春考汽修 - AI自适应刷题")

    with st.sidebar:
        if "student" not in st.session_state:
            st.session_state.student = DEFAULT_STUDENT
        student = st.text_input("学生姓名", key="student").strip() or DEFAULT_STUDENT
        st.divider()

    pending = st.session_state.pop("_pending_page", None)
    if pending:
        st.session_state.page = pending

    page = st.sidebar.radio("功能导航", PAGES, key="page")

    if page == "📝 刷题":
        render_quiz_page(student)
    elif page == "📕 错题本":
        render_wrong_book(student)
    else:
        render_analytics(student)


if __name__ == "__main__":
    main()
