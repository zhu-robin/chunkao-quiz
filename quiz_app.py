"""
quiz_app.py — 春考汽修 · AI 自适应刷题

本版要点：
1. 单选 / 多选自动区分：优先读库里的 qtype 字段，缺失时回退到「答案字母数」
2. 首次启动自动给 questions 表补 qtype 列并回填，老库可直接用，无需手工迁移
3. AI 出题失败不再无限重试，失败给「跳过」出口
4. AI 返回值统一校验 / 归一化，答案大小写、分隔符、乱序都能正确判分
5. 每题独立 widget key，杜绝选项相同导致串题
6. 变式题保留原章节 topic，不会出现 AI-AI-xxx 脱纲
7. 区分「题库为空」和「本轮完成」两种状态；答错的原题排回队尾再练一次
8. DB 连接上下文管理、WAL、绝对路径定位 tiku.db

依赖：streamlit / openai / syllabus.py（SYLLABUS_DICT, get_syllabus）
"""

import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

import streamlit as st
from openai import OpenAI

from syllabus import SYLLABUS_DICT, get_syllabus

# ============================== 基础配置 ==============================
st.set_page_config(page_title="春考AI自适应题库", page_icon="🧠", layout="centered")

DB_PATH = Path(__file__).resolve().parent / "tiku.db"
API_BASE = "https://api.deepseek.com"
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")   # 官方模型：deepseek-chat / deepseek-reasoner
QUESTIONS_PER_ROUND = 3
RANDOM_CHAPTER = "随机抽题"
AI_SOURCE = "AI-错题衍生"
LETTERS = "ABCDEF"
MULTI_ALIASES = ("multi", "multiple", "多选", "多选题")
PREFIX_RE = re.compile(r"^\s*[A-Fa-f]\s*[.、,，:：)）]\s*")


# ============================== 通用工具 ==============================
def normalize_answer(ans) -> str:
    """'bcd' / 'B,C,D' / ['B','C'] / 'DCB' 统一成 'BCD'。"""
    if isinstance(ans, (list, tuple, set)):
        ans = "".join(str(a) for a in ans)
    return "".join(sorted(set(re.findall(r"[A-F]", str(ans).upper()))))


def guess_qtype(answer) -> str:
    """没有 qtype 时的兜底推断：答案超过 1 个字母即多选。"""
    return "multi" if len(normalize_answer(answer)) > 1 else "single"


def parse_options(raw) -> list:
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, list) or not data:
        raise ValueError("options 字段不是非空列表")
    return [str(x) for x in data]


def is_multi(q: dict) -> bool:
    """题型判定：qtype 明确标了多选就是多选；否则看答案字母数（取并集，绝不漏判）。"""
    qt = str(q.get("qtype") or "").strip().lower()
    if qt in MULTI_ALIASES:
        return True
    return len(normalize_answer(q["answer"])) > 1


def choice_letters(q: dict, picked) -> str:
    """把用户选中的选项文本换算成字母（按列表下标，不依赖 'A. ' 前缀）。"""
    picked = [picked] if isinstance(picked, str) else list(picked)
    idx = [q["options"].index(p) for p in picked if p in q["options"]]
    return "".join(sorted(chr(65 + i) for i in idx))


# ============================== 数据库 ==============================
@contextmanager
def db():
    """统一连接管理：异常回滚、必定关闭、10s 锁等待。"""
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
    """建表 + 补 qtype 列 + 回填 + 开启 WAL。老库第一次运行会自动完成迁移。"""
    with db() as conn:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                stem TEXT NOT NULL,
                options TEXT NOT NULL,
                answer TEXT NOT NULL,
                explanation TEXT,
                difficulty TEXT,
                topic TEXT,
                qtype TEXT
            )
            """
        )

        cols = {row[1] for row in conn.execute("PRAGMA table_info(questions)")}
        if "qtype" not in cols:
            try:
                conn.execute("ALTER TABLE questions ADD COLUMN qtype TEXT")
            except sqlite3.OperationalError:
                pass  # 并发/已存在，忽略

        # 只回填还没标注的题，你手工改过的 qtype 不会被覆盖
        todo = conn.execute("SELECT id, answer FROM questions WHERE qtype IS NULL OR qtype = ''").fetchall()
        for rid, ans in todo:
            conn.execute("UPDATE questions SET qtype = ? WHERE id = ?", (guess_qtype(ans), rid))
    return True


# ============================== 题目读取 ==============================
def load_questions(chapter: str, limit: int = QUESTIONS_PER_ROUND) -> list:
    sql = "SELECT * FROM questions"
    params = []
    if chapter != RANDOM_CHAPTER:
        sql += " WHERE topic = ?"
        params.append(chapter)
    sql += " ORDER BY RANDOM() LIMIT ?"
    params.append(limit)

    try:
        with db() as conn:
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as e:
        st.error(f"读取题库失败：{e}")
        st.stop()

    questions = []
    for r in rows:
        keys = r.keys()
        try:
            options = parse_options(r["options"])
        except Exception:
            continue
        answer = normalize_answer(r["answer"])
        if not answer or any(c not in LETTERS[: len(options)] for c in answer):
            continue  # 脏数据跳过，不让它把整页搞崩
        questions.append(
            {
                "id": r["id"],
                "stem": r["stem"],
                "options": options,
                "answer": answer,
                "explanation": r["explanation"] or "",
                "topic": r["topic"] or "未分类",
                "qtype": (r["qtype"] if "qtype" in keys else None),
                "_uid": uuid.uuid4().hex[:8],
            }
        )
    return questions


# ============================== AI 变式题 ==============================
@st.cache_resource(show_spinner=False)
def get_client() -> OpenAI:
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        st.error("❌ 未找到 DEEPSEEK_API_KEY。请在 .streamlit/secrets.toml 中配置，或设置同名环境变量。")
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
    base_topic = topic[3:] if topic.startswith("AI-") else topic  # 兼容旧数据的 AI- 前缀

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
    """校验 + 归一化 AI 输出，不合法一律抛 ValueError（由调用方转成 ai_error）。"""
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

    # 按位置重新打标，防止模型把 C 放在第一个
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
    """入库，按题干去重（防止刷新/双击重复插入）。"""
    with db() as conn:
        if conn.execute("SELECT 1 FROM questions WHERE stem = ? LIMIT 1", (q["stem"],)).fetchone():
            return
        conn.execute(
            """INSERT INTO questions
               (subject, stem, options, answer, explanation, difficulty, topic, qtype)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                AI_SOURCE,
                q["stem"],
                json.dumps(q["options"], ensure_ascii=False),
                q["answer"],
                q["explanation"],
                q["difficulty"],
                topic,
                guess_qtype(q["answer"]),
            ),
        )


def generate_variant(wrong_q: dict) -> dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": build_prompt(wrong_q)}],
        temperature=0.7,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    new_q = parse_ai_question(json.loads(resp.choices[0].message.content))
    new_q["topic"] = wrong_q["topic"]      # 保留原章节，不加 AI- 前缀
    new_q["qtype"] = guess_qtype(new_q["answer"])
    new_q["_uid"] = uuid.uuid4().hex[:8]
    return new_q


# ============================== 状态流转 ==============================
def start_round(chapter: str) -> None:
    qs = load_questions(chapter)
    st.session_state.queue = qs
    st.session_state.chapter = chapter
    st.session_state.score = 0
    st.session_state.phase = "answering" if qs else "empty"
    st.session_state.result = None
    st.session_state.ai_done = False
    st.session_state.ai_error = None
    st.session_state.party = False


def next_question() -> None:
    res = st.session_state.get("result") or {}
    q = st.session_state.queue.pop(0)

    # 答错的题（变式题除外）排回队尾再练一次，只重排一次防止死循环
    if not res.get("ok") and not q.get("_retry"):
        again = dict(q, _uid=uuid.uuid4().hex[:8], _retry=True)
        st.session_state.queue.append(again)

    st.session_state.phase = "answering" if st.session_state.queue else "finished"
    st.session_state.result = None
    st.session_state.ai_done = False
    st.session_state.ai_error = None
    st.rerun()


# ============================== 页面渲染 ==============================
def render_empty() -> None:
    st.warning(f"⚠️ 题库里暂时没有【{st.session_state.chapter}】分类的可用题目。")
    st.caption("可能原因：该章节还没录题，或 options / answer 字段格式不正确（options 需为 JSON 数组）。")
    if st.button("重新加载", type="primary", use_container_width=True):
        start_round(st.session_state.chapter)
        st.rerun()


def render_finished() -> None:
    if not st.session_state.get("party"):
        st.balloons()
        st.session_state.party = True
    st.header("🎉 本轮完成！")
    st.write(f"你的得分：**{st.session_state.score}**")
    st.caption(f"章节：{st.session_state.chapter}")
    if st.button("再来一轮", type="primary", use_container_width=True):
        start_round(st.session_state.chapter)
        st.rerun()


def render_feedback(q: dict) -> None:
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

    # ---- 答错：生成一次变式题，失败则给跳过出口 ----
    if not st.session_state.ai_done and not st.session_state.ai_error:
        with st.spinner("🤖 AI 导师正在结合考纲，为你定制同考点变式题..."):
            try:
                variant = generate_variant(q)
                save_question(variant, topic=q["topic"])
                st.session_state.queue.insert(1, variant)
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
        if st.button("跳过本题 ➡️", type="primary", use_container_width=True):
            next_question()


def render_question() -> None:
    q = st.session_state.queue[0]
    editable = st.session_state.phase == "answering"
    multi = is_multi(q)

    st.write(f"**当前得分：{st.session_state.score}** ｜ 待做题目：{len(st.session_state.queue)} 道")
    st.divider()
    st.subheader(q["stem"])
    st.caption(f"所属大纲章节：{q['topic']} ｜ 题型：{'多选题' if multi else '单选题'}")

    key = f"choice_{q['_uid']}"          # 每题唯一 key，避免串题/预选

    if multi:
        st.info("📝 提示：本题为多项选择题")
        picked = st.multiselect(
            "请选择你的答案（可多选）：", q["options"], key=key, disabled=not editable
        )
    else:
        picked = st.radio(
            "请选择你的答案：", q["options"], index=None, key=key, disabled=not editable
        )

    if not editable:
        render_feedback(q)
        return

    if st.button("提交答案", type="primary", use_container_width=True):
        if not picked:
            st.warning("请先选择一个选项！")
            return
        correct = normalize_answer(q["answer"])
        chosen = choice_letters(q, picked)
        ok = chosen == correct
        if ok:
            st.session_state.score += 1
        st.session_state.result = {"chosen": chosen, "correct": correct, "ok": ok}
        st.session_state.phase = "graded"
        st.rerun()


def main() -> None:
    ensure_db()

    st.title("🧠 春考汽修 - AI自适应刷题")

    chapter_list = [RANDOM_CHAPTER] + list(SYLLABUS_DICT.keys())
    with st.sidebar:
        st.header("⚙️ 刷题设置")
        selected = st.selectbox("选择练习的知识点章节：", chapter_list)
        st.divider()
        if st.button("🔄 重新开始本轮", use_container_width=True):
            start_round(selected)
            st.rerun()
        st.caption("答错后 AI 会按考纲生成同考点变式题；刷新页面会重置本轮进度。")

    # 首次进入或切换章节时初始化
    if st.session_state.get("chapter") != selected:
        start_round(selected)

    phase = st.session_state.phase
    if phase == "empty":
        render_empty()
    elif phase == "finished":
        render_finished()
    else:
        render_question()


if __name__ == "__main__":
    main()