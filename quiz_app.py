import streamlit as st
import sqlite3
import json
from openai import OpenAI

# 1. 初始化 DeepSeek（请替换成你真实的 API Key）
client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"], 
    base_url="https://api.deepseek.com",
)

st.set_page_config(page_title="春考AI自适应题库", page_icon="🧠", layout="centered")
st.title("🧠 春考汽修 - AI自适应刷题")

# 2. 核心功能：让 AI 根据错题现场出题
def generate_variant_from_error(wrong_q):
    # 去数据库里找 2 道真题作为风格参考
    conn = sqlite3.connect('tiku.db')
    cursor = conn.cursor()
    cursor.execute("SELECT stem, options, explanation FROM questions WHERE stem != ? ORDER BY RANDOM() LIMIT 2", (wrong_q['stem'],))
    examples = cursor.fetchall()
    conn.close()

    example_str = ""
    for ex in examples:
        example_str += f"【风格参考真题】\n题干：{ex[0]}\n选项：{ex[1]}\n解析：{ex[2]}\n\n"

    prompt = f"""你是一个山东省春考车辆维修类专业的资深命题专家。
请先阅读以下你过去的命题风格：
{example_str}

现在，学生刚刚做错了下面这道题：
【学生错题】
题干：{wrong_q['stem']}
选项：{wrong_q['options']}
正确答案：{wrong_q['answer']}

请深度分析该错题的考点，并模仿上述【风格参考真题】的设问方式、难度和专业术语，生成 1 道全新的变式选择题。
要求：必须输出严格的 JSON 对象，格式如下：
{{
  "stem": "题干内容",
  "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
  "answer": "A",
  "explanation": "详细的解析",
  "difficulty": "medium"
}}
"""
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    new_q = json.loads(response.choices[0].message.content)
    
    # 把 AI 新生成的题存入数据库
    conn = sqlite3.connect('tiku.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO questions (subject, stem, options, answer, explanation, difficulty)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', ("AI-错题衍生", new_q['stem'], json.dumps(new_q['options'], ensure_ascii=False), new_q['answer'], new_q['explanation'], new_q['difficulty']))
    conn.commit()
    conn.close()
    
    return new_q

# 3. 初始加载题目
def load_initial_questions():
    conn = sqlite3.connect('tiku.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, stem, options, answer, explanation FROM questions ORDER BY RANDOM() LIMIT 3")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "stem": r[1], "options": json.loads(r[2]), "answer": r[3], "explanation": r[4]} for r in rows]

# 初始化 Session State
if "question_queue" not in st.session_state:
    st.session_state.question_queue = load_initial_questions()
    st.session_state.score = 0
    st.session_state.submitted = False
    st.session_state.user_choice = None
    st.session_state.ai_generated = False 

questions = st.session_state.question_queue

# 页面主逻辑
if len(questions) > 0:
    q = questions[0] 
    
    st.write(f"**当前得分：{st.session_state.score}** | 待做题目：{len(questions)} 道")
    st.divider()

    st.subheader(f"{q['stem']}")

    user_choice = st.radio(
        "请选择你的答案：",
        q['options'],
        index=None,
        disabled=st.session_state.submitted
    )

    if not st.session_state.submitted:
        if st.button("提交答案", type="primary", use_container_width=True):
            if user_choice is None:
                st.warning("请先选择一个选项！")
            else:
                st.session_state.user_choice = user_choice
                st.session_state.submitted = True
                if user_choice[0] == q['answer']:
                    st.session_state.score += 1
                st.rerun()

    if st.session_state.submitted:
        if st.session_state.user_choice[0] == q['answer']:
            st.success("✅ 回答正确！")
            st.info(f"**解析：**\n{q['explanation']}")
            
            if st.button("下一题 ➡️", type="primary", use_container_width=True):
                st.session_state.question_queue.pop(0)
                st.session_state.submitted = False
                st.session_state.user_choice = None
                st.session_state.ai_generated = False
                st.rerun()
        else:
            st.error(f"❌ 回答错误！正确答案是：{q['answer']}")
            st.info(f"**解析：**\n{q['explanation']}")
            
            if not st.session_state.ai_generated:
                st.warning("🤖 AI 导师发现你这个知识点薄弱，正在为你量身定制一道变式题，请稍候...")
                with st.spinner("AI 正在疯狂出题中..."):
                    try:
                        new_q = generate_variant_from_error(q)
                        st.session_state.question_queue.insert(1, new_q)
                        st.session_state.ai_generated = True
                    except Exception as e:
                        st.error(f"AI 出题失败: {e}")
                st.rerun()
            else:
                if st.button("挑战 AI 定制变式题 ➡️", type="primary", use_container_width=True):
                    st.session_state.question_queue.pop(0)
                    st.session_state.submitted = False
                    st.session_state.user_choice = None
                    st.session_state.ai_generated = False
                    st.rerun()

else:
    st.balloons()
    st.header("🎉 恭喜通关！")
    st.write(f"你的最终得分是：**{st.session_state.score}**")
    if st.button("重新开始", type="primary"):
        st.session_state.question_queue = load_initial_questions()
        st.session_state.score = 0
        st.session_state.submitted = False
        st.session_state.user_choice = None
        st.session_state.ai_generated = False
        st.rerun()