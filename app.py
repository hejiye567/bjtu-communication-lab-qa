"""
北京交通大学通信原理实验答疑系统 - Streamlit前端
部署版本：单文件结构，适配Streamlit Cloud
"""

import os
import json
import requests
from typing import List, Dict, Optional

import streamlit as st

# ============ RAG引擎（内嵌，无需外部导入） ============

class RAGEngine:
    """RAG引擎：知识库检索 + 大模型生成"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_knowledge_base(self) -> List[Dict]:
        """加载知识库"""
        default_kb = [
            {
                "id": "1",
                "title": "示波器基本操作",
                "content": "示波器使用时需要注意：1. 先打开电源预热；2. 调整触发模式为自动或正常；3. 调节时基和幅度使波形清晰显示；4. 使用探头时注意接地。",
                "keywords": ["示波器", "操作", "使用", "波形"],
                "category": "仪器操作"
            },
            {
                "id": "2", 
                "title": "波形不稳定的原因",
                "content": "示波器波形不稳定常见原因：1. 触发源选择错误；2. 触发电平设置不当；3. 信号频率与时基不匹配；4. 探头接触不良。解决方法：检查触发设置，调整触发电平至波形中间位置，确保探头接触良好。",
                "keywords": ["波形", "不稳定", "触发", "抖动"],
                "category": "故障排查"
            },
            {
                "id": "3",
                "title": "AM调制实验",
                "content": "AM调制实验步骤：1. 连接信号源和调制器；2. 设置载波频率和调制信号频率；3. 调节调制深度（建议30%-80%）；4. 观察调制波形和频谱。注意事项：避免过调制（调制深度>100%），否则会出现失真。",
                "keywords": ["AM", "调制", "载波", "调制深度"],
                "category": "实验内容"
            },
            {
                "id": "4",
                "title": "频谱分析仪使用",
                "content": "频谱分析仪基本操作：1. 设置中心频率和扫频宽度；2. 调节分辨率带宽（RBW）；3. 使用标记（Marker）测量频率和幅度；4. 注意输入信号功率不要超过设备上限，避免损坏仪器。",
                "keywords": ["频谱", "分析仪", "频率", "幅度"],
                "category": "仪器操作"
            },
            {
                "id": "5",
                "title": "滤波器实验常见问题",
                "content": "滤波器实验中常见问题：1. 截止频率与设计值偏差——检查元件参数是否准确；2. 通带纹波过大——可能是元件Q值不够；3. 阻带衰减不足——检查接地是否良好，增加级数。",
                "keywords": ["滤波器", "截止频率", "纹波", "衰减"],
                "category": "故障排查"
            },
            {
                "id": "6",
                "title": "采样定理实验",
                "content": "采样定理实验要点：1. 采样频率必须大于信号最高频率的2倍（奈奎斯特准则）；2. 当采样频率低于2倍信号频率时会出现混叠现象；3. 观察不同采样率下信号重建效果；4. 使用抗混叠滤波器减少混叠。",
                "keywords": ["采样", "奈奎斯特", "混叠", "采样频率"],
                "category": "实验内容"
            }
        ]
        return default_kb
    
    def _search_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict]:
        """在知识库中检索相关内容"""
        query_lower = query.lower()
        results = []
        
        for item in self.knowledge_base:
            score = 0
            for keyword in item["keywords"]:
                if keyword.lower() in query_lower:
                    score += 1
            if item["title"].lower() in query_lower:
                score += 2
            if any(word in item["content"].lower() for word in query_lower.split()):
                score += 0.5
                
            if score > 0:
                results.append({"item": item, "score": score})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return [r["item"] for r in results[:top_k]]
    
    def _call_llm_api(self, query: str, context: str = "") -> str:
        """调用大模型API生成回答"""
        system_prompt = """你是北京交通大学通信原理实验的智能答疑助手。你的任务是帮助学生解答通信原理实验中的问题。

规则：
1. 如果提供了参考资料，请优先基于参考资料回答，确保准确性
2. 如果没有参考资料或参考资料不相关，请基于你的知识回答，但要明确说明
3. 回答要简洁、准确、实用，适合实验场景
4. 如果涉及操作步骤，请分点说明
5. 如果不确定，请诚实告知学生需要咨询教师
"""

        if context:
            user_prompt = f"""参考资料：
{context}

学生问题：{query}

请基于参考资料回答学生的问题。如果参考资料不足以回答，请补充你的知识，但要注明。"""
        else:
            user_prompt = f"学生问题：{query}\n\n请回答这个问题。注意：知识库中暂未找到相关内容，以下回答基于通用知识。"

        try:
            response = self._call_siliconflow_api(system_prompt, user_prompt)
            if response:
                return response
        except Exception as e:
            print(f"API调用失败: {e}")
        
        return "[系统提示] 大模型服务暂时不可用，请稍后重试或联系管理员检查API配置。"
    
    def _call_siliconflow_api(self, system_prompt: str, user_prompt: str) -> str:
        """调用硅基流动API"""
        url = "https://api.siliconflow.cn/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
        if not self.api_key:
            url = "https://api.siliconflow.cn/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": "deepseek-ai/DeepSeek-V2.5",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API返回错误: {response.status_code} - {response.text}")
    
    def query(self, question: str) -> Dict:
        """主查询接口：RAG完整流程"""
        relevant_docs = self._search_knowledge_base(question)
        
        context = ""
        sources = []
        if relevant_docs:
            context_parts = []
            for doc in relevant_docs:
                context_parts.append(f"【{doc['title']}】{doc['content']}")
                sources.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "category": doc["category"]
                })
            context = "\n\n".join(context_parts)
        
        answer = self._call_llm_api(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "has_kb_match": len(relevant_docs) > 0,
            "kb_match_count": len(relevant_docs)
        }


# ============ 页面配置 ============
st.set_page_config(
    page_title="北京交通大学通信原理实验答疑系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 自定义CSS样式 ============
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
    }
    .main-title {
        text-align: center;
        color: #1e3a8a;
        font-size: 2.2em;
        font-weight: bold;
        margin-bottom: 5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-title {
        text-align: center;
        color: #64748b;
        font-size: 1.1em;
        margin-bottom: 20px;
    }
    .school-badge {
        text-align: center;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: 50px;
        border-bottom-right-radius: 5px;
    }
    .assistant-message {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-right: 50px;
        border-bottom-left-radius: 5px;
    }
    .source-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85em;
        margin: 2px;
        border: 1px solid #93c5fd;
    }
    .welcome-card {
        background: linear-gradient(135deg, #dbeafe, #e0e7ff);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid #93c5fd;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============ 初始化RAG引擎 ============
@st.cache_resource
def init_engine():
    return RAGEngine()

# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: white;">🎓 实验智答</h2>
        <p style="color: rgba(255,255,255,0.8);">AI辅助答疑系统</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 系统状态")
    st.markdown("""
    <div style="color: white;">
        🟢 大模型服务：在线<br>
        🟢 知识库：已加载<br>
        🟢 RAG引擎：运行中
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📚 知识库")
    st.markdown("""
    <div style="color: white;">
        • 仪器操作：2条<br>
        • 故障排查：2条<br>
        • 实验内容：2条<br>
        <br>
        <small>知识库持续扩充中...</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ❓ 使用说明")
    st.markdown("""
    <div style="color: rgba(255,255,255,0.9); font-size: 0.9em;">
        1. 在下方输入框输入实验相关问题<br><br>
        2. 系统优先检索知识库，再结合大模型生成回答<br><br>
        3. 可点击快速问题直接提问<br><br>
        4. 回答中的蓝色标签为知识库引用来源
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="color: rgba(255,255,255,0.7); font-size: 0.8em; text-align: center;">
        北京交通大学<br>
        电子信息工程学院<br>
        通信工程专业实验室<br><br>
        2026年大学生创新创业训练项目
    </div>
    """, unsafe_allow_html=True)

# ============ 主界面 ============
st.markdown("""
<div class="school-badge">
    🏫 北京交通大学 · 电子信息工程学院 · 通信工程专业实验室
</div>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">📡 通信原理实验答疑系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于RAG技术的AI智能辅助答疑 | 优先调用专属知识库，确保回答准确可靠</p>', unsafe_allow_html=True)

st.markdown("---")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = init_engine()

# 欢迎消息
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 欢迎来到通信原理实验答疑系统！</h3>
        <p>我是你的AI实验助手，可以帮你解答通信原理实验中的各种问题，包括：</p>
        <ul>
            <li>🔧 <b>仪器操作</b>：示波器、频谱分析仪等仪器的使用方法</li>
            <li>🔍 <b>故障排查</b>：实验过程中遇到的问题诊断和解决</li>
            <li>📖 <b>实验内容</b>：AM调制、采样定理、滤波器等实验原理和步骤</li>
        </ul>
        <p><b>💡 提示：</b>系统会优先检索实验室专属知识库，结合大模型为你提供准确、可靠的回答。</p>
    </div>
    """, unsafe_allow_html=True)

# 快速问题区域
st.markdown("### ⚡ 快速提问")
quick_questions = [
    "示波器波形不稳定怎么办？",
    "AM调制的调制深度怎么设置？",
    "什么是采样定理？",
    "频谱分析仪怎么用？",
    "滤波器截止频率偏差怎么解决？"
]

cols = st.columns(len(quick_questions))
for i, question in enumerate(quick_questions):
    with cols[i]:
        if st.button(question, key=f"quick_{i}", use_container_width=True):
            st.session_state.current_question = question

st.markdown("---")

# 显示聊天历史
st.markdown("### 💬 对话记录")

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <b>👤 你：</b><br>{message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if message.get("sources"):
            sources_html = "<div style='margin-top: 10px;'><small>📚 知识库引用：</small>"
            for source in message["sources"]:
                sources_html += f"<span class='source-badge'>{source['title']}</span>"
            sources_html += "</div>"
        
        match_status = ""
        if message.get("has_kb_match"):
            match_status = f"<small style='color: #22c55e;'>✓ 基于知识库回答（匹配{message['kb_match_count']}条）</small>"
        else:
            match_status = "<small style='color: #f59e0b;'>⚠ 基于通用知识回答（知识库暂未收录）</small>"
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <b>🤖 实验助手：</b><br>{message['content']}
            {sources_html}
            <div style="margin-top: 8px;">{match_status}</div>
        </div>
        """, unsafe_allow_html=True)

# 输入区域
st.markdown("---")

if "current_question" in st.session_state:
    default_value = st.session_state.current_question
    del st.session_state.current_question
else:
    default_value = ""

with st.form(key="question_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "输入你的实验问题：",
            value=default_value,
            placeholder="例如：示波器波形不稳定怎么办？",
            key="user_question",
            label_visibility="collapsed"
        )
    with col2:
        submit_button = st.form_submit_button("🚀 提问", use_container_width=True)

if submit_button and user_input.strip():
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip()
    })
    
    with st.spinner("🤖 正在检索知识库并生成回答..."):
        try:
            result = st.session_state.rag_engine.query(user_input.strip())
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "has_kb_match": result["has_kb_match"],
                "kb_match_count": result["kb_match_count"]
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"抱歉，系统出现错误：{str(e)}\n\n请检查网络连接或稍后重试。",
                "sources": [],
                "has_kb_match": False,
                "kb_match_count": 0
            })
    
    st.rerun()

# 底部操作栏
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("💾 导出记录", use_container_width=True):
        if st.session_state.messages:
            export_text = ""
            for msg in st.session_state.messages:
                role = "用户" if msg["role"] == "user" else "助手"
                export_text += f"【{role}】\n{msg['content']}\n\n"
            
            st.download_button(
                label="📥 下载对话记录",
                data=export_text,
                file_name="实验答疑记录.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("暂无对话记录可导出")

with col3:
    if st.button("ℹ️ 关于系统", use_container_width=True):
        st.info("""
        **实验智答 v1.0**
        
        基于RAG（检索增强生成）技术构建，优先调用通信工程专业实验专属知识库，结合大语言模型为学生提供准确、可靠的实验答疑服务。
        
        技术栈：Python + Streamlit + RAG + 大模型API
        
        项目团队：何佳源、邓坤、郭朝龙
        指导教师：周春月
        """)
