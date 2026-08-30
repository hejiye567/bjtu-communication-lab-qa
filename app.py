"""
北京交通大学通信原理实验答疑系统 - Streamlit前端
部署版本：单文件结构，适配Streamlit Cloud
支持多模态（文字+图片）输入
"""

import os
import io
import base64
import json
import requests
from typing import List, Dict, Optional

import streamlit as st

# ============ RAG引擎（内嵌，无需外部导入） ============

class RAGEngine:
    """RAG引擎：知识库检索 + 大模型生成（支持多模态）"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[Dict]:
        """加载知识库"""
        default_kb = [
            # ---- 仪器操作类 ----
            {
                "id": "1",
                "title": "示波器基本操作",
                "content": "示波器使用时需要注意：1. 先打开电源预热5-10分钟，使电路稳定；2. 调整触发模式为自动(Auto)或正常(Normal)；3. 调节时基(Time/Div)和幅度(Volts/Div)旋钮使波形清晰显示在屏幕中央；4. 使用探头时注意接地夹要可靠连接到被测电路的地线；5. 耦合方式选择：DC耦合可观察直流和交流成分，AC耦合仅观察交流成分；6. 测量前先用校准信号(通常为1kHz方波)校准探头。",
                "keywords": ["示波器", "操作", "使用", "波形", "探头", "时基", "耦合"],
                "category": "仪器操作"
            },
            {
                "id": "2",
                "title": "波形不稳定的原因及解决方法",
                "content": "示波器波形不稳定常见原因：1. 触发源(Source)选择错误——应选择与被测信号相同的通道作为触发源；2. 触发电平(Level)设置不当——调整触发电平至波形幅度的中间位置；3. 信号频率与时基不匹配——调节Time/Div使屏幕显示2-4个完整周期；4. 探头接触不良或接地不良——检查探头补偿螺丝和接地夹；5. 触发耦合方式不正确——对于带有直流分量的信号使用DC耦合。解决步骤：先检查触发源→调整触发电平→调节时基→检查探头连接。",
                "keywords": ["波形", "不稳定", "触发", "抖动", "滚动", "同步"],
                "category": "故障排查"
            },
            {
                "id": "3",
                "title": "频谱分析仪使用方法",
                "content": "频谱分析仪基本操作：1. 设置中心频率(Center Frequency)为待测信号频率；2. 设置扫频宽度(Span)以覆盖感兴趣的频率范围；3. 调节分辨率带宽(RBW)——RBW越小频率分辨率越高但扫描越慢，一般设为Span的1/100至1/10；4. 调节参考电平(Reference Level)使信号峰值在屏幕上部但不超出；5. 使用标记(Marker)功能精确测量频率和幅度；6. 视频带宽(VBW)用于平滑显示，一般设为RBW的1/10至1/3。注意：输入信号功率不要超过分析仪最大输入功率(通常为+30dBm)，避免损坏仪器。",
                "keywords": ["频谱", "分析仪", "频率", "幅度", "RBW", "标记", "扫频"],
                "category": "仪器操作"
            },
            {
                "id": "4",
                "title": "信号发生器使用方法",
                "content": "信号发生器(函数发生器)使用：1. 选择输出波形类型(正弦/方波/三角波)；2. 设置频率——注意单位选择(Hz/kHz/MHz)；3. 设置输出幅度(Vpp或dBm)——注意阻抗选择(通常50欧姆)；4. 设置直流偏置(Offset)——如需要；5. 开启输出(Output)开关；6. 对于AM/FM调制信号，设置调制参数(调制频率、调制深度/频偏)。注意：连接负载前先关闭输出，避免带电操作损坏仪器。",
                "keywords": ["信号发生器", "函数发生器", "频率", "幅度", "调制", "波形"],
                "category": "仪器操作"
            },
            # ---- 实验内容类 ----
            {
                "id": "5",
                "title": "AM调制实验",
                "content": "AM(调幅)实验要点：1. 连接信号源(载波)和调制信号(基带)到调制器输入端；2. 载波频率通常设为100kHz-1MHz，调制信号频率设为1kHz-10kHz；3. 调制深度(调幅系数ma)计算公式：ma=(Vmax-Vmin)/(Vmax+Vmin)×100%，建议设置在30%-80%之间；4. 用示波器观察时域调制波形，测量Vmax和Vmin计算调制深度；5. 过调制(ma>100%)会导致包络失真，无法正确解调；6. 用频谱分析仪观察：载波频率fc处有主峰，fc±fm处有边带峰，边带幅度比载波低6mA(dB)。实验中常见问题：调制深度调节过大导致失真，载波和调制信号幅度比不合适。",
                "keywords": ["AM", "调制", "载波", "调制深度", "调幅", "包络", "边带"],
                "category": "实验内容"
            },
            {
                "id": "6",
                "title": "FM调制实验",
                "content": "FM(调频)实验要点：1. FM信号的瞬时频率随调制信号幅度变化而变化；2. 最大频偏Δf与调制信号幅度成正比；3. 调制指数β=Δf/fm，其中fm为调制信号频率；4. 卡森带宽公式：BW=2(Δf+fm)，即包含主要能量的频带宽度；5. FM相比AM抗干扰能力更强但占用带宽更大；6. 用频谱仪观察时，FM信号的谱线遵循贝塞尔函数分布，载波幅度在特定β值时可能消失(如β≈2.405)。实验中注意：频率调制器需要足够的频偏范围，调制信号幅度不宜过大。",
                "keywords": ["FM", "调频", "频偏", "调制指数", "卡森", "带宽", "贝塞尔"],
                "category": "实验内容"
            },
            {
                "id": "7",
                "title": "采样定理与Nyquist准则",
                "content": "采样定理(奈奎斯特定理)实验：1. 采样频率fs必须大于信号最高频率fmax的2倍，即fs≥2fmax，否则会产生混叠(aliasing)；2. 临界频率2fmax称为奈奎斯特频率；3. 当fs<2fmax时，高频信号频谱会折叠到低频段，产生混叠失真；4. 实验中通过改变采样率观察不同效果：fs>2fmax时信号可完整恢复，fs<2fmax时出现混叠；5. 实际应用中采样频率通常取fmax的5-10倍以保证信号质量；6. 抗混叠滤波器(低通滤波器)应在采样前使用，截止频率设为fs/2。实验步骤：产生正弦信号→设置不同采样率→观察重建信号→比较混叠现象。",
                "keywords": ["采样", "奈奎斯特", "混叠", "采样频率", "Nyquist", "抗混叠"],
                "category": "实验内容"
            },
            {
                "id": "8",
                "title": "滤波器设计与实验",
                "content": "滤波器实验要点：1. 低通滤波器(LPF)：允许低频通过，阻高频，截止频率fc处衰减3dB；2. 高通滤波器(HPF)：允许高频通过，阻低频；3. 带通滤波器(BPF)：允许某频段通过；4. RC低通滤波器截止频率fc=1/(2πRC)；5. 滤波器阶数越高，阻带衰减越快(每阶-20dB/十倍频)；6. 实验中常见问题：截止频率与理论值偏差——检查元件参数精度和温度特性；通带纹波过大——使用更高Q值元件或增加级数；阻带衰减不足——增加滤波器阶数或检查接地。测量方法：用频谱仪扫频观察幅频特性曲线。",
                "keywords": ["滤波器", "截止频率", "纹波", "衰减", "RC", "低通", "高通", "带通"],
                "category": "实验内容"
            },
            # ---- 故障排查类 ----
            {
                "id": "9",
                "title": "实验中信号无输出的排查",
                "content": "信号无输出常见排查步骤：1. 检查电源——确认仪器已开机且电源指示灯亮；2. 检查输出开关——很多信号发生器有独立的Output按钮；3. 检查线缆连接——BNC接头是否拧紧，同轴线是否断裂；4. 检查阻抗匹配——50欧姆输出应接50欧姆负载；5. 检查频率设置——频率是否超出仪器量程；6. 检查幅度设置——是否设为0V或过低；7. 检查衰减器——是否意外开启了-20dB/-40dB衰减；8. 用万用表测量输出端是否有直流电压。建议按从简到繁的顺序逐一排查。",
                "keywords": ["无输出", "信号", "排查", "故障", "线缆", "衰减", "阻抗"],
                "category": "故障排查"
            },
            {
                "id": "10",
                "title": "噪声干扰问题排查",
                "content": "实验中噪声干扰排查：1. 50Hz工频干扰——检查接地是否良好，使用差分测量减少共模干扰；2. 高频电磁干扰——远离干扰源，使用屏蔽线缆，缩短接地线长度；3. 探头引入噪声——使用探头接地套筒而非鳄鱼夹减小环路面积；4. 电源纹波——使用线性电源而非开关电源，或加滤波电路；5. 接触不良导致噪声——清洁BNC接头，检查连接是否紧固；6. 数字噪声串扰——数字电路和模拟电路分开供电和接地。降低噪声的方法：改善接地、使用屏蔽线、增加滤波电容、差分测量。",
                "keywords": ["噪声", "干扰", "50Hz", "工频", "接地", "屏蔽", "纹波"],
                "category": "故障排查"
            },
            {
                "id": "11",
                "title": "数字调制实验(ASK/FSK/PSK)",
                "content": "数字调制实验：1. ASK(幅移键控)：用数字信号控制载波幅度，0表示无载波输出，1表示有载波输出；2. FSK(频移键控)：用数字信号控制载波频率，0和1分别对应两个不同频率；3. PSK(相移键控)：用数字信号控制载波相位，BPSK中0和1分别对应0度和180度相位；4. QPSK(四相移相键控)使用4种相位，频带利用率比BPSK高一倍；5. 实验中注意：观察眼图判断信号质量，码元速率与载波频率的关系(载波频率通常为码元速率的4-8倍)；6. 相干解调需要恢复载波同步，非相干解调较简单但性能较差。常见问题：相位不连续导致频谱扩展，码间串扰导致误码率升高。",
                "keywords": ["ASK", "FSK", "PSK", "QPSK", "数字调制", "键控", "眼图", "解调"],
                "category": "实验内容"
            },
            {
                "id": "12",
                "title": "PCM脉冲编码调制实验",
                "content": "PCM(脉冲编码调制)实验：1. PCM过程包括三个步骤：采样→量化→编码；2. 采样：将连续信号离散化，采样频率fs≥2fmax(奈奎斯特准则)；3. 量化：将采样值映射到有限的离散电平，量化位数n决定量化电平数N=2^n；4. 编码：将量化后的电平用二进制码表示，n位PCM码组有2^n个量化级；5. 量化噪声：量化信噪比SNR≈6n(dB)，即每增加1位编码信噪比提高6dB；6. 实验中观察：改变量化位数(8bit/16bit)听音质差异，观察量化误差波形；7. A律/μ律压扩用于非均匀量化，改善小信号信噪比。常见问题：采样率过低导致混叠，量化位数太少导致信号失真。",
                "keywords": ["PCM", "脉冲编码", "采样", "量化", "编码", "量化噪声", "信噪比", "A律"],
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
            query_words = query_lower.replace("？", "").replace("?", "").replace("的", " ").replace("了", " ").split()
            for word in query_words:
                if len(word) >= 2 and word in item["content"].lower():
                    score += 0.5

            if score > 0:
                results.append({"item": item, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return [r["item"] for r in results[:top_k]]

    def _call_llm_api(self, query: str, context: str = "", image_data: str = None) -> str:
        """调用大模型API生成回答（支持多模态）"""
        system_prompt = """你是北京交通大学通信原理实验的智能答疑助手，服务于电子信息工程学院通信工程专业的学生。你的任务是帮助学生解答通信原理实验中的各种问题。

专业背景：
- 课程：通信原理实验
- 实验内容涵盖：AM/FM调制解调、采样定理、滤波器设计、数字调制(ASK/FSK/PSK)、PCM编码、示波器和频谱分析仪使用等
- 学生为本科二三年级，有基本通信原理知识但实操经验有限

回答规则：
1. 如果提供了知识库参考资料，必须优先基于参考资料回答，确保与实验室实际设备和流程一致
2. 如果没有参考资料，可以基于通用知识回答，但需明确标注"以下基于通用知识，具体参数请参照实验指导书"
3. 涉及公式时要写出完整公式（如调制深度ma=(Vmax-Vmin)/(Vmax+Vmin)×100%）
4. 涉及操作步骤时分点详细说明，包括仪器旋钮名称和设置值
5. 涉及故障排查时给出明确的排查步骤和顺序
6. 如果提供了实验仪器照片，仔细分析照片中的仪器型号、面板设置、波形显示等信息，据此给出针对性建议
7. 回答要专业准确，使用标准通信术语，避免模糊表述
8. 如果问题超出通信原理实验范围，礼貌说明并建议咨询相关课程教师
9. 涉及安全注意事项时用"⚠注意"标注
"""

        if context:
            user_prompt = f"""参考资料（来自实验室专属知识库）：
{context}

学生问题：{query}

请基于上述参考资料回答学生的问题。如果参考资料不足以完全回答，可以补充通用知识，但需注明哪些是知识库内容、哪些是补充内容。如果学生上传了实验仪器的照片，请结合照片信息综合分析。"""
        else:
            user_prompt = f"""学生问题：{query}

请回答这个问题。注意：知识库中暂未找到直接相关内容，以下回答基于通用知识，具体参数请参照实验指导书。如果学生上传了实验仪器的照片，请结合照片信息综合分析。"""

        try:
            response = self._call_sensenova_api(system_prompt, user_prompt, image_data)
            if response:
                return response
        except Exception as e:
            print(f"SenseNova API调用失败: {e}，尝试备用API...")

        try:
            response = self._call_siliconflow_api(system_prompt, user_prompt)
            if response:
                return response
        except Exception as e:
            print(f"备用API调用失败: {e}")

        return "[系统提示] 大模型服务暂时不可用，请稍后重试或联系管理员检查API配置。"

    def _call_sensenova_api(self, system_prompt: str, user_prompt: str, image_data: str = None) -> str:
        """
        调用SenseNova大模型API（商汤，兼容OpenAI格式）
        模型：sensenova-6.7-flash-lite（支持文字+图片多模态）
        免费额度：公测期间每5小时1500次调用
        """
        api_key = os.environ.get("SENSENOVA_API_KEY", "")

        if not api_key:
            raise Exception("SenseNova API未配置，请在Streamlit Secrets中设置 SENSENOVA_API_KEY")

        url = "https://token.sensenova.cn/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        if image_data:
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        else:
            user_content = user_prompt

        payload = {
            "model": "sensenova-6.7-flash-lite",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.6,
            "max_tokens": 2000,
            "stream": False
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"SenseNova API返回错误: {response.status_code} - {response.text}")

    def _call_siliconflow_api(self, system_prompt: str, user_prompt: str) -> str:
        """调用硅基流动API（备用，纯文本）"""
        url = "https://api.siliconflow.cn/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }

        if not self.api_key:
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

    def query(self, question: str, image_data: str = None) -> Dict:
        """主查询接口：RAG完整流程（支持多模态）"""
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

        answer = self._call_llm_api(question, context, image_data)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "has_kb_match": len(relevant_docs) > 0,
            "kb_match_count": len(relevant_docs),
            "has_image": image_data is not None
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
    .multimodal-tag {
        display: inline-block;
        background: #f0fdf4;
        color: #166534;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85em;
        margin: 2px;
        border: 1px solid #86efac;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============ 初始化RAG引擎 ============
# 清除旧缓存，确保使用新版RAGEngine（多模态版）
if hasattr(st, 'cache_resource'):
    st.cache_resource.clear()

@st.cache_resource
def init_engine_v2():
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
        🟢 多模态识别：已启用<br>
        🟢 知识库：已加载<br>
        🟢 RAG引擎：运行中
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📚 知识库")
    st.markdown("""
    <div style="color: white;">
        • 仪器操作：3条<br>
        • 故障排查：3条<br>
        • 实验内容：6条<br>
        <br>
        <small>知识库持续扩充中...</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ❓ 使用说明")
    st.markdown("""
    <div style="color: rgba(255,255,255,0.9); font-size: 0.9em;">
        <b>📝 文字提问：</b><br>
        在输入框输入实验相关问题<br><br>
        <b>📷 图片提问：</b><br>
        上传实验仪器的照片，AI会识别图片内容并给出针对性建议<br><br>
        <b>⚡ 快速提问：</b><br>
        点击预设问题直接提问<br><br>
        <b>📚 蓝色标签：</b><br>
        表示知识库引用来源
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
st.markdown('<p class="sub-title">基于RAG技术的AI智能辅助答疑 | 支持文字+图片多模态输入 | 优先调用专属知识库确保准确可靠</p>', unsafe_allow_html=True)

st.markdown("---")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = init_engine_v2()

# 欢迎消息
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 欢迎来到通信原理实验答疑系统！</h3>
        <p>我是你的AI实验助手，支持文字和图片两种提问方式，可以帮你解答通信原理实验中的各种问题：</p>
        <ul>
            <li>🔧 <b>仪器操作</b>：示波器、频谱分析仪、信号发生器等仪器的使用方法</li>
            <li>🔍 <b>故障排查</b>：实验过程中遇到的问题诊断和解决</li>
            <li>📖 <b>实验内容</b>：AM/FM调制、采样定理、滤波器、数字调制、PCM编码等实验原理和步骤</li>
            <li>📷 <b>图片识别</b>：上传实验仪器照片，AI自动识别面板设置和波形并给出建议</li>
        </ul>
        <p><b>💡 提示：</b>系统会优先检索实验室专属知识库，结合多模态大模型为你提供准确、可靠的回答。</p>
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
        image_html = ""
        if message.get("image"):
            image_html = f"<br><img src='{message['image']}' style='max-width:200px; border-radius:10px; margin-top:10px;'/>"
        st.markdown(f"""
        <div class="chat-message user-message">
            <b>👤 你：</b><br>{message['content']}
            {image_html}
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

        multimodal_tag = ""
        if message.get("has_image"):
            multimodal_tag = "<span class='multimodal-tag'>📷 含图片分析</span>"

        st.markdown(f"""
        <div class="chat-message assistant-message">
            <b>🤖 实验助手：</b> {multimodal_tag}<br>{message['content']}
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

# 图片上传区域
st.markdown("##### 📷 上传实验图片（可选）")
uploaded_image = st.file_uploader(
    "上传示波器截图、仪器面板照片等",
    type=["png", "jpg", "jpeg"],
    key="image_uploader",
    label_visibility="collapsed"
)

# 处理图片
image_data_url = None
if uploaded_image is not None:
    image_bytes = uploaded_image.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    ext = uploaded_image.type.split("/")[-1]
    image_data_url = f"data:image/{ext};base64,{image_b64}"
    st.image(image_bytes, caption="已上传的图片", width=300)

# 文字输入
with st.form(key="question_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "输入你的实验问题：",
            value=default_value,
            placeholder="例如：示波器波形不稳定怎么办？（也可上传图片后提问）",
            key="user_question",
            label_visibility="collapsed"
        )
    with col2:
        submit_button = st.form_submit_button("🚀 提问", use_container_width=True)

if submit_button and (user_input.strip() or image_data_url):
    question_text = user_input.strip() if user_input.strip() else "请分析这张实验图片"

    st.session_state.messages.append({
        "role": "user",
        "content": question_text,
        "image": image_data_url
    })

    with st.spinner("🤖 正在检索知识库并生成回答..."):
        try:
            result = st.session_state.rag_engine.query(question_text, image_data_url)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "has_kb_match": result["has_kb_match"],
                "kb_match_count": result["kb_match_count"],
                "has_image": result.get("has_image", False)
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"抱歉，系统出现错误：{str(e)}\n\n请检查网络连接或稍后重试。",
                "sources": [],
                "has_kb_match": False,
                "kb_match_count": 0,
                "has_image": False
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
        **实验智答 v2.0（多模态版）**

        基于RAG（检索增强生成）技术构建，支持文字+图片多模态输入。

        **核心能力：**
        - 📚 专属知识库（12条通信原理实验条目）
        - 📷 图片识别（示波器波形、仪器面板等）
        - 🧠 多模态大模型（SenseNova 6.7 Flash-Lite）
        - 🔍 智能检索 + 知识增强生成

        技术栈：Python + Streamlit + RAG + SenseNova多模态大模型API

        项目团队：何佳源、邓坤、郭朝龙
        指导教师：周春月
        """)
