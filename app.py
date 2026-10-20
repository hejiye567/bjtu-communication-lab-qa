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
            },            # ---- 通信原理实验指导书（10个核心实验）----
            {
                "id": "13",
                "title": "实验一 AMI/HDB3码型变换实验",
                "content": "AMI/HDB3码型变换实验（M03时分复用&基带传输模块 + 13号载波同步及位同步模块）。实验目的：掌握AMI和HDB3码的编译规则，了解滤波法位同步的作用。编码规则：AMI码——遇到0输出0，遇到1则交替输出+1和-1；HDB3码是AMI的改进型，将4个连0用取代节000V或B00V代替，使连0不超过3个，保证定时提取。HDB3取代规则：两相邻V码间有奇数个1时用000V，偶数个1(含0)时用B00V。译码：AMI将±1变1、0变0；HDB3找到极性与前一符号相同的传号A，将A及其前3个码元清0。关键测试点：TH16(编码输入数据)、TH18(HDB3输出)、TH19(AMI输出)、TH23(单极性码)、TH26(HDB3译码输出)。设置：D1为PN15、256KHz；编码输出波形设为RZ；13号模块S3拨为0011提取512K同步时钟。思考要点：HDB3相比AMI的优势是限制连0数；单极性码在频域有直流分量而双极性码无直流分量。",
                "keywords": ["AMI", "HDB3", "码型变换", "基带传输", "编译码", "连0", "取代节", "M03", "TH18", "TH19"],
                "category": "实验内容"
            },
            {
                "id": "14",
                "title": "实验二 时分复用与解复用实验",
                "content": "时分复用与解复用实验（M01信源编译码 + M03时分复用&基带传输 + 13号模块）。实验目的：掌握时分复用概念及工作原理。帧结构：256KHz模式下每帧4个时隙（32比特）——第0时隙为帧头01110010，第1时隙DIN1(M01的PCM编码输出)，第2时隙DIN2(信号源D1)，第3时隙为拨码开关信号；2048KHz模式下每帧32个时隙（256比特），时隙位置可任意搬移。复用速率由拨码设置，位同步由13号模块数字锁相环提取。关键测试点：M03的TH6(复用输出)、TH12(DOUT1解复用输出)、TH10(FSOUT帧同步)、M01的TH13(PCM译码输出)。设置：A1为2KHz正弦波、D1为64KHz数字终端；PCM编码选A律；13号S3拨0100(256K位同步)；复用帧头01110010。同步指示灯亮表示正常，失步/捕获灯亮需检查连线和拨码。可设置拨码为01110010观察假同步现象。",
                "keywords": ["时分复用", "解复用", "帧结构", "时隙", "PCM", "帧同步", "M03", "M01", "FSOUT", "假同步"],
                "category": "实验内容"
            },
            {
                "id": "15",
                "title": "实验三 FSK调制及解调实验",
                "content": "FSK调制及解调实验（9号数字调制解调模块 + M06综合实验模块）。实验目的：掌握键控法产生FSK信号和非相干解调原理。调制原理：基带信号与载波1(256KHz)相乘得1电平ASK，基带取反与载波2(128KHz)相乘得0电平ASK，相加合成FSK输出。解调采用过零检测法：已调信号经过零检测识别载波频率变化，经上/下沿单稳触发相加、低通滤波、门限判决恢复基带信号。2FSK表达式：e(t)=s1(t)cosω1t+s2(t)cosω2t。关键测试点：TH1(基带信号)、TH4(调制输出)、TP6(单稳相加输出)、TP7(LPF-FSK低通输出)、TH8(FSK解调输出)。设置：D1为PN15、32KHz；A1为128KHz正弦波；A2为256KHz正弦波；9号S1拨0000。抗噪测试：改变高斯白噪声幅度观察眼图变化。注意：FSK可用过零检测法解调是因为不同频率过零率不同；若两路载波频率设反，解调仍可工作但判决极性反转。",
                "keywords": ["FSK", "频移键控", "调制", "解调", "过零检测", "非相干", "9号模块", "眼图", "抗噪"],
                "category": "实验内容"
            },
            {
                "id": "16",
                "title": "实验四 BPSK调制及解调实验",
                "content": "BPSK调制及解调实验（9号模块 + 13号载波同步及位同步模块 + M06）。实验目的：掌握BPSK调制解调原理、载波恢复与位定时恢复。调制原理：基带1电平与256KHz载波相乘，0电平与256KHz反相载波相乘，叠加得BPSK输出。BPSK表达式：e(t)=Acos(ωt+φn)，φn=0表示0，φn=π表示1。解调需相干载波（由13号科斯塔斯环提取），已调信号与相干载波相乘后经低通滤波和门限判决恢复基带。关键测试点：TH1(基带信号)、TH4(调制输出)、13号TH1(SIN相干载波)、TP6(LPF-BPSK判决前抽样点)、TH8(BPSK解调输出)。设置：D1为PN15、32KHz；A2为256KHz正弦波；9号S1拨0000。眼图观测：以CLK触发测LPF-BPSK，调13号W1使眼图稳定、眼皮薄、眼睛张开大。相位模糊问题：多次按13号复位键，相干载波相位可能跳变180°，导致解调输出0/1颠倒。",
                "keywords": ["BPSK", "相移键控", "相干解调", "载波同步", "相位模糊", "科斯塔斯环", "眼图", "9号模块", "13号模块"],
                "category": "实验内容"
            },
            {
                "id": "17",
                "title": "实验五 DBPSK调制及解调实验",
                "content": "DBPSK(差分相移键控)调制及解调实验（9号 + 13号 + M06模块）。实验目的：掌握DBPSK原理及差分编码的作用。调制原理：基带信号先经差分编码得到相对码，再将相对码与载波及反相载波相乘得DBPSK输出。DBPSK用前后码元的相对相位差表示信息：Δφ=0表示1，Δφ=π表示0。相比BPSK增加了差分编码和差分译码模块（13号数字锁相环BS2提供差分译码时钟）。差分编码的作用：消除BPSK的相位模糊问题——即使相干载波相位跳变180°，相对相位差不变，解调不会出现0/1颠倒。关键测试点：TH1(基带信号)、NRZ-Q(差分编码输出)、TH4(调制输出)、TH8(DBPSK解调输出)、13号SIN(相干载波)、BS2(位同步)。设置：D1为PN15、32KHz；A2为256KHz正弦波；9号S1拨0100；13号S3拨0111。对比BPSK：DBPSK抗相位模糊但误码率略高，差分编码增加电路复杂度。",
                "keywords": ["DBPSK", "差分相移键控", "差分编码", "相位模糊", "相对相位", "9号模块", "BS2"],
                "category": "实验内容"
            },
            {
                "id": "18",
                "title": "实验六 汉明编译码实验",
                "content": "汉明编译码实验（M02信道编译码模块）。实验目的：掌握汉明码编译码原理和检错纠错原理。汉明码是能纠正1位错码的线性分组码，(7,4)汉明码参数：码长n=7，信息位k=4，监督位r=3，最小码距d0=3，纠错能力t=1。监督关系：S1=α6⊕α5⊕α4⊕α2，S2=α6⊕α5⊕α3⊕α1，S3=α6⊕α4⊕α3⊕α0。校正子S1S2S3指示错码位置：001→α0，010→α1，100→α2，011→α3，101→α4，110→α5，111→α6，000→无错。编码效率η=k/n，n大时接近1。关键测试点：M02的TH1(编码输入数据)、TH5(编码输出)、TH10(译码输出)、TP2(帧同步)、TP1(插错指示)、误码指示。设置：D1为数字终端、32KHz，S1=10100000；FS输出为FS 32bit；编码速率32K。实验可插入1位错误验证纠错能力，插入多位错误验证检错能力。注意编码速率须与数据速率一致，否则译码错误。",
                "keywords": ["汉明码", "编译码", "线性分组码", "纠错", "检错", "校正子", "M02", "监督位", "码距"],
                "category": "实验内容"
            },
            {
                "id": "19",
                "title": "实验七 滤波法位同步实验",
                "content": "滤波法位同步实验（13号载波同步及位同步模块 + M03）。实验目的：掌握滤波法提取位同步原理及对信息码的要求。原理：输入HDB3单极性码经256KHz窄带滤波器(BPF)滤出同步分量，经门限判决提取位同步信号；由于其他频率成分干扰导致占空比不均，再经模拟锁相环(CD4046)平滑处理。数字256K时钟经4分频得64K送入鉴相器A脚，VCO经8分频得64K送入鉴相器B脚，故VCO输出512KHz。S2开关上拨选择滤波法。关键测试点：BPF-Out(带通滤波输出)、门限判决输出、鉴相输入1、鉴相输入2、BS1(滤波法位同步输出)。设置：D1为PN15、256KHz；13号S2上拨、S4拨1000。滤波法对信息码要求：基带信号需有足够的定时分量（如归零码或经非线性处理后的码），否则BPF无法滤出位同步分量。",
                "keywords": ["滤波法", "位同步", "窄带滤波", "锁相环", "CD4046", "BPF", "BS1", "13号模块"],
                "category": "实验内容"
            },
            {
                "id": "20",
                "title": "实验八 数字锁相环法位同步实验",
                "content": "数字锁相环法位同步实验（13号模块）。实验目的：掌握数字锁相环提取位同步原理。数字锁相环(DPLL)由三部分组成：①超前-滞后型数字鉴相器(LL-DPD)——比较输入码元与本地时钟相位，输出超前/滞后极性(Sign)和有效性(AbsVal)；②数字环路滤波器(DLF)——加减计数逻辑(CntLgc)滤波去噪，模值M=1024，达边界0或M时输出扣脉冲(Deduct)或插脉冲(Insert)控制指令；③数控振荡器(DCO)——由高速振荡器(64MHz)、相位调节器(PhsAdj)、分频器组成，根据Deduct/Insert指令调整本地时钟相位，扣脉冲使相位前移、插脉冲使相位后移。S3拨码控制分频比：0000→4096K，0011→512K，0100→256K，0111→32K。关键测试点：数字锁相环输入、输入跳变指示、鉴相输出、插入指示、扣除指示、BS2(恢复位同步)。设置：D1为PN15、256KHz；13号S3拨0100。BS2恢复时钟存在抖动，抖动区间为一个高速时钟周期(15.625ns)，增大DLF模值M可减小抖动但增加捕捉时间。",
                "keywords": ["数字锁相环", "位同步", "DPLL", "鉴相器", "DCO", "数控振荡器", "超前滞后", "BS2", "13号模块"],
                "category": "实验内容"
            },
            {
                "id": "21",
                "title": "实验九 载波同步实验",
                "content": "载波同步实验（9号 + 13号模块）。实验目的：掌握科斯塔斯环提取载波的方法，了解相位模糊现象。原理：科斯塔斯环(同相正交环)将BPSK信号分别与VCO输出载波及90°移相载波相乘，经低通滤波得到I、Q两路，再相乘得误差电压v7≈(1/4)m²(t)·θ，该电压与相位误差θ成正比，调整VCO相位使θ趋近于0，恢复相干载波。载波同步性能指标：高效率（直接法不发导频，效率高）、高精度（稳态相差+随机相差）、同步建立时间快、保持时间长。关键测试点：13号TH2(载波同步输入)、SIN(恢复载波输出)、I/Q路低通输出、压控电压。设置：D1为PN15、32KHz；A2为256KHz正弦波；9号S1拨0000(BPSK)。调节13号压控偏置电位器使SIN与A2同相。相位模糊：科斯塔斯环恢复的载波可能有0°或180°两种稳定相位，导致BPSK解调0/1颠倒，解决方法是采用DBPSK差分编码。",
                "keywords": ["载波同步", "科斯塔斯环", "Costas", "相位模糊", "VCO", "相干载波", "SIN", "13号模块"],
                "category": "实验内容"
            },
            {
                "id": "22",
                "title": "实验十 帧同步提取实验",
                "content": "帧同步提取实验（M03时分复用&基带传输模块）。实验目的：掌握巴克码识别原理、同步保护原理及假同步/漏同步概念。原理：帧同步通过在帧结构中插入特殊的帧头码(如01110010巴克码组)，接收端用巴克码识别器检测帧头位置，识别器输出与本地帧结构比对，实现失步→捕获→同步三态切换。同步保护：为防止假同步(把数据误判为帧头)和漏同步(真帧头未检出)，采用捕捉态和维持态两种工作模式——捕捉态要求连续多帧检测到帧头才确认同步，维持态允许偶尔漏检但保持同步。关键测试点：M03的TH6(复用输出)、TH10(FSOUT帧同步输出)、同步/失步/捕获指示灯。设置：选择时分复用实验，帧头设为01110010。假同步观察方法：设置拨码开关数据为01110010，反复切换复用速率256K-2048K-256K，观察第3时隙数据是否出现错位。M03同步指示灯亮表示正常同步。",
                "keywords": ["帧同步", "巴克码", "假同步", "漏同步", "同步保护", "捕捉态", "维持态", "FSOUT", "M03"],
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
- 实验内容涵盖（共10个核心实验）：AMI/HDB3码型变换、时分复用与解复用、FSK调制解调、BPSK调制解调、DBPSK调制解调、汉明编译码、滤波法位同步、数字锁相环法位同步、载波同步、帧同步提取；以及AM/FM调制、采样定理、滤波器设计、PCM编码等基础实验
- 实验设备：信号源-综合信号测试仪、M01信源编译码、M02信道编译码、M03时分复用&基带传输、9号数字调制解调、13号载波同步及位同步、M06综合实验模块、双踪示波器
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
        模型：sensenova-6.8-flash-lite（支持文字+图片多模态）
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
            "model": "sensenova-6.8-flash-lite",
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
        color: #ffffff;
        font-size: 2.2em;
        font-weight: bold;
        margin-bottom: 5px;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.35);
    }
    .sub-title {
        text-align: center;
        color: #ffffff;
        font-size: 1.05em;
        font-weight: 500;
        margin-bottom: 20px;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.35);
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
        font-size: 0.95rem;
        line-height: 1.65;
    }
    /* 限制AI回答中markdown元素的字号 */
    .assistant-message h1, .assistant-message h2, .assistant-message h3,
    .assistant-message h4, .assistant-message h5, .assistant-message h6 {
        font-size: 1.05rem !important;
        font-weight: 600;
        color: #1e3a8a;
        margin: 8px 0 4px 0;
        padding: 0;
        border: none;
    }
    .assistant-message p, .assistant-message li,
    .assistant-message ol, .assistant-message ul {
        font-size: 0.95rem !important;
        line-height: 1.65;
        margin: 4px 0;
    }
    .assistant-message strong { color: #1e3a8a; }
    .assistant-message code { font-size: 0.85rem; }
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
    /* 顶栏透明但保留侧边栏展开按钮 */
    header {
        background: transparent !important;
    }
    /* 只隐藏顶栏右侧的工具按钮（Deploy等），保留左侧侧边栏开关 */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
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
        <h2 style="color: #1e3a8a;">🎓 实验智答</h2>
        <p style="color: #475569;">AI辅助答疑系统</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 系统状态")
    st.markdown("""
    <div style="color: #334155;">
        🟢 大模型服务：在线<br>
        🟢 多模态识别：已启用<br>
        🟢 知识库：已加载<br>
        🟢 RAG引擎：运行中
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📚 知识库")
    st.markdown("""
    <div style="color: #334155;">
        • 仪器操作：3条<br>
        • 故障排查：3条<br>
        • 实验内容：16条<br>
        <br>
        <small style="color: #64748b;">含通信原理实验指导书10个核心实验</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ❓ 使用说明")
    st.markdown("""
    <div style="color: #334155; font-size: 0.9em;">
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
    <div style="color: #64748b; font-size: 0.8em; text-align: center;">
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
            st.session_state.pending_question = question

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
            placeholder="例如：示波器波形不稳定怎么办？（也可上传图片后提问）",
            key="user_question",
            label_visibility="collapsed"
        )
    with col2:
        submit_button = st.form_submit_button("🚀 提问", use_container_width=True)

# 快速提问点击后存入 pending_question，这里取出并直接提交
pending_question = st.session_state.pop("pending_question", None)

if pending_question or (submit_button and (user_input.strip() or image_data_url)):
    if pending_question:
        question_text = pending_question
    else:
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
        - 📚 专属知识库（22条通信原理实验条目，含指导书10个核心实验）
        - 📷 图片识别（示波器波形、仪器面板等）
        - 🧠 多模态大模型（SenseNova 6.8 Flash-Lite）
        - 🔍 智能检索 + 知识增强生成

        技术栈：Python + Streamlit + RAG + SenseNova多模态大模型API

        项目团队：何佳源、邓坤、郭朝龙
        指导教师：周春月
        """)
