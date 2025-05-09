import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings('ignore')

# 直接设置matplotlib字体路径来解决中文显示问题
def set_matplotlib_font_path():
    # 检查常见的Microsoft YaHei字体路径
    possible_paths = [
        'C:/Windows/Fonts/msyh.ttc',  # Windows 标准路径
        'C:/Windows/Fonts/msyh.ttf',
        '/usr/share/fonts/windows/msyh.ttf'  # Linux 可能路径
    ]
    for path in possible_paths:
        if os.path.exists(path):
            # 直接为matplotlib添加字体文件路径
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(path)
            # 刷新字体缓存
            fm._rebuild()
            return path
    return None

# 应用启动时设置字体路径
font_path = set_matplotlib_font_path()

# 辅助函数：配置中文字体
def setup_chinese_fonts(font_path=None):
    # 如果没有传入字体路径，检查常见路径
    if font_path is None:
        # 检查多个可能的微软雅黑字体路径
        possible_paths = [
            'C:/Windows/Fonts/msyh.ttc',  # Windows 标准路径
            'C:/Windows/Fonts/msyh.ttf',
            '/usr/share/fonts/windows/msyh.ttf'  # Linux 可能路径
        ]
        for path in possible_paths:
            if os.path.exists(path):
                font_path = path
                break
    
    # 设置matplotlib字体，只用Microsoft YaHei
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.family': 'Microsoft YaHei'})
    
    # 如果找到了字体文件路径，使用它
    if font_path:
        chinese_font = FontProperties(fname=font_path)
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        # 为所有matplotlib文本元素设置字体
        from matplotlib import rc
        rc('font', family='Microsoft YaHei', weight='normal', size=9)
        # 特别处理shap瀑布图中的字体
        import matplotlib as mpl
        mpl.rcParams['font.family'] = ['Microsoft YaHei']
        mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        mpl.rcParams['axes.unicode_minus'] = False
        return chinese_font
    return None

# 设置matplotlib中文字体，只用Microsoft YaHei
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.family': 'Microsoft YaHei'})
# 确保plotly也能显示中文
import plotly.io as pio
pio.templates.default = "simple_white"
# 设置plotly的默认字体为Microsoft YaHei
pio.templates["simple_white"].layout.font.family = "Microsoft YaHei"

# 应用启动时设置中文字体
chinese_font = setup_chinese_fonts(font_path)

# 设置页面配置
st.set_page_config(
    page_title="胃癌术后生存预测",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Microsoft YaHei', serif;
        padding: 0.8rem 0;
        border-bottom: 2px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.2rem;
        color: white;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        font-family: 'Microsoft YaHei', serif;
    }
    .description {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
        font-family: 'Microsoft YaHei', sans-serif;
    }
    .section-container {
        padding: 0.8rem;
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 0.8rem;
        height: 100%;
        font-family: 'Microsoft YaHei', sans-serif;
    }
    .results-container {
        padding: 0.8rem;
        background-color: #F0F9FF;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 0.8rem;
        border: 1px solid #93C5FD;
        height: 100%;
        font-family: 'Microsoft YaHei', sans-serif;
    }
    .metric-card {
        background-color: #F0F9FF;
        padding: 0.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
        font-family: 'Microsoft YaHei', sans-serif;
    }
    .disclaimer {
        font-size: 0.75rem;
        color: #6B7280;
        text-align: center;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #E5E7EB;
        font-family: 'Microsoft YaHei', sans-serif;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        border-radius: 0.3rem;
        border: none;
        margin-top: 0.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
    /* 改善小型设备上的响应式布局 */
    @media (max-width: 1200px) {
        .main-header {
            font-size: 1.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
        }
    }
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    /* 优化指标显示 */
    .stMetric {
        background-color: transparent;
        padding: 5px;
        border-radius: 5px;
    }
    /* 改进分割线 */
    hr {
        margin: 0.8rem 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.1), rgba(0,0,0,0));
    }
    /* 仪表盘和SHAP图中的文字加深 */
    .js-plotly-plot .plotly .gtitle {
        font-weight: bold !important;
        fill: #000000 !important;
    }
    .js-plotly-plot .plotly .g-gtitle {
        font-weight: bold !important;
        fill: #000000 !important;
    }
    /* 图表背景 */
    .stPlotlyChart, .stImage {
        background-color: white !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-weight: bold !important;
        color: #1E3A8A !important;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: bold !important;
        font-size: 0.9rem !important;
    }
    /* 紧凑化滑块和单选按钮 */
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        padding: 0.2rem 0.5rem;
        min-height: auto;
    }
    div.stSlider {
        padding-top: 0.3rem;
        padding-bottom: 0.5rem;
    }
    /* 紧凑化标签 */
    p {
        margin-bottom: 0.3rem;
    }
    div.stMarkdown p {
        margin-bottom: 0.3rem;
    }
    /* 美化进度条区域 */
    .progress-container {
        background-color: #f0f7ff;
        border-radius: 0.3rem;
        padding: 0.4rem;
        margin-bottom: 0.5rem;
        border: 1px solid #dce8fa;
    }
    
    /* 改善左右对齐 */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* 确保滑块组件对齐 */
    .stSlider > div {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* 缩小图表外边距 */
    .stPlotlyChart > div, .stImage > img {
        margin: 0 auto !important;
        padding: 0 !important;
    }
    
    /* 使侧边栏更紧凑 */
    section[data-testid="stSidebar"] div.stMarkdown p {
        margin-bottom: 0.2rem;
    }
    
    /* 更紧凑的标题 */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    
    /* 使结果区域更紧凑 */
    .results-container > div {
        margin-bottom: 0.4rem !important;
    }
</style>
""", unsafe_allow_html=True)

# 加载保存的随机森林模型
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf1.pkl')
        # 添加模型信息
        if hasattr(model, 'n_features_in_'):
            st.session_state['model_n_features'] = model.n_features_in_
            st.session_state['model_feature_names'] = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        return model
    except Exception as e:
        st.error(f"⚠️ 模型文件 'rf.pkl' 加载错误: {str(e)}。请确保模型文件在正确的位置。")
        return None

model = load_model()

# 侧边栏配置和调试信息
with st.sidebar:
    st.markdown("### 模型信息")
    if model is not None and hasattr(model, 'n_features_in_'):
        st.info(f"模型期望特征数量: {model.n_features_in_}")
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            st.write("模型期望特征列表:", expected_features)
    
    st.markdown("---")
    st.markdown("### 应用说明")
    st.markdown("""
    本应用基于随机森林算法构建，通过分析胃癌患者的关键临床特征，预测术后三年内的死亡风险。

    **使用方法：**
    1. 在右侧输入患者特征
    2. 点击"开始预测"按钮
    3. 查看预测结果与解释
    """)

# 特征范围定义
feature_ranges = {
    "术中出血量": {"type": "numerical", "min": 0.000, "max": 800.000, "default": 50, 
                                 "description": "手术期间的出血量 (ml)", "unit": "ml"},
    "CEA": {"type": "numerical", "min": 0, "max": 150.000, "default": 8.68, 
           "description": "癌胚抗原水平", "unit": "ng/ml"},
    "白蛋白": {"type": "numerical", "min": 1.0, "max": 80.0, "default": 38.60, 
               "description": "血清白蛋白水平", "unit": "g/L"},
    "TNM分期": {"type": "categorical", "options": [1, 2, 3, 4], "default": 2, 
                 "description": "肿瘤分期", "unit": ""},
    "年龄": {"type": "numerical", "min": 25, "max": 90, "default": 76, 
           "description": "患者年龄", "unit": "岁"},
    "术中肿瘤最大直径": {"type": "numerical", "min": 0.2, "max": 20, "default": 4, 
                          "description": "肿瘤最大直径", "unit": "cm"},
    "淋巴血管侵犯": {"type": "categorical", "options": [0, 1], "default": 1, 
                              "description": "淋巴血管侵犯 (0=否, 1=是)", "unit": ""},
}

# 特征顺序定义 - 确保与模型训练时的顺序一致
if model is not None and hasattr(model, 'feature_names_in_'):
    feature_input_order = list(model.feature_names_in_)
    feature_ranges_ordered = {}
    for feature in feature_input_order:
        if feature in feature_ranges:
            feature_ranges_ordered[feature] = feature_ranges[feature]
        else:
            # 模型需要但UI中没有定义的特征
            with st.sidebar:
                st.warning(f"模型要求特征 '{feature}' 但在UI中未定义")
    
    # 检查UI中定义但模型不需要的特征
    for feature in feature_ranges:
        if feature not in feature_input_order:
            with st.sidebar:
                st.warning(f"UI中定义的特征 '{feature}' 不在模型要求的特征中")
    
    # 使用排序后的特征字典
    feature_ranges = feature_ranges_ordered
else:
    # 如果模型没有feature_names_in_属性，使用原来的顺序
    feature_input_order = list(feature_ranges.keys())

# 应用标题和描述
st.markdown('<h1 class="main-header">胃癌术后三年生存预测模型</h1>', unsafe_allow_html=True)

# 创建两列布局，调整为更合适的比例
col1, col2 = st.columns([3.5, 6.5], gap="small")

with col1:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">患者特征输入</h2>', unsafe_allow_html=True)
    
    # 动态生成输入项 - 更紧凑布局
    feature_values = {}
    
    for feature in feature_input_order:
        properties = feature_ranges[feature]
        
        # 显示特征描述 - 根据变量类型生成不同的帮助文本
        if properties["type"] == "numerical":
            help_text = f"{properties['description']} ({properties['min']}-{properties['max']} {properties['unit']})"
            
            # 为数值型变量创建滑块 - 使用更紧凑的布局
            value = st.slider(
                label=f"{feature}",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                step=0.1,
                help=help_text,
                # 使布局更紧凑
            )
        elif properties["type"] == "categorical":
            # 对于分类变量，只使用描述作为帮助文本
            help_text = f"{properties['description']}"
            
            # 为分类变量创建单选按钮
            if feature == "TNM分期":
                options_display = {1: "I期", 2: "II期", 3: "III期", 4: "IV期"}
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    format_func=lambda x: options_display[x],
                    help=help_text,
                    horizontal=True
                )
            elif feature == "淋巴血管侵犯":
                options_display = {0: "否", 1: "是"}
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    format_func=lambda x: options_display[x],
                    help=help_text,
                    horizontal=True
                )
            else:
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    help=help_text,
                    horizontal=True
                )
                
        feature_values[feature] = value
    
    # 预测按钮
    predict_button = st.button("开始预测", help="点击生成预测结果")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if predict_button and model is not None:
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">预测结果</h2>', unsafe_allow_html=True)
        
        # 准备模型输入
        features_df = pd.DataFrame([feature_values])
        
        # 确保特征顺序与模型训练时一致
        if hasattr(model, 'feature_names_in_'):
            # 检查是否所有需要的特征都有值
            missing_features = [f for f in model.feature_names_in_ if f not in features_df.columns]
            if missing_features:
                st.error(f"缺少模型所需的特征: {missing_features}")
                st.stop()
            
            # 按模型训练时的特征顺序重排列特征
            features_df = features_df[model.feature_names_in_]
        
        # 转换为numpy数组
        features_array = features_df.values
        
        with st.spinner("计算预测结果..."):
            try:
                # 模型预测
                predicted_class = model.predict(features_array)[0]
                predicted_proba = model.predict_proba(features_array)[0]
                
                # 提取预测的类别概率
                death_probability = predicted_proba[1] * 100  # 假设1表示死亡类
                survival_probability = 100 - death_probability
                
                # 创建概率显示 - 进一步减小尺寸
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = death_probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "", 'font': {'size': 14, 'family': 'Microsoft YaHei', 'color': 'black', 'weight': 'bold'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont': {'color': 'black', 'size': 9, 'family': 'Microsoft YaHei'}},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'green'},
                            {'range': [30, 70], 'color': 'orange'},
                            {'range': [70, 100], 'color': 'red'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.6,
                            'value': death_probability}}))
                
                fig.update_layout(
                    height=160,  # 进一步减小高度
                    margin=dict(l=5, r=5, t=5, b=5),  # 减小顶部边距
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font={'family': "Microsoft YaHei", 'color': 'black', 'size': 11},
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 创建风险类别显示
                risk_category = "低风险"
                risk_color = "green"
                if death_probability > 30 and death_probability <= 70:
                    risk_category = "中等风险"
                    risk_color = "orange"
                elif death_probability > 70:
                    risk_category = "高风险"
                    risk_color = "red"
                
                # 显示风险类别和概率 - 使用浅色背景代替白色
                st.markdown(f"""
                <div style="text-align: center; margin: -0.2rem 0 0.3rem 0;">
                    <span style="font-size: 1.1rem; font-family: 'Microsoft YaHei'; color: {risk_color}; font-weight: bold;">
                        {risk_category}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # 显示具体概率数值 - 放入浅色背景容器
                st.markdown(f"""
                <div class="progress-container">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                        <div style="text-align: center; width: 48%;">
                            <div style="font-size: 0.9rem; font-weight: bold; color: #1E3A8A; font-family: 'Microsoft YaHei';">三年生存概率</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #10B981; font-family: 'Microsoft YaHei';">{survival_probability:.1f}%</div>
                        </div>
                        <div style="text-align: center; width: 48%;">
                            <div style="font-size: 0.9rem; font-weight: bold; color: #1E3A8A; font-family: 'Microsoft YaHei';">三年死亡风险</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #EF4444; font-family: 'Microsoft YaHei';">{death_probability:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 添加SHAP可视化部分 - 减小间距
                st.markdown('<hr style="margin:0.3rem 0;">', unsafe_allow_html=True)
                st.markdown('<h2 class="sub-header">预测结果解释</h2>', unsafe_allow_html=True)
                
                try:
                    with st.spinner("正在生成SHAP解释图..."):
                        # 使用最新版本的SHAP API，采用最简洁、最兼容的方式
                        explainer = shap.Explainer(model)
                        
                        # 计算SHAP值
                        shap_values = explainer(features_df)
                        
                        # 使用matplotlib中文字体解决中文显示问题
                        plt.rcParams.update({'font.family': 'Microsoft YaHei'})
                        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        # 使用waterfall图，进一步减小尺寸
                        plt.figure(figsize=(6.5, 3), dpi=200, facecolor='#f8f9fa')  # 使用浅灰背景
                        
                        # 加载微软雅黑字体
                        chinese_font = setup_chinese_fonts(font_path)
                        
                        # 在绘制SHAP瀑布图之前确保字体设置正确
                        import shap.plots as shap_plots
                        # 尝试修改SHAP内部使用的字体
                        from matplotlib import rcParams
                        rcParams['font.family'] = 'Microsoft YaHei'
                        rcParams['font.sans-serif'] = ['Microsoft YaHei']
                        
                        # 自定义SHAP瀑布图显示
                        # 检查是否有字体文件
                        if chinese_font:
                            # 尝试修改SHAP内部绘图函数中的默认字体设置
                            import types
                            
                            # 备份原始的waterfall函数用于后续还原
                            original_waterfall = shap.plots.waterfall
                            
                            # 获取特征名称列表，确保显示中文
                            feature_names = features_df.columns.tolist()
                            
                            # 创建一个定制版本的waterfall函数
                            def custom_waterfall(shap_values, max_display=10, show=True):
                                # 调用原始函数
                                plot = original_waterfall(shap_values, max_display=max_display, show=False)
                                
                                # 获取当前图形的所有文本元素并设置字体
                                for ax in plt.gcf().get_axes():
                                    # 设置标题和坐标轴标签
                                    if ax.get_title():
                                        ax.set_title(ax.get_title(), fontproperties=chinese_font)
                                    ax.set_xlabel(ax.get_xlabel(), fontproperties=chinese_font)
                                    ax.set_ylabel(ax.get_ylabel(), fontproperties=chinese_font)
                                    
                                    # 设置刻度标签
                                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                                        label.set_fontproperties(chinese_font)
                                    
                                    # 设置所有文本元素的字体
                                    for text in ax.texts:
                                        text.set_fontproperties(chinese_font)
                                
                                if show:
                                    plt.show()
                                return plot
                            
                            # 暂时替换SHAP的waterfall函数
                            shap.plots.waterfall = custom_waterfall
                        
                        # 对于多分类模型，选择死亡类(索引1)
                        if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
                            # 多分类情况 - 选择第二个类别(通常是正类/死亡类)
                            shap.plots.waterfall(shap_values[0, :, 1], max_display=7, show=False)
                        else:
                            # 二分类或回归情况
                            shap.plots.waterfall(shap_values[0], max_display=7, show=False)
                        
                        # 还原原始的waterfall函数
                        if chinese_font:
                            shap.plots.waterfall = original_waterfall
                        
                        # 手动修改标题 - 使用中文字体
                        plt.title("特征对预测的影响", fontsize=12, fontproperties=chinese_font if chinese_font else 'Microsoft YaHei', fontweight='bold', color='black')
                        
                        # 修改图中所有文本元素的字体
                        for ax in plt.gcf().get_axes():
                            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                        ax.get_xticklabels() + ax.get_yticklabels()):
                                if item is not None:
                                    if chinese_font:
                                        item.set_fontproperties(chinese_font)
                                    else:
                                        item.set_fontproperties('Microsoft YaHei')
                            
                            # 尝试获取所有文本元素并设置字体
                            for text in ax.texts:
                                if chinese_font:
                                    text.set_fontproperties(chinese_font)
                                else:
                                    text.set_fontproperties('Microsoft YaHei')
                            
                            # 获取特征名称列表
                            feature_names = features_df.columns.tolist()
                            
                            # 尝试直接修改文本元素中的特征名称
                            for text in ax.texts:
                                text_content = text.get_text()
                                for feature in feature_names:
                                    if feature in text_content:
                                        # 确保使用正确的中文名称并设置字体
                                        text.set_text(text_content)
                                        if chinese_font:
                                            text.set_fontproperties(chinese_font)
                                        else:
                                            text.set_fontproperties('Microsoft YaHei')
                        
                        plt.tight_layout()
                        
                        # 保存并显示图
                        plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=200, facecolor='#f8f9fa', 
                                    pad_inches=0.1)
                        plt.close()
                        st.image("shap_waterfall_plot.png")
                        
                        # 添加简要解释 - 更紧凑，使用浅色背景
                        st.markdown("""
                        <div style="background-color: #f0f7ff; padding: 5px; border-radius: 3px; margin-top: 3px; font-size: 0.8rem; border: 1px solid #dce8fa; font-family: 'Microsoft YaHei', sans-serif;">
                          <p style="margin:0; font-family: 'Microsoft YaHei', sans-serif;"><strong>图表解释:</strong> 红色条表示该特征增加死亡风险，蓝色条表示该特征降低死亡风险。数值表示对预测结果的贡献大小。</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as shap_error:
                    st.error(f"生成SHAP图时出错: {str(shap_error)}")
                    st.warning("无法生成SHAP解释图，请联系技术支持。")
                
            except Exception as e:
                st.error(f"预测过程中发生错误: {str(e)}")
                st.warning("请检查输入数据是否与模型期望的特征匹配，或联系开发人员获取支持。")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # 当没有点击预测按钮时，不显示任何内容
        pass

# 添加页脚说明
st.markdown("""
<div class="disclaimer">
    <p style="font-family: 'Microsoft YaHei', sans-serif;">📋 免责声明：本预测工具仅供临床医生参考，不能替代专业医疗判断。预测结果应结合患者的完整临床情况进行综合评估。</p>
    <p style="font-family: 'Microsoft YaHei', sans-serif;">© 2025 | 开发版本 v1.1.0</p>
</div>
""", unsafe_allow_html=True) 