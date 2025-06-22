import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pickle

# --- 这部分字体设置代码保持不变 ---
sns.set_style('whitegrid')
# 注意：请确保你的字体路径正确
try:
    font_path = '../.font/Arial Unicode.ttf' # 假设字体文件在当前目录，如果不是请修改路径
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    print(f"成功注册字体，字体名为: {font_name}")
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
except FileNotFoundError:
    print("警告：未找到指定的字体文件，将使用默认字体。")


date = {
    'both': '06221654',
    'agg': '06221729',
    'time': '06221751',
}

for type, time in date.items():
    print(f"特征工程类型: {type}")

    # --- 数据加载部分保持不变 ---
    with open('Processed_Data/' + type + 'train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    feature_names = train_df.drop('label', axis=1).columns.tolist()
    print("train_df 加载成功")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(f'model/XGBModel/XGBModel_{time}.json')
    print("XGBoost 模型加载成功。")

    importances = xgb_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n特征重要性 (Top 10):")
    print(feature_importance_df.head(10))

    # =============================================================
    # ================ 开始修改绘图部分 ============================
    # =============================================================

    # 1. 准备用于绘图的数据，只取前20个
    top_n = 10
    plot_data = feature_importance_df.head(top_n).sort_values('importance', ascending=True)

    # 2. 调整画布大小，以更好地容纳大字体
    plt.figure(figsize=(14, 10)) 
    
    # 3. 【核心修改】使用 plt.barh 并设置 height 参数使横条变细
    plt.barh(plot_data['feature'], plot_data['importance'], height=0.5) # <-- 修改点1: height从默认0.8改为0.5

    # 4. 【核心修改】为所有文本元素设置更大的字体
    plt.title(f'XGBoost 特征重要性 (Top {top_n}) - 类型: {type}', fontsize=18) # <-- 修改点2: 增大标题字体
    plt.xlabel('重要性得分 (Gain)', fontsize=15) # <-- 修改点3: 增大X轴标签字体
    plt.ylabel('特征名称', fontsize=15)           # <-- 修改点4: 增大Y轴标签字体
    plt.xticks(fontsize=17)                     # <-- 修改点5: 增大X轴刻度字体
    plt.yticks(fontsize=17)                     # <-- 修改点6: 增大Y轴特征名称字体

    # 5. 优化网格线和布局
    plt.grid(True, axis='x', linestyle='--', alpha=0.7) # 只保留垂直网格线，更清爽
    plt.tight_layout() # 自动调整元素间距，防止标签重叠

    # 6. 保存图像
    plt.savefig(f"analysis/xgb_{type}_feature_importance.png", dpi=300) # 增加dpi参数以提高保存图片的清晰度
    plt.show() # 在屏幕上显示图像