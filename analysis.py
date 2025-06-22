import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 设置绘图风格
sns.set_style('whitegrid')
font_path = '../.font/Arial Unicode.ttf'


fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()
print(f"成功注册字体，字体名为: {font_name}")
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False 

# 1. 数据加载 (请将 'your_dataset.csv' 替换为你的文件名)
df = pd.read_csv('Data/train.csv')

if not df.empty:
    print("数据维度 (行, 列):", df.shape)
    print("\n各列数据类型及非空值数量:")
    df.info()

    print("\n数值型特征描述性统计:")
    print(df.describe())

    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percentage (%)': missing_percentage})
    print("\n缺失值统计:")
    print(missing_info[missing_info['Missing Values'] > 0].sort_values(by='Percentage (%)', ascending=False))

    time_cols = ['order_pay_time', 'goods_list_time', 'goods_delist_time']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

if not df.empty:
    # 1. 用户地域分布 (Top 15 省份)
    try:
        plt.figure(figsize=(12, 6))
        df['customer_province'].value_counts().nlargest(15).plot(kind='bar')
        plt.title('Customer Province Distribution(Top 15)')
        plt.xlabel('Province')
        plt.ylabel('Number')
        plt.xticks(rotation=45)
        plt.savefig("analysis/province_distri.png")
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!!用户地域分布")

    # 2. 订单金额分布
    try:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['order_total_payment'], bins=50, kde=True)
        plt.title('Order Price Distribution')
        plt.xlabel('Ordelr Price')
        plt.ylabel('Number')
        plt.xlim(0, df['order_total_payment'].quantile(0.95))
        plt.savefig("analysis/order_price.png")
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!!订单金额分布")

    # 3. 商品价格分布
    try:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['goods_price'], bins=50, kde=True)
        plt.title('Goods Price Distribuion')
        plt.xlabel('Goeds Price')
        plt.ylabel('Number')
        plt.xlim(0, df['goods_price'].quantile(0.95)) # 显示95%分位数以内的分布
        plt.savefig("analysis/price_distri.png")
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!!商品价格分布")
    
    # 4. 订单状态分布
    try:
        plt.figure(figsize=(10, 5))
        df['order_status'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Order Status Distribution')
        plt.ylabel('')
        plt.savefig("analysis/order_status.png")
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!!订单状态分布")

if not df.empty:
    # 1. 用户购买频次分析
    try:
        customer_purchase_freq = df.groupby('customer_id')['order_id'].nunique().sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        sns.histplot(customer_purchase_freq, bins=50, kde=False)
        plt.title('Customer Shopping Times Distribution')
        plt.xlabel('Shopping Times')
        plt.ylabel('Customer Number')
        plt.yscale('log')
        plt.savefig("analysis/shopping_times.png")
        print("\n用户购买次数描述性统计:")
        print(customer_purchase_freq.describe())
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!!用户购买频次分析")

    # 2. 用户总消费金额分析
    try:
        customer_payment = df.groupby('customer_id')['order_total_payment'].sum().sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        sns.histplot(customer_payment, bins=50, kde=True)
        plt.title('Customer Shopping Amount Distribution')
        plt.xlabel('Total Amount')
        plt.ylabel('Customer Number')
        plt.xlim(0, customer_payment.quantile(0.95))
        plt.savefig("analysis/shopping_amount.png")
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!!用户总消费金额分析")

    # 3. 整体销售趋势 (按月)
    if 'order_pay_time' in df.columns:
        try:
            df_time = df.set_index('order_pay_time')
            monthly_sales = df_time['order_total_payment'].resample('M').sum()
            plt.figure(figsize=(15, 6))
            monthly_sales.plot()
            plt.title('Month Level Amount Tendency')
            plt.xlabel('Month')
            plt.ylabel('Total Amount')
            plt.savefig("analysis/month_prefer.png")
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!整体销售趋势 (按月)")

    # 4. 购买时间偏好 (按小时)
    if 'order_pay_time' in df.columns:
        try:
            plt.figure(figsize=(12, 6))
            df['order_hour'] = df['order_pay_time'].dt.hour
            sns.countplot(x='order_hour', data=df, palette='viridis')
            plt.title('Hour Level Amount Distribution')
            plt.xlabel('Hour ')
            plt.ylabel('Total Amount')
            plt.savefig("analysis/hour_prefer.png")
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!购买时间偏好 (按小时)")

    # 5. 数值变量相关性热图
    try:
        plt.figure(figsize=(15, 15))
        corr_cols = ['order_total_num', 'order_amount', 'order_total_payment', 'order_total_discount',
                    'order_detail_goods_num', 'order_detail_amount', 'order_detail_payment', 'goods_price']
        correlation_matrix = df[corr_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Core Variables Relation Heatmap')
        plt.savefig("analysis/var_rel.png")
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!!数值变量相关性热图")