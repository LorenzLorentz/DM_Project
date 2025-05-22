import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def preprocess(data:pd.DataFrame) -> pd.DataFrame:
    feature = pd.DataFrame(
        data.groupby("customer_id")["customer_gender"].last().fillna(0)
    )

    feature[['goods_id_last','goods_status_last','goods_price_last','goods_has_discount_last','goods_list_time_last','goods_delist_time_last']] = data.groupby('customer_id')[['goods_id','goods_status','goods_price','goods_has_discount','goods_list_time','goods_delist_time']].last()

    feature[['order_total_num_last','order_amount_last','order_total_payment_last','order_total_discount_last','order_pay_time_last','order_status_last','order_count_last','is_customer_rate_last','order_detail_status_last', 'order_detail_goods_num_last', 'order_detail_amount_last','order_detail_payment_last', 'order_detail_discount_last']] = data.groupby('customer_id')[['order_total_num', 'order_amount','order_total_payment', 'order_total_discount', 'order_pay_time', 'order_status', 'order_count', 'is_customer_rate','order_detail_status', 'order_detail_goods_num', 'order_detail_amount','order_detail_payment', 'order_detail_discount']].last()

    feature[['member_id_last','member_status_last','is_member_actived_last']] = data.groupby('customer_id')[['member_id','member_status','is_member_actived']].last()

    feature[['goods_price_min','goods_price_max','goods_price_mean','goods_price_std']] = data.groupby('customer_id',as_index = False)['goods_price'].agg({'goods_price_min':'min','goods_price_max':'max','goods_price_mean':'mean','goods_price_std':'std'}).drop(['customer_id'],axis=1)

    feature[['order_total_payment_min','order_total_payment_max','order_total_payment_mean','order_total_payment_std']]= data.groupby('customer_id',as_index = False)['order_total_payment'].agg({'order_total_payment_min':'min','order_total_payment_max':'max','order_total_payment_mean':'mean','order_total_payment_std':'std'}).drop(['customer_id'],axis=1)

    feature[['order_count']] = data.groupby('customer_id',as_index = False)['order_id'].count().drop(['customer_id'],axis=1)

    feature[['goods_count']] = data.groupby('customer_id',as_index = False)['goods_id'].count().drop(['customer_id'],axis=1)

    feature['customer_province'] = data.groupby('customer_id')['customer_province'].last()

    feature['customer_city'] = data.groupby('customer_id')['customer_city'].last()

    feature[['is_customer_rate_mean','is_customer_rate_sum']] = data.groupby('customer_id')['is_customer_rate'].agg([('is_customer_rate_mean',np.mean), ('is_customer_rate_sum',np.sum)])

    feature['discount'] = feature['order_detail_amount_last']/feature['order_detail_payment_last']

    feature[['member_status_mean','member_status_sum']] = data.groupby('customer_id')['member_status'].agg([('member_status_mean',np.mean), ('member_status_sum',np.sum)])

    feature[['order_detail_discount_mean','order_detail_discount_sum']] = data.groupby('customer_id')['order_detail_discount'].agg([('order_detail_discount_mean',np.mean), ('order_detail_discount_sum',np.sum)])      

    feature[['goods_status_mean','goods_status_sum']] = data.groupby('customer_id')['goods_status'].agg([('goods_status_mean',np.mean), ('goods_status_sum',np.sum)])

    feature[['is_member_actived_mean','is_member_actived_sum']] = data.groupby('customer_id')['is_member_actived'].agg([('is_member_actived_mean',np.mean), ('is_member_actived_sum',np.sum)])  

    feature[['order_status_mean','order_status_sum']] = data.groupby('customer_id')['order_status'].agg([('order_status_mean',np.mean), ('order_status_sum',np.sum)])

    feature['order_detail_count'] = data.groupby('customer_id')['customer_id'].count()
 
    feature[['goods_has_discount_mean','goods_has_discount_sum']] = data.groupby('customer_id')['goods_has_discount'].agg([('goods_has_discount_mean',np.mean), ('goods_has_discount_sum',np.sum)])

    feature[['order_total_payment_mean','order_total_payment_sum']] = data.groupby('customer_id')['order_total_payment'].agg([('order_total_payment_mean',np.mean), ('order_total_payment_sum',np.sum)])

    feature[['order_total_num_mean','order_total_num_sum']] = data.groupby('customer_id')['order_total_num'].agg([('order_total_num_mean',np.mean), ('order_total_num_sum',np.sum)])

    feature['order_pay_time_last'] = pd.to_datetime(feature['order_pay_time_last'])
    feature['order_pay_time_last_m'] = feature['order_pay_time_last'].dt.month
    feature['order_pay_time_last_d'] = feature['order_pay_time_last'].dt.day
    feature['order_pay_time_last_h'] = feature['order_pay_time_last'].dt.hour
    feature['order_pay_time_last_min'] = feature['order_pay_time_last'].dt.minute
    feature['order_pay_time_last_s'] = feature['order_pay_time_last'].dt.second
    feature['order_pay_time_last_weekday'] = feature['order_pay_time_last'].dt.weekday

    t_min=pd.to_datetime('2012-10-11 00:00:00')
    feature['order_pay_time_last_diff'] = (feature['order_pay_time_last'] - t_min).dt.days

    feature['goods_list_time_last'] =pd.to_datetime(feature['goods_list_time_last'])    
    feature['goods_list_time_diff'] = (feature['goods_list_time_last']-t_min).dt.days

    feature['goods_delist_time_last'] =pd.to_datetime(feature['goods_delist_time_last'])    
    feature['goods_delist_time_diff'] = (feature['goods_delist_time_last'] - t_min).dt.days

    feature['goods_time_diff'] =  feature['goods_delist_time_diff'] - feature['goods_list_time_diff']
    
    feature.drop(['goods_list_time_last','goods_delist_time_last','order_pay_time_last'],axis=1,inplace=True)

    feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature.fillna(-999, inplace=True)

    return feature

def label_encode(data:pd.DataFrame | list[pd.DataFrame], feature:str):
    encoder = LabelEncoder()    
    if isinstance(data, list):
        for data_ in data:
            data_[feature] = data_[feature].astype("str") 
            data_[feature] = encoder.fit_transform(data_[feature])
    else:
        data[feature] = data[feature].astype("str") 
        data[feature] = encoder.fit_transform(data[feature])

def main(data_path:str, output_path:str):
    data_raw = pd.read_csv(os.path.join(data_path, "train.csv"))
    data = data_raw[data_raw["order_pay_time"] <= "2013-07-31 23:59:59"]
    label = set(data_raw[data_raw["order_pay_time"] > "2013-07-31 23:59:59"]["customer_id"].dropna())

    train = preprocess(data)
    train['label'] = train.index.map(lambda x:int(x in label))

    test = preprocess(data_raw)

    label_encode([train, test], "customer_province")
    label_encode([train, test], "customer_city")
    
    train.to_pickle(os.path.join(output_path, "train.pkl"))
    test.to_pickle(os.path.join(output_path, "test.pkl"))

if __name__ == "__main__":
    main("Data", "Processed_Data")