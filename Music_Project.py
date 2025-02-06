from datetime import datetime, timedelta,date
import pandas as pd
import numpy as np
from __future__ import division
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import plotly.offline as pyoff
import plotly.graph_objs as go

#------------------------------------------------------------------------------#
#                      PHÂN LOẠI KHÁCH HÀNG THEO RFM SCORE                     #
#------------------------------------------------------------------------------#

url=r'D:\DAS 2046 HCM\Silde and Assignment\Mock Project\Dataset.csv'
# url='https://drive.google.com/uc?id=' + url.split('/')[-2]
tx_data = pd.read_csv(url, encoding='unicode_escape')


print(tx_data.info())
print(tx_data.describe())

tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
tx_data['BillingCountry'].value_counts()

# ==============================Danh sách khách hàng===============================================#
tx_user = pd.DataFrame(tx_data['CustomerId'].unique())
tx_user.columns = ['CustomerId']
tx_user.info()

#-----------------------------------------------------------------#
#                             Recency                             #
#-----------------------------------------------------------------#
# Lấy ngày mua hàng gần nhất của từng khách hàng
tx_max_purchase = tx_data.groupby('CustomerId').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerId','MaxPurchaseDate']
tx_max_purchase.info()

tx_max_purchase['MaxPurchaseDate'] = pd.to_datetime(tx_max_purchase['MaxPurchaseDate'])

tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerId','Recency']], on='CustomerId')

tx_user.Recency.describe()

tx_recency = tx_user[['CustomerId','Recency']]

# Phân cụm khách hàng theo Recency
kmeans = KMeans(n_clusters=5).fit(tx_recency[['Recency']])

tx_user['RecencyCluster'] = kmeans.predict(tx_recency[['Recency']])

tx_user['RecencyCluster'].value_counts()

# sse = {}
# for k in range(1, 11):
#   kmeans = KMeans(n_clusters=k).fit(tx_user[['Recency']])
#   tx_recency['clusters'] = kmeans.labels_
#   sse[k] = kmeans.inertia_
#   print(k,"-> ",kmeans.inertia_)

# import seaborn as sns
# plt.title('The Elbow Method')
# plt.xlabel('k')
# plt.ylabel('SSE')
# sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
# plt.show()

#-----------------------------------------------------------------#
#                             Frequency                           #
#-----------------------------------------------------------------#
# Đếm số lần quay lại của khách hàng
tx_frequency = tx_data.groupby('CustomerId').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerId','Frequency']

# Phân cụm khách hàng theo Frequency
kmeans = KMeans(n_clusters=5).fit(tx_frequency[['Frequency']])
tx_frequency['FrequencyCluster'] = kmeans.predict(tx_frequency[['Frequency']])

#-----------------------------------------------------------------#
#                             Monetary                            #
#-----------------------------------------------------------------#
# tính tổng số tiền khách hàng chi trả
tx_data['Revenue'] = tx_data['Total']
tx_revenue = tx_data.groupby('CustomerId').Revenue.sum().reset_index()

tx_revenue.columns = ['CustomerId','Monetary']

# Phân cụm khách hàng theo Monetary
kmeans = KMeans(n_clusters=5).fit(tx_revenue[['Monetary']])
tx_revenue['MonetaryCluster'] = kmeans.predict(tx_revenue[['Monetary']])

tx_user_frequency = pd.merge(tx_user, tx_frequency, on='CustomerId')

tx_user_rfm = pd.merge(tx_user_frequency, tx_revenue, on='CustomerId')

#-----------Chuẩn hóa giá trị theo thứ tự điểm cụm-----------#
#-----------------------------Recency------------------------#
# Tính giá trị trung bình của Recency trong mỗi cụm
cluster_means = tx_user_rfm.groupby('RecencyCluster')['Recency'].mean().sort_values(ascending=False)
print(cluster_means)

# Tạo ánh xạ lại các cụm: Nhóm với trung bình LỚN nhất sẽ được đánh số 1, tiếp theo là 2, ...
cluster_map = {cluster: rank + 1 for rank, cluster in enumerate(cluster_means.index)}

print(cluster_map)

# Gán lại nhãn cụm theo thứ tự mong muốn
tx_user_rfm['RecencyClusterSort'] = tx_user_rfm['RecencyCluster'].map(cluster_map)
cluster_means = tx_user_rfm.groupby('RecencyClusterSort')['Recency'].mean().sort_values()
print(cluster_means)

#-----------------------------Frequency------------------------#
# Tính giá trị trung bình của Frequency trong mỗi cụm
cluster_means = tx_user_rfm.groupby('FrequencyCluster')['Frequency'].mean().sort_values()
print(cluster_means)

# Tạo ánh xạ lại các cụm: Nhóm với trung bình NHỎ nhất sẽ được đánh số 1, tiếp theo là 2, ...
cluster_map = {cluster: rank + 3 for rank, cluster in enumerate(cluster_means.index)}

print(cluster_map)

# Gán lại nhãn cụm theo thứ tự mong muốn
tx_user_rfm['FrequencyClusterSort'] = tx_user_rfm['FrequencyCluster'].map(cluster_map)
cluster_means = tx_user_rfm.groupby('FrequencyClusterSort')['Frequency'].mean().sort_values()
print(cluster_means)

#-----------------------------Monetary------------------------#
# Tính giá trị trung bình của Monetary trong mỗi cụm
cluster_means = tx_user_rfm.groupby('MonetaryCluster')['Monetary'].mean().sort_values()
print(cluster_means)

# Tạo ánh xạ lại các cụm: Nhóm với trung bình NHỎ nhất sẽ được đánh số 1, tiếp theo là 2, ...
cluster_map = {cluster: rank + 1 for rank, cluster in enumerate(cluster_means.index)}

print(cluster_map)

# Gán lại nhãn cụm theo thứ tự mong muốn
tx_user_rfm['MonetaryClusterSort'] = tx_user_rfm['MonetaryCluster'].map(cluster_map)
cluster_means = tx_user_rfm.groupby('MonetaryClusterSort')['Monetary'].mean().sort_values()
print(cluster_means)

# Gộp các điểm cụm thành RFM score
tx_user_rfm['RFM_Score'] = tx_user_rfm.RecencyClusterSort.astype(str) + tx_user_rfm.FrequencyClusterSort.astype(str) + tx_user_rfm.MonetaryClusterSort.astype(str)

#=======================Mapping RFM score sang Customer Segment=========================#
# Bảng định nghĩa từ Scores sang Segment
score_to_segment = {
    '01_Champions': [555, 554, 544, 545, 454, 455, 445],
    '02_Loyal': [543, 444, 435, 355, 354, 345, 344, 335],
    '03_Potential Loyalist': [553, 551, 552, 541, 542, 533, 532, 531, 452, 451, 442, 441, 431, 453, 433, 432, 423, 353, 352, 351, 342, 341, 333, 323],
    '04_New Customers': [512, 511, 422, 421, 412, 411, 311],
    '05_Promising': [525, 524, 523, 522, 521, 515, 514, 513, 425, 424, 413, 414, 415, 315, 314, 313],
    '06_Need Attention': [535, 534, 443, 434, 343, 334, 325, 324],
    '07_About To Sleep': [331, 321, 312, 221, 213, 231, 241, 251],
    '08_At Risk': [255, 254, 245, 244, 253, 252, 243, 242, 235, 234, 225, 224, 153, 152, 145, 143, 142, 135, 134, 133, 125, 124],
    '09_Cannot Lose Them': [155, 154, 144, 214, 215, 115, 114, 113],
    '10_Hibernating customers': [332, 322, 231, 241, 251, 233, 232, 223, 222, 132, 123, 122, 212, 211],
    '11_Lost customers': [111, 112, 121, 131, 141, 151]
}

# Đảo ngược mapping để dễ tra cứu
mapping = {score: segment for segment, scores in score_to_segment.items() for score in scores}
# Map RFM_Score sang Segment
tx_user_rfm['Segment'] = tx_user_rfm['RFM_Score'].astype(int).map(mapping)

grouped = tx_user_rfm.groupby('Segment').agg(Count=('CustomerId', 'count')
                                                    , Avg_Recency=('Recency', 'mean')
                                                    , Avg_Frequency=('Frequency', 'mean')
                                                    , Avg_Monetary=('Monetary', 'mean')
                                                    )
print(grouped)
print(grouped.sort_values('Count'))


tx_user_rfm.to_csv(r'D:\DAS 2046 HCM\Silde and Assignment\Mock Project\CustomerSegment.csv',index=False)

#------------------------------------------------------------------------------#
#             DỰ ĐOÁN NGÀY MUA HÀNG TIẾP THEO CỦA KHÁCH HÀNG                   #
#------------------------------------------------------------------------------#
tx_data['InvoiceDate']=pd.to_datetime(tx_data['InvoiceDate']).dt.date
tx_3y = tx_data[(tx_data.InvoiceDate < date(2025,1,1))].reset_index(drop=True)

print(tx_3y.info())

print(tx_data.info())

# Lấy ra ngày order
tx_day_order = tx_3y[['CustomerId','InvoiceDate']]
tx_day_order['InvoiceDay'] = pd.to_datetime(tx_3y['InvoiceDate']).dt.date
tx_day_order = tx_day_order.sort_values(['CustomerId','InvoiceDate'])

# Loại bỏ những ngày trùng
tx_day_order = tx_day_order.drop_duplicates(subset=['CustomerId','InvoiceDay'],keep='first')

# Tìm số ngày giữa 3 lần mua gần nhất của khách hàng
tx_day_order['PrevInvoiceDate'] = tx_day_order.groupby('CustomerId')['InvoiceDay'].shift(1)
tx_day_order['T2InvoiceDate'] = tx_day_order.groupby('CustomerId')['InvoiceDay'].shift(2)
tx_day_order['T3InvoiceDate'] = tx_day_order.groupby('CustomerId')['InvoiceDay'].shift(3)

tx_day_order['PrevInvoiceDate'] = pd.to_datetime(tx_day_order['PrevInvoiceDate'])
tx_day_order['T2InvoiceDate'] = pd.to_datetime(tx_day_order['T2InvoiceDate'])
tx_day_order['T3InvoiceDate'] = pd.to_datetime(tx_day_order['T3InvoiceDate'])
tx_day_order['InvoiceDay'] = pd.to_datetime(tx_day_order['InvoiceDay'])

tx_day_order['DayDiff'] = (tx_day_order['InvoiceDay'] - tx_day_order['PrevInvoiceDate']).dt.days
tx_day_order['DayDiff2'] = (tx_day_order['InvoiceDay'] - tx_day_order['T2InvoiceDate']).dt.days
tx_day_order['DayDiff3'] = (tx_day_order['InvoiceDay'] - tx_day_order['T3InvoiceDate']).dt.days

tx_day_diff = tx_day_order.groupby('CustomerId').agg({'DayDiff': ['mean','std']}).reset_index()
tx_day_diff.columns = ['CustomerId', 'DayDiffMean','DayDiffStd']

# Ngày order gần nhất của khách hàng, loại những ngày lần 2 trở đi
tx_day_order_last = tx_day_order.drop_duplicates(subset=['CustomerId'],keep='last')
print(tx_day_order.info())
print(tx_day_order_last.info())

tx_day_order_last = tx_day_order_last.dropna()
tx_day_order_last = pd.merge(tx_day_order_last, tx_day_diff, on='CustomerId')

# Lấy khách hàng duy nhất
tx_user = pd.DataFrame(tx_3y['CustomerId'].unique())
tx_user.columns = ['CustomerId']

# Lấy ngày mua hàng gần nhất của khách hàng
tx_last_purchase = tx_3y.groupby('CustomerId').InvoiceDate.max().reset_index()
tx_last_purchase.columns = ['CustomerId','MaxPurchaseDate']


# Lấy dữ liệu 1 năm tiếp theo để tính khoảng thời gian khách hàng quay lại
tx_next = tx_data[(tx_data.InvoiceDate >= date(2025,1,1)) & (tx_data.InvoiceDate < date(2026,1,1))].reset_index(drop=True)
tx_next_first_purchase = tx_next.groupby('CustomerId').InvoiceDate.min().reset_index()
tx_next_first_purchase.columns = ['CustomerId','MinPurchaseDate']

# Tìm ngày gần nhất khách hàng quay trở lại mua
tx_purchase_dates = pd.merge(tx_last_purchase,tx_next_first_purchase,on='CustomerId',how='left')

tx_purchase_dates['MaxPurchaseDate'] = pd.to_datetime(tx_purchase_dates['MaxPurchaseDate'])
tx_purchase_dates['MinPurchaseDate'] = pd.to_datetime(tx_purchase_dates['MinPurchaseDate'])

tx_purchase_dates['NextPurchaseDay'] = (tx_purchase_dates['MinPurchaseDate'] - tx_purchase_dates['MaxPurchaseDate']).dt.days

tx_user = pd.merge(tx_user, tx_purchase_dates[['CustomerId','NextPurchaseDay']],on='CustomerId',how='left')

# Đặt cho những người chưa quay lại mua giá trị là 9999
tx_user.info()
tx_user = tx_user.fillna(9999)

tx_user = pd.merge(tx_user, tx_day_order_last[['CustomerId','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='CustomerId')
tx_user.info()

#=======Chuyển đổi cột có kiểu dữ liệu chuỗi thành số=======#
tx_class = tx_user.copy()
tx_class = pd.get_dummies(tx_class)
tx_class.info()

#============Định nghĩa các khoảng thời gian khách hàng quay lại theo cụm============#
# >500 ngày   : 0
# >200 ngày   : 1
# <= 200 ngày : 2
tx_class['NextPurchaseDayRange'] = 2
tx_class.loc[tx_class.NextPurchaseDay>200,'NextPurchaseDayRange'] = 1
tx_class.loc[tx_class.NextPurchaseDay>500,'NextPurchaseDayRange'] = 0

tx_class.NextPurchaseDayRange.value_counts()
# NextPurchaseDayRange
# 0    35
# 1    16
# 2     8
tx_class.NextPurchaseDayRange.value_counts()/len(tx_user)*100
#NextPurchaseDayRange
# 0  :  59.322034 ~= 59.32%
# 1  :  27.118644 ~= 27.11%
# 2  :  13.559322 ~= 13.56%


# Chia tập dữ liệu
tx_class = tx_class.drop('NextPurchaseDay',axis=1)

X, y = tx_class.drop('NextPurchaseDayRange',axis=1), tx_class.NextPurchaseDayRange

# from __future__ import division
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree,DecisionTreeRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tìm thuật toán dự đoán tốt nhất
models = []
models.append(("LR",LogisticRegression()))
models.append(("RF",RandomForestClassifier()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))

for name,model in models:
    kfold = KFold(n_splits=2)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    print(name, ': ',cv_result)
# LR :  [0.85714286 1.        ]
# RF :  [0.85714286 1.        ]
# Dtree :  [0.85714286 1.        ]
# XGB :  [0.80952381 1.        ]

#======Chọn DecisionTRee======#
tree_model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

y_pred_train = tree_model.predict(X_train)
X_train['Predict']=y_pred_train
# Đánh giá mô hình
accuracy = accuracy_score(y_train, y_pred_train)
print(f"Độ chính xác (Accuracy): {accuracy:.2f}")
print("\nBáo cáo phân loại:")
print(classification_report(y_train, y_pred_train))

y_pred = tree_model.predict(X_test)
X_test['predict'] = y_pred

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác (Accuracy): {accuracy:.2f}")
print("\nBáo cáo phân loại:")
print(classification_report(y_test,y_pred))


# Dữ liệu chỉ có 59 khách hàng.
# Khó có điểm chính xác 100% => check lại quá trình.
# Nên vẽ thêm biểu đồ để visualize công thức.