# Music-Business-Analysis
# 1. Tổng quan đồ án
## 1.1. Bài toán:
1.	Phân tích tình hình kinh doanh nhạc số theo Doanh thu, Sản phẩm, Khách hàng và Nhân viên trong 4 năm trở lại đây.
2.	Dự đoán khoảng bao lâu thì khách hàng sẽ quay lại mua hàng.
## 1.2. Quá trình thực hiện đồ án:**
- Giai đoạn 1: Nhóm giải quyết bài toán phân tích tình hình kinh doanh nhạc số trong 4 năm trở lại đây; nhóm lấy code slq từ Chinook database trên github, sau đó dùng công cụ Sql Server Management Studio để tạo database DB_Music trên database local.
o	Cấu trúc dữ liệu:
 
o	Vấn đề: Dữ liệu còn thiếu khá nhiều, format ngày tháng năm chưa đồng nhất, kiểu dữ liệu là số nhưng hiện là kiểu chữ. Vì thế phải làm sạch dữ liệu, định dạng lại ngày tháng năm, điền những dữ liệu trống bằng chuỗi “Unknow”.
- Giai đoạn 2: Nhóm cùng nhau đặt ra các câu hỏi để từ đó tìm ra insight có liên quan để phân tích tình hình kinh doanh nhạc số trong 4 năm trở lại đây. Sau đó trực quan hóa dữ liệu bằng công cụ Power BI.
- Giai đoạn 3: Nhóm tiến hành giải quyết bài toán dự đoán bao lâu thì khách hàng sẽ quay lại mua hàng bằng thuật toán Decision Tree Classifier trên công cụ Visual Studio Code.
# 2. Phân tích chi tiết
## 2.1. Bài toán 1:
### 2.1.1 Mô tả dữ liệu:
**Track**
•	TrackId: Mã số bài hát
•	Name: Tên bài hát
•	AlbumId: Mã số Album
•	MediaTypeId: Mã số loại phương tiện truyền thông
•	GenreId: Mã dòng nhạc
•	Composer: Đơn vị biên soạn
•	Milliseconds: số giây bài hát
•	Bytes: Dung lượng bài hát
•	UnitPrice: Giá bài hát

**MediaType:**
•	MediaTypeId: Mã loại phương tiện truyền thông.
•	MediaType_Name: Tên loại phương tiện truyền thông.

**Genre:**
•	GenreId: Mã dòng nhạc
•	Name: Tên dòng nhạc

**Artist:**
•	ArtistId: Mã nghệ sĩ
•	Name: Tên nghệ sĩ

**Album:**
•	AlbumId: Mã album
•	Title: Tên album
•	ArtistId: Mã nghệ sĩ

**Playlist:**
•	PlaylistId: Mã danh sách phát.
•	Playlist_Name: Tên danh sách phát.

**PlaylistTrack:**
•	PlaylistId: Mã danh sách phát
•	TrackId: Mã bài hát.

**Invoice:**
•	InvoiceId: Mã hóa đơn
•	CustomerId: Mã khách hàng mua hóa đơn
•	InvoiceDate: Ngày mua hàng.
•	BillingAddress: Địa chỉ mua hàng.
•	BillingCity: Thành phố nơi mua hàng.
•	BillingState: Bang nơi mua hàng.
•	BillingCountry: Quốc gia nơi mua hàng
•	BillingPostalCode: Mã quốc gia nơi mua hàng
•	Total: tổng tiền hóa đơn.

**InvoiceLine:**
•	InvoiceLineId: Mã dòng hóa đơn (có nhiều dòng hóa đơn trong 1 hóa đơn mua hàng)
•	InvoiceId: Mã hóa đơn
•	TrackId: Mã bài hát
•	UnitPrice: Giá bán bài hát
•	Quantity: số lượng mua bài hát.

**Customer:**
•	CustomerId: Mã khách hàng.
•	FirstName: Tên khách hàng.
•	LastName: Họ khách hàng.
•	Company: Công ty nơi khác hàng làm việc.
•	Address: địa chỉ khách hàng.
•	City: Thành phố nơi khách hàng ở.
•	State: Bang nơi khách hàng ở.
•	Country: Quốc gia nơi khách hàng ở.
•	PostalCode: Mã quốc gia của khách hàng.
•	Phone: số điện thoại khách hàng.
•	Fax: mã fax khách hàng.
•	Email: email khách hàng.
•	SupportRepId: Mã nhân viên đã hỗ trợ khách hàng.

**Employee:**
•	EmployeeId: Mã nhân viên
•	LastName: Họ nhân viên
•	FirstName: Tên nhân viên
•	Title: Chức vụ
•	ReportsTo: Mã nhân viên của người quản lý trực tiếp mà nhân viên báo cáo công việc.
•	BirthDate: Ngày sinh nhân viên.
•	HireDate: Ngày vào làm nhân viên.
•	Address: Địa chỉ nhân viên.
•	City: Thành phố của nhân viên.
•	State: Bang của nhân viên.
•	Country: Quốc gia của nhân viên.
•	PostalCode: Mã quốc gia của nhân viên.
•	Phone: Số điện thoại của nhân viên.
•	Fax: Fax của nhân viên.
•	Email: email của nhân viên.

### 2.1.2. Tiền xử lý dữ liệu: 
Sử dụng công cụ Microsoft Excel để xử lý: định dạng lại các cột ngày tháng thành 1 định dạng chung là yyyy-MM-dd, định dạng các cột vừa có dữ liệu dạng chữ và dạng số thành 1 dạng chung là kiểu dữ liệu chuỗi (text), điền những dòng dữ liệu trống bằng chuỗi “Unkow”.
Hình ảnh ví dụ:
 
Dữ liệu được lưu thành file **Music_Project.xlsx**

2.1.3. Phân loại khách hàng dựa trên bảng RFM Score:
Tìm Recency khách hàng:
tx_max_purchase = tx_data.groupby('CustomerId').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerId','MaxPurchaseDate']
tx_max_purchase.info()

tx_max_purchase['MaxPurchaseDate'] = pd.to_datetime(tx_max_purchase['MaxPurchaseDate'])

tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

Phân cụm Recency thành điểm bằng thuật toán Kmeans:
kmeans = KMeans(n_clusters=5).fit(tx_recency[['Recency']])

tx_user['RecencyCluster'] = kmeans.predict(tx_recency[['Recency']])

Tìm Frequency khách hàng:
tx_frequency = tx_data.groupby('CustomerId').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerId','Frequency']

Phân cụm Frequency thành điểm bằng thuật toán Kmeans
kmeans = KMeans(n_clusters=5).fit(tx_frequency[['Frequency']])
tx_frequency['FrequencyCluster'] = kmeans.predict(tx_frequency[['Frequency']])

Tìm Monetary khách hàng:
tx_data['Revenue'] = tx_data['Total']
tx_revenue = tx_data.groupby('CustomerId').Revenue.sum().reset_index()

Phân cụm Monetary thành điểm bằng thuật toán Kmeans:
kmeans = KMeans(n_clusters=5).fit(tx_revenue[['Monetary']])
tx_revenue['MonetaryCluster'] = kmeans.predict(tx_revenue[['Monetary']])

Gộp các điểm cụm thành RFM score
tx_user_rfm['RFM_Score'] = tx_user_rfm.RecencyClusterSort.astype(str) + tx_user_rfm.FrequencyClusterSort.astype(str) + tx_user_rfm.MonetaryClusterSort.astype(str)

Định nghĩa Score sang Segment và mapping từ RFM Score để phân loại khách hàng:
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

Kết quả được lưu vào file CustomerSegment.csv


2.1.4. Trực quan hóa dữ liệu
Tình hình doanh thu theo thời gian:
 

Tình hình doanh thu theo bài hát:
 
Tình hình doanh thu theo khách hàng:
 

Thống kê thông tin nhân viên:
 
Phần trăm quay lại của khách hàng theo thời gian:
 

Tình hình doanh thu theo các đối tượng khác:
 
2.1.5 Kết luận
- Doanh thu tăng trưởng không ổn định qua các năm, có những thời điểm giảm mạnh. Các thời điểm doanh thu tăng mạnh là 01/2022, 04/2023, 06/2023, 11/2025. Giảm mạnh ở các thời điểm 11/2023 và 02/2025.
- Dự báo doanh thu đến 2026 cho thấy có xu hướng giảm và có khả năng phục hồi nhẹ.
- Tuy nhiên tổng số bài hát bán được giữa các tháng không chênh lệch cao (33-35 bài hát)  số lượng bài hát bán ra ổn định.
- Vì tập khách hàng tập trung chủ yếu là người Châu Âu và Châu Mỹ nên những bài hát thuộc dòng nhạc Rock và Latin được ưa chuộng nhất.
- Tập khách hàng Potential Loyalist là nhiều nhất, chiếm đến 51% trên tổng 8 tập khách hàng. Nên đầu tư giữ chân các khách hàng tiềm năng và triển khai chương trình tri ân các khách hàng thuộc tập khách hàng Champion.
- Phần trăm khách hàng quay lại tương đối cao, những khách hàng bắt đầu mua từ năm 2022 quay lại mua lên đến 50%, tuy nhiên lượng khách hàng còn ít (59 khách hàng) và thời gian quay lại còn cách khá xa (từ 3 tháng trở lên) nên còn cần phải quảng bá để tiếp cận nhiều khách hàng hơn.
2.2. Bài toán 2
2.2.1. Mô tả dữ liệu
Từ file Music_Project.xlsx sẽ chỉ lấy những thông tin về hóa đơn, khách hàng, ngày mua hàng, số lượng mua, tổng tiền, địa chỉ mua hàng để giải quyết bài toán.
Các thu thập dữ liệu: Dùng công cụ SQL Server Management Studio → trỏ đến database DB_Music → Chạy câu truy vấn sau: 
Select I.InvoiceId,CustomerId,InvoiceDate,BillingCountry,Total,sum(quantity) as Quantity
from Invoice I 
join InvoiceLine IL on I.InvoiceId = IL.InvoiceId
GROUP By I.InvoiceId,CustomerId,InvoiceDate,BillingCountry,Total
ORDER BY I.InvoiceId

→ Sau đó lưu vào file Dataset.csv  
 
Tổng quan dữ liệu
2.2.2. Mô tả các biến
•	InvoiceId: Mã hóa đơn.
•	CustomerId: Mã khách hàng mua hóa đơn.
•	InvoiceDate: Ngày mua hàng.
•	BillingCountry: Quốc gia nơi khách mua hàng.
•	Total: tổng tiền hóa đơn khách hàng phải thoanh toán.
•	Quantiy: Số lượng bài hát khách hàng mua.
2.2.3. Tiền xử lý dữ liệu:
- Xử lý định dạng cột InvoiceDate thành kiểu Datetime:
tx_data['InvoiceDate']=pd.to_datetime(tx_data['InvoiceDate'])

 
→ Có 412 dòng dữ liệu.
- Lấy dữ liệu 4 năm trở lại đây:
tx_3y = tx_data[(tx_data.InvoiceDate < date(2025,1,1))].reset_index(drop=True)

 
→ Có 332 dòng dữ liệu
2.2.4. Tiến hành hành bài toán dự đoán:
Bước 1: Tìm số ngày giữa 3 lần mua gần nhất của khách hàng
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

Bước 2: Lấy ngày mua hàng gần nhất của khách hàng
tx_last_purchase = tx_3y.groupby('CustomerId').InvoiceDate.max().reset_index()
tx_last_purchase.columns = ['CustomerId','MaxPurchaseDate']

Bước 3: Lấy dữ liệu 1 năm tiếp theo để tính khoảng thời gian khách hàng quay lại
tx_next = tx_data[(tx_data.InvoiceDate >= date(2025,1,1)) & (tx_data.InvoiceDate < date(2026,1,1))].reset_index(drop=True)
tx_next_first_purchase = tx_next.groupby('CustomerId').InvoiceDate.min().reset_index()
tx_next_first_purchase.columns = ['CustomerId','MinPurchaseDate']

Bước 4: Tìm ngày gần nhất khách hàng quay trở lại mua
tx_purchase_dates = pd.merge(tx_last_purchase,tx_next_first_purchase,on='CustomerId',how='left')

tx_purchase_dates['MaxPurchaseDate'] = pd.to_datetime(tx_purchase_dates['MaxPurchaseDate'])
tx_purchase_dates['MinPurchaseDate'] = pd.to_datetime(tx_purchase_dates['MinPurchaseDate'])

tx_purchase_dates['NextPurchaseDay'] = (tx_purchase_dates['MinPurchaseDate'] - tx_purchase_dates['MaxPurchaseDate']).dt.days

tx_user = pd.merge(tx_user, tx_purchase_dates[['CustomerId','NextPurchaseDay']],on='CustomerId',how='left')

Bước 5: Đặt cho những người chưa quay lại mua tính tới 1 năm tiếp theo giá trị là 9999
tx_user = tx_user.fillna(9999)

Bước 6: Định nghĩa các khoảng thời gian khách hàng quay lại theo cụm
	Trên 500 ngày mới quay trở lại thì thuộc cụm 0
	Trên 200 ngày mới quay trở lại thì thuộc cụm 1
	Từ 200 ngày trở xuống thì thuộc cụm 2
tx_class['NextPurchaseDayRange'] = 2
tx_class.loc[tx_class.NextPurchaseDay>200,'NextPurchaseDayRange'] = 1
tx_class.loc[tx_class.NextPurchaseDay>500,'NextPurchaseDayRange'] = 0
→ Số lượng khách hàng theo cụm:
tx_class.NextPurchaseDayRange.value_counts()
 
→ Tỷ lệ khách hàng trong các cụm:
tx_class.NextPurchaseDayRange.value_counts()

 
Bước 7: Chạy thuật toán dự đoán khoảng thời gian:
Bước 7.1: Xác định biến phụ thuộc X, y:
tx_class = tx_class.drop('NextPurchaseDay',axis=1)
X, y = tx_class.drop('NextPurchaseDayRange',axis=1), tx_class.NextPurchaseDayRange
Bước 7.2: Chia tập dữ liệu thành 2 tập: 
	Tập Train: X_train, y_train
	Tập test: X_test, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Bước 7.3: Training data
	Import thư viện của thuật toán:
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree,DecisionTreeRegressor
	Chạy thuật toán training:
tree_model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

Bước 8: Kết quả dự đoán và đánh giá mô hình
	Kết quả dự đoán:
y_pred_train = tree_model.predict(X_train)
X_train['Predict']=y_pred_train
	Đánh giá mô hình tập Train:
# Đánh giá mô hình
accuracy = accuracy_score(y_train, y_pred_train)
print(f"Độ chính xác (Accuracy): {accuracy:.2f}")
print("\nBáo cáo phân loại:")
print(classification_report(y_train, y_pred_train))

 
	Áp dụng để lấy kết quả dự đoán tập Test:
y_pred = tree_model.predict(X_test)
X_test['predict'] = y_pred
	Đánh giá mô hình tập Test:
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác (Accuracy): {accuracy:.2f}")
print("\nBáo cáo phân loại:")
print(classification_report(y_test,y_pred))

 
 Dựa trên kết quả tập Test ta thấy được mô hình đạt độ chính xác 100% thời gian khách hàng quay lại thực tế như hình dưới đây. Suy ra mô hình đã hoạt động rất tốt.
 

2.2.5. Kết luận
- Mô hình sử dụng dự đoán hoạt động rất tốt do có độ chính xác cao (100%)
- Thời gian để khách hàng quay lại mua hàng là khá cao, một phần vì sản phẩm không có tính hư hao, chỉ mang tính chất về tinh thần cho khách hàng.
- Lượng khác hàng quay lại dưới 200 ngày còn thấp (chỉ chiếm 17%), khách hàng sau hơn 500 ngày quay lại thì khá cao (chiếm 61%)
