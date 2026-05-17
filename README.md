# stroke-prediction-project
# 🛠️ Chi tiết Các Module Mã nguồn (Source Code Breakdown)
## 🧱 Module 1: NumPy & Linear Algebra Matrix (src/numpy_tasks.py)
Tập trung vào tối ưu hóa tính toán vectơ, xử lý mảng đa chiều mà không sử dụng vòng lặp thuần (vectorized operations):

Q1 & Q2: Tải các đặc trưng số liên tục (age, avg_glucose_level, bmi), thực hiện xử lý giá trị khuyết (NaN) bằng kỹ thuật điền khuyết giá trị trung bình (np.nanmean) và xuất thống kê mô tả (Mean, Median, Std).

Q3 & Q7: Xây dựng cấu hình chuẩn hóa Min-Max và Z-score chuẩn hóa dữ liệu từ đầu (from scratch) để đưa các biến về cùng thang đo.

Q4 & Q5: Sử dụng Boolean Indexing lọc tệp bệnh nhân nguy cơ cao và tính toán ma trận tương quan phi tham số.

Q6 & Q8: Vectơ hóa hoàn toàn phép tính khoảng cách Pairwise Euclidean và độ tương đồng góc Cosine giữa các vectơ bệnh nhân.

Q9 & Q10: Tự phát triển thuật toán giảm chiều dữ liệu Phân tích thành phần chính (Manual PCA) và chia lô xử lý dữ liệu lớn (Batch Processing).

## 📐 Module 2: Math for AI & Optimization (src/math_tasks.py)
Vận dụng các nền tảng toán học nâng cao nhằm xây dựng bộ não tối ưu cho các mô hình AI:

Q1 & Q2: Biểu diễn không gian ma trận, tính toán Hạng ma trận (Matrix Rank) và giải phương trình Hồi quy tuyến tính bằng Phương trình chuẩn tắc (Normal Equation).

Q3 & Q9: Tính toán xác suất nền và xác suất có điều kiện dựa trên Định lý Bayes, hỗ trợ phân tầng rủi ro sức khỏe chuyển hóa.

Q4 & Q5: Tính đạo hàm riêng (Gradient) của hàm mất mát Bình phương sai số trung bình (MSE Loss) và thiết lập ma trận hiệp biến.

Q6 & Q7: Phân rã ma trận thông qua Trị riêng/Vectơ riêng (Eigen Decomposition) và Phân rã giá trị đơn lẻ (SVD Analysis).

Q8 & Q10: Tự tay lập trình thuật toán tối ưu hóa Gradient Descent để giám sát đường cong hội tụ Loss, đồng thời đánh giá tác động chống học vẹt (Overfitting) của L1 (Lasso) vs L2 (Ridge) Regularization.

## 🐼 Module 3: Pandas & Advanced EDA (src/pandas_tasks.py)
Thực hiện các chiến lược xử lý dữ liệu phức tạp, kỹ nghệ đặc trưng và trực quan hóa phân tích cấu trúc:

Q1 & Q2: Kiểm tra phân phối dữ liệu (Inspection) và thanh lọc các bản ghi phân loại dị biệt hoặc nhiễu hệ thống.

Q3 & Q6: Phân tích dữ liệu đa chiều sử dụng kỹ thuật gom nhóm phức hợp (Aggregation) và Bảng xoay Pivot Table (Work Type vs Stroke).

Q4: Tạo lập đặc trưng mới (Feature Engineering): Glucose-to-BMI Ratio, đóng vai trò là một dấu ấn sinh học cấu trúc nâng cao.

Q7 & Q8: Phân đoạn dữ liệu liên tục thành các biến danh mục tầng (Age Groups & BMI Categories) bằng hàm pd.cut.

Q5 & Q9: Thiết lập hệ thống trực quan hóa tự động lưu trữ và hiển thị đồ thị phân phối (Histograms), biểu đồ hộp (Boxplots) và biểu đồ nhiệt (Correlation Heatmaps).

Q10: Thực hiện mã hóa One-Hot Encoding (pd.get_dummies) kèm cấu hình triệt tiêu đa cộng tuyến (drop_first=True) để xuất bản tập dữ liệu hoàn hảo.

## 📊 Insights Cốt lõi Từ Dự Án (Core Project Insights)
Khung rủi ro theo độ tuổi (Age Risk Horizon): Số liệu chứng minh đột quỵ có xu hướng tập trung cực kỳ dày đặc ở nhóm người cao tuổi (Senior). Các ca mắc ở độ tuổi thanh thiếu niên (Child/Adult) chiếm tỷ lệ rất thấp và được coi là các biến dị biệt thống kê.

Sát thủ thầm lặng (Hypertension & Glucose Anomalies): Kiểm định Bayes và biểu đồ Boxplot đồng nhất khẳng định: Bệnh nhân có tiền sử cao huyết áp và mức đường huyết trung bình cao (kèm dải biến thiên ngoại lai rộng) sở hữu tỷ lệ đột quỵ áp đảo. Đây là hai thuộc tính dự báo mang trọng số cao nhất.

Tính trung lập của Giới tính (Gender Neutrality): Ma trận tương quan cho thấy thuộc tính Giới tính hoàn toàn không có mối liên hệ tuyến tính hay tác động mang ý nghĩa thống kê đến rủi ro đột quỵ.

# 🚀 Hướng dẫn Chạy Dự án (Quick Start Guide)
1. Cài đặt Môi trường
Đảm bảo máy tính của bạn đã cài đặt Python 3.8+ và các thư viện cốt lõi:

Bash
pip install numpy pandas matplotlib seaborn scikit-learn
2. Sử dụng trên Jupyter Notebook (analysis.ipynb)
Mở file Notebook lên, hệ thống đã cấu hình sẵn lệnh ma thuật tự động nạp mã nguồn từ thư mục src/:

Python
# Import toàn bộ pipeline
import sys
sys.path.append('..')
from src.numpy_tasks import *
from src.math_tasks import *
from src.pandas_tasks import *
Bạn chỉ cần bấm Run All trên thanh công cụ của Notebook để chứng kiến toàn bộ hệ thống số liệu chạy tuần tự, in kết quả tính toán toán học, hiển thị các biểu đồ rực rỡ và xuất file dữ liệu sạch ra thư mục data/processed/.
