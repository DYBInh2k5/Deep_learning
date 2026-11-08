# Deep Learning — Ghi chú và thực hành

Tổng hợp kiến thức và bài tập trong repository: nền tảng học máy, toán học, perceptron, TensorFlow/Keras, phân loại, hồi quy và generalization.

## Mục lục

- [Tổng quan](#tổng-quan)
- [Mạng nơ-ron và quá trình học](#mạng-nơ-ron-và-quá-trình-học)
- [Toán học, perceptron và activation](#toán-học-perceptron-và-activation)
- [TensorFlow và Keras](#tensorflow-và-keras)
- [Quy trình học máy](#quy-trình-học-máy)
- [Phân loại, hồi quy và loss](#phân-loại-hồi-quy-và-loss)
- [Đánh giá mô hình](#đánh-giá-mô-hình)
- [Generalization và regularization](#generalization-và-regularization)
- [Notebook](#notebook)
- [Cài đặt](#cài-đặt)

## Tổng quan

Deep Learning là nhánh Machine Learning dùng mạng nơ-ron nhiều tầng để học biểu diễn từ dữ liệu. Mô hình học weights và bias để giảm loss. So với cách làm dựa nhiều vào feature engineering thủ công, mạng sâu có thể học đặc trưng, nhưng cần dữ liệu phù hợp, tiền xử lý đúng và đánh giá cẩn thận.

Các notebook dùng Python, NumPy, Pandas, Matplotlib, scikit-learn, TensorFlow và Keras; bài tập sử dụng MNIST, IMDB, Reuters, Iris, Breast Cancer Wisconsin, California Housing và Boston Housing.

## Mạng nơ-ron và quá trình học

Mạng gồm input layer, hidden layers và output layer. Neuron tính `z = w · x + b`, sau đó `a = activation(z)`, với `x` là đầu vào, `w` là trọng số, `b` là bias.

Quá trình huấn luyện:

1. **Forward pass:** tính đầu ra/dự đoán từ dữ liệu đầu vào.
2. **Loss:** đo sai khác giữa dự đoán và nhãn thật.
3. **Backpropagation:** dùng quy tắc dây chuyền tính gradient của loss theo tham số.
4. **Optimizer:** cập nhật weights và bias nhằm giảm loss.

**Epoch** là một lượt qua tập train; **batch size** là số mẫu dùng cho một cập nhật; **learning rate** quyết định độ lớn bước cập nhật. Optimization làm giảm loss trên train; **generalization** là dự đoán tốt trên dữ liệu mới. **Underfitting** nghĩa là mô hình chưa học đủ; **overfitting** là khớp train tốt nhưng hoạt động kém trên dữ liệu chưa thấy. Dấu hiệu thường gặp là train loss tiếp tục giảm trong khi validation loss ngừng giảm hoặc tăng.

## Toán học, perceptron và activation

Tích vô hướng `x · w = Σ xᵢwᵢ` kết hợp đặc trưng theo trọng số. Gradient descent cập nhật tham số theo:

`w ← w − η · ∂L/∂w`

Trong đó `L` là loss và `η` là learning rate. Learning rate quá lớn có thể làm quá trình học dao động/phân kỳ; quá nhỏ làm học chậm.

Perceptron tính `z = w · x + b`, rồi dùng step function dự đoán lớp. Quy tắc cập nhật minh họa: `w ← w + η(y − ŷ)x`, `b ← b + η(y − ŷ)`. Perceptron biểu diễn ranh giới tuyến tính; notebook thực hành với logic OR.

| Activation | Đặc điểm | Ứng dụng thường gặp |
| --- | --- | --- |
| Step | Ngưỡng đầu ra thành 0/1 | Perceptron căn bản |
| ReLU: `max(0, x)` | Phi tuyến, đơn giản | Hidden layer |
| Sigmoid: `1/(1+e⁻ˣ)` | Đầu ra trong (0, 1) | Phân loại nhị phân |
| Softmax | Vector xác suất có tổng bằng 1 | Phân loại đa lớp |

Phi tuyến cho phép mạng học quan hệ phức tạp; xếp nhiều phép biến đổi tuyến tính mà không có activation vẫn chỉ tạo ra một phép biến đổi tuyến tính.

## TensorFlow và Keras

- **Tensor** là mảng nhiều chiều; xem `shape` và `dtype` để hiểu kích thước và kiểu dữ liệu.
- `tf.constant()` tạo tensor hằng; `tf.Variable()` tạo biến có thể cập nhật.
- `tf.GradientTape()` ghi lại phép toán để TensorFlow tính gradient tự động.
- Keras cung cấp API xây mạng; `Sequential` xếp layer tuần tự, `Dense` là layer kết nối đầy đủ.
- `compile()` cấu hình optimizer/loss/metrics; `fit()` train; `evaluate()` đánh giá; `predict()` dự đoán.
- `history.history` lưu loss và metrics theo epoch, gồm cả validation nếu có.

## Quy trình học máy

1. Xác định mục tiêu, input, label và tiêu chí đánh giá.
2. Khám phá shape, kiểu dữ liệu, phân bố nhãn, giá trị thiếu và mất cân bằng lớp.
3. Tách train, validation, test. Train học tham số; validation chọn mô hình/hyperparameter/epoch; chỉ dùng test để đánh giá sau cùng.
4. Tiền xử lý nhất quán. Tính mean/std trên train rồi áp dụng cho validation/test để tránh rò rỉ dữ liệu.
5. Tạo baseline đơn giản để so sánh.
6. Chọn kiến trúc, output, loss, optimizer và metric đúng với mục tiêu.
7. Train và theo dõi đường cong train/validation.
8. Chọn cấu hình bằng validation, đánh giá cuối cùng trên test.

Các phép tiền xử lý được thực hành: min-max scaling (ví dụ pixel chia 255), standardization `(x − mean_train)/std_train`, vector hóa sequence thành multi-hot, one-hot encoding nhãn và K-fold validation khi dữ liệu ít.

## Phân loại, hồi quy và loss

| Bài toán | Output điển hình | Loss phù hợp | Ví dụ |
| --- | --- | --- | --- |
| Nhị phân | 1 neuron, sigmoid | `binary_crossentropy` | IMDB, Breast Cancer |
| Đa lớp, nhãn số nguyên | C neuron, softmax | `sparse_categorical_crossentropy` | MNIST, Reuters |
| Đa lớp, nhãn one-hot | C neuron, softmax | `categorical_crossentropy` | Iris, Reuters |
| Hồi quy | 1 neuron tuyến tính | `mse` (hoặc loss hồi quy khác) | Housing |

Sparse categorical nhận nhãn nguyên, còn categorical thường nhận nhãn one-hot. Hồi quy dự đoán số liên tục nên dùng sai số như MAE/MSE thay vì accuracy.

## Đánh giá mô hình

- **Accuracy:** tỷ lệ dự đoán đúng; có thể gây hiểu nhầm khi dữ liệu mất cân bằng.
- **Confusion matrix:** số lượng dự đoán đúng/sai theo từng lớp.
- **Precision:** phần dự đoán dương là đúng; **recall:** phần mẫu dương thật được phát hiện.
- **F1-score:** trung bình điều hòa của precision và recall.
- **MAE:** sai số tuyệt đối trung bình; **MSE:** bình phương sai số trung bình, nhấn mạnh lỗi lớn.
- **Learning curves:** train/validation loss và metrics qua epoch giúp phát hiện hội tụ, underfitting và overfitting.
- **Baseline:** mốc đơn giản như luôn đoán lớp phổ biến nhất; mô hình cần cải thiện mốc này.

Không chọn mô hình chỉ dựa trên training accuracy. Dùng test để lặp lại việc chọn hyperparameter làm mất đi tính độc lập của đánh giá test.

## Generalization và regularization

- **Noise features:** đặc trưng nhiễu có thể khiến mô hình học tương quan ngẫu nhiên; đặc trưng toàn số 0 không bổ sung thông tin.
- **Random labels:** mạng có thể ghi nhớ nhãn train ngẫu nhiên; train tốt không chứng minh generalization tốt.
- **Capacity:** mô hình nhỏ dễ underfit; mô hình lớn có thể overfit và tốn tài nguyên. Chọn theo validation.
- **L2:** cộng khoản phạt tỷ lệ với tổng bình phương weights vào loss; hệ số quá lớn có thể gây underfit.
- **Dropout:** tắt ngẫu nhiên activation khi train để hạn chế phụ thuộc neuron; tắt ở inference. Tỷ lệ quá cao làm khó học.
- **Early stopping:** dừng khi validation không còn cải thiện; `restore_best_weights=True` giữ weights tốt nhất.
- **Learning-rate tuning:** so sánh các giá trị trên validation; quá cao làm loss bất ổn, quá thấp làm hội tụ chậm.

Các kỹ thuật cần được so sánh trên cùng cách chia dữ liệu và tiêu chí validation.

## Ví dụ Keras

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=20, batch_size=32,
                    validation_data=(x_val, y_val))
test_loss, test_accuracy = model.evaluate(x_test, y_test)
probabilities = model.predict(x_test)
predicted_labels = (probabilities >= 0.5).astype("int32")
```

Đây là ví dụ nhị phân. Đa lớp thường dùng softmax cùng categorical hoặc sparse categorical cross-entropy; hồi quy thường dùng output tuyến tính và loss hồi quy.

## Notebook

| File | Nội dung |
| --- | --- |
| `C1-ex.ipynb` | Python list/tuple/dictionary, NumPy, Pandas. |
| `C2-ex (1).ipynb` | Perceptron, activation, MNIST, Iris, classification report, confusion matrix. |
| `L3-ex.ipynb` | Tensor/slicing và mạng hai hidden layer cho phân loại nhị phân, đa lớp. |
| `22301763_L4_Ex.ipynb` | TensorFlow/Keras, GradientTape, gradient descent, phân loại nhị phân. |
| `C4_Ex.ipynb` | IMDB, Iris, Reuters, Boston Housing, validation và K-fold. |
| `ex5.ipynb` | Generalization, noise, random labels, baseline, capacity, learning rate, L2, dropout, early stopping. |
| `lab9.ipynb` | Lab tổng hợp lý thuyết, toán, Keras, phân loại và hồi quy. |
| `Ex (1).ipynb` | Bài thực hành tổng hợp chương 1–4. |
| `ID_FullName_Exam_A.ipynb` | Đề IMDB nhị phân và Reuters 5 lớp, chọn epoch và đánh giá. |

Tên file notebook được giữ nguyên như repository.

## Cài đặt

Mở notebook với Jupyter Notebook, JupyterLab hoặc Google Colab; chạy cell theo thứ tự. Cài thư viện:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
```

Dataset có thể cần tải ở lần chạy đầu. California Housing qua scikit-learn cần mạng; phiên bản TensorFlow phụ thuộc Python và phần cứng.

