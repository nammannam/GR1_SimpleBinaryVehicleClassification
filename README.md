# GR1_SimpleBinaryVehicleClassification

## Thông tin dự án
**Giảng viên hướng dẫn:** TS.Đỗ Công Thuần

**Sinh viên thực hiện:**
- Họ tên: Nguyễn Khánh Nam
- Mã số sinh viên: 20225749
- Lớp: Việt Nhật 03 - K67

**Môn học:** Nghiên cứu đồ án 1 - GR1
---

## Mô tả dự án
Dự án phân loại nhị phân phương tiện giao thông sử dụng TensorFlow/Keras để phân biệt giữa xe hơi (car) và máy bay (airplane) từ bộ dữ liệu CIFAR-10.

**Bộ dữ liệu:** [CIFAR-10 Dataset](https://huggingface.co/datasets/uoft-cs/cifar10) - Hugging Face

## Tính năng chính
- Model CNN đã được huấn luyện và lưu sẵn
- Phân loại nhị phân: Car vs Airplane
- Hỗ trợ test với ảnh mới
- Ghi log training với TensorBoard

## Yêu cầu hệ thống
- Anaconda hoặc Miniconda
- GPU NVIDIA (khuyến nghị, có hỗ trợ CUDA)
- Python 3.8

### Cài đặt CUDA và cuDNN (cho GPU)
Nếu bạn muốn sử dụng GPU để tăng tốc training:

1. **CUDA Toolkit 11.2**: [Download CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)
2. **cuDNN 8.1**: [Download cuDNN 8.1 for CUDA 11.2](https://developer.nvidia.com/cudnn)

*Lưu ý: Cần đăng ký tài khoản NVIDIA Developer để tải cuDNN*

## Hướng dẫn cài đặt và chạy

### Bước 1: Tạo môi trường Python từ file env.yaml

```bash
# Tạo môi trường từ file env.yaml
conda env create -f env.yaml

# Kích hoạt môi trường
conda activate ImageClassification_testing
```

### Bước 2: Khởi chạy Jupyter Notebook

```bash
# Khởi động jupyter notebook
jupyter lab
```

### Bước 3: Chạy notebook

1. Mở file `VehicleClassification.ipynb`
2. **Model đã được huấn luyện và lưu sẵn**, bạn chỉ cần chạy **2 cell cuối cùng** để test:
   - Cell thứ 2 từ cuối lên: Load model đã lưu
   - Cell cuối cùng: Test với ảnh mẫu

## Cấu trúc thư mục

```
├── VehicleClassification.ipynb    # Notebook chính
├── env.yaml                       # File cấu hình môi trường
├── modelSave/                     # Thư mục chứa model đã huấn luyện
│   └── BinaryVehicleClassification.h5
├── data_to_newTesting/           # Dữ liệu test
│   └── VehicleClassification/    # Ảnh test cho phân loại xe
├── logs/                         # TensorBoard logs
└── README.md                     # File hướng dẫn này
```

## Cách sử dụng

### Test nhanh (Model đã sẵn sàng)
1. Kích hoạt môi trường: `conda activate ImageClassification_testing`
2. Chạy jupyter: `jupyter notebook`
3. Mở `VehicleClassification.ipynb`
4. Chạy 2 cell cuối cùng trong section "Full Testing"

### Huấn luyện lại model (tuỳ chọn)
Nếu bạn muốn huấn luyện lại model từ đầu:
1. Chạy tất cả các cell từ đầu notebook
2. Thời gian training tuỳ vào GPU

### Thêm ảnh test mới
1. Thêm ảnh vào thư mục `data_to_newTesting/VehicleClassification/`
2. Sửa đường dẫn trong cell cuối cùng
3. Chạy cell để xem kết quả phân loại

## Thông tin model
- **Kiến trúc**: CNN với 3 lớp Conv2D + BatchNormalization + MaxPooling
- **Input**: Ảnh 32x32x3 (RGB)
- **Output**: Sigmoid (0: Airplane, 1: Car)
- **Accuracy**: ~90% trên validation set
- **Training time**: ~100 epochs

## Lưu ý
- Model sử dụng GPU nếu có sẵn, fallback về CPU
- Ảnh input sẽ được resize về 32x32 pixels
- Model đã được lưu sẵn tại `modelSave/BinaryVehicleClassification.h5`

## Troubleshooting

### Lỗi CUDA/GPU
```bash
# Kiểm tra GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Lỗi môi trường
```bash
# Xoá và tạo lại môi trường
conda env remove -n ImageClassification_testing
conda env create -f env.yaml
```

### Lỗi Jupyter
```bash
# Cài đặt lại jupyter trong môi trường
conda activate ImageClassification_testing
conda install jupyter
```
