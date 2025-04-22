# Project-Video-Semantic-Segmentation

## Mô tả bài toán

* Mục tiêu bài toán: Đánh nhãn cho mỗi pixel trong từng khung hình thuộc về những lớp tương ứng, phân loại các vùng của khung hình thành các đối tượng hoặc cảnh vật cụ thể, ví dụ như xe cộ, người, cây cối, v.v., trong bối cảnh động của video.
* Đầu vào: Là một chuỗi các khung hình thể hiện cho một video.
* Đầu ra: Đầu ra là một chuỗi các khung hình đã được đánh nhãn cho từng pixel tương ứng với mỗi khung hình đầu vào.

## Một số mô hình

* Mask R-CNN: Sử dụng một mô hình backbone để trích xuất dữ liệu từ khung hình đầu vào. Tiếp đó sử dụng một mạng Region Proposal Network (RPN) để đưa ra vị trí các ô có khả năng chứa đối tượng, Các vùng đề xuất này sau đó được đưa qua một lớp ROIAlign để tạo ra các đặc trưng có kích thước cố định. Cuối cùng, các đặc trưng này đi qua các Fully Connected (FC) layers để dự đoán nhãn lớp (label) của đối tượng, tọa độ bounding box (bounding box regression) và thêm một nhánh riêng biệt để dự đoán mặt nạ (mask) phân đoạn của từng đối tượng ở cấp độ pixel.
* DeepLabv3+: Sử dụng mô hình Encoder-Decoder, trong đó encoder là DeepLabv3 gồm một mạng backbone CNN (thường là ResNet hoặc Xception) để trích xuất đặc trưng, kết hợp với ASPP (Atrous Spatial Pyramid Pooling) để thu thập thông tin ở nhiều mức độ ngữ cảnh. Decoder sử dụng bilinear upsampling để tăng kích thước ảnh, đồng thời kết hợp với đặc trưng từ một tầng nông của encoder để giữ chi tiết biên tốt hơn. Cuối cùng, các lớp CNN bổ sung giúp làm mịn và cải thiện chất lượng phân đoạn.

## Bộ dữ liệu dự kiến sử dụng: 

* Bộ dữ liệu huấn luyện: dansbecker/cityscapes-image-pairs (có sẵn trên kaggle)
* Bộ dữ liệu kiểm thử: Tự quay trên đường phố.

## Mô hình: 
* Là nhánh semantic head kết hợp với backbone efficient net (có thay đổi một chút) được tham khảo từ paper: https://arxiv.org/pdf/2004.02307
* Backbone: Efficient net B5
* Tiêu chí đánh giá: mIOU, AP

## Trạng thái: Đang thực hiện
