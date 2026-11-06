# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** Python: 3.13.18
PyTorch: 2.11.0+cpu
Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Record này nói về toàn bộ bức ảnh (image-level classification), không phải một vật thể riêng biệt.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Danh sách tên lớp được định nghĩa sẵn trong taxonomy đi kèm checkpoint, không phải do model tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   Ba trường này có vai trò khác nhau: class_id Mã định danh của lớp
   class_name Tên dễ đọc của lớp
   taxonomy_name Bộ phân loại/taxonomy mà lớp thuộc về
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Guideline cần quy định rõ tiêu chí chọn một nhãn duy nhất cho toàn bộ ảnh.
6. Vì sao không được lấy `score` của model làm ground truth?
   score là điểm tin cậy do mô hình dự đoán, không phải nhãn đúng đã được xác minh.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
   {
   "class_name": "person",
   "score": 0.91,
   "bbox_xyxy": [x1, y1, x2, y2],
   "bbox_width": x2 - x1,
   "bbox_height": y2 - y1
   }
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   Bounding box được biểu diễn bởi bốn tọa độ:
   bbox_xyxy = [x1, y1, x2, y2]
   x1: tọa độ x của góc trên-trái.
   y1: tọa độ y của góc trên-trái.
   x2: tọa độ x của góc dưới-phải.
   y2: tọa độ y của góc dưới-phải.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Khi hạ confidence threshold, mô hình giữ lại nhiều prediction hơn, bao gồm cả những prediction có độ tin cậy thấp.
   Reviewer cần:
    - Kiểm tra các bounding box mới được giữ lại.
    - Xác định các box đúng và sai.
    - Phát hiện vật thể bị bỏ sót hoặc nhận diện nhầm.
    - Kiểm tra box có bao trọn vật thể đúng cách không.
    - Xử lý các trường hợp trùng box hoặc box không rõ ràng.
      Vì vậy, hạ ngưỡng có thể tăng khả năng phát hiện vật thể nhưng cũng làm tăng khối lượng công việc kiểm duyệt.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Có thể viết guideline như sau:
   Người gán nhãn cần vẽ bounding box bao quanh toàn bộ phần vật thể nhìn thấy được. Box phải sát với biên ngoài của vật thể, không bao gồm các vùng nền không cần thiết. Với vật thể bị che khuất, chỉ bao phần nhìn thấy được, không tự suy đoán phần bị che. Với vật thể bị cắt mép ảnh, box được phép chạm biên ảnh.
   Đây là quy tắc áp dụng được cho nhiều ảnh, không chỉ ảnh kitchen.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Với vật thể bị che khuất, chỉ gán nhãn phần vật thể nhìn thấy được. Không vẽ box dựa trên phần bị che mà người gán nhãn không thể quan sát.

    Nếu không thể xác định vật thể đó thuộc lớp nào hoặc ranh giới box không rõ, cần escalation lên Lab Coach để thống nhất cách xử lý.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
   instance_id = kitchen-001
   class_name = person
   score = 0.899318
   polygon_point_count = 348
   5 điểm đầu của polygon_xy (đơn vị pixel, ảnh 640×427):
   [446.0, 70.0]
   [445.0, 71.0]
   [444.0, 71.0]
   [443.0, 72.0]
   [442.0, 72.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Bounding box chỉ xác định vùng hình chữ nhật bao quanh vật thể, trong khi polygon mô tả đường biên và hình dạng thực tế của vật thể.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   instance_id là mã định danh của một đối tượng cụ thể trong kết quả segmentation.
   Câu “instance_id không phải class ID và không phải tracking ID” có nghĩa:

- Hai đối tượng cùng lớp bowl có thể có class_id giống nhau.
- Nhưng chúng là hai instance khác nhau nên có instance_id khác nhau.
- tracking_id thường được sử dụng trong bài toán tracking để duy trì danh tính đối tượng qua nhiều frame.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Người gán nhãn cần vẽ đường biên mask sát với đường viền thực tế của vật thể, chỉ bao gồm các pixel thuộc về vật thể đó. Không bao gồm vùng nền hoặc pixel thuộc vật thể khác. Với các vùng bị che khuất, chỉ gán nhãn phần có thể quan sát được, không tự suy đoán phần bị che. Các đường biên cần được vẽ nhất quán giữa các ảnh.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   Dining table và bowl.
   Mình chọn escalation nếu trường hợp này chưa được quy định trong guideline, vì việc tự suy đoán có thể làm giảm tính nhất quán của dữ liệu segmentation.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                               | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                   | Annotator làm gì?                                                               | Reviewer xem gì?                                                                                                                                         |
| --------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một nhãn cho cả ảnh (vd: kitchen, traffic). Đơn vị nhãn = 1 ảnh.            | Ảnh có nhiều vật thể nên nhãn có thể mơ hồ. Ví dụ ảnh kitchen có person, bowl, oven, spoon, dining table ?                           | Chọn đúng 1 nhãn theo guideline                                                 | Nhãn có khớp guideline không, các ảnh nhập nhằng có được xử lý nhất quán không.                                                                          |
| Phát hiện vật thể     | Mỗi vật thể = 1 bbox bbox_xyxy + class_name. Đơn vị nhãn = 1 box.           | traffic-017 và traffic-022 (đều car, bbox gần như trùng nhau: [311.65, 231.04, ...] và [311.54, 232.06, ...]) là box trùng lặp.      | Vẽ 1 box chặt cho mỗi vật thể thật, chọn đúng class, không vẽ trùng.            | Box có thiếu/thừa/trùng không, box có sát vật thể không, class có đúng không.                                                                            |
| Instance segmentation | Mỗi vật thể = 1 polygon (polygon_xy) + class. Đơn vị nhãn = 1 mask/polygon. | kitchen-005 (dining table, 598 điểm) có đoạn nhảy bất thường, vd từ [251, 399] sang [25, 351], và polygon bao cả vùng có bowl/người. | Vẽ viền quanh vật thể, xử lý vùng bị che khuất, tách từng instance (không gộp). | Polygon có tự cắt hoặc nhảy điểm không, mask có tràn sang vật khác không, có bị tách/gộp instance sai không. Đối chiếu trên segmentation_prediction.png. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Chỉ dùng ảnh và dữ liệu mẫu của bài lab (COCO) trong môi trường local, không tải lên, chia sẻ hay đưa dữ liệu ra ngoài phạm vi bài lab, và không dùng ảnh chứa thông tin cá nhân nhạy cảm (khuôn mặt, biển số, giấy tờ...) khi chưa được phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Mình sẽ dừng thao tác ngay, không xử lý hay lưu thêm, rồi báo cho giảng viên/người phụ trách lab (hoặc mentor/trưởng nhóm dự án) để họ quyết định xử lý, ví dụ xóa hoặc thay bằng dữ liệu hợp lệ.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
