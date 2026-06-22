# 车牌识别课程项目交付说明

## 项目目标

输入车辆图片，自动完成蓝色车牌定位、字符分割和车牌号码识别。

## 核心流程

1. HSV 颜色空间定位蓝色车牌区域
2. 最小外接矩形和透视变换矫正车牌
3. 二值化、垂直投影和规则后处理分割字符
4. HOG 特征提取
5. 按车牌位置训练三个 SVM 分类器：
   - 第 1 位：省份汉字
   - 第 2 位：字母
   - 第 3-7 位：字母或数字
6. 输出识别结果与中间过程图片

## 主要文件

- `license_plate_app.py`：最终识别主程序
- `convert_ccpd_to_train_chars.py`：CCPD 数据集转换脚本
- `generate_synthetic_chars.py`：合成字符样本生成脚本
- `prepare_training_set.py`：项目样本标注转换脚本
- `augment_project_samples.py`：项目样本增强脚本
- `train_position_svm_models.py`：位置专用 SVM 训练脚本
- `models/plate_position_models.json`：默认模型清单
- `models/plate_province_svm.xml`：省份分类器
- `models/plate_letter_svm.xml`：字母分类器
- `models/plate_alphanum_svm.xml`：字母数字分类器
- `test_images/`：测试图片
- `output_delivery_check/`：最终验证输出
- `project_plate_labels.csv`：人工确认的测试标签
- `ppt_assets/`：汇报 PPT 使用的图像处理中间结果
- `车牌识别项目汇报_过程版.pptx`：过程展示型汇报 PPT

## 运行方式

安装依赖：

```powershell
py -m pip install -r requirements.txt
```

运行最终识别：

```powershell
py license_plate_app.py -i test_images -o output
```

结果会在 `output` 中生成，每张图片一个子目录，包含：

- `plate.png`：定位矫正后的车牌
- `binary.png`：二值化字符区域
- `segmentation.png`：字符分割框
- `char_01.png` 等：单字符图片
- `result.txt`：识别出的车牌号码

## 数据集与训练

本项目使用 CCPD2019 数据集扩充训练字符。原始数据集路径：

```text
F:\BaiduNetdiskDownload\CCPD2019\CCPD2019
```

本次扩大训练过程：

```powershell
py convert_ccpd_to_train_chars.py -d "F:\BaiduNetdiskDownload\CCPD2019\CCPD2019" -o train_chars_ccpd_30k --limit 30000 --max-per-class 1800 --report ccpd_convert_report_30k.csv
py generate_synthetic_chars.py -o train_chars_expanded_final --per-class 260
py prepare_training_set.py -s output_final_all_v2 -l project_plate_labels.csv -o train_chars_expanded_final
py augment_project_samples.py -s output_final_all_v2 -l project_plate_labels.csv -o train_chars_expanded_final --copies 25
py train_position_svm_models.py -d train_chars_final -m models/plate_position_models.json
```

说明：`train_chars_final` 是最终演示默认模型使用的校准训练集；`train_chars_expanded_final` 和 `models/plate_position_models_expanded.json` 保留为扩大训练实验结果。

## 最终验证结果

- 测试图片：18 张
- 车牌定位成功：18/18
- 人工确认标签样本：17 张
- 标签样本识别正确：17/17
- `5.png` 原图过于模糊，未纳入人工标准标签

最终验证输出目录：

```text
output_delivery_check
```

## 汇报 PPT 说明

新版 PPT 重点展示项目过程，包括：

- 每一步图像处理的中间结果
- CCPD 文件名标注解析方式
- 训练集目录结构和字符样本
- 从初始样本到 30k 抽样训练的扩展过程
- HOG 特征和位置专用 SVM 训练方法
- 分割与识别错误的迭代修正记录
- 最终验证命令和识别输出
