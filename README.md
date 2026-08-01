# MultiProCom

MultiProCom 根据连续 5 个历史相机与毫米波雷达观测，预测未来 8 个时刻的波束索引。

## 最新方法设计

模型首先将相机图像转换为运动组件框，并通过集合注意力编码目标几何与运动信息；雷达支路分别编码 range–Doppler、range–angle、delay–Doppler 和 power map。RAMF 根据视觉组件可用性和两种模态特征自适应融合信息，同时保持雷达主导。

融合后的历史序列由并行预测器生成 8 步基础 logits。最终 AFSP 使用 Strong-light 源场景标签估计波束转移矩阵，并将上一预测步的概率分布传播为下一步的转移先验：

\[
\hat{l}_m=l_m+\beta\log(p_{m-1}T+\epsilon).
\]

该设计保留每个预测步对多模态历史的直接依赖，同时引入有序的未来状态约束。最终配置使用 \(N=5\)、\(M=8\)、9 个候选波束和 \(\beta=0.4\)。

## 训练

一键训练脚本为：

```bash
bash run_latest_multiprocom_training.sh
```

默认数据目录为：

```text
/home/ybpeng/Data/ActualMulData/dataset_multimodal_data
```


## 部署推理

部署脚本接收按时间顺序排列的 5 张图像和 5 个雷达 ADC `.bin` 文件：

```bash
python infer_multiprocom.py \
  --images image_t1.jpg image_t2.jpg image_t3.jpg image_t4.jpg image_t5.jpg \
  --radar-adc adc_t1.bin adc_t2.bin adc_t3.bin adc_t4.bin adc_t5.bin \
  --device cuda \
  --output-json prediction.json
```

输出 JSON 中的 `beam_indices` 为未来 8 步预测结果，默认采用 0-based 索引。硬件码本从 1 开始编号时增加 `--beam-index-base 1`。在线部署时可使用 `--previous-image image_t0.jpg` 提供历史窗口前一帧，以提取第一个历史时刻的运动组件。

若需要加载重新训练的权重：

```bash
python infer_multiprocom.py \
  --images image_t1.jpg image_t2.jpg image_t3.jpg image_t4.jpg image_t5.jpg \
  --radar-adc adc_t1.bin adc_t2.bin adc_t3.bin adc_t4.bin adc_t5.bin \
  --checkpoint experiments/latest_training_run/multiprocom/selected_checkpoint.pt \
  --device cuda
```
