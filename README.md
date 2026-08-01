# MultiProCom

MultiProCom predicts the beam indices for the next eight time steps from five consecutive historical camera and millimetre-wave radar observations.

## Latest method design

The camera frames are first converted into motion-component bounding boxes. A set-attention encoder then captures their geometric and motion information. The radar branch separately encodes the range–Doppler, range–angle, delay–Doppler, and power maps. RAMF adaptively fuses the two modalities according to visual-proposal availability and the encoded modality features while retaining radar as the dominant stream.

A parallel predictor maps the fused historical sequence to base logits for eight future steps. The final AFSP estimates a beam-transition matrix from the Strong-light source-scene labels and propagates the predictive distribution from the preceding step as a transition prior:

\[
\hat{l}_m=l_m+\beta\log(p_{m-1}T+\epsilon).
\]

This design preserves the direct dependence of every prediction step on the multimodal history while introducing ordered future-state consistency. The final configuration uses \(N=5\), \(M=8\), nine candidate beams, and \(\beta=0.4\). AFSP introduces no additional trainable parameters; it stores only a \(9\times9\) transition matrix and two scalar buffers.

## Training

Run the complete training pipeline with:

```bash
bash run_latest_multiprocom_training.sh
```

The script executes the following experiments in order:

1. Train MultiProCom: train the parallel prediction anchor for 140 epochs and construct the source-transition AFSP;
2. Train the modality baselines: R-only and V-only;
3. Train the ablation models: w/o RAMF and w/o AFSP with the GRU-AR decoder.

The default dataset directory is:

```text
/home/ybpeng/Data/ActualMulData/dataset_multimodal_data
```

The runtime configuration can be changed through environment variables:

```bash
DATA_ROOT=/path/to/dataset \
GPU_ID=0 \
RUN_ROOT=experiments/latest_training_run \
bash run_latest_multiprocom_training.sh
```

## Deployment inference

The deployment script accepts five chronologically ordered camera images and five radar ADC `.bin` files:

```bash
python infer_multiprocom.py \
  --images image_t1.jpg image_t2.jpg image_t3.jpg image_t4.jpg image_t5.jpg \
  --radar-adc adc_t1.bin adc_t2.bin adc_t3.bin adc_t4.bin adc_t5.bin \
  --device cuda \
  --output-json prediction.json
```

The `beam_indices` field in the output JSON contains the eight future predictions and uses zero-based indices by default. Add `--beam-index-base 1` when the hardware codebook is one-based. For online deployment, `--previous-image image_t0.jpg` can provide the frame immediately preceding the history window so that motion components are also extracted for the first historical step.

To load a newly trained checkpoint:

```bash
python infer_multiprocom.py \
  --images image_t1.jpg image_t2.jpg image_t3.jpg image_t4.jpg image_t5.jpg \
  --radar-adc adc_t1.bin adc_t2.bin adc_t3.bin adc_t4.bin adc_t5.bin \
  --checkpoint experiments/latest_training_run/multiprocom/selected_checkpoint.pt \
  --device cuda
```
