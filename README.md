# MultiProCom

This repository contains the retained training, evaluation, deployment, and
real-system analysis pipeline for MultiProCom. The model uses five historical
camera/radar observations to predict eight future beam indices.

## Current implementation

- `multimodal_encoders.py`: motion-component visual encoder and four-view radar
  encoder.
- `multiprocom.py`: the original learned-gate RAMF plus GRU autoregressive
  decoder, retained as the GRU-AR ablation and for the full-training paper
  experiment.
- `source_transition_afsp.py`: the final lightweight AFSP. It regularizes the
  parallel future logits with a source-estimated beam-transition prior.
- `train_multiprocom.py`: common model construction and training utilities.
- `train_baseline_fixed_epoch_cross_scene.py`: 140-epoch Strong-light training
  for R-only, V-only, w/o RAMF, and the parallel-decoder anchor.
- `search_source_only_transition_afsp.py`: reproduces the final V23
  source-transition AFSP.
- `evaluate_latest_cross_scene_all_metrics.py`: evaluates all retained methods
  using Top-1/3/5, ±1-beam accuracy, beam MAE, normalized beam-power loss, and
  spectral-efficiency ratio.
- `infer_multiprocom.py`: deployment inference from five camera frames and five
  raw radar ADC frames.

## Retained artifacts

- `experiments/afsp_v23_source_only_transition_final/`: final MultiProCom
  checkpoint and AFSP-selection records.
- `experiments/baseline_cross_scene_epoch140/`: fixed-epoch comparison models.
- `experiments/strong_light_generalization_early_stop/`: retained GRU-AR
  ablation checkpoint.
- `experiments/latest_cross_scene_all_metrics/`: latest four-scene predictive
  performance tables.
- `experiments/multiprocom/`: full-training paper figures, compact checkpoints,
  preprocessing assets, metrics, and complexity results.
- `logs/{Strong_light,Dim light,Obstruction}/`: the latest three repeated
  hardware runs for each control method.
- `experiments/repeated_system_log_results/`,
  `experiments/actual_control_lead/`, and
  `experiments/control_overhead_results/`: current system-performance,
  proactive-control lead, and control-overhead results.

Raw datasets remain external. The default actual-data root is:

```text
/home/ybpeng/Data/ActualMulData/dataset_multimodal_data
```

Generated radar maps under
`experiments/multiprocom/assets/precomputed_radar_maps/` are kept locally and
excluded from Git.

## Main commands

Evaluate the retained cross-scene models:

```bash
python evaluate_latest_cross_scene_all_metrics.py --device cuda
```

Regenerate the latest hardware-log analyses:

```bash
python plot_repeated_system_log_metrics.py
python analyze_actual_control_lead.py
python plot_control_overhead_metrics.py
```

The full-data resubstitution experiment can still be reproduced with:

```bash
bash run_multiprocom_experiments.sh
```

## Deployment inference

Provide five synchronized camera images and five radar ADC `.bin` files in
chronological order:

```bash
python infer_multiprocom.py \
  --images image_t1.jpg image_t2.jpg image_t3.jpg image_t4.jpg image_t5.jpg \
  --radar-adc adc_t1.bin adc_t2.bin adc_t3.bin adc_t4.bin adc_t5.bin \
  --device cuda \
  --output-json prediction.json
```

The default checkpoint is the final V23 source-transition AFSP. `beam_indices`
contains the eight predictions and is zero-based by default; use
`--beam-index-base 1` for a one-based hardware codebook. In a streaming setup,
`--previous-image image_t0.jpg` supplies the frame immediately before the
history window for first-step motion extraction.
