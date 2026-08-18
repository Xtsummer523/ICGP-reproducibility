# ICGP Reproducibility Archive

Release date: 2026-08-18  
Author: Tian Xia

This archive contains the data, code, model weights, figure sources, and manuscript files supporting *ICGP: Intent-Conditioned Neighbor-Response Evaluation for Local Multi-Robot Planning under Observation Noise*.

## Contents

- `data/legacy_main/`: 960 selected clean-condition records.
- `data/gaussian_observation_study/`: 3,840 Gaussian-noise records and paired summaries.
- `data/autoregressive_observation_study/`: 1,920 AR(1) records.
- `data/short_dropout_probability_005/` and `data/short_dropout_probability_010/`: two 1,920-record dropout studies.
- `data/prediction_quality/`: 1,440 case-level prediction diagnostics.
- `data/factorial_ablation/`: 3,840 records from the fully crossed architecture-by-conditioning ablation, training data, manifests, four weights, shard outputs, and summaries.
- `data/ros2_gazebo_rviz/`: the retained, single-seed ROS 2/Gazebo/RViz diagnostic and synchronized logs.
- `code/`: simulator, statistical checks, figure-generation code, ROS 2 metrics, and the factorial-ablation runner.
- `figure_sources/fig1/`: editable draw.io source and manuscript export for Fig. 1.
- `manuscript/`: the checked LaTeX source, vector figures, references, and PDF.

## Verify

From the archive root:

```powershell
pip install -r requirements.txt
python code/verify_archive.py
```

The check validates row counts, balanced designs, duplicate keys, file hashes, and the 3,840-record factorial corpus.

## Re-run

Run the observation-noise simulator from `code/observation_noise_experiment/`. Run the crossed ablation from the archive root:

```powershell
python code/factorial_ablation/source_snapshot/scripts/run_architecture_conditioning_ablation.py --mode train --output-dir reproduced/factorial
python code/factorial_ablation/source_snapshot/scripts/run_architecture_conditioning_ablation.py --mode evaluate --output-dir reproduced/factorial --seeds 30
python code/factorial_ablation/source_snapshot/scripts/run_architecture_conditioning_ablation.py --mode merge --output-dir reproduced/factorial --seeds 30
```

Full runs may take time. Model-side latency is hardware-dependent and should not be compared as a byte-identical output.

## Boundaries

The ten-seed Gazebo boundary study includes the retained 30-run summaries and 20 paired-difference rows, but its historical per-step raw logs are not locally available. The separate single-seed ROS 2/Gazebo/RViz diagnostic includes synchronized raw logs.

Fig. 1 is an editable vector schematic manually redrawn and adjusted by Tian Xia in draw.io. The editable source and vector export are included for reproducibility.

See `RELEASE_CHECKLIST.md` before depositing the archive.
