# Geometry of a Nightingale Song

[![Project Status](https://img.shields.io/badge/status-experimental-orange.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#)

## Overview

Geometry of a Nightingale Song is a research / analysis project that studies the structure of nightingale vocalizations using geometric and topological data-analysis techniques. The goal is to transform audio recordings of nightingale song into geometric representations (point clouds, manifolds, graphs), then analyze and visualize the structure, motifs, and variability using dimensionality reduction, graph/geodesic methods, and topological summaries.

This repository contains code, notebooks, scripts, and documentation to reproduce the analysis and visualizations, plus an accompanying video demonstration (replace the placeholder with your video link below).

---

## Video Demonstration

Replace `VIDEO_URL_HERE` with the YouTube link (or direct URL) for your project's video. If you use YouTube, you can embed a clickable thumbnail like this:

[![Watch the demo video](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](VIDEO_URL_HERE)

Or link directly:
- Demo video: VIDEO_URL_HERE

If you provide a YouTube URL I can insert the correct thumbnail and the VIDEO_ID automatically.

---

## Key Features

- Audio preprocessing and high-quality spectrogram generation
- Feature extraction (MFCCs, spectral features, pitch contours)
- Time-warping and alignment of song syllables
- Construction of k-NN graphs and geodesic-distance embeddings
- Dimensionality reduction (PCA, t-SNE, UMAP) for visual exploration
- Topological Data Analysis (persistent homology) to capture shape features
- Clustering and motif discovery
- Interactive notebooks and static visualizations
- Reproducible scripts for pipeline stages

---

## Motivation & Scientific Goals

Nightingale songs are rich with structure at multiple time scales: motifs, syllables, and long-range patterns. By mapping short audio segments to points in feature space we may reveal geometric structure (clusters, trajectories, loops) that correspond to song motifs, transitions, or performance variability. Topological measures can detect global organization (e.g., loops for cyclic patterns) that complements local cluster-based analysis.

Primary scientific questions:
- Do nightingale syllables populate low-dimensional manifolds?
- Can geometric/topological structure reveal song motifs and transitions?
- How stable are motifs across individuals or recording sessions?

---

## Dataset

This project expects collections of nightingale recordings (WAV or FLAC). Common sources:
- Xeno-canto (https://www.xeno-canto.org) — community-contributed bird recordings
- Macaulay Library (Cornell) — curated recordings with metadata

Organize your dataset under the `data/` folder:
- `data/raw/` — raw audio files (one WAV/FLAC per recording)
- `data/metadata.csv` — CSV with columns (filename, species, location, date, recorder, notes)

Note: This repository does not include copyrighted third-party recordings. Always respect licensing and attribution requirements from data providers.

---

## High-level Pipeline

1. Data ingestion and audio normalization
2. Segmentation: detect syllables/notes or process fixed windows
3. Feature extraction for each segment:
   - MFCCs, spectral centroid, bandwidth, rolloff
   - Pitch (F0) / fundamental frequency contours
   - Delta/delta-delta features or time-aggregated statistics
4. Optional DTW/time-warping to align motifs
5. Build point cloud (each segment = a point in feature space)
6. Construct k-NN graph and compute geodesic distances (Isomap-like)
7. Dimensionality reduction (PCA / UMAP / t-SNE) for visualization
8. Compute persistent homology (e.g., Vietoris–Rips) for topology
9. Cluster and annotate motifs
10. Produce figures, interactive plots, and video demo

---

## Typical File / Folder Layout

- data/
  - raw/
  - processed/
  - metadata.csv
- notebooks/
  - 01-data-prep.ipynb
  - 02-feature-extraction.ipynb
  - 03-geometry-embeddings.ipynb
  - 04-topology.ipynb
- src/
  - audio/
    - preprocess.py
    - segment.py
    - features.py
  - geometry/
    - knn_graph.py
    - embeddings.py
  - topology/
    - persistent_homology.py
  - viz/
    - plot_utils.py
- scripts/
  - run_preprocess.sh
  - run_full_pipeline.sh
- results/
  - figures/
  - video/
- requirements.txt
- environment.yml
- README.md

If your repository differs, please adapt the layout and the commands below.

---

## Installation

Recommended: use a conda environment.

Conda:
```bash
conda create -n nightingale python=3.10 -y
conda activate nightingale
conda install -c conda-forge ffmpeg jupyterlab nodejs -y
pip install -r requirements.txt
```

Pip-only (virtualenv):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Example contents of requirements.txt:
- numpy
- scipy
- librosa
- matplotlib
- seaborn
- scikit-learn
- umap-learn
- pandas
- jupyter
- plotly
- ripser
- gudhi (optional)
- scikit-image
- soundfile
- pybind11 (if compiling custom C++ components)

If you rely on GPU acceleration for UMAP/PCA/etc., consult the package docs.

---

## Quick Start — Commands

1. Prepare data (put raw audio in `data/raw/` and metadata in `data/metadata.csv`)
2. Preprocess audio and create spectrograms:
```bash
python src/audio/preprocess.py --input_dir data/raw --output_dir data/processed --sr 22050 --hop_length 512
```
3. Extract features for segments:
```bash
python src/audio/features.py --input_dir data/processed --out_file data/features/features.pkl
```
4. Compute embeddings and save visualizations:
```bash
python src/geometry/embeddings.py --features data/features/features.pkl --method umap --out results/embeddings/umap.npy --plot results/figures/umap.png
```
5. Compute topology summaries:
```bash
python src/topology/persistent_homology.py --features data/features/features.pkl --out results/topology/persistence.json
```
6. Open notebooks for exploration:
```bash
jupyter lab notebooks/
```

Adjust arguments to match your scripts.

---

## Notebooks

Notebooks in `notebooks/` demonstrate the end-to-end analyses:
- 01-data-prep.ipynb: loading, normalization, segmentation
- 02-feature-extraction.ipynb: MFCCs, spectral measures
- 03-geometry-embeddings.ipynb: PCA, t-SNE, UMAP, Isomap visualizations
- 04-topology.ipynb: Compute and visualize persistence diagrams

Use these notebooks both to reproduce figures and to iterate on methods.

---

## Reproducibility

- Set RNG seeds where applicable (e.g., numpy, scikit-learn, UMAP).
- Record package versions: run `pip freeze > requirements-freeze.txt`.
- Document dataset provenance and exact audio files used.
- For time-consuming steps, cache intermediate files in `data/processed/`.

---

## Example Results (What to expect)

- 2D UMAP/PCA scatter plots showing clusters of syllable types
- Spectrogram mosaics for cluster centroids
- Persistence diagrams and barcodes highlighting 0D (connected) and 1D (loop) features
- A video animation that traverses a trajectory in embedding space while showing the corresponding waveform/spectrogram side-by-side

---

## How to Add the Video Properly

1. Host the video on YouTube or in the repo `results/video/` (large files in repo are discouraged).
2. For YouTube:
   - Replace `VIDEO_URL_HERE` in the Video Demonstration section with `https://www.youtube.com/watch?v=VIDEO_ID`.
   - Optionally add a thumbnail link like `https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg`.
3. For a local file:
   - Add the video file under `results/video/`.
   - In GitHub README, link to the file: `[Demo video](results/video/demo.mp4)` (GitHub will render a player for some video formats).
4. Tell me the URL and I'll update the exact embed code.

---

## Contributing

Contributions are welcome. Suggested workflow:
1. Open an issue describing your idea or bug.
2. Fork the repository and create a feature branch.
3. Add tests and update notebooks or scripts.
4. Submit a pull request describing changes.

Please follow the code style and document new features in the README and notebooks.

---

## License

This project is released under the MIT License. See LICENSE for details.

---

## Acknowledgements & References

- Librosa: McFee et al., for audio feature extraction
- UMAP: McInnes et al., for manifold learning
- Ripser / GUDHI: for persistent homology computation
- Data: Xeno-canto / Macaulay Library (credit original recordists and licenses)

Suggested readings:
- M. McInnes, L. Healy, J. Melville. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction."
- H. Edelsbrunner, J. Harer. "Computational Topology: An Introduction."

---

## Contact

If you want help customizing the README or embedding the video, provide:
- The video URL (YouTube or hosted file)
- Any project-specific text you want included (author names, affiliations, dataset DOIs)
- Any figure or screenshots to include in `assets/`

Maintainer: dhiraj7kr (GitHub: @dhiraj7kr)

---

Thank you for working on this interesting intersection of bioacoustics and geometry. Provide the video link and any project-specific details you want included (author list, grant/funding, dataset citations), and I will update the README to include them exactly.
