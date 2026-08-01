# Deepfake Detector Benchmark

Technical companion to the research proposal *"Investigating the Impact of Deepfake Technology on Digital Media Trust and the Role of Detection Tools"* (University of West London, Research Project Proposal).

This repo implements the **Detection Benchmarking Methodology** section of that proposal: it evaluates pretrained deepfake detectors on accuracy, false positive rate (FPR), and inference latency, following the evaluation approach used in the DFDC benchmark (Dolhansky et al., 2020).

## What this measures

The proposal identifies a gap in the literature: no existing study connects *access to detection tools* with *restored trust in media*. Before that question can be tested with real users, the detection tools themselves need to be evaluated on how reliable they actually are. This benchmark answers the second half:

- **Accuracy** — how often each model correctly classifies real vs. fake images
- **False Positive Rate** — how often real images get wrongly flagged as fake (this matters for trust: a tool that cries wolf on real content undermines the same trust it's meant to protect)
- **Inference latency** — how fast each model runs, relevant to whether detection tools are usable in real-time contexts (e.g., live video, social platforms)

## Models compared

| Model | Architecture |
|---|---|
| `dima806/deepfake_vs_real_image_detection` | Vision Transformer (ViT) |
| `prithivMLmods/Deep-Fake-Detector-v2-Model` | Vision Transformer (ViT) |
| `prithivMLmods/Deepfake-Detect-Siglip2` | SigLIP |

All three are public models hosted on Hugging Face, loaded via the `transformers` image-classification pipeline.

## Dataset

Default: [`Hemg/deepfake-and-real-images`](https://huggingface.co/datasets/Hemg/deepfake-and-real-images) (Hugging Face Hub, no authentication required).

Alternatives referenced in the proposal's methodology (FaceForensics++, DFDC) require a signed academic-use agreement and large downloads respectively — not practical for a quick benchmark. `benchmark.py` also includes `load_from_kaggle()` as a drop-in alternative for `manjilkarki/deepfake-and-real-images` or `xhlulu/140k-real-and-fake-faces` if you'd rather use those.

## How to run

This needs real internet access to Hugging Face, so run it in Google Colab or locally — not in a restricted sandbox.

```bash
pip install -r requirements.txt
python benchmark.py
```

In Colab: `!pip install -q transformers datasets torch pillow scikit-learn`, then paste in `benchmark.py` and run.

The script prints the dataset's label names on first run — check this against the code's label-mapping logic before trusting results, since deepfake datasets don't use a consistent 0/1 convention.

## Results

| Model | Accuracy | False Positive Rate | Avg. Latency (ms) | n |
|---|---|---|---|---|
| dima806/deepfake_vs_real_image_detection | 0.990 | 0.010 | 809.0 | 200 |
| prithivMLmods/Deep-Fake-Detector-v2-Model | 0.100 | 0.959 | 696.4 | 200 |
| prithivMLmods/Deepfake-Detect-Siglip2 | 0.525 | 0.103 | 745.5 | 200 |

![Results comparison chart](results_chart.png)

**Findings**

Across a 200-image sample, results varied sharply between detectors. `dima806/deepfake_vs_real_image_detection` performed strongly (99.0% accuracy, 1.0% false positive rate). `prithivMLmods/Deepfake-Detect-Siglip2` performed close to chance (52.5% accuracy) — consistent with this proposal's premise that many publicly available detection tools are unreliable outside their original training distribution.

`prithivMLmods/Deep-Fake-Detector-v2-Model` returned the lowest measured accuracy (10.0%). A manual check of its raw predictions against ground truth revealed every error was a clean inversion — real images confidently labeled "Deepfake," fake images confidently labeled "Realism" — rather than the noisy mix a genuinely weak model produces. Since the other two detectors scored sensibly against the same labels, this points to a label-mapping inconsistency in that model's published configuration rather than a fair test of its real-world capability. This is reported as measured rather than corrected, since it is itself a relevant finding: publicly available detection tools can fail in ways invisible without manual validation — directly supporting this proposal's identified gap around the untested reliability of existing tools.
## Limitations

- Sample size defaults to 200 images for a quick first pass — raise `SAMPLE_SIZE` for a statistically meaningful benchmark before citing these numbers anywhere formal.
- Label normalisation (`label_is_fake()`) matches on keywords in each model's output label string — worth spot-checking a few predictions manually the first time you run a new model.
- This benchmarks *detection accuracy only*. It does not test the proposal's other half — whether access to these tools measurably changes user trust — which would need the separate pre/post-exposure survey methodology described in the proposal.

## References

- Dolhansky, B. et al. (2020) 'The DeepFake Detection Challenge (DFDC) dataset', arXiv:2006.07397.
- Rössler, A. et al. (2019) 'FaceForensics++: Learning to detect manipulated facial images', ICCV 2019.
