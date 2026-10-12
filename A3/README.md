# Assignment 3: Visual Recognition with Deep Learning

**Individual** · **125 points**

A3 develops an offline PyTorch transfer-learning pipeline for a balanced six-class visual-pattern dataset. Task details and points are in `A3.ipynb`; installation and general workflow are in the repository README.

## A3 workflow

```bash
conda activate cs5243
jupyter lab /path/to/workspace/A3/A3.ipynb
python /path/to/workspace/common-setup/scripts/validate_submission.py --assignment A3 --preflight
```

Implement reusable graded functions in `src/student_code.py`. Use the notebook for experiments, artifacts, and analysis. The required pipeline uses the bundled dataset and local pretrained weights; it must not download data or weights.

## Dataset and models

The fixed CC0 dataset contains six balanced classes: checker, circles, diagonal, dots, horizontal, and vertical. It has 720 training, 180 validation, and 180 test images. The test split is reserved for final evaluation and robustness comparisons; do not use it for model or hyperparameter selection.

Required architecture names are `mobilenet_v3_small` and `efficientnet_b0`. Build the torchvision architecture without requested weights, load the corresponding local state dictionary, and then replace its classifier. This preserves offline ImageNet transfer learning.

## External datasets (Experiments 8-9)

Experiments 8 and 9 fine-tune the same pipeline on two real-world datasets pulled from the Hugging Face Hub with the `datasets` library — the one approved exception to the fully offline policy above. **`datasets` is the only new package A3 needs.** If you already set up `cs5243` before this update, you do not need to redo the full `common-setup` install — just activate your existing environment and add the one package:

```bash
conda activate cs5243
pip install datasets
```

(everything else A3 uses, including `scikit-learn` for the lesion-resplit helper below, was already part of `cs5243`.) Setting up `cs5243` for the first time already includes `datasets` via `environment.yml`, so this step is only for students updating an existing environment.

- `marmal88/skin_cancer` — 7 real dermatoscopic diagnostic classes (HAM10000-derived), 10,015 images, string label column `dx`. After loading it, pass it to `a3_tools.resplit_by_lesion`, which builds lesion-level train/validation/test splits (7153/1431/1431 images) shared by the whole class. Chosen to surface class imbalance: some diagnoses are far more common than others, so overall accuracy alone is misleading and the per-class metrics from Task 4 matter.
- `timm/eurosat-rgb` — 10 land-use/land-cover classes from Sentinel-2 satellite imagery, ~27k images across its own train/validation/test splits, integer label column `label`. Chosen to surface domain shift: the pretrained backbone learned features from everyday photos, not overhead satellite tiles.

The first download of `marmal88/skin_cancer` is about 3.7 GB and needs roughly 7 GB of free disk once `datasets` builds its local cache; `timm/eurosat-rgb` is about 90 MB. Start the download early on a reliable connection; later runs reuse the cache. The cache lives outside the assignment folder (by default `~/.cache/huggingface`; set `HF_HOME` to move it). Never place it inside `A3/`, because it must not be submitted.

Implement `build_external_dataloaders` once in `src/student_code.py` and reuse it for both. Do not request any other network resource, and continue to use only the bundled local pretrained weights for the model itself — only the two named datasets are downloaded. The Skin Cancer dataset is for this controlled classroom exercise only; do not present its output as medical guidance.

## Artifact formats

- `experiment_metrics.csv`: cumulative long-form records with `experiment, method, condition, metric, value, units`. Append later experiments without deleting earlier records; additional useful columns are allowed.
- `predictions.csv`: `sample_id, split, true_label, predicted_label, confidence` with one test record per sample for the submitted checkpoint.
- `model_metadata.json`: `architecture`, `num_classes`, `class_names`, `checkpoint`, `parameter_counts.total`, `parameter_counts.trainable`, `seed`, and `selected_by`. The selection description must identify validation evidence, not test performance.
- `trained_model.pt`: a loadable state dictionary for the submitted model, no optimizer state, at most 25 MB.

## Public tests

```bash
pytest A3/tests_public -q
```

Skipped implementation tests are expected in the untouched starter. Applicable tests should pass after implementation; public tests do not guarantee full correctness.

## Official resources

These references explain the relevant tools and APIs. They do not prescribe solutions to the assignment.

- PyTorch: [data loading](https://docs.pytorch.org/docs/stable/data.html), [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [saving and loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), and [transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- torchvision: [models and weights](https://docs.pytorch.org/vision/stable/models.html), [MobileNetV3](https://docs.pytorch.org/vision/stable/models/mobilenetv3.html), [EfficientNet](https://docs.pytorch.org/vision/stable/models/efficientnet.html), and [transforms](https://docs.pytorch.org/vision/stable/transforms.html)
- scikit-learn: [classification metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html) and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- Matplotlib: [`imshow`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
- Pandas: [`DataFrame.to_csv`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
- Python: [`json.dump`](https://docs.python.org/3/library/json.html#json.dump)
