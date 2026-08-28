# UTSA CS 5243 Computer Vision Assignments

Frozen Version 1.0 infrastructure for the individual A0-A5 assignments. A0 verifies setup and submission workflow; A1 is **Image Representation and Visual Exploration**; A2 is **Image Enhancement & Computational Photography**; A3 is **Visual Recognition with Deep Learning**; A4 is **Image Registration and Augmented Reality**; and A5 is **Stereo Vision and 3D Scene Reconstruction**. Historical notebooks under `legacy_assignments/` are instructor references and are not part of the student release.

## Start here

`<COURSE_ROOT>` means the folder containing this README. Paths are quoted so every command can be run from any working directory.

### First-time setup

1. Install a current Miniforge or Conda distribution for your operating system.
2. Open a new terminal and create the one course environment:

   ```bash
   conda env create --file "<COURSE_ROOT>/environment.yml"
   conda activate cs5243
   python -m pip install --editable "<COURSE_ROOT>"
   python -m pytest "<COURSE_ROOT>/assignments/A#/tests_public" -q
   ```

3. Launch JupyterLab at the repository:

   ```bash
   jupyter lab "<COURSE_ROOT>"
   ```

Use the platform guide for [Windows 11](docs/setup-windows-11.md), [Apple Silicon macOS](docs/setup-macos-apple-silicon.md), or [Ubuntu](docs/setup-ubuntu.md) if installation needs platform-specific troubleshooting.

### Everyday assignment workflow

1. Activate the environment: `conda activate cs5243`.
2. Launch JupyterLab with `jupyter lab "<COURSE_ROOT>"`.
3. Complete the assigned notebook and source files individually.
4. Run `python "<COURSE_ROOT>/scripts/validate_submission.py" --assignment A# --preflight` when starting a substantive assignment.
5. Use repository-relative paths and keep datasets outside the assignment folder.
6. Restart the kernel, run all cells, inspect the results, and save.
7. Export HTML without re-executing. The `classic` template preserves the figure descriptions used by the assignment notebooks:

   ```bash
   jupyter nbconvert --to html --template classic "<COURSE_ROOT>/assignments/A#/A#.ipynb" --output A#.html
   ```

8. Validate and package from any directory:

   ```bash
   python "<COURSE_ROOT>/scripts/validate_submission.py" --assignment A#
   python "<COURSE_ROOT>/scripts/package_submission.py" --assignment A#
   ```

The packager stops on validation errors. Warnings remain in the report for review. The predictable output is `LastName_FirstName_A#.zip` beside the assignment directory, containing one folder with the same name.

### Update the environment

When the instructor publishes an updated `environment.yml`:

```bash
conda activate cs5243
conda env update --name cs5243 --file "<COURSE_ROOT>/environment.yml" --prune
python -m pip install --editable "<COURSE_ROOT>"
python -m pytest "<COURSE_ROOT>/assignments/A#/tests_public" -q
```

Do not create a second course environment unless the instructor specifically directs you to do so.

## Canvas submission checklist

Open the ZIP before uploading and confirm:

- there is exactly one top-level folder named `LastName_FirstName_A#`;
- the executed notebook and rendered HTML are present;
- `src/` contains your required source files;
- `outputs/` contains only selected figures, tables, or videos;
- `validation-report.json` reports zero errors;
- no datasets, environments, caches, checkpoints, hidden files, or temporary files are present.

A PDF export is not required.

## Course infrastructure rules

- A0-A5 are individual assignments. Research-project policy is outside this repository.
- Required work uses environment version 2026.1. Do not require or depend on additional packages without written instructor approval.
- The environment does not establish a GPU requirement or a CPU-only policy. Generic PyTorch packages may use hardware supported by the student's system and installation.
- Use Pillow or OpenCV for supported assignment image I/O. On some Apple Silicon installations, importing torchvision may emit a harmless optional JPEG-extension warning; see [troubleshooting](docs/troubleshooting.md).
- Generative AI may support conceptual clarification, debugging, code suggestions, and documentation interpretation, but meaningful use must be disclosed in every assignment. Students remain responsible for understanding, testing, and explaining all work. AI may not replace experimental analysis, failure analysis, reflection, or interpretation, and may not bypass required implementations. Students may be asked to explain submitted work.
- Never submit datasets, model caches, environments, large trained models, or instructor-only material.

More detail is available in the [student workflow and troubleshooting guide](docs/student-workflow.md). Instructor authoring guidance is in [docs/assignment-authoring.md](docs/assignment-authoring.md).
