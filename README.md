# Vehicle Classification with ResNet

**Amirsepehr Shamloo**

I built this project to classify vehicle images into eight classes using transfer learning. I used pretrained ResNet models and fine-tuned them on my vehicle dataset. Most of the work is in Jupyter notebooks, including data checks, training, model comparison, and testing.

The project runs on CPU. I used Python, PyTorch, Torchvision, and VS Code. My aim was to compare different training methods, choose a good model using validation results, and then check it on images kept outside training.

## 1. About the project

### Vehicle classes

Each image has one vehicle label. These are the folder names used in the project:

| Label | Vehicle type |
| --- | --- |
| `ambulance` | Ambulance |
| `autobus` | Bus |
| `kamyun` | Truck |
| `kamyunet` | Light truck |
| `minibus` | Minibus |
| `savari` | Passenger car |
| `taxi` | Taxi |
| `vanet` | Pickup truck |

In the main experiment, the names `neysan`, `nysan`, and `nissan` are mapped to `vanet`. There is also a separate experiment where Neysan images are kept outside training. The model still has eight output classes and is expected to predict these images as `vanet`.

### Data preparation

I combined images from two datasets. Their original `train`, `test`, and `unclean` folders were treated as source folders. I made a new split after combining and checking the data, rather than using the original split.

The main preparation steps are:

1. Read the images from all five source folders.
2. Check that images can be opened and class names are valid.
3. Standardize the class names.
4. Check duplicates and keep the unique images.
5. Save image information in CSV manifests.
6. Create a stratified train, validation, and test split with seed 42.
7. Check that image paths and file hashes do not overlap between splits.

A manifest is a table containing image paths, labels, and other information. The training code reads these tables instead of copying every image into a new folder. Stratification keeps approximately the same class proportions in each split.

The main run used **3,512 unique images**:

| Split | Images | Approximate share |
| --- | ---: | ---: |
| Training | 2,458 | 70% |
| Validation | 527 | 15% |
| Test | 527 | 15% |

These counts describe the recorded run. A changed dataset can produce different counts, so the audit output should always be checked before training.

### Image processing

Images are converted to RGB and resized to **224 × 224 pixels**. The resize keeps the original proportions and adds padding to make the image square. This avoids stretching the vehicle.

The images are normalized using ImageNet mean and standard deviation. During training, random horizontal flips and small changes in brightness, contrast, and saturation add some variation. Validation, testing, and prediction use preprocessing without random augmentation.

### Models and experiments

I compared **ResNet18, ResNet34, and ResNet50**. Each model starts with ImageNet pretrained weights. This lets the model use image features it has already learned, instead of learning everything from the beginning.

I trained the models in stages:

| Stage | Layers that learn |
| --- | --- |
| FC only | Final classification layer |
| FC + Layer 4 | Final layer and last ResNet block |
| FC + Layers 3–4 | Final layer and last two ResNet blocks |

The earlier layers stay frozen. Fine-tuning more of the later layers helps the model adapt its features to vehicle images, but it also takes more CPU time.

After comparing the models, I continued experiments with ResNet18. I compared dropout values **0, 0.25, and 0.5**, standard and balanced batches, and Cross-Entropy and Binary Cross-Entropy loss. The BCE version uses one-hot class targets. At prediction time, the output with the highest score gives the single predicted class.

I selected dropout 0.25, balanced sampling, and BCE for the final workflow. Balanced batches contain two images from each of the eight classes, giving a batch size of 16. The notebooks also save training curves and wrong predictions so I can inspect the results.

### Final model and results

After model selection, I combined training and validation into **2,985 images** and continued training for eight fixed epochs. The 527 test images stayed separate.

| Final training setting | Value |
| --- | --- |
| Model | ResNet18 |
| Trainable layers | `layer3`, `layer4`, `fc` |
| Dropout | 0.25 |
| Loss | BCE |
| Sampling | Balanced |
| Optimizer | AdamW |
| Batch size | 16 |
| Refit epochs | 8 |
| Layer 3 learning rate | 0.0000015 |
| Layer 4 learning rate | 0.000005 |
| FC learning rate | 0.000015 |
| Weight decay | 0.0001 |
| Scheduler during refit | None; fixed learning rates |
| Seed | 42 |

The saved final test results are:

| Metric | Result |
| --- | ---: |
| Test accuracy | **97.53%** |
| Macro precision | 97.46% |
| Macro recall | 97.55% |
| Macro F1 | 97.46% |
| Correct predictions | 514 / 527 |
| Wrong predictions | 13 |

These results belong to `resnet18_trainval_final.pt`. Macro metrics give equal importance to each class. The confusion matrix and error images help show which vehicles are confused with each other.

There is also an optional all-data model trained on all 3,512 images. It is intended for later use on new images. Since it has seen the original test images, the 97.53% held-out result should not be reported as its test accuracy.

### Separate Neysan experiment

In this experiment, Neysan images are excluded from training and validation. The remaining images are split 80% for training and 20% for validation. The saved recipe records 2,615 training images and 654 validation images. After model selection, these are combined for a ten-epoch final refit.

The final Neysan test contains **243 images**. The model correctly predicted **228** as `vanet`, giving **93.83% accuracy on this Neysan-only test**. This is a separate result from the eight-class test accuracy.

Images with a sigmoid score below 80% are copied for human review. The saved test has 39 such images. All eligible test images are included in the accuracy calculation, including those below the threshold. The score is not a calibrated probability of being correct.

## 2. How to run and use the project

### What is included

The repository contains notebooks, two evaluation scripts, package requirements, a PDF report, and saved figures and JSON results.

**The dataset, CSV manifests, and `.pt` model weights are not included in this GitHub checkout.** They are excluded by `.gitignore`. To train, you need the dataset. To evaluate without training, you need the matching saved checkpoint. Cloning the repository alone is not enough to run model prediction.

### Step 1: Download the repository

Install Python and Git. For notebook work, install VS Code with its Python and Jupyter extensions. Then open a terminal and run:

```powershell
git clone https://github.com/Amirsepehr1996/vehicle_classification.git
cd vehicle_classification
```

You can also download the ZIP from GitHub and extract it. The commands below should be run from the project root, the folder containing `src` and `requirements.txt`.

### Step 2: Create a Python environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ipykernel jupyter
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

If PowerShell blocks activation, you can use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install ipykernel jupyter
```

`requirements.txt` records the package versions from my environment, including PyTorch 2.14.0 and Torchvision 0.29.0. Installation depends on whether those versions support your Python and operating system. If pip cannot find a pinned version, use a compatible environment or install an available compatible package set and record the changes. Results from a different environment may vary.

Check the main imports:

```powershell
python -c "import torch, torchvision, pandas, sklearn; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

The scripts use CPU, so `CUDA: False` is fine. If the requirements file cannot be read, it is stored as UTF-16 in this checkout. You can create a UTF-8 copy and install from it:

```powershell
python -c "from pathlib import Path; p=Path('requirements.txt'); Path('requirements_utf8.txt').write_text(p.read_text(encoding='utf-16'), encoding='utf-8')"
python -m pip install -r requirements_utf8.txt
```

### Step 3: Prepare the data folders

For the main experiment, place the original datasets under `data/raw`. The audit expects these five folders:

| Required source folder | Contents |
| --- | --- |
| `data/raw/dataset1/train` | Class folders containing images |
| `data/raw/dataset1/test` | Class folders containing images |
| `data/raw/dataset1/unclean` | Class folders containing images |
| `data/raw/datasetv2_TrainUclean/train` | Class folders containing images |
| `data/raw/datasetv2_TrainUclean/unclean` | Class folders containing images |

For example, an ambulance image can be at `data/raw/dataset1/train/ambulance/image01.jpg`. Keep images inside their class folders. Supported image extensions are JPG, JPEG, PNG, BMP, and WebP.

If your dataset has a different structure, update `SOURCES` and the raw-data path in the audit notebook before running it. Check the printed folder and class information before continuing.

### Step 4: Set the notebook paths and kernel

Open the project folder in VS Code. Open a notebook from `src` and select the `.venv` Python environment using **Select Kernel**.

Several notebooks contain my local path:

```python
PROJECT_ROOT = Path(
    r"D:\Maktab_Sharif_Projects\traffic-vehicle-classification"
)
```

Replace it with the full path of your downloaded project, for example:

```python
PROJECT_ROOT = Path(r"C:\Projects\vehicle_classification")
```

Do this in every notebook you use. The audit and split notebooks use `Path.cwd().parent`, which expects the working directory to be `src`. To avoid a wrong path, you can replace this with the same explicit project path.

CSV manifests store image paths. If you move the project or dataset after creating them, those paths can become invalid. Regenerate the manifests from the new location before training.

### Step 5: Run the main notebooks in order

Run the cells from top to bottom in each notebook. Later notebooks depend on files created by earlier ones.

| Order | Notebook | What it does |
| ---: | --- | --- |
| 1 | `src/data_audit.ipynb` | Checks images, standardizes labels, and creates unique-image manifests |
| 2 | `src/data_split.ipynb` | Creates the stratified 70/15/15 split |
| 3 | `src/model_training.ipynb` | Trains ResNet18, ResNet34, and ResNet50 in stages |
| 4 | `src/model_evaluation.ipynb` | Compares models, runs dropout/sampling/loss experiments, and continues BCE training |
| 5 | `src/final_training.ipynb` | Refits the selected model on training + validation |
| 6 | `src/final_test.ipynb` | Evaluates the refitted model on the separate test set |
| 7, optional | `src/all_data_training.ipynb` | Trains an additional model using all images and provides single-image prediction |

After the split, check for `train.csv`, `validation.csv`, and `test.csv` under `data/manifests`. Training then creates `.pt` checkpoints and history files under `outputs`.

`model_evaluation.ipynb` includes additional training, so it takes longer than just calculating metrics. Complete its selected BCE continuation before running `final_training.ipynb`, which expects `outputs/bce_continuation/resnet18_bce_continued_best.pt`.

CPU training can take considerable time. Keep the computer awake and save your notebooks. Some later training cells refuse to overwrite existing results. For a fresh experiment, back up the old outputs and choose new output paths. A recovery checkpoint does not automatically mean training resumes; check the notebook code before restarting an interrupted run.

### Step 6: Evaluate new labeled images

If you already have a compatible checkpoint, you can use `mentor_evaluation.py` without repeating training. This script expects a ResNet18 BCE checkpoint containing the model settings, class mapping, and training SHA-256 hashes. The all-data notebook saves these hashes in `resnet18_all_data_final.pt`.

Create a folder such as `mentor_data`. Inside it, create class folders using the labels listed above and add your new images. The `--data-dir` argument must point to the folder directly containing those class folders.

From the project root:

```powershell
python src/mentor_evaluation.py --checkpoint outputs/all_data_model/resnet18_all_data_final.pt --data-dir mentor_data --output-dir mentor_results_run1 --batch-size 16
```

Replace the checkpoint path with the location of your compatible model. The output folder must be new or empty. For another run, use a different name such as `mentor_results_run2`.

The script excludes images whose exact file hashes match training images. It also reports repeated copies in the evaluation set, but does not remove those copies. This overlap check detects identical files; it does not guarantee detection of cropped or recompressed versions of training images.

The main output files are:

| File | How to use it |
| --- | --- |
| `metrics.json` | Read accuracy, macro metrics, and image counts |
| `predictions.csv` | Check each true label, prediction, score, and correctness |
| `errors.csv` | Inspect only the wrong predictions |
| `classification_report.csv` | Compare results for each class |
| `confusion_matrix.csv` | See which classes are confused |
| `excluded_training_overlap.csv` | See images excluded because they were used in training |
| `overlap_summary.json` | Check supplied, excluded, and remaining image counts |

Use images from all eight classes for a full eight-class evaluation. The script reports missing classes, and its macro metrics include all eight classes even when some have no test images.

### Step 7: Predict one image

The last section of `all_data_training.ipynb` loads the all-data checkpoint and defines `predict_image`. After its required setup and prediction cells have run, use:

```python
result = predict_image(r"C:\Images\new_vehicle.jpg")
print(result)
```

It returns `predicted_class`, `confidence_percent`, and `confidence_method`. Replace the path with your image. The example already in the notebook uses an image from its training manifest, so replace that example with a new image when checking practical performance.

You do not need to repeat training when the all-data checkpoint already exists. Run the cells needed to define the paths, imports, device, resize helper, and prediction function. This notebook function is the provided single-image interface; the mentor script is intended for labeled-folder evaluation.

### Step 8: Run the Neysan experiment

Prepare a separate dataset under `data/raw_without_neysan_in_train`, with the same five source folders. Keep Neysan images in separate `neysan` folders so the audit can identify and hold them out. It also recognizes `nysan` and `nissan` as Neysan aliases. Images left inside ordinary `vanet` folders are treated as Vanet training candidates, so check this organization carefully.

Run these notebooks in order:

1. `src/neysan_data_audit.ipynb`
2. `src/neysan_model_training.ipynb`
3. `src/neysan_final_training.ipynb`
4. `src/neysan_evaluation.ipynb`

For additional Neysan-only images, use:

```powershell
python src/neysan_mentor_evaluation.py --checkpoint outputs/neysan_experiment/trainval_refit/resnet18_neysan_trainval_final.pt --data-dir new_neysan_images --output-dir neysan_results_run1 --batch-size 16
```

The input folder must contain only Neysan images, although subfolders are allowed. Every image has the expected label `vanet`. Keep the new, empty output folder outside the input folder.

Along with metrics and predictions, the script saves a `human_check_manifest.csv` and copies images below the 80% score threshold into `need human check`. The original images stay in place, and these review images still count in the evaluation.

### Common problems

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Select the environment where you installed the packages |
| Dataset or CSV not found | Check `PROJECT_ROOT`, source folders, and the notebook order |
| Image paths do not exist | Regenerate manifests after moving the data |
| Checkpoint not found | Supply the weights or finish the earlier training stages |
| Training hashes missing | Use a checkpoint exported with `training_sha256` |
| Unknown class folder | Use one of the eight labels or supported aliases |
| Output folder is not empty | Choose a new results folder |
| No images remain after overlap checking | Supply images that were not used in training |
| Training is slow or memory is limited | Start with ResNet18; use a smaller evaluation batch if needed |

For balanced training, batch size 16 is part of the two-images-per-class sampler. Changing it requires updating the sampler too. Evaluation scripts allow a smaller batch, such as `--batch-size 8`, without changing the trained model.

The notebooks contain the full workflow, and `report.pdf` explains the project in more detail. Saved figures and JSON files under `outputs` can be viewed even when the dataset and checkpoints are unavailable.
