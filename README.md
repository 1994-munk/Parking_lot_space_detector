# Parking Lot Space Detector

Classifies the state of individual parking spaces (empty / occupied / anything
else your dataset labels) and turns that into a per-lot occupancy report with an
annotated image.

Trained on the Roboflow [Parking Lot v10](https://universe.roboflow.com/kaisha/parking-lot-vagxk)
multi-class export: **6,155 cropped space images, 640×640, CC BY 4.0**, labelled
per space rather than per-car.

Because the dataset is *classification* (one crop = one space), the project has
two layers:

1. **A crop classifier** — MobileNetV3/ResNet/EfficientNet fine-tune, with
   imbalance-aware loss, augmentation, early stopping and a confusion matrix.
2. **A lot-level scorer** — give it a lot photo plus a JSON of slot boxes, and it
   classifies every slot, draws green/red boxes, and reports occupancy %.

## 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Use the plain `pip install -r requirements.txt` (or the PyTorch selector) if you
have a CUDA GPU or an Apple Silicon Mac.

## 2. Dataset layout

Point `--data` at the exported folder:

```
Parking Lot.v10i.multiclass/
    train/empty/*.jpg      train/occupied/*.jpg
    valid/...
    test/...
```

The tree you already have — `...\Downloads\Parking Lot.v10i.multiclass` — is
exactly what's needed. Class names are discovered from the folder names, so a
2-class export (`empty`, `occupied`) or a 3-class one works unchanged. If there
is no `valid/` folder, a stratified validation split is carved out of `train`
(`--val-fraction`, default 0.2). If you only have the `train` folder, that's
fine too; point `--data` at the parent export folder whenever possible so
`test/` stays untouched for the final evaluation.

## 3. Train

Windows (your path):

```bat
python -m parking_lot.train --data "C:\Users\DELL\Downloads\Parking Lot.v10i.multiclass" ^
    --out runs/parking-v10 --epochs 12 --img-size 224 --batch-size 64
```

Linux/macOS:

```bash
python -m parking_lot.train \
    --data "$HOME/Downloads/Parking Lot.v10i.multiclass" \
    --out runs/parking-v10 --epochs 12 --img-size 224 --batch-size 64
```

`--model` choices: `mobilenet_v3_small` (default, fastest), `mobilenet_v3_large`,
`shufflenet_v2_x1_0`, `resnet18`, `resnet34`, `resnet50`, `efficientnet_b0`.
A laptop CPU manages a few minutes per epoch with the default backbone; a GPU
finishes all 12 epochs in a couple of minutes.

Useful flags:

| flag | why |
| --- | --- |
| `--no-class-weights` | dataset is balanced, so drop inverse-frequency weighting |
| `--no-pretrained` | offline / no weight download (trains from scratch) |
| `--patience 0` | disable early stopping |
| `--img-size 320` | small or distant spaces benefit from more resolution |
| `--lr 1e-4` | gentler fine-tune when the head is small |

Outputs in `--out`:

```
best.pt / last.pt      checkpoints (arch, class names, img size, normalisation)
history.csv            per-epoch loss/accuracy/macro-F1
metrics.json           best epoch + validation report
config.json            exact arguments used
```

## 4. Evaluate

```bash
python -m parking_lot.evaluate \
    --data "C:\Users\DELL\Downloads\Parking Lot.v10i.multiclass" \
    --checkpoint runs/parking-v10/best.pt --out runs/parking-v10
```

Uses `test/` when present (otherwise `valid/`), prints precision/recall/F1 per
class plus the most common mistakes, and writes `confusion_matrix.png` and
`report_test.json`. Report those test numbers, not the training log — the
export's `test/` split is the only unbiased estimate.

## 5. Classify individual crops

```bash
python -m parking_lot.predict --checkpoint runs/parking-v10/best.pt --topk 2 some_space.jpg
python -m parking_lot.predict --checkpoint runs/parking-v10/best.pt crops/ --json crops.json
```

## 6. Score a whole lot

Describe the slots of one camera view in JSON (absolute pixels by default; set
`"normalized": true` for 0–1 coordinates):

```json
{
  "image": "lot.jpg",
  "slots": [
    {"id": "A1", "box": [120, 240, 210, 330]},
    {"id": "A2", "box": [220, 240, 310, 330]}
  ]
}
```

```bash
python -m parking_lot.slots --checkpoint runs/parking-v10/best.pt \
    --slots demo/slots.json --out demo/annotated.jpg
```

It prints free/occupied counts and occupancy %, saves the annotated image and a
JSON with each slot's label, confidence and absolute box. Classes whose names
contain `empty`, `vacant`, `free`, `available`, `unoccupied`, `no car` or `open`
are treated as free (override with `--empty-classes empty,vacant`). Slots under
0.6 confidence are listed so a human can check them.

**No whole-lot photos?** Build a synthetic one from dataset crops to try the
pipeline end to end:

```bash
python scripts/make_demo_lot.py --data "C:\Users\DELL\Downloads\Parking Lot.v10i.multiclass" \
    --out demo --rows 3 --cols 5
python -m parking_lot.slots --checkpoint runs/parking-v10/best.pt --slots demo/slots.json
```

For a real lot, draw each slot's rectangle once in any annotation tool
(CVAT, Roboflow, labelImg…) and export the boxes to the JSON above. Slots are
fixed per camera, so this is a one-time step.

## Project layout

```
parking_lot/
  data.py        class discovery, splits, transforms, dataloaders
  model.py       torchvision backbones + classifier head swap
  train.py       fine-tuning loop (AMP, cosine LR, early stopping, class weights)
  evaluate.py    test metrics + confusion matrix plot
  predict.py     batch/CLI inference on crops
  slots.py       lot-level occupancy from slot boxes + annotated overlay
  inference.py   checkpoint loading and batched probabilities
  metrics.py     confusion matrix / precision-recall-F1 helpers
scripts/make_demo_lot.py   synthetic lot image + slots.json from dataset crops
```

## Known limits

- This is **not** a space detector: it does not find new slots in an arbitrary
  photo. Slot geometry is input (JSON), the model decides each slot's state.
  Automatic slot discovery needs detection/segmentation labels, which this
  export does not contain.
- Accuracy is tied to the camera angle and lot in the training data. A new
  mounting point or lot usually needs a fresh labelled sample (a few hundred
  crops is enough to fine-tune `best.pt`).
- The export's 640×640 stretch preprocessing is baked into the images; keep
  `--img-size` reasonable (224–320) rather than enormous.
- Crops at the edge of a slot include bits of neighbouring cars, which drives
  most empty/occupied confusions — tighten the boxes if you see that.
