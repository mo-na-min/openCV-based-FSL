# Static FSL Translator (v4) - pure OpenCV

Recognises static Filipino Sign Language **letters A-Z and numbers 0-9** from a webcam.
Only `opencv-python` and `numpy` are used. There is no machine learning and no deep learning:
no model, no training, no weights. The DATASETS folder is used as a **reference gallery**
that live frames are compared against with plain distance measures.

## Requirements
- Python 3.9+
- A webcam
- `pip install -r requirements.txt`

## Folder layout
```
fsl_app_v4/
  app.py            run this
  evaluate.py       accuracy report on your DATASETS
  config.py         every tunable number
  segmentation.py   skin segmentation, calibration, background capture, wrist trim
  features.py       palm/finger analysis + descriptors (HOG, radial, Hu, geometry)
  dataset.py        loads DATASETS/<label>/*.jpg, caches features
  matcher.py        nearest-reference matching
  rules.py          geometry-only fallback (no dataset)
  stabilizer.py     smoothing + hold-to-commit
  gui.py            light-mode interface
  DATASETS/         <- put your dataset here
      A/ *.jpg
      B/ *.jpg
      ...
      0/ *.jpg ... 9/ *.jpg
```
Folder names must be a single letter A-Z or digit 0-9. Other folders are ignored.

## Run
```
python app.py
python app.py --camera 1      # another camera
python app.py --rebuild       # re-extract dataset features
python app.py --no-dataset    # geometry rules only (weak)
python evaluate.py            # leave-one-out accuracy + confusions
```
The first start extracts features from DATASETS and caches them in `cache/`; later starts are instant.

## Using it (for best accuracy)
1. Good, even light on your hand; plain background behind it.
2. Click **Calibrate skin** (or press `K`), cover the blue box with your palm for 3 s.
3. Optional: **Capture background** (`B`), remove your hand for 3 s. Helps a lot with skin-coloured walls.
4. Let your forearm run off the bottom edge of the box so the wrist can be trimmed.
5. Use the **All / Letters / Numbers** switch. Pairs such as V/2, W/3, B/4, O/0 can look identical,
   so pick the scope you are signing.
6. Hold a sign steady about 1.2 s and it is added to the text.

| Key | Action | Key | Action |
|---|---|---|---|
| `K` | calibrate skin | `1` `2` `3` | All / Letters / Numbers |
| `B` | capture background | `Space` | add space |
| `R` | reset calibration | `Backspace` | delete last |
| `Q` / `Esc` | quit | `X` | clear text |

## How it recognises a sign
1. Segment the hand (YCrCb + HSV skin colour, optional background subtraction, morphology).
2. Trim the forearm, find the palm centre (distance transform) and count fingers with a circular scan.
3. Describe the hand: HOG of the outline, HOG of the interior texture (finger overlaps, thumb),
   polar radial signature, Hu moments, geometry (solidity, holes, finger count).
4. Compare against every reference in DATASETS (each also mirrored for left/right hands);
   score each class by its 3 closest references.
5. Vote over the last 10 frames and require a steady hold before committing.

## Tuning
All in `config.py`: `W_*` (feature weights), `K_NEAREST`, `REJECT_DISTANCE`, `MIN_CONFIDENCE`,
`FINGER_MIN_LENGTH`, `DWELL_SECONDS`. Run `python evaluate.py` after each change.
`dataset.py` has `USE_GRABCUT` (OpenCV's classic GrabCut) as a fallback for photos where skin
colour segmentation fails; set it to `False` if your instructor wants colour thresholding only.
# openCV-based-FSL
