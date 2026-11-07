"""Central configuration for the static FSL (letters A-Z, numbers 0-9) app.

Everything tunable lives here so the other modules stay free of magic numbers.
Pure OpenCV + NumPy: no machine learning, no deep learning, no training.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "DATASETS"          # DATASETS/<label>/*.jpg
CACHE_DIR = BASE_DIR / "cache"
CACHE_PATH = CACHE_DIR / "reference_features.npz"
HISTORY_DIR = BASE_DIR / "history"           # snapshots of committed signs

VALID_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUMBERS = set("0123456789")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --------------------------------------------------------------------------
# Camera / region of interest
# --------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
ROI_SIZE_FRAC = 0.82        # ROI is a square, side = this * frame height
ROI_CENTER_X_FRAC = 0.50    # horizontal centre of the ROI in the frame
ROI_CENTER_Y_FRAC = 0.50

# --------------------------------------------------------------------------
# Segmentation (skin colour in YCrCb + HSV, optional background subtraction)
# --------------------------------------------------------------------------
CR_RANGE = (133, 173)       # default skin range, used until the user calibrates
CB_RANGE = (77, 127)
HSV_S_MIN = 20
HSV_V_MIN = 40
CALIB_MARGIN = 10           # widen the sampled Cr/Cb range by this much
BG_DIFF_THRESHOLD = 22      # grey-level difference that counts as "not background"
MIN_COMPONENT_FRAC = 0.25   # keep blobs >= this fraction of the largest blob
MIN_HAND_AREA_FRAC = 0.04   # hand must cover at least this fraction of the ROI
WRIST_TRIM_FACTOR = 1.15    # cut the forearm this many palm-radii below the palm

# --------------------------------------------------------------------------
# Hand analysis
# --------------------------------------------------------------------------
FINGER_MIN_LENGTH = 1.50    # fingers are counted on circles from this many palm-radii outwards
FINGER_UNIT_WIDTH = 0.45    # typical finger width in palm-radii (for pressed-together fingers)
RADIAL_BINS = 36

# --------------------------------------------------------------------------
# Descriptor + matcher (reference matching, no training)
# --------------------------------------------------------------------------
NORM_SIZE = 64              # hand crops are normalised to NORM_SIZE x NORM_SIZE
MIRROR_AUGMENT = True       # also store a mirrored copy of each reference (left/right hand)
DATASET_MAX_SIDE = 320      # reference images are down-scaled to this before analysis
CACHE_VERSION = 4           # bump when descriptor code changes to invalidate the cache

# Weights for combining feature distances (each distance is scaled to 0..1)
W_HOG_GRAY = 0.30           # interior structure: finger overlaps, thumb position
W_HOG_MASK = 0.25           # outline shape
W_RADIAL = 0.25             # where the fingers point (polar signature)
W_HU = 0.08                 # global shape moments
W_GEOM = 0.12               # solidity, holes, finger count ...
HOG_COS_SCALE = 0.60
RADIAL_SCALE = 0.60
HU_SCALE = 3.0
K_NEAREST = 3               # class score = mean of the K closest references of that class
REJECT_DISTANCE = 0.55      # best class distance above this => "unknown"
CONF_ABS_SCALE = 0.60
CONF_MARGIN_SCALE = 0.35
MIN_CONFIDENCE = 0.35       # below this the sign is not shown / committed

# --------------------------------------------------------------------------
# Temporal stabiliser
# --------------------------------------------------------------------------
SMOOTH_WINDOW = 10          # frames considered for the vote
MIN_VOTE_SHARE = 0.55       # winner needs this share of the weighted votes
DWELL_SECONDS = 1.2         # hold a sign this long to commit it to the text
REPEAT_SECONDS = 1.8        # holding the same sign this long adds it again
