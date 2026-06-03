import logging
import os

# Lazy imports for multiprocessing compatibility
decord = None
OpenAI = None

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = os.environ.get("ANNOTATE_MODEL", "qwen-vl-plus")
DEFAULT_BASE_URL = os.environ.get(
    "ANNOTATE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Video processing defaults
DEFAULT_ANALYSIS_FPS = 4.0     # Higher FPS for analysis
DEFAULT_REFINEMENT_FPS = 1.0   # Lower FPS for refinement
DEFAULT_MAX_FRAMES = 100000
DEFAULT_RESIZE_WIDTH = 512
DEFAULT_JPEG_QUALITY = 75
DEFAULT_BATCH_SIZE = 64        # Batch size for chunked frame reading (memory safety)
MIN_API_FRAMES = 4             # Minimum frames per API request

# Video base directory (single source of truth)
VIDEO_BASE_DIR = os.environ.get("VIDEO_BASE_DIR", "./data/videos")

# API retry settings
MAX_RETRIES = 15
RATE_LIMIT_DELAY = 0.0         # Delay (seconds) after each successful API call
RATE_LIMIT_BACKOFF_CAP = 300   # Max backoff (seconds) for rate-limit retries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
