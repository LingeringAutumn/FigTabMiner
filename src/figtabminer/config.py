import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

# Rendering
RENDER_ZOOM = 2.0  # Zoom factor for PDF rendering (2.0 = 144 DPI approx)
PREVIEW_MAX_SIZE = (800, 800)

# OCR
# FIGTABMINER_OCR_GPU: "auto" | "true"/"false"
OCR_GPU = os.getenv("FIGTABMINER_OCR_GPU", "auto").strip().lower()

# Keywords for heuristic detection
CAPTION_KEYWORDS = [
    "Figure", "Fig.", "Table", "Tab.", "Scheme", "Chart", 
    "图", "表"
]

# AI / Science Keywords
SUBTYPE_KEYWORDS = {
    "microscopy": ["SEM", "TEM", "AFM", "Microscope", "Micrograph", "Scale bar"],
    "spectrum": ["XRD", "FTIR", "Raman", "UV-Vis", "Spectrum", "Spectra", "Absorbance", "Transmittance", "Intensity"],
    "line_plot": ["vs", "dependence", "function of", "plot", "curve"],
}

# Regex Patterns for Condition Extraction
CONDITION_PATTERNS = {
    "temperature": r"(\d+(\.\d+)?\s*(K|°C|deg\s*C))",
    "wavelength": r"(\d+(\.\d+)?\s*(nm|μm))",
    "wavenumber": r"(\d+(\.\d+)?\s*cm\^-1)",
    "pressure": r"(\d+(\.\d+)?\s*(Pa|kPa|MPa|bar|Torr))",
    "time": r"(\d+(\.\d+)?\s*(s|min|h|hours|seconds))",
    "ph": r"(pH\s*=?\s*\d+(\.\d+)?)",
}

MATERIAL_PATTERNS = [
    r"\b[A-Z][a-z]?\d*([A-Z][a-z]?\d*)*\b",  # Simple chemical formula heuristic
]

# Fallback / Thresholds
CAPTION_SEARCH_WINDOW = 300  # pixels vertical distance to search for caption
