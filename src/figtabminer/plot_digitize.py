import cv2
import numpy as np
import pandas as pd
from . import utils

logger = utils.setup_logging(__name__)

def digitize_line_plot(image_path: str, x_min: float, x_max: float, y_min: float, y_max: float) -> pd.DataFrame:
    """
    Semi-automatic digitization of a line plot.
    1. Read image
    2. Convert to grayscale & binary
    3. Find connected components (curves)
    4. Select largest curve (heuristic for main data)
    5. Map pixels to data coordinates
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
            
        h, w, _ = img.shape
        
        # 1. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding - assume dark curve on light background
        # Invert so curve is white
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Find Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filter noise: ignore small components and the background (label 0)
        # We assume the curve is one of the largest components.
        # Let's skip label 0 (background)
        
        largest_area = 0
        largest_label = -1
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Heuristic: Curve should have some width/height ratio or just be large
            if area > largest_area:
                largest_area = area
                largest_label = i
        
        if largest_label == -1:
            return pd.DataFrame(columns=["x", "y"])
            
        # 3. Extract points for the largest component
        # Create mask
        mask = (labels == largest_label).astype(np.uint8) * 255
        
        # Get coordinates of all non-zero pixels
        ys, xs = np.where(mask > 0)
        
        # 4. Digitization: For each unique X, find median Y
        # This assumes single-valued function y=f(x)
        
        pixel_data = {}
        for x, y in zip(xs, ys):
            if x not in pixel_data:
                pixel_data[x] = []
            pixel_data[x].append(y)
            
        # Aggregate
        curve_pixels = []
        for x in sorted(pixel_data.keys()):
            y_median = np.median(pixel_data[x])
            curve_pixels.append((x, y_median))
            
        # Downsample if too many points
        if len(curve_pixels) > 300:
            step = len(curve_pixels) // 300
            curve_pixels = curve_pixels[::step]
            
        # 5. Map to Data Coordinates
        # Assume the image represents the full plot area defined by x_min...y_max
        # If the crop includes axis labels, this will be inaccurate.
        # But for this baseline, we assume the user provides the bounds FOR THE IMAGE AREA.
        # Or we assume the image IS the plot area.
        
        data_points = []
        for px, py in curve_pixels:
            # Map px (0..w) to (x_min..x_max)
            data_x = x_min + (px / w) * (x_max - x_min)
            
            # Map py (0..h) to (y_max..y_min) -> Note: pixel y grows DOWN, graph y grows UP
            # py=0 -> y_max, py=h -> y_min
            data_y = y_max - (py / h) * (y_max - y_min)
            
            data_points.append({"x": data_x, "y": data_y})
            
        return pd.DataFrame(data_points)

    except Exception as e:
        logger.error(f"Digitization failed: {e}")
        return pd.DataFrame(columns=["x", "y"])
