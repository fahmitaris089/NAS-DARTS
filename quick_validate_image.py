#!/usr/bin/env python3
"""
Quick validation tool for individual palm vein images
Use during capture session to validate quality before continuing
"""

import cv2
import numpy as np
import sys
from pathlib import Path

def validate_image(image_path: str) -> dict:
    """Quick validation of single image"""
    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Cannot read image'}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Sharpness
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = laplacian.var()
    
    # Vein visibility
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Contrast
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    contrast_ratio = std_intensity / (mean_intensity + 1e-6)
    
    # Quality assessment
    quality = 'EXCELLENT'
    issues = []
    
    if lap_var < 60:
        quality = 'REJECTED'
        issues.append(f'Too blurry (Laplacian={lap_var:.1f}, need >60)')
    elif lap_var < 100:
        quality = 'ACCEPTABLE'
        issues.append(f'Slightly soft (Laplacian={lap_var:.1f}, prefer >100)')
    
    if edge_density < 0.01:
        if quality == 'EXCELLENT':
            quality = 'ACCEPTABLE'
        issues.append(f'Weak vein pattern (edge_density={edge_density:.4f}, prefer >0.015)')
    
    if contrast_ratio < 0.2:
        if quality == 'EXCELLENT':
            quality = 'ACCEPTABLE'
        issues.append(f'Low contrast (ratio={contrast_ratio:.3f}, prefer >0.25)')
    
    return {
        'quality': quality,
        'laplacian_var': lap_var,
        'edge_density': edge_density,
        'contrast_ratio': contrast_ratio,
        'mean_intensity': mean_intensity,
        'issues': issues
    }

def print_result(result: dict, image_path: str):
    """Print colored validation result"""
    colors = {
        'EXCELLENT': '\033[92m',  # Green
        'ACCEPTABLE': '\033[93m',  # Yellow
        'REJECTED': '\033[91m',    # Red
    }
    reset = '\033[0m'
    
    quality = result.get('quality', 'UNKNOWN')
    color = colors.get(quality, '')
    
    print(f"\n{'='*60}")
    print(f"Image: {Path(image_path).name}")
    print(f"Quality: {color}{quality}{reset}")
    print(f"{'='*60}")
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"📊 Metrics:")
    print(f"  Sharpness (Laplacian):  {result['laplacian_var']:.2f}")
    print(f"  Vein visibility (edges): {result['edge_density']:.4f}")
    print(f"  Contrast ratio:          {result['contrast_ratio']:.3f}")
    print(f"  Mean intensity:          {result['mean_intensity']:.1f}")
    
    if result['issues']:
        print(f"\n⚠️  Issues:")
        for issue in result['issues']:
            print(f"  - {issue}")
    else:
        print(f"\n✅ No issues detected!")
    
    print(f"{'='*60}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 quick_validate_image.py <image_path>")
        print("\nExample:")
        print("  python3 quick_validate_image.py dataset_multi_distance/835/25cm/final/image_001.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    result = validate_image(image_path)
    print_result(result, image_path)
    
    # Exit code based on quality
    if result.get('quality') == 'REJECTED':
        sys.exit(1)
    elif result.get('quality') == 'ACCEPTABLE':
        sys.exit(0)
    else:  # EXCELLENT
        sys.exit(0)

if __name__ == '__main__':
    main()
