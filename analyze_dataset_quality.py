#!/usr/bin/env python3
"""
Dataset Quality Analysis for Multi-Distance Palm Vein
Analyzes image quality, distribution, and provides recommendations
"""

import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetQualityAnalyzer:
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.results = defaultdict(list)
        
    def compute_laplacian_variance(self, image: np.ndarray) -> float:
        """Compute Laplacian variance for sharpness measurement"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def compute_vein_visibility(self, image: np.ndarray) -> dict:
        """Estimate vein pattern visibility using edge detection and contrast"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge density (vein pattern complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Local contrast (vein vs background)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        contrast_ratio = std_intensity / (mean_intensity + 1e-6)
        
        # Histogram spread (dynamic range)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_spread = np.sum(hist > 0) / 256.0
        
        return {
            'edge_density': edge_density,
            'contrast_ratio': contrast_ratio,
            'histogram_spread': hist_spread,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity
        }
    
    def analyze_image(self, image_path: Path) -> dict:
        """Analyze single image quality"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        metrics = {
            'path': str(image_path),
            'filename': image_path.name,
            'shape': image.shape,
            'laplacian_var': self.compute_laplacian_variance(image),
        }
        
        # Add vein visibility metrics
        vein_metrics = self.compute_vein_visibility(image)
        metrics.update(vein_metrics)
        
        return metrics
    
    def analyze_distance_folder(self, distance: str, user_id: str = '835'):
        """Analyze all images in a distance folder"""
        final_dir = self.dataset_root / user_id / distance / 'final'
        
        if not final_dir.exists():
            print(f"⚠️  {distance} final directory not found")
            return
        
        images = list(final_dir.glob('*.png'))
        print(f"\n📊 Analyzing {distance}: {len(images)} images")
        
        for img_path in images:
            metrics = self.analyze_image(img_path)
            if metrics:
                metrics['distance'] = distance
                self.results[distance].append(metrics)
    
    def generate_report(self) -> dict:
        """Generate comprehensive quality report"""
        report = {
            'summary': {},
            'per_distance': {},
            'recommendations': []
        }
        
        total_images = 0
        all_laplacian = []
        all_edge_density = []
        
        for distance, metrics_list in self.results.items():
            count = len(metrics_list)
            total_images += count
            
            laplacian_vars = [m['laplacian_var'] for m in metrics_list]
            edge_densities = [m['edge_density'] for m in metrics_list]
            contrast_ratios = [m['contrast_ratio'] for m in metrics_list]
            
            all_laplacian.extend(laplacian_vars)
            all_edge_density.extend(edge_densities)
            
            report['per_distance'][distance] = {
                'count': count,
                'laplacian_var': {
                    'mean': np.mean(laplacian_vars),
                    'std': np.std(laplacian_vars),
                    'min': np.min(laplacian_vars),
                    'max': np.max(laplacian_vars)
                },
                'edge_density': {
                    'mean': np.mean(edge_densities),
                    'std': np.std(edge_densities)
                },
                'contrast_ratio': {
                    'mean': np.mean(contrast_ratios),
                    'std': np.std(contrast_ratios)
                }
            }
        
        report['summary'] = {
            'total_images': total_images,
            'total_distances': len(self.results),
            'avg_images_per_distance': total_images / len(self.results) if self.results else 0,
            'global_laplacian_mean': np.mean(all_laplacian),
            'global_laplacian_std': np.std(all_laplacian),
            'global_edge_density_mean': np.mean(all_edge_density)
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: dict) -> list:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Check sample count
        min_count = min(d['count'] for d in report['per_distance'].values())
        max_count = max(d['count'] for d in report['per_distance'].values())
        avg_count = report['summary']['avg_images_per_distance']
        
        if avg_count < 15:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Sample Size',
                'issue': f'Average {avg_count:.1f} images per distance is below recommended 15-20',
                'action': f'Capture additional {int(15 - avg_count)} images per distance, prioritize distances with <10 samples'
            })
        
        # Check distribution balance
        if max_count - min_count > 5:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Distribution Balance',
                'issue': f'Unbalanced distribution: {min_count}-{max_count} images per distance',
                'action': 'Balance dataset by capturing more images for under-represented distances'
            })
        
        # Check image quality
        for distance, metrics in report['per_distance'].items():
            lap_mean = metrics['laplacian_var']['mean']
            if lap_mean < 60:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Image Quality',
                    'issue': f'{distance}: Low sharpness (Laplacian={lap_mean:.1f}, threshold=60)',
                    'action': f'Re-capture {distance} with better focus or increase stable-frames parameter'
                })
            
            edge_mean = metrics['edge_density']['mean']
            if edge_mean < 0.05:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Vein Visibility',
                    'issue': f'{distance}: Low vein pattern visibility (edge_density={edge_mean:.4f})',
                    'action': f'Adjust exposure/contrast settings for {distance} captures'
                })
        
        # Check critical distances (22cm and 32cm - boundary cases)
        for critical_dist in ['22cm', '32cm']:
            if critical_dist in report['per_distance']:
                if report['per_distance'][critical_dist]['count'] < 12:
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'Critical Distance',
                        'issue': f'{critical_dist} is boundary distance with only {report["per_distance"][critical_dist]["count"]} samples',
                        'action': f'Increase {critical_dist} samples to at least 12 for better boundary robustness'
                    })
        
        return recommendations
    
    def visualize_results(self, output_dir: Path):
        """Generate visualization plots"""
        output_dir.mkdir(exist_ok=True)
        
        # Prepare data for plotting
        distances = []
        laplacian_vars = []
        edge_densities = []
        contrast_ratios = []
        counts = []
        
        for distance in sorted(self.results.keys()):
            for metrics in self.results[distance]:
                distances.append(distance)
                laplacian_vars.append(metrics['laplacian_var'])
                edge_densities.append(metrics['edge_density'])
                contrast_ratios.append(metrics['contrast_ratio'])
            counts.append(len(self.results[distance]))
        
        # Plot 1: Sample distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sample count per distance
        ax = axes[0, 0]
        sorted_distances = sorted(self.results.keys())
        sample_counts = [len(self.results[d]) for d in sorted_distances]
        ax.bar(sorted_distances, sample_counts, color='steelblue')
        ax.axhline(y=15, color='red', linestyle='--', label='Target: 15 samples')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Sample Count')
        ax.set_title('Sample Distribution per Distance')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Laplacian variance (sharpness)
        ax = axes[0, 1]
        data_lap = [self.results[d] for d in sorted_distances]
        lap_data = [[m['laplacian_var'] for m in d] for d in data_lap]
        bp = ax.boxplot(lap_data, labels=sorted_distances, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
        ax.axhline(y=60, color='red', linestyle='--', label='Min threshold: 60')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Laplacian Variance')
        ax.set_title('Image Sharpness per Distance')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Edge density (vein visibility)
        ax = axes[1, 0]
        edge_data = [[m['edge_density'] for m in d] for d in data_lap]
        bp = ax.boxplot(edge_data, labels=sorted_distances, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Edge Density')
        ax.set_title('Vein Pattern Visibility per Distance')
        ax.grid(axis='y', alpha=0.3)
        
        # Contrast ratio
        ax = axes[1, 1]
        contrast_data = [[m['contrast_ratio'] for m in d] for d in data_lap]
        bp = ax.boxplot(contrast_data, labels=sorted_distances, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightyellow')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Contrast Ratio')
        ax.set_title('Image Contrast per Distance')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_quality_analysis.png', dpi=150)
        print(f"\n📈 Visualization saved: {output_dir / 'dataset_quality_analysis.png'}")
        plt.close()
    
    def save_report(self, output_path: Path):
        """Save detailed report to JSON"""
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved: {output_path}")
        return report
    
    def print_report(self, report: dict):
        """Print human-readable report"""
        print("\n" + "="*80)
        print("📊 DATASET QUALITY ANALYSIS REPORT")
        print("="*80)
        
        # Summary
        summary = report['summary']
        print(f"\n📈 SUMMARY")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Total distances: {summary['total_distances']}")
        print(f"  Avg images per distance: {summary['avg_images_per_distance']:.1f}")
        print(f"  Global sharpness (Laplacian): {summary['global_laplacian_mean']:.2f} ± {summary['global_laplacian_std']:.2f}")
        print(f"  Global vein visibility (edge density): {summary['global_edge_density_mean']:.4f}")
        
        # Per-distance details
        print(f"\n📏 PER-DISTANCE METRICS")
        for distance in sorted(report['per_distance'].keys()):
            metrics = report['per_distance'][distance]
            print(f"\n  {distance}:")
            print(f"    Samples: {metrics['count']}")
            print(f"    Sharpness: {metrics['laplacian_var']['mean']:.2f} ± {metrics['laplacian_var']['std']:.2f} (range: {metrics['laplacian_var']['min']:.2f}-{metrics['laplacian_var']['max']:.2f})")
            print(f"    Vein visibility: {metrics['edge_density']['mean']:.4f} ± {metrics['edge_density']['std']:.4f}")
            print(f"    Contrast: {metrics['contrast_ratio']['mean']:.4f} ± {metrics['contrast_ratio']['std']:.4f}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS")
        if not report['recommendations']:
            print("  ✅ Dataset quality is acceptable!")
        else:
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"\n  {i}. [{rec['priority']}] {rec['category']}")
                print(f"     Issue: {rec['issue']}")
                print(f"     Action: {rec['action']}")
        
        print("\n" + "="*80)


def main():
    dataset_root = Path('/Users/fahmitaris/Downloads/NAS-DARTS/dataset_multi_distance')
    output_dir = Path('/Users/fahmitaris/Downloads/NAS-DARTS/dataset_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 Starting dataset quality analysis...")
    
    analyzer = DatasetQualityAnalyzer(dataset_root)
    
    # Analyze all distances
    distances = ['22cm', '25cm', '27cm', '30cm', '32cm']
    for distance in distances:
        analyzer.analyze_distance_folder(distance)
    
    # Generate and save report
    report = analyzer.save_report(output_dir / 'quality_report.json')
    
    # Print human-readable report
    analyzer.print_report(report)
    
    # Generate visualizations
    analyzer.visualize_results(output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
