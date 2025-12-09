#!/usr/bin/env python3
"""
SGLang Benchmark Results Analyzer and Plotter

Parses benchmark JSON results and creates comparison plots.
Compares current run against reference runs (e.g., different devices, configurations).

Usage:
    python analyze_benchmarks.py <benchmark_dir> [--reference <ref_dir_or_json>] [--output <output_dir>]
    python analyze_benchmarks.py /path/to/benchmarks_sglang/ --reference /path/to/reference_benchmarks/
    python analyze_benchmarks.py results.json --reference baseline.json --label "B200_sglang" --ref-label "MI355X"
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib numpy")


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    # Config
    input_len: int = 0
    output_len: int = 0
    concurrency: int = 0
    num_prompts: int = 0
    tp_size: int = 0  # Will be read from JSON, 0 means unknown
    
    # Metrics
    throughput: float = 0.0  # requests/s
    output_throughput: float = 0.0  # tokens/s
    input_throughput: float = 0.0  # tokens/s
    total_throughput: float = 0.0  # tokens/s (input + output)
    
    # Latency (ms)
    mean_ttft: float = 0.0  # time to first token
    median_ttft: float = 0.0
    p99_ttft: float = 0.0
    
    mean_tpot: float = 0.0  # time per output token
    median_tpot: float = 0.0
    p99_tpot: float = 0.0
    
    mean_itl: float = 0.0  # inter-token latency
    median_itl: float = 0.0
    p99_itl: float = 0.0
    
    mean_e2e_latency: float = 0.0  # end-to-end latency
    median_e2e_latency: float = 0.0
    p99_e2e_latency: float = 0.0
    
    # Metadata
    source_file: str = ""
    device: str = ""
    label: str = ""
    
    @property
    def group_key(self) -> str:
        """Key for grouping similar configs."""
        if self.tp_size > 0:
            return f"ISL={self.input_len} | OSL={self.output_len} | TP={self.tp_size}"
        return f"ISL={self.input_len} | OSL={self.output_len}"
    
    @property
    def scenario_name(self) -> str:
        return f"isl{self.input_len}_osl{self.output_len}_c{self.concurrency}"


@dataclass 
class BenchmarkDataset:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    label: str = ""
    device: str = ""
    
    def add_result(self, result: BenchmarkResult):
        result.device = self.device
        result.label = self.label
        self.results.append(result)
    
    def get_by_group(self) -> Dict[str, List[BenchmarkResult]]:
        """Group results by config (ISL/OSL/TP)."""
        groups = defaultdict(list)
        for r in self.results:
            groups[r.group_key].append(r)
        return dict(groups)


class BenchmarkParser:
    """Parse SGLang benchmark JSON files."""
    
    @staticmethod
    def parse_filename(filename: str) -> dict:
        """Extract config from filename like 'isl1024_osl128_c4_np100_20241209.json'."""
        config = {}
        
        patterns = {
            'input_len': r'isl(\d+)',
            'output_len': r'osl(\d+)',
            'concurrency': r'c(\d+)',
            'num_prompts': r'np(\d+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                config[key] = int(match.group(1))
        
        return config
    
    @staticmethod
    def parse_json(filepath: str) -> Optional[BenchmarkResult]:
        """Parse a single benchmark JSON file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                # Handle Infinity values (not valid JSON)
                content = content.replace(': Infinity', ': null')
                content = content.replace(':Infinity', ': null')
                data = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not parse {filepath}: {e}")
            return None
        
        result = BenchmarkResult(source_file=filepath)
        
        # Extract from filename first
        filename_config = BenchmarkParser.parse_filename(os.path.basename(filepath))
        result.input_len = filename_config.get('input_len', 0)
        result.output_len = filename_config.get('output_len', 0)
        result.concurrency = filename_config.get('concurrency', 0)
        result.num_prompts = filename_config.get('num_prompts', 0)
        
        if not isinstance(data, dict):
            return result
        
        # SGLang bench_serving output format (flat structure)
        # Config fields
        result.input_len = int(data.get('random_input_len', data.get('input_len', result.input_len)) or result.input_len)
        result.output_len = int(data.get('random_output_len', data.get('output_len', result.output_len)) or result.output_len)
        result.concurrency = int(data.get('max_concurrency', data.get('concurrency', result.concurrency)) or result.concurrency)
        result.num_prompts = int(data.get('completed', data.get('num_prompts', result.num_prompts)) or result.num_prompts)
        result.tp_size = int(data.get('tp_size', data.get('tensor_parallel_size', result.tp_size)) or result.tp_size)
        
        # Throughput metrics (flat fields)
        result.throughput = float(data.get('request_throughput', 0) or 0)
        result.output_throughput = float(data.get('output_throughput', data.get('output_token_throughput', 0)) or 0)
        result.input_throughput = float(data.get('input_throughput', data.get('input_token_throughput', 0)) or 0)
        result.total_throughput = float(data.get('total_token_throughput', 0) or 0)
        if result.total_throughput == 0:
            result.total_throughput = result.input_throughput + result.output_throughput
        
        # Calculate request throughput from duration if not present
        if result.throughput == 0:
            duration = float(data.get('duration', 0) or 0)
            completed = int(data.get('completed', 0) or 0)
            if duration > 0 and completed > 0:
                result.throughput = completed / duration
        
        # Latency metrics - try flat fields first (SGLang format)
        # E2E Latency
        result.mean_e2e_latency = float(data.get('mean_e2e_latency_ms', data.get('mean_latency_ms', 0)) or 0)
        result.median_e2e_latency = float(data.get('median_e2e_latency_ms', data.get('median_latency_ms', 0)) or 0)
        result.p99_e2e_latency = float(data.get('p99_e2e_latency_ms', data.get('p99_latency_ms', 0)) or 0)
        
        # TTFT (Time to First Token)
        result.mean_ttft = float(data.get('mean_ttft_ms', 0) or 0)
        result.median_ttft = float(data.get('median_ttft_ms', 0) or 0)
        result.p99_ttft = float(data.get('p99_ttft_ms', 0) or 0)
        
        # TPOT (Time Per Output Token)
        result.mean_tpot = float(data.get('mean_tpot_ms', 0) or 0)
        result.median_tpot = float(data.get('median_tpot_ms', 0) or 0)
        result.p99_tpot = float(data.get('p99_tpot_ms', 0) or 0)
        
        # ITL (Inter-Token Latency)
        result.mean_itl = float(data.get('mean_itl_ms', 0) or 0)
        result.median_itl = float(data.get('median_itl_ms', 0) or 0)
        result.p99_itl = float(data.get('p99_itl_ms', 0) or 0)
        
        # Also try nested format (vLLM and some other tools)
        if result.median_e2e_latency == 0:
            e2e = data.get('e2e_latency_ms', data.get('request_latency_ms', {}))
            if isinstance(e2e, dict):
                result.mean_e2e_latency = float(e2e.get('mean', e2e.get('avg', 0)) or 0)
                result.median_e2e_latency = float(e2e.get('median', e2e.get('p50', 0)) or 0)
                result.p99_e2e_latency = float(e2e.get('p99', 0) or 0)
        
        if result.median_ttft == 0:
            ttft = data.get('ttft_ms', data.get('time_to_first_token_ms', {}))
            if isinstance(ttft, dict):
                result.mean_ttft = float(ttft.get('mean', 0) or 0)
                result.median_ttft = float(ttft.get('median', ttft.get('p50', 0)) or 0)
                result.p99_ttft = float(ttft.get('p99', 0) or 0)
        
        # Handle nested config section (some formats)
        config = data.get('config', data.get('args', {}))
        if isinstance(config, dict):
            if result.input_len == 0:
                result.input_len = int(config.get('random_input_len', config.get('input_len', 0)) or 0)
            if result.output_len == 0:
                result.output_len = int(config.get('random_output_len', config.get('output_len', 0)) or 0)
            if result.concurrency == 0:
                result.concurrency = int(config.get('max_concurrency', 0) or 0)
            if result.tp_size == 0:
                result.tp_size = int(config.get('tp_size', config.get('tensor_parallel_size', 0)) or 0)
        
        return result
    
    @staticmethod
    def load_directory(dirpath: str, label: str = "", device: str = "") -> BenchmarkDataset:
        """Load all benchmark JSONs from a directory."""
        dataset = BenchmarkDataset(label=label, device=device)
        dirpath = Path(dirpath)
        
        if not dirpath.exists():
            print(f"Warning: Directory not found: {dirpath}")
            return dataset
        
        json_files = list(dirpath.glob("*.json"))
        if not json_files:
            # Try subdirectories
            json_files = list(dirpath.glob("**/*.json"))
        
        for json_file in sorted(json_files):
            # Skip non-benchmark files
            if 'config' in json_file.name.lower() or 'sanity' in json_file.name.lower():
                continue
            
            result = BenchmarkParser.parse_json(str(json_file))
            if result and (result.throughput > 0 or result.output_throughput > 0):
                dataset.add_result(result)
        
        print(f"Loaded {len(dataset.results)} benchmarks from {dirpath}")
        return dataset


class BenchmarkPlotter:
    """Create comparison plots for benchmark results."""
    
    # Available color palette for dynamic assignment
    COLOR_PALETTE = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    
    # Available line styles for different devices
    LINE_STYLES = ['-', '--', ':', '-.']
    MARKERS = ['o', 'x', 's', '^', 'D', 'v', 'p', '*']
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Dynamic color mapping - assigned at runtime based on data
        self._config_color_map: Dict[str, str] = {}
        self._device_style_map: Dict[str, dict] = {}
    
    def _assign_colors_to_configs(self, all_groups: set) -> None:
        """Dynamically assign colors to configuration groups found in data."""
        self._config_color_map = {}
        sorted_groups = sorted(all_groups)  # Sort for consistent assignment
        for i, group_key in enumerate(sorted_groups):
            color_idx = i % len(self.COLOR_PALETTE)
            self._config_color_map[group_key] = self.COLOR_PALETTE[color_idx]
    
    def _assign_styles_to_devices(self, devices: List[str]) -> None:
        """Dynamically assign line styles to devices found in data."""
        self._device_style_map = {}
        for i, device in enumerate(devices):
            style_idx = i % len(self.LINE_STYLES)
            marker_idx = i % len(self.MARKERS)
            self._device_style_map[device] = {
                'linestyle': self.LINE_STYLES[style_idx],
                'marker': self.MARKERS[marker_idx]
            }
    
    def _get_config_color(self, group_key: str) -> str:
        """Get color for a configuration group."""
        if group_key in self._config_color_map:
            return self._config_color_map[group_key]
        # Fallback: assign new color dynamically
        color_idx = len(self._config_color_map) % len(self.COLOR_PALETTE)
        self._config_color_map[group_key] = self.COLOR_PALETTE[color_idx]
        return self._config_color_map[group_key]
    
    def _get_device_style(self, device_key: str) -> dict:
        """Get line style for a device."""
        if device_key in self._device_style_map:
            return self._device_style_map[device_key]
        # Fallback: assign new style dynamically
        idx = len(self._device_style_map)
        style_idx = idx % len(self.LINE_STYLES)
        marker_idx = idx % len(self.MARKERS)
        self._device_style_map[device_key] = {
            'linestyle': self.LINE_STYLES[style_idx],
            'marker': self.MARKERS[marker_idx]
        }
        return self._device_style_map[device_key]
    
    def plot_throughput_vs_latency(
        self,
        datasets: List[BenchmarkDataset],
        title: str = "Output tokens/s vs Mean E2E latency",
        model_name: str = "",
        output_file: str = "throughput_vs_latency.png"
    ):
        """
        Create throughput vs latency plot comparing multiple datasets.
        Similar to the reference image with concurrency labels.
        """
        if not HAS_MATPLOTLIB:
            print("Skipping plot: matplotlib not available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Collect all unique groups across datasets
        all_groups = set()
        for ds in datasets:
            for result in ds.results:
                all_groups.add(result.group_key)
        
        # Collect all devices
        all_devices = []
        for ds in datasets:
            device_key = ds.device or ds.label or f"Dataset_{datasets.index(ds)}"
            if device_key not in all_devices:
                all_devices.append(device_key)
        
        # Dynamically assign colors and styles based on actual data
        self._assign_colors_to_configs(all_groups)
        self._assign_styles_to_devices(all_devices)
        
        # Plot each group
        legend_handles = []
        plotted_groups = set()
        plotted_devices = set()
        
        for ds_idx, ds in enumerate(datasets):
            device_key = ds.device or ds.label or f"Dataset_{ds_idx}"
            style = self._get_device_style(device_key)
            
            groups = ds.get_by_group()
            
            for group_key, results in groups.items():
                color = self._get_config_color(group_key)
                
                # Sort by concurrency
                results = sorted(results, key=lambda r: r.concurrency)
                
                # Extract data points
                latencies = []
                throughputs = []
                concurrencies = []
                
                for r in results:
                    latency = r.mean_e2e_latency if r.mean_e2e_latency > 0 else r.median_e2e_latency
                    throughput = r.output_throughput if r.output_throughput > 0 else r.total_throughput
                    
                    if latency > 0 and throughput > 0:
                        latencies.append(latency)
                        throughputs.append(throughput)
                        concurrencies.append(r.concurrency)
                
                if not latencies:
                    continue
                
                # Plot line
                line, = ax.plot(
                    latencies, throughputs,
                    color=color,
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markersize=6,
                    linewidth=1.5,
                    label=f"{group_key} - {device_key}"
                )
                
                # Add concurrency labels
                for lat, thr, conc in zip(latencies, throughputs, concurrencies):
                    ax.annotate(
                        str(conc),
                        (lat, thr),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                        alpha=0.8
                    )
                
                # Track for legend
                if group_key not in plotted_groups:
                    plotted_groups.add(group_key)
                
                if device_key not in plotted_devices:
                    plotted_devices.add(device_key)
        
        # Create custom legend
        # Group legend (colors)
        group_handles = []
        for group_key in sorted(plotted_groups):
            color = self._get_config_color(group_key)
            handle = mlines.Line2D([], [], color=color, linestyle='-', linewidth=2, label=group_key)
            group_handles.append(handle)
        
        # Device legend (line styles)
        device_handles = []
        for device_key in all_devices:
            if device_key in plotted_devices:
                style = self._get_device_style(device_key)
                handle = mlines.Line2D(
                    [], [], color='black',
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markersize=6,
                    linewidth=1.5,
                    label=device_key
                )
                device_handles.append(handle)
        
        # Add legends
        legend1 = ax.legend(
            handles=group_handles,
            title="Configuration",
            loc='lower right',
            bbox_to_anchor=(0.98, 0.02),
            fontsize=9
        )
        ax.add_artist(legend1)
        
        if len(device_handles) > 1:
            legend2 = ax.legend(
                handles=device_handles,
                title="Device",
                loc='lower right',
                bbox_to_anchor=(0.98, 0.25),
                fontsize=9
            )
        
        # Labels and title
        full_title = title
        if model_name:
            full_title = f"{title}\n{model_name}"
        ax.set_title(full_title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Mean E2E latency (ms)", fontsize=12)
        ax.set_ylabel("Output tokens/s", fontsize=12)
        
        # Grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        
        # Also save as PDF for quality
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved plot: {pdf_path}")
        
        plt.close()
    
    def plot_throughput_comparison(
        self,
        datasets: List[BenchmarkDataset],
        title: str = "Throughput Comparison",
        output_file: str = "throughput_comparison.png"
    ):
        """Bar chart comparing throughput across datasets."""
        if not HAS_MATPLOTLIB:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Collect all scenarios
        all_scenarios = set()
        for ds in datasets:
            for r in ds.results:
                all_scenarios.add((r.input_len, r.output_len, r.concurrency))
        
        scenarios = sorted(all_scenarios)
        x = np.arange(len(scenarios))
        width = 0.8 / len(datasets)
        
        for i, ds in enumerate(datasets):
            throughputs = []
            for isl, osl, conc in scenarios:
                matching = [r for r in ds.results 
                           if r.input_len == isl and r.output_len == osl and r.concurrency == conc]
                if matching:
                    tp = matching[0].output_throughput or matching[0].total_throughput
                    throughputs.append(tp)
                else:
                    throughputs.append(0)
            
            label = ds.device or ds.label or f"Dataset {i+1}"
            offset = (i - len(datasets)/2 + 0.5) * width
            bars = ax.bar(x + offset, throughputs, width, label=label, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, throughputs):
                if val > 0:
                    ax.annotate(
                        f'{val:.0f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=7, rotation=45
                    )
        
        ax.set_xlabel('Scenario (ISL, OSL, Concurrency)')
        ax.set_ylabel('Output Tokens/s')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"({s[0]},{s[1]},{s[2]})" for s in scenarios], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        plt.close()
    
    def plot_latency_breakdown(
        self,
        datasets: List[BenchmarkDataset],
        title: str = "Latency Breakdown",
        output_file: str = "latency_breakdown.png"
    ):
        """Plot latency components (TTFT, TPOT, ITL) comparison."""
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = [
            ('median_ttft', 'Time to First Token (ms)', axes[0]),
            ('median_tpot', 'Time per Output Token (ms)', axes[1]),
            ('median_itl', 'Inter-Token Latency (ms)', axes[2]),
        ]
        
        for attr, ylabel, ax in metrics:
            all_scenarios = set()
            for ds in datasets:
                for r in ds.results:
                    all_scenarios.add(r.concurrency)
            
            scenarios = sorted(all_scenarios)
            x = np.arange(len(scenarios))
            width = 0.8 / max(len(datasets), 1)
            
            for i, ds in enumerate(datasets):
                values = []
                for conc in scenarios:
                    matching = [r for r in ds.results if r.concurrency == conc]
                    if matching:
                        val = getattr(matching[0], attr, 0)
                        values.append(val)
                    else:
                        values.append(0)
                
                label = ds.device or ds.label or f"Dataset {i+1}"
                offset = (i - len(datasets)/2 + 0.5) * width
                ax.bar(x + offset, values, width, label=label, alpha=0.8)
            
            ax.set_xlabel('Concurrency')
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        plt.close()


def export_to_csv(datasets: List[BenchmarkDataset], output_dir: str) -> None:
    """Export benchmark results to CSV files."""
    import csv
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export combined CSV with all datasets
    combined_csv = output_path / "benchmark_results.csv"
    
    fieldnames = [
        'input_len', 'output_len', 'concurrency', 'num_prompts', 'tp_size',
        'output_throughput', 'input_throughput', 'total_throughput', 'request_throughput',
        'mean_e2e_latency_ms', 'median_e2e_latency_ms', 'p99_e2e_latency_ms',
        'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
        'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms',
        'mean_itl_ms', 'median_itl_ms', 'p99_itl_ms',
        'source_file'
    ]
    
    with open(combined_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for ds in datasets:
            for r in ds.results:
                row = {
                    'input_len': r.input_len,
                    'output_len': r.output_len,
                    'concurrency': r.concurrency,
                    'num_prompts': r.num_prompts,
                    'tp_size': r.tp_size,
                    'output_throughput': r.output_throughput,
                    'input_throughput': r.input_throughput,
                    'total_throughput': r.total_throughput,
                    'request_throughput': r.throughput,
                    'mean_e2e_latency_ms': r.mean_e2e_latency,
                    'median_e2e_latency_ms': r.median_e2e_latency,
                    'p99_e2e_latency_ms': r.p99_e2e_latency,
                    'mean_ttft_ms': r.mean_ttft,
                    'median_ttft_ms': r.median_ttft,
                    'p99_ttft_ms': r.p99_ttft,
                    'mean_tpot_ms': r.mean_tpot,
                    'median_tpot_ms': r.median_tpot,
                    'p99_tpot_ms': r.p99_tpot,
                    'mean_itl_ms': r.mean_itl,
                    'median_itl_ms': r.median_itl,
                    'p99_itl_ms': r.p99_itl,
                    'source_file': r.source_file
                }
                writer.writerow(row)
    
    print(f"Saved CSV: {combined_csv}")
    
    # Also export per-dataset CSVs
    for ds in datasets:
        if not ds.results:
            continue
        
        device_name = (ds.device or ds.label or "dataset").replace(" ", "_").replace("/", "_")
        per_device_csv = output_path / f"benchmark_{device_name}.csv"
        
        with open(per_device_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in ds.results:
                row = {
                    'input_len': r.input_len,
                    'output_len': r.output_len,
                    'concurrency': r.concurrency,
                    'num_prompts': r.num_prompts,
                    'tp_size': r.tp_size,
                    'output_throughput': r.output_throughput,
                    'input_throughput': r.input_throughput,
                    'total_throughput': r.total_throughput,
                    'request_throughput': r.throughput,
                    'mean_e2e_latency_ms': r.mean_e2e_latency,
                    'median_e2e_latency_ms': r.median_e2e_latency,
                    'p99_e2e_latency_ms': r.p99_e2e_latency,
                    'mean_ttft_ms': r.mean_ttft,
                    'median_ttft_ms': r.median_ttft,
                    'p99_ttft_ms': r.p99_ttft,
                    'mean_tpot_ms': r.mean_tpot,
                    'median_tpot_ms': r.median_tpot,
                    'p99_tpot_ms': r.p99_tpot,
                    'mean_itl_ms': r.mean_itl,
                    'median_itl_ms': r.median_itl,
                    'p99_itl_ms': r.p99_itl,
                    'source_file': r.source_file
                }
                writer.writerow(row)
        
        print(f"Saved CSV: {per_device_csv}")


def print_summary_table(datasets: List[BenchmarkDataset]):
    """Print a summary comparison table."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    
    # Header
    header = f"{'Scenario':<30} "
    for ds in datasets:
        label = (ds.device or ds.label or "Dataset")[:15]
        header += f"| {label:>15} "
    print(header)
    print("-" * 100)
    
    # Collect all scenarios
    all_scenarios = set()
    for ds in datasets:
        for r in ds.results:
            all_scenarios.add((r.input_len, r.output_len, r.concurrency))
    
    for isl, osl, conc in sorted(all_scenarios):
        scenario = f"ISL={isl}, OSL={osl}, C={conc}"
        row = f"{scenario:<30} "
        
        for ds in datasets:
            matching = [r for r in ds.results 
                       if r.input_len == isl and r.output_len == osl and r.concurrency == conc]
            if matching:
                r = matching[0]
                tp = r.output_throughput or r.total_throughput
                row += f"| {tp:>12.1f} t/s "
            else:
                row += f"| {'N/A':>15} "
        
        print(row)
    
    print("=" * 100)


def download_remote_dir(remote_path: str, remote_host: str) -> str:
    """Download benchmark directory from remote host."""
    import tempfile
    local_dir = tempfile.mkdtemp(prefix="sglang_bench_")
    
    cmd = ["scp", "-r", f"{remote_host}:{remote_path}", local_dir]
    print(f"Downloading: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Find the actual downloaded directory
        contents = os.listdir(local_dir)
        if contents:
            return os.path.join(local_dir, contents[0])
        return local_dir
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e.stderr}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot SGLang benchmark results"
    )
    parser.add_argument(
        "benchmark_path",
        help="Path to benchmark JSON file or directory"
    )
    parser.add_argument(
        "--reference", "-r",
        action="append",
        help="Reference benchmark path(s) for comparison (can specify multiple)"
    )
    parser.add_argument(
        "--label", "-l",
        default="Current",
        help="Label for the main benchmark dataset"
    )
    parser.add_argument(
        "--ref-label",
        action="append",
        help="Labels for reference datasets (in order)"
    )
    parser.add_argument(
        "--device", "-d",
        default="",
        help="Device name for main dataset (e.g., B200_sglang)"
    )
    parser.add_argument(
        "--ref-device",
        action="append",
        help="Device names for reference datasets"
    )
    parser.add_argument(
        "--remote",
        help="Remote host (user@host) if benchmark is on remote machine"
    )
    parser.add_argument(
        "--output", "-o",
        default="./benchmark_plots_and_csv",
        help="Output directory for plots and CSV files"
    )
    parser.add_argument(
        "--model", "-m",
        default="",
        help="Model name for plot titles"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting, only print summary"
    )
    
    args = parser.parse_args()
    
    # Load main benchmark
    benchmark_path = args.benchmark_path
    if args.remote:
        benchmark_path = download_remote_dir(args.benchmark_path, args.remote)
    
    datasets = []
    
    # Load main dataset
    if os.path.isdir(benchmark_path):
        main_ds = BenchmarkParser.load_directory(
            benchmark_path, 
            label=args.label,
            device=args.device or args.label
        )
    else:
        main_ds = BenchmarkDataset(label=args.label, device=args.device or args.label)
        result = BenchmarkParser.parse_json(benchmark_path)
        if result:
            main_ds.add_result(result)
    
    datasets.append(main_ds)
    
    # Load reference datasets
    if args.reference:
        ref_labels = args.ref_label or []
        ref_devices = args.ref_device or []
        
        for i, ref_path in enumerate(args.reference):
            label = ref_labels[i] if i < len(ref_labels) else f"Reference_{i+1}"
            device = ref_devices[i] if i < len(ref_devices) else label
            
            if os.path.isdir(ref_path):
                ref_ds = BenchmarkParser.load_directory(ref_path, label=label, device=device)
            else:
                ref_ds = BenchmarkDataset(label=label, device=device)
                result = BenchmarkParser.parse_json(ref_path)
                if result:
                    ref_ds.add_result(result)
            
            datasets.append(ref_ds)
    
    # Print summary
    print_summary_table(datasets)
    
    # Export to CSV
    export_to_csv(datasets, args.output)
    
    # Generate plots
    if not args.no_plot and HAS_MATPLOTLIB:
        plotter = BenchmarkPlotter(output_dir=args.output)
        
        # Main throughput vs latency plot
        plotter.plot_throughput_vs_latency(
            datasets,
            title="Output tokens/s vs Mean E2E latency",
            model_name=args.model,
            output_file="throughput_vs_latency.png"
        )
        
        # Throughput comparison bar chart
        plotter.plot_throughput_comparison(
            datasets,
            title=f"Throughput Comparison{' - ' + args.model if args.model else ''}",
            output_file="throughput_comparison.png"
        )
        
        # Latency breakdown
        plotter.plot_latency_breakdown(
            datasets,
            title=f"Latency Breakdown{' - ' + args.model if args.model else ''}",
            output_file="latency_breakdown.png"
        )
        
        print(f"\nPlots and CSV saved to: {args.output}/")
    elif not HAS_MATPLOTLIB:
        print("\nInstall matplotlib for plots: pip install matplotlib numpy")
        print(f"CSV files saved to: {args.output}/")


if __name__ == "__main__":
    main()
