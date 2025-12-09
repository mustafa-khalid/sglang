#!/usr/bin/env python3
"""
SGLang Server Log Analyzer

Downloads server logs and analyzes them for performance improvement opportunities.
Identifies configuration issues, bottlenecks, and suggests flag changes.

Usage:
    python analyze_server_logs.py <log_file_or_remote_path> [--remote <user@host>]
    python analyze_server_logs.py /path/to/sglang_server.log
    python analyze_server_logs.py /home/user/logs_sglang/20241209/sglang_server.log --remote user@remote-host
"""

import argparse
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PerformanceIssue:
    severity: str  # "critical", "warning", "info"
    category: str
    message: str
    suggestion: str
    flag_change: Optional[str] = None


@dataclass
class AnalysisResult:
    issues: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    current_config: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)


class SGLangLogAnalyzer:
    """Analyzes SGLang server logs for performance optimization opportunities."""

    # Patterns to extract configuration and metrics
    PATTERNS = {
        # Server configuration
        "tp_size": r"tensor.parallel.*?(\d+)|--tp-size\s+(\d+)|tp_size[=:\s]+(\d+)",
        "dp_size": r"data.parallel.*?(\d+)|--data-parallel-size\s+(\d+)|dp_size[=:\s]+(\d+)",
        "max_running_requests": r"max.running.requests[=:\s]+(\d+)",
        "mem_fraction": r"mem.fraction.static[=:\s]+([\d.]+)",
        "chunked_prefill": r"chunked.prefill.size[=:\s]+(\d+)",
        "max_prefill_tokens": r"max.prefill.tokens[=:\s]+(\d+)",
        "kv_cache_dtype": r"kv.cache.dtype[=:\s]+(\S+)",
        "attention_backend": r"attention.backend[=:\s]+(\S+)",
        "quantization": r"quantization[=:\s]+(\S+)",
        "cuda_graph_max_bs": r"cuda.graph.max.bs[=:\s]+(\d+)",
        
        # Performance metrics
        "throughput": r"throughput[=:\s]+([\d.]+)\s*(?:tok(?:en)?s?/s|tokens per second)",
        "latency_p50": r"(?:p50|median)\s*(?:latency)?[=:\s]*([\d.]+)\s*(?:ms|s)",
        "latency_p99": r"p99\s*(?:latency)?[=:\s]*([\d.]+)\s*(?:ms|s)",
        "batch_size": r"(?:current|avg|average)\s*batch\s*size[=:\s]*([\d.]+)",
        "gpu_memory": r"gpu.*?memory.*?([\d.]+)\s*(?:GB|%)",
        "cache_hit_rate": r"cache.*?hit.*?rate[=:\s]*([\d.]+)",
        
        # Warnings and errors
        "oom": r"(out of memory|OOM|CUDA out of memory|RuntimeError.*memory)",
        "cuda_graph_fail": r"(cuda graph.*fail|failed to capture|graph capture)",
        "nccl_error": r"(NCCL error|nccl.*timeout|collective.*timeout)",
        "slow_tokenizer": r"(tokenizer.*slow|slow.*tokenization)",
        "scheduler_bottleneck": r"(scheduler.*bottleneck|waiting.*schedule|schedule.*delay)",
        "prefill_bottleneck": r"(prefill.*queue|waiting.*prefill|prefill.*backlog)",
        "decode_bottleneck": r"(decode.*bottleneck|decode.*stall)",
        "memory_pressure": r"(memory pressure|low memory|memory warning)",
        "fragmentation": r"(memory fragmentation|fragmented)",
        "allreduce_slow": r"(allreduce.*slow|collective.*slow|nccl.*slow)",
        
        # Feature detection
        "flashinfer": r"(flashinfer|FlashInfer)",
        "triton_moe": r"(triton.*moe|moe.*triton)",
        "cutlass_moe": r"(cutlass.*moe|moe.*cutlass|SGLANG_CUTLASS_MOE)",
        "dp_attention": r"(dp.attention|enable.dp.attention|data.parallel.attention)",
        "radix_cache": r"(radix.cache|disable.radix.cache)",
        "custom_allreduce": r"(custom.all.?reduce|disable.custom.all.?reduce)",
        "jit_deepgemm": r"(jit.deepgemm|SGL_ENABLE_JIT_DEEPGEMM)",
        "trtllm_mla": r"(trtllm.mla|tensorrt.*mla|attention.*trtllm)",
    }

    # Known optimal configurations for different scenarios
    B200_DEEPSEEK_OPTIMAL = {
        "SGL_ENABLE_JIT_DEEPGEMM": "0",
        "SGLANG_CUTLASS_MOE": "1",
        "--attention-backend": "trtllm-mla",
        "--enable-dp-attention": True,
        "--data-parallel-size": "8",
        "--tp-size": "8",
        "--max-running-requests": "3072",
        "--mem-fraction-static": "0.89",
        "--kv-cache-dtype": "fp8_e4m3",
        "--disable-radix-cache": True,
        "--chunked-prefill-size": "32768",
        "--max-prefill-tokens": "32768",
    }

    def __init__(self):
        self.result = AnalysisResult()

    def download_log(self, remote_path: str, remote_host: str) -> str:
        """Download log file from remote host via scp."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            local_path = f.name

        cmd = ["scp", f"{remote_host}:{remote_path}", local_path]
        print(f"Downloading: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Downloaded to: {local_path}")
            return local_path
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e.stderr}")
            sys.exit(1)

    def extract_value(self, pattern: str, text: str) -> Optional[str]:
        """Extract first matching group from pattern."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Return first non-None group
            for group in match.groups():
                if group:
                    return group
        return None

    def analyze_log(self, log_path: str) -> AnalysisResult:
        """Analyze the server log file."""
        print(f"\nAnalyzing: {log_path}\n")
        
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Extract configuration
        self._extract_config(content)
        
        # Extract metrics
        self._extract_metrics(content)
        
        # Analyze for issues
        self._analyze_errors(content, lines)
        self._analyze_performance_patterns(content, lines)
        self._analyze_configuration(content)
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.result

    def _extract_config(self, content: str):
        """Extract server configuration from logs."""
        config_patterns = [
            "tp_size", "dp_size", "max_running_requests", "mem_fraction",
            "chunked_prefill", "max_prefill_tokens", "kv_cache_dtype",
            "attention_backend", "quantization", "cuda_graph_max_bs"
        ]
        
        for key in config_patterns:
            value = self.extract_value(self.PATTERNS[key], content)
            if value:
                self.result.current_config[key] = value

        # Detect enabled features
        features = ["flashinfer", "triton_moe", "cutlass_moe", "dp_attention",
                    "radix_cache", "custom_allreduce", "jit_deepgemm", "trtllm_mla"]
        
        for feature in features:
            if re.search(self.PATTERNS[feature], content, re.IGNORECASE):
                self.result.current_config[f"feature_{feature}"] = True

    def _extract_metrics(self, content: str):
        """Extract performance metrics from logs."""
        metric_patterns = ["throughput", "latency_p50", "latency_p99", 
                          "batch_size", "gpu_memory", "cache_hit_rate"]
        
        for key in metric_patterns:
            value = self.extract_value(self.PATTERNS[key], content)
            if value:
                self.result.metrics[key] = float(value)

    def _analyze_errors(self, content: str, lines: list):
        """Analyze for errors and critical issues."""
        
        # OOM errors
        if re.search(self.PATTERNS["oom"], content, re.IGNORECASE):
            self.result.issues.append(PerformanceIssue(
                severity="critical",
                category="Memory",
                message="Out of memory errors detected",
                suggestion="Reduce memory usage by lowering batch sizes or using more aggressive quantization",
                flag_change="--mem-fraction-static 0.85 OR --max-running-requests <lower_value>"
            ))

        # NCCL errors
        if re.search(self.PATTERNS["nccl_error"], content, re.IGNORECASE):
            self.result.issues.append(PerformanceIssue(
                severity="critical",
                category="Communication",
                message="NCCL communication errors or timeouts detected",
                suggestion="Check GPU interconnect, consider disabling custom allreduce",
                flag_change="--disable-custom-all-reduce"
            ))

        # CUDA graph failures
        if re.search(self.PATTERNS["cuda_graph_fail"], content, re.IGNORECASE):
            self.result.issues.append(PerformanceIssue(
                severity="warning",
                category="CUDA Graphs",
                message="CUDA graph capture failures detected",
                suggestion="Reduce CUDA graph max batch size or disable if unstable",
                flag_change="--cuda-graph-max-bs 64 OR --disable-cuda-graph"
            ))

    def _analyze_performance_patterns(self, content: str, lines: list):
        """Analyze for performance bottleneck patterns."""
        
        # Scheduler bottleneck
        if re.search(self.PATTERNS["scheduler_bottleneck"], content, re.IGNORECASE):
            self.result.issues.append(PerformanceIssue(
                severity="warning",
                category="Scheduling",
                message="Scheduler appears to be a bottleneck",
                suggestion="Increase max running requests to improve scheduling throughput",
                flag_change="--max-running-requests 3072"
            ))

        # Prefill bottleneck
        if re.search(self.PATTERNS["prefill_bottleneck"], content, re.IGNORECASE):
            self.result.issues.append(PerformanceIssue(
                severity="warning",
                category="Prefill",
                message="Prefill queue backlog detected",
                suggestion="Increase chunked prefill size or enable data-parallel attention",
                flag_change="--chunked-prefill-size 32768 --enable-dp-attention"
            ))

        # Decode bottleneck
        if re.search(self.PATTERNS["decode_bottleneck"], content, re.IGNORECASE):
            self.result.issues.append(PerformanceIssue(
                severity="warning",
                category="Decode",
                message="Decode phase bottleneck detected",
                suggestion="Enable optimized attention backend for decode",
                flag_change="--attention-backend trtllm-mla"
            ))

        # Memory fragmentation
        if re.search(self.PATTERNS["fragmentation"], content, re.IGNORECASE):
            self.result.issues.append(PerformanceIssue(
                severity="warning",
                category="Memory",
                message="Memory fragmentation detected",
                suggestion="Enable expandable segments in PyTorch allocator",
                flag_change="ENV: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ))

        # Slow allreduce
        if re.search(self.PATTERNS["allreduce_slow"], content, re.IGNORECASE):
            self.result.issues.append(PerformanceIssue(
                severity="warning",
                category="Communication",
                message="Slow allreduce operations detected",
                suggestion="Try FlashInfer allreduce fusion or check NVLink topology",
                flag_change="--enable-flashinfer-allreduce-fusion"
            ))

        # Slow tokenizer
        if re.search(self.PATTERNS["slow_tokenizer"], content, re.IGNORECASE):
            self.result.issues.append(PerformanceIssue(
                severity="info",
                category="Tokenization",
                message="Tokenization may be slow",
                suggestion="Ensure TOKENIZERS_PARALLELISM is enabled",
                flag_change="ENV: TOKENIZERS_PARALLELISM=true"
            ))

    def _analyze_configuration(self, content: str):
        """Analyze configuration against known optimal settings."""
        config = self.result.current_config
        
        # Check for suboptimal MoE backend
        if config.get("feature_triton_moe") and not config.get("feature_cutlass_moe"):
            self.result.issues.append(PerformanceIssue(
                severity="warning",
                category="MoE Backend",
                message="Using Triton MoE backend instead of CUTLASS",
                suggestion="CUTLASS MoE is typically faster on B200",
                flag_change="ENV: SGLANG_CUTLASS_MOE=1 (remove --moe-runner-backend triton)"
            ))

        # Check attention backend
        if not config.get("feature_trtllm_mla"):
            self.result.issues.append(PerformanceIssue(
                severity="info",
                category="Attention",
                message="Not using TensorRT-LLM MLA attention backend",
                suggestion="TRT-LLM MLA can improve DeepSeek performance",
                flag_change="--attention-backend trtllm-mla"
            ))

        # Check DP attention
        if not config.get("feature_dp_attention"):
            self.result.issues.append(PerformanceIssue(
                severity="info",
                category="Parallelism",
                message="Data-parallel attention not enabled",
                suggestion="DP attention can improve throughput with multiple GPUs",
                flag_change="--enable-dp-attention --data-parallel-size 8"
            ))

        # Check max running requests
        max_req = config.get("max_running_requests")
        if max_req and int(max_req) < 1024:
            self.result.issues.append(PerformanceIssue(
                severity="warning",
                category="Concurrency",
                message=f"Low max_running_requests ({max_req})",
                suggestion="Increase for better throughput on high-memory GPUs",
                flag_change="--max-running-requests 3072"
            ))

        # Check KV cache dtype
        kv_dtype = config.get("kv_cache_dtype")
        if kv_dtype and kv_dtype not in ["fp8_e4m3", "fp8"]:
            self.result.issues.append(PerformanceIssue(
                severity="info",
                category="Memory",
                message=f"KV cache dtype is '{kv_dtype}'",
                suggestion="FP8 KV cache can reduce memory and improve throughput",
                flag_change="--kv-cache-dtype fp8_e4m3"
            ))

        # Check memory fraction
        mem_frac = config.get("mem_fraction")
        if mem_frac and float(mem_frac) < 0.85:
            self.result.issues.append(PerformanceIssue(
                severity="info",
                category="Memory",
                message=f"Memory fraction is {mem_frac} (conservative)",
                suggestion="Can increase to 0.89 for more KV cache capacity",
                flag_change="--mem-fraction-static 0.89"
            ))

        # Check for JIT DeepGEMM (can cause issues)
        if config.get("feature_jit_deepgemm"):
            # Check if it's explicitly disabled
            if "JIT_DEEPGEMM=0" not in content and "JIT_DEEPGEMM=false" not in content.lower():
                self.result.issues.append(PerformanceIssue(
                    severity="info",
                    category="GEMM",
                    message="JIT DeepGEMM may be enabled",
                    suggestion="Disable JIT DeepGEMM for stability on B200",
                    flag_change="ENV: SGL_ENABLE_JIT_DEEPGEMM=0"
                ))

    def _generate_recommendations(self):
        """Generate prioritized recommendations."""
        
        # Sort issues by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_issues = sorted(self.result.issues, 
                               key=lambda x: severity_order.get(x.severity, 3))
        
        # Generate flag recommendations
        add_flags = []
        remove_flags = []
        env_vars = []
        
        for issue in sorted_issues:
            if issue.flag_change:
                if issue.flag_change.startswith("ENV:"):
                    env_vars.append(issue.flag_change.replace("ENV:", "").strip())
                elif "remove" in issue.flag_change.lower():
                    parts = issue.flag_change.split("remove")
                    if len(parts) > 1:
                        remove_flags.append(parts[1].strip())
                else:
                    add_flags.append(issue.flag_change)

        self.result.recommendations = {
            "add_flags": list(set(add_flags)),
            "remove_flags": list(set(remove_flags)),
            "env_vars": list(set(env_vars)),
        }

    def print_report(self):
        """Print the analysis report."""
        print("=" * 70)
        print("SGLANG SERVER LOG ANALYSIS REPORT")
        print("=" * 70)
        
        # Current configuration
        print("\n📋 DETECTED CONFIGURATION:")
        print("-" * 40)
        if self.result.current_config:
            for key, value in sorted(self.result.current_config.items()):
                print(f"  {key}: {value}")
        else:
            print("  (No configuration detected)")
        
        # Metrics
        print("\n📊 PERFORMANCE METRICS:")
        print("-" * 40)
        if self.result.metrics:
            for key, value in sorted(self.result.metrics.items()):
                print(f"  {key}: {value}")
        else:
            print("  (No metrics detected)")
        
        # Issues
        print("\n🔍 IDENTIFIED ISSUES:")
        print("-" * 40)
        
        if not self.result.issues:
            print("  ✅ No issues detected")
        else:
            for issue in self.result.issues:
                icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
                print(f"\n  {icon} [{issue.severity.upper()}] {issue.category}")
                print(f"     Message: {issue.message}")
                print(f"     Suggestion: {issue.suggestion}")
                if issue.flag_change:
                    print(f"     Flag: {issue.flag_change}")
        
        # Recommendations
        print("\n" + "=" * 70)
        print("📝 RECOMMENDED CHANGES:")
        print("=" * 70)
        
        recs = self.result.recommendations
        
        if recs.get("env_vars"):
            print("\n🌍 Environment Variables to set:")
            for env in recs["env_vars"]:
                print(f"  export {env}")
        
        if recs.get("add_flags"):
            print("\n➕ Flags to ADD:")
            for flag in recs["add_flags"]:
                print(f"  {flag}")
        
        if recs.get("remove_flags"):
            print("\n➖ Flags to REMOVE:")
            for flag in recs["remove_flags"]:
                print(f"  {flag}")
        
        # B200 DeepSeek optimal config reference
        print("\n" + "=" * 70)
        print("📌 REFERENCE: Optimal B200 DeepSeek-R1 Configuration")
        print("=" * 70)
        print("""
Environment:
  SGL_ENABLE_JIT_DEEPGEMM=0
  SGLANG_CUTLASS_MOE=1

Server flags:
  --attention-backend trtllm-mla
  --enable-dp-attention
  --data-parallel-size 8
  --tp-size 8
  --max-running-requests 3072
  --mem-fraction-static 0.89
  --kv-cache-dtype fp8_e4m3
  --disable-radix-cache
  --chunked-prefill-size 32768
  --max-prefill-tokens 32768
""")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SGLang server logs for performance optimization"
    )
    parser.add_argument(
        "log_path",
        help="Path to log file (local or remote)"
    )
    parser.add_argument(
        "--remote", "-r",
        help="Remote host (user@host) if log is on remote machine"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for report (default: stdout)"
    )
    
    args = parser.parse_args()
    
    analyzer = SGLangLogAnalyzer()
    
    # Download if remote
    if args.remote:
        log_path = analyzer.download_log(args.log_path, args.remote)
    else:
        log_path = args.log_path
        if not Path(log_path).exists():
            print(f"Error: Log file not found: {log_path}")
            sys.exit(1)
    
    # Analyze
    analyzer.analyze_log(log_path)
    
    # Print report
    analyzer.print_report()


if __name__ == "__main__":
    main()
