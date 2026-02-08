namespace GpuStreamingDemo.Core;

/// <summary>
/// Encapsulates the results of a GPU benchmark run.
/// </summary>
/// <param name="TaskName">The name of the benchmark task that was executed.</param>
/// <param name="Accelerator">The type of accelerator used for the benchmark.</param>
/// <param name="DeviceName">The specific device name (e.g. "NVIDIA GeForce RTX 4070 Ti").</param>
/// <param name="BatchSize">The number of elements processed in each batch.</param>
/// <param name="Iterations">The number of benchmark iterations executed.</param>
/// <param name="AvgLatencyMs">The average latency per iteration in milliseconds.</param>
/// <param name="P95LatencyMs">The 95th percentile latency in milliseconds.</param>
/// <param name="ThroughputElementsPerSec">The overall throughput in elements per second.</param>
public sealed record BenchmarkResult(
    string TaskName,
    AcceleratorKind Accelerator,
    string DeviceName,
    int BatchSize,
    int Iterations,
    double AvgLatencyMs,
    double P95LatencyMs,
    double ThroughputElementsPerSec
);
