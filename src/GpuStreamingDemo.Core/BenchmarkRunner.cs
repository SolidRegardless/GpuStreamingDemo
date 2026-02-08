using System.Diagnostics;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Executes GPU compute benchmarks using pluggable <see cref="IBenchmarkTask"/> implementations.
/// Manages accelerator lifecycle and timing of warmup and benchmark iterations.
/// </summary>
public static class GpuBenchmarkRunner
{
    /// <summary>
    /// Runs a GPU benchmark for the given task on the specified accelerator kind.
    /// </summary>
    public static BenchmarkResult Run(
        IBenchmarkTask task,
        AcceleratorKind acceleratorKind,
        int batchSize,
        int iterations,
        int warmupIterations)
    {
        var (context, accelerator) = AcceleratorFactory.Create(acceleratorKind);

        try
        {
            return RunOnAccelerator(task, acceleratorKind, accelerator.Name, accelerator, batchSize, iterations, warmupIterations);
        }
        finally
        {
            accelerator.Dispose();
            context.Dispose();
        }
    }

    /// <summary>
    /// Runs a GPU benchmark for the given task on a specific device by index.
    /// </summary>
    public static BenchmarkResult RunOnDevice(
        IBenchmarkTask task,
        int deviceIndex,
        int batchSize,
        int iterations,
        int warmupIterations)
    {
        var (context, accelerator, deviceName) = AcceleratorFactory.CreateByIndex(deviceIndex);
        var kind = accelerator.AcceleratorType switch
        {
            AcceleratorType.CPU => AcceleratorKind.Cpu,
            AcceleratorType.Cuda => AcceleratorKind.Cuda,
            AcceleratorType.OpenCL => AcceleratorKind.OpenCL,
            _ => AcceleratorKind.Auto
        };

        try
        {
            return RunOnAccelerator(task, kind, deviceName, accelerator, batchSize, iterations, warmupIterations);
        }
        finally
        {
            accelerator.Dispose();
            context.Dispose();
        }
    }

    private static BenchmarkResult RunOnAccelerator(
        IBenchmarkTask task,
        AcceleratorKind acceleratorKind,
        string deviceName,
        ILGPU.Runtime.Accelerator accelerator,
        int batchSize,
        int iterations,
        int warmupIterations)
    {
        task.Setup(accelerator, batchSize);

        // Warmup
        for (int i = 0; i < warmupIterations; i++)
            task.Execute();

        // Benchmark
        var latencies = new double[iterations];
        var sw = new Stopwatch();

        for (int i = 0; i < iterations; i++)
        {
            sw.Restart();
            task.Execute();
            sw.Stop();
            latencies[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(latencies);

        var avgMs = latencies.Average();
        var p95Ms = latencies[(int)(latencies.Length * 0.95)];
        var totalSeconds = latencies.Sum() / 1000.0;
        var throughput = (double)batchSize * iterations / totalSeconds;

        return new BenchmarkResult(
            task.Name,
            acceleratorKind,
            deviceName,
            batchSize,
            iterations,
            avgMs,
            p95Ms,
            throughput
        );
    }
}
