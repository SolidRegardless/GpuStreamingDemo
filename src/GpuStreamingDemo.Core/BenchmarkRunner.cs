using System.Diagnostics;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Executes GPU compute benchmarks with a specified accelerator.
/// Measures latency and throughput of GPU kernel execution.
/// </summary>
public static class GpuBenchmarkRunner
{
    /// <summary>
    /// Runs a GPU benchmark using the specified accelerator.
    /// </summary>
    /// <param name="accelerator">The accelerator kind to use (CPU, CUDA, OpenCL, or Auto).</param>
    /// <param name="batchSize">The number of elements to process per iteration.</param>
    /// <param name="iterations">The number of timed benchmark iterations to execute.</param>
    /// <param name="warmupIterations">The number of warmup iterations before benchmarking (for JIT and kernel compilation).</param>
    /// <returns>A <see cref="BenchmarkResult"/> containing latency and throughput metrics.</returns>
    public static BenchmarkResult Run(
        AcceleratorKind accelerator,
        int batchSize,
        int iterations,
        int warmupIterations)
    {
        using var engine = new GpuComputeEngine(accelerator);

        var input = new float[batchSize];
        var output = new float[batchSize];
        var rnd = new Random(42);

        for (int i = 0; i < batchSize; i++)
            input[i] = (float)rnd.NextDouble();

        // ---- Warmup (JIT + kernel compile) ----
        for (int i = 0; i < warmupIterations; i++)
            engine.ProcessBatch(input, output, batchSize, 1.1f, 0.9f);

        var latencies = new double[iterations];
        var sw = new Stopwatch();

        // ---- Benchmark ----
        for (int i = 0; i < iterations; i++)
        {
            sw.Restart();
            engine.ProcessBatch(input, output, batchSize, 1.1f, 0.9f);
            sw.Stop();

            latencies[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(latencies);

        var avgMs = latencies.Average();
        var p95Ms = latencies[(int)(latencies.Length * 0.95)];

        var totalElements = (double)batchSize * iterations;
        var totalSeconds = latencies.Sum() / 1000.0;
        var throughput = totalElements / totalSeconds;

        return new BenchmarkResult(
            accelerator,
            batchSize,
            iterations,
            avgMs,
            p95Ms,
            throughput
        );
    }
}
