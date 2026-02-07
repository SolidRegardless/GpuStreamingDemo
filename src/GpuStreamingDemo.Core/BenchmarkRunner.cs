using System.Diagnostics;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Executes GPU compute benchmarks using pluggable <see cref="IBenchmarkTask"/> implementations.
/// Manages accelerator lifecycle and timing of warmup and benchmark iterations.
/// </summary>
public static class GpuBenchmarkRunner
{
    /// <summary>
    /// Runs a GPU benchmark for the given task on the specified accelerator.
    /// </summary>
    /// <param name="task">The benchmark task to execute.</param>
    /// <param name="acceleratorKind">The accelerator kind to use.</param>
    /// <param name="batchSize">The number of elements to process per iteration.</param>
    /// <param name="iterations">The number of timed benchmark iterations.</param>
    /// <param name="warmupIterations">The number of warmup iterations before timing.</param>
    /// <returns>A <see cref="BenchmarkResult"/> containing latency and throughput metrics.</returns>
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
                batchSize,
                iterations,
                avgMs,
                p95Ms,
                throughput
            );
        }
        finally
        {
            accelerator.Dispose();
            context.Dispose();
        }
    }
}
