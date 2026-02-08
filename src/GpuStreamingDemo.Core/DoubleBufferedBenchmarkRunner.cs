using System.Diagnostics;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Result of a double-buffered benchmark run, extending the standard result
/// with overlap metrics.
/// </summary>
/// <param name="TaskName">The name of the benchmark task.</param>
/// <param name="Accelerator">The accelerator type used.</param>
/// <param name="DeviceName">The specific device name (e.g. "NVIDIA GeForce RTX 4070 Ti").</param>
/// <param name="BatchSize">Elements per batch.</param>
/// <param name="Iterations">Number of timed iterations.</param>
/// <param name="AvgLatencyMs">Average latency per iteration in milliseconds.</param>
/// <param name="P95LatencyMs">95th percentile latency in milliseconds.</param>
/// <param name="ThroughputElementsPerSec">Overall throughput in elements per second.</param>
/// <param name="SingleBufferAvgMs">Average latency of the single-buffered baseline.</param>
/// <param name="SpeedupFactor">Throughput improvement factor vs single-buffered.</param>
/// <param name="OverlapPercentage">Estimated percentage of transfer time hidden by overlap.</param>
public sealed record DoubleBufferedResult(
    string TaskName,
    AcceleratorKind Accelerator,
    string DeviceName,
    int BatchSize,
    int Iterations,
    double AvgLatencyMs,
    double P95LatencyMs,
    double ThroughputElementsPerSec,
    double SingleBufferAvgMs,
    double SpeedupFactor,
    double OverlapPercentage
);

/// <summary>
/// Executes double-buffered GPU benchmarks using <see cref="IDoubleBufferedTask"/> implementations.
/// Overlaps host-to-device and device-to-host transfers with kernel execution using two
/// alternating buffer slots on separate ILGPU streams.
/// </summary>
public static class DoubleBufferedBenchmarkRunner
{
    /// <summary>
    /// Runs a double-buffered benchmark, measuring throughput improvement vs single-buffered execution.
    /// </summary>
    /// <param name="task">The double-buffered benchmark task.</param>
    /// <param name="acceleratorKind">The accelerator to use.</param>
    /// <param name="batchSize">Elements per batch.</param>
    /// <param name="iterations">Number of timed iterations (must be ≥ 2).</param>
    /// <param name="warmupIterations">Number of warmup iterations.</param>
    /// <returns>A <see cref="DoubleBufferedResult"/> with latency, throughput, and overlap metrics.</returns>
    public static DoubleBufferedResult Run(
        IDoubleBufferedTask task,
        AcceleratorKind acceleratorKind,
        int batchSize,
        int iterations,
        int warmupIterations)
    {
        var (context, accelerator) = AcceleratorFactory.Create(acceleratorKind);

        try
        {
            using var streamA = accelerator.CreateStream();
            using var streamB = accelerator.CreateStream();

            task.SetupDoubleBuffered(accelerator, batchSize, streamA, streamB);

            // --- Phase 1: Single-buffered baseline (full H2D → compute → D2H) ---
            var sw = new Stopwatch();

            // Warmup single-buffered
            for (int i = 0; i < warmupIterations; i++)
            {
                task.CopyToDeviceAsync(0, streamA);
                streamA.Synchronize();
                task.LaunchKernel(0, streamA);
                streamA.Synchronize();
                task.CopyFromDeviceAsync(0, streamA);
                streamA.Synchronize();
            }

            var singleLatencies = new double[iterations];
            for (int i = 0; i < iterations; i++)
            {
                sw.Restart();
                task.CopyToDeviceAsync(0, streamA);
                streamA.Synchronize();
                task.LaunchKernel(0, streamA);
                streamA.Synchronize();
                task.CopyFromDeviceAsync(0, streamA);
                streamA.Synchronize();
                sw.Stop();
                singleLatencies[i] = sw.Elapsed.TotalMilliseconds;
            }

            Array.Sort(singleLatencies);
            var singleAvgMs = singleLatencies.Average();

            // --- Phase 2: Double-buffered ---

            // Warmup double-buffered
            for (int i = 0; i < warmupIterations; i++)
            {
                int slot = i % 2;
                var stream = slot == 0 ? streamA : streamB;
                task.CopyToDeviceAsync(slot, stream);
                stream.Synchronize();
                task.LaunchKernel(slot, stream);
                stream.Synchronize();
                task.CopyFromDeviceAsync(slot, stream);
                stream.Synchronize();
            }

            // Timed double-buffered: true pipelined overlap.
            // Each stream queues: H2D → Kernel → D2H for its slot.
            // While stream A executes kernel(A), stream B is uploading batch(B).
            // We measure total wall-clock time for all iterations and per-iteration latency.
            var doubleLatencies = new double[iterations];

            // Prime: queue first batch entirely on stream A
            task.CopyToDeviceAsync(0, streamA);
            task.LaunchKernel(0, streamA);
            task.CopyFromDeviceAsync(0, streamA);

            sw.Restart();

            for (int i = 0; i < iterations; i++)
            {
                int currentSlot = i % 2;
                int nextSlot = (i + 1) % 2;
                var currentStream = currentSlot == 0 ? streamA : streamB;
                var nextStream = nextSlot == 0 ? streamA : streamB;

                var iterStart = sw.Elapsed.TotalMilliseconds;

                // Queue next batch on the other stream (overlaps with current stream's work)
                if (i < iterations - 1)
                {
                    task.CopyToDeviceAsync(nextSlot, nextStream);
                    task.LaunchKernel(nextSlot, nextStream);
                    task.CopyFromDeviceAsync(nextSlot, nextStream);
                }

                // Wait for the current batch to complete
                currentStream.Synchronize();

                doubleLatencies[i] = sw.Elapsed.TotalMilliseconds - iterStart;
            }

            sw.Stop();

            Array.Sort(doubleLatencies);

            var doubleAvgMs = doubleLatencies.Average();
            var p95Ms = doubleLatencies[(int)(doubleLatencies.Length * 0.95)];
            var totalSeconds = doubleLatencies.Sum() / 1000.0;
            var throughput = (double)batchSize * iterations / totalSeconds;

            // Calculate overlap metrics
            var speedup = singleAvgMs / doubleAvgMs;
            // Overlap% = how much of the transfer time was hidden
            // If single = transfer + compute and double ≈ max(transfer, compute),
            // then overlap% ≈ (1 - double/single) * 100 / (transfer_fraction)
            // Simplified: just report the speedup-derived overlap
            var overlapPct = Math.Max(0, (1.0 - (doubleAvgMs / singleAvgMs)) * 100.0);

            return new DoubleBufferedResult(
                task.Name + " (2xBuf)",
                acceleratorKind,
                accelerator.Name,
                batchSize,
                iterations,
                doubleAvgMs,
                p95Ms,
                throughput,
                singleAvgMs,
                speedup,
                overlapPct
            );
        }
        finally
        {
            accelerator.Dispose();
            context.Dispose();
        }
    }
}
