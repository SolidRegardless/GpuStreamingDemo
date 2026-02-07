using ILGPU.Runtime;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Defines a self-contained GPU benchmark task that can be plugged into the benchmark harness.
/// Each implementation manages its own GPU buffers and kernels.
/// </summary>
public interface IBenchmarkTask : IDisposable
{
    /// <summary>
    /// Gets the short name of the benchmark task (e.g. "AffineTransform").
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets a human-readable description of what the task computes.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Allocates GPU buffers and compiles kernels for the given accelerator and batch size.
    /// </summary>
    /// <param name="accelerator">The ILGPU accelerator to use.</param>
    /// <param name="batchSize">The number of elements to process per iteration.</param>
    void Setup(Accelerator accelerator, int batchSize);

    /// <summary>
    /// Executes one iteration of the benchmark, including synchronisation.
    /// </summary>
    void Execute();
}
