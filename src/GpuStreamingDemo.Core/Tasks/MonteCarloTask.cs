using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that estimates Pi using Monte-Carlo random sampling on the GPU.
/// Each thread runs multiple trials, counting hits inside the unit circle.
/// Represents financial simulation / Monte-Carlo risk model workloads.
/// </summary>
public sealed class MonteCarloTask : IBenchmarkTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<int>, int>? _kernel;
    private MemoryBuffer1D<int, Stride1D.Dense>? _output;
    private int _batchSize;
    private const int TrialsPerThread = 256;

    /// <inheritdoc />
    public string Name => "MonteCarloPi";

    /// <inheritdoc />
    public string Description => "Monte-Carlo Pi estimation (256 trials/thread) — financial simulation proxy";

    /// <inheritdoc />
    public void Setup(Accelerator accelerator, int batchSize)
    {
        _accelerator = accelerator;
        _batchSize = batchSize;

        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, int>(Kernels.MonteCarloPi);

        _output = accelerator.Allocate1D<int>(batchSize);
    }

    /// <inheritdoc />
    public void Execute()
    {
        _kernel!(_batchSize, _output!.View, TrialsPerThread);
        _accelerator!.Synchronize();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _output?.Dispose();
    }
}
