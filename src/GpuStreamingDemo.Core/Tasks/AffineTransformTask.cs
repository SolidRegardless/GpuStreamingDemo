using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that executes the affine transform kernel: y[i] = a * x[i] + b.
/// </summary>
public sealed class AffineTransformTask : IBenchmarkTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<float>, ArrayView<float>, float, float>? _kernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _input;
    private MemoryBuffer1D<float, Stride1D.Dense>? _output;
    private int _batchSize;

    /// <inheritdoc />
    public string Name => "AffineTransform";

    /// <inheritdoc />
    public string Description => "y[i] = a * x[i] + b";

    /// <inheritdoc />
    public void Setup(Accelerator accelerator, int batchSize)
    {
        _accelerator = accelerator;
        _batchSize = batchSize;

        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, float, float>(Kernels.AffineTransform);

        _input = accelerator.Allocate1D<float>(batchSize);
        _output = accelerator.Allocate1D<float>(batchSize);

        // Initialise input data
        var data = new float[batchSize];
        var rnd = new Random(42);
        for (int i = 0; i < batchSize; i++)
            data[i] = (float)rnd.NextDouble();
        _input.CopyFromCPU(data);
    }

    /// <inheritdoc />
    public void Execute()
    {
        _kernel!(_batchSize, _input!.View, _output!.View, 1.1f, 0.9f);
        _accelerator!.Synchronize();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _input?.Dispose();
        _output?.Dispose();
    }
}
