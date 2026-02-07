using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that executes the classic BLAS SAXPY operation: y[i] = a * x[i] + y[i].
/// </summary>
public sealed class SaxpyTask : IBenchmarkTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<float>, ArrayView<float>, float>? _kernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _x;
    private MemoryBuffer1D<float, Stride1D.Dense>? _y;
    private float[]? _yData;
    private int _batchSize;

    /// <inheritdoc />
    public string Name => "Saxpy";

    /// <inheritdoc />
    public string Description => "y[i] = a * x[i] + y[i] (BLAS SAXPY)";

    /// <inheritdoc />
    public void Setup(Accelerator accelerator, int batchSize)
    {
        _accelerator = accelerator;
        _batchSize = batchSize;

        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, float>(Kernels.Saxpy);

        _x = accelerator.Allocate1D<float>(batchSize);
        _y = accelerator.Allocate1D<float>(batchSize);

        var rnd = new Random(42);
        var xData = new float[batchSize];
        _yData = new float[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            xData[i] = (float)rnd.NextDouble();
            _yData[i] = (float)rnd.NextDouble();
        }
        _x.CopyFromCPU(xData);
    }

    /// <inheritdoc />
    public void Execute()
    {
        // Reset y each iteration so results don't explode
        _y!.CopyFromCPU(_yData!);
        _kernel!(_batchSize, _x!.View, _y!.View, 2.0f);
        _accelerator!.Synchronize();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _x?.Dispose();
        _y?.Dispose();
    }
}
