using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that executes vector addition: z[i] = x[i] + y[i].
/// </summary>
public sealed class VectorAddTask : IBenchmarkTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _kernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _x;
    private MemoryBuffer1D<float, Stride1D.Dense>? _y;
    private MemoryBuffer1D<float, Stride1D.Dense>? _z;
    private int _batchSize;

    /// <inheritdoc />
    public string Name => "VectorAdd";

    /// <inheritdoc />
    public string Description => "z[i] = x[i] + y[i]";

    /// <inheritdoc />
    public void Setup(Accelerator accelerator, int batchSize)
    {
        _accelerator = accelerator;
        _batchSize = batchSize;

        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(Kernels.VectorAdd);

        _x = accelerator.Allocate1D<float>(batchSize);
        _y = accelerator.Allocate1D<float>(batchSize);
        _z = accelerator.Allocate1D<float>(batchSize);

        var rnd = new Random(42);
        var data = new float[batchSize];
        for (int i = 0; i < batchSize; i++) data[i] = (float)rnd.NextDouble();
        _x.CopyFromCPU(data);
        for (int i = 0; i < batchSize; i++) data[i] = (float)rnd.NextDouble();
        _y.CopyFromCPU(data);
    }

    /// <inheritdoc />
    public void Execute()
    {
        _kernel!(_batchSize, _x!.View, _y!.View, _z!.View);
        _accelerator!.Synchronize();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _x?.Dispose();
        _y?.Dispose();
        _z?.Dispose();
    }
}
