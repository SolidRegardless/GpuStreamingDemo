using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that computes Mandelbrot set escape iterations on the GPU.
/// The batch size determines the total number of pixels (width × height).
/// </summary>
public sealed class MandelbrotTask : IBenchmarkTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<int>, int, int>? _kernel;
    private MemoryBuffer1D<int, Stride1D.Dense>? _output;
    private int _batchSize;
    private int _width;

    /// <inheritdoc />
    public string Name => "Mandelbrot";

    /// <inheritdoc />
    public string Description => "Mandelbrot set escape iterations (max 1000)";

    /// <inheritdoc />
    public void Setup(Accelerator accelerator, int batchSize)
    {
        _accelerator = accelerator;
        _batchSize = batchSize;
        _width = (int)Math.Sqrt(batchSize);
        if (_width < 1) _width = 1;

        // Adjust batch size to be width * height
        int height = batchSize / _width;
        _batchSize = _width * height;

        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, int, int>(Kernels.Mandelbrot);

        _output = accelerator.Allocate1D<int>(_batchSize);
    }

    /// <inheritdoc />
    public void Execute()
    {
        _kernel!(_batchSize, _output!.View, _width, 1000);
        _accelerator!.Synchronize();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _output?.Dispose();
    }
}
