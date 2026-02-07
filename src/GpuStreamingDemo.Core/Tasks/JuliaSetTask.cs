using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that computes Julia set escape iterations on the GPU.
/// Uses a fixed complex constant c = -0.7 + 0.27015i for a visually interesting set.
/// Represents fractal generation beyond Mandelbrot.
/// </summary>
public sealed class JuliaSetTask : IBenchmarkTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<int>, int, int, float, float>? _kernel;
    private MemoryBuffer1D<int, Stride1D.Dense>? _output;
    private int _batchSize;
    private int _width;

    /// <inheritdoc />
    public string Name => "JuliaSet";

    /// <inheritdoc />
    public string Description => "Julia set fractal (c = -0.7 + 0.27015i, max 1000 iter)";

    /// <inheritdoc />
    public void Setup(Accelerator accelerator, int batchSize)
    {
        _accelerator = accelerator;
        _width = (int)Math.Sqrt(batchSize);
        if (_width < 1) _width = 1;
        int height = batchSize / _width;
        _batchSize = _width * height;

        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, int, int, float, float>(Kernels.JuliaSet);

        _output = accelerator.Allocate1D<int>(_batchSize);
    }

    /// <inheritdoc />
    public void Execute()
    {
        _kernel!(_batchSize, _output!.View, _width, 1000, -0.7f, 0.27015f);
        _accelerator!.Synchronize();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _output?.Dispose();
    }
}
