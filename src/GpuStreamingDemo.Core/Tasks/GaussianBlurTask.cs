using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that applies a horizontal Gaussian blur to a synthetic image on the GPU.
/// Represents image pipeline and computer vision workloads.
/// </summary>
public sealed class GaussianBlurTask : IBenchmarkTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<float>, ArrayView<float>, int, int>? _kernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _input;
    private MemoryBuffer1D<float, Stride1D.Dense>? _output;
    private int _batchSize;
    private int _width;
    private const int BlurRadius = 16;

    /// <inheritdoc />
    public string Name => "GaussianBlur";

    /// <inheritdoc />
    public string Description => "Horizontal Gaussian blur (radius 16) — image processing";

    /// <inheritdoc />
    public void Setup(Accelerator accelerator, int batchSize)
    {
        _accelerator = accelerator;
        _width = (int)Math.Sqrt(batchSize);
        if (_width < 1) _width = 1;
        int height = batchSize / _width;
        _batchSize = _width * height;

        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(Kernels.GaussianBlur);

        _input = accelerator.Allocate1D<float>(_batchSize);
        _output = accelerator.Allocate1D<float>(_batchSize);

        // Synthetic image: gradient with noise
        var rnd = new Random(42);
        var data = new float[_batchSize];
        for (int i = 0; i < _batchSize; i++)
            data[i] = (i % _width) / (float)_width + (float)rnd.NextDouble() * 0.1f;
        _input.CopyFromCPU(data);
    }

    /// <inheritdoc />
    public void Execute()
    {
        _kernel!(_batchSize, _input!.View, _output!.View, _width, BlurRadius);
        _accelerator!.Synchronize();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _input?.Dispose();
        _output?.Dispose();
    }
}
