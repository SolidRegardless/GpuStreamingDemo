using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that executes the affine transform kernel: y[i] = a * x[i] + b.
/// Supports both single-buffered and double-buffered execution modes.
/// </summary>
public sealed class AffineTransformTask : IDoubleBufferedTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<float>, ArrayView<float>, float, float>? _kernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _input;
    private MemoryBuffer1D<float, Stride1D.Dense>? _output;
    private int _batchSize;

    // Double-buffered resources (two slots)
    private readonly MemoryBuffer1D<float, Stride1D.Dense>?[] _dbInputs = new MemoryBuffer1D<float, Stride1D.Dense>?[2];
    private readonly MemoryBuffer1D<float, Stride1D.Dense>?[] _dbOutputs = new MemoryBuffer1D<float, Stride1D.Dense>?[2];
    private readonly float[][] _hostInputs = new float[2][];
    private readonly float[][] _hostOutputs = new float[2][];
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, float, float>? _streamKernel;

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
    public void SetupDoubleBuffered(Accelerator accelerator, int batchSize, AcceleratorStream streamA, AcceleratorStream streamB)
    {
        _accelerator = accelerator;
        _batchSize = batchSize;

        _streamKernel = accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, float, float>(Kernels.AffineTransform);

        var rnd = new Random(42);

        for (int slot = 0; slot < 2; slot++)
        {
            _dbInputs[slot] = accelerator.Allocate1D<float>(batchSize);
            _dbOutputs[slot] = accelerator.Allocate1D<float>(batchSize);
            _hostInputs[slot] = new float[batchSize];
            _hostOutputs[slot] = new float[batchSize];

            // Fill host input with test data
            for (int i = 0; i < batchSize; i++)
                _hostInputs[slot][i] = (float)rnd.NextDouble();
        }
    }

    /// <inheritdoc />
    public void CopyToDeviceAsync(int slot, AcceleratorStream stream)
    {
        _dbInputs[slot]!.CopyFromCPU(stream, _hostInputs[slot]);
    }

    /// <inheritdoc />
    public void LaunchKernel(int slot, AcceleratorStream stream)
    {
        _streamKernel!(stream, _batchSize, _dbInputs[slot]!.View, _dbOutputs[slot]!.View, 1.1f, 0.9f);
    }

    /// <inheritdoc />
    public void CopyFromDeviceAsync(int slot, AcceleratorStream stream)
    {
        _dbOutputs[slot]!.CopyToCPU(stream, _hostOutputs[slot]);
    }

    /// <inheritdoc />
    public float[] GetHostOutput(int slot) => _hostOutputs[slot];

    /// <inheritdoc />
    public void Dispose()
    {
        _input?.Dispose();
        _output?.Dispose();
        for (int i = 0; i < 2; i++)
        {
            _dbInputs[i]?.Dispose();
            _dbOutputs[i]?.Dispose();
        }
    }
}
