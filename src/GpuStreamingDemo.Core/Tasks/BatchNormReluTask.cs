using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that applies batch normalisation followed by ReLU activation on the GPU.
/// Represents AI feature extraction and inference layer workloads.
/// </summary>
public sealed class BatchNormReluTask : IBenchmarkTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<float>, ArrayView<float>, float, float, float, float>? _kernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _input;
    private MemoryBuffer1D<float, Stride1D.Dense>? _output;
    private int _batchSize;

    /// <inheritdoc />
    public string Name => "BatchNormReLU";

    /// <inheritdoc />
    public string Description => "Batch normalisation + ReLU activation — AI inference";

    /// <inheritdoc />
    public void Setup(Accelerator accelerator, int batchSize)
    {
        _accelerator = accelerator;
        _batchSize = batchSize;

        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, float, float, float, float>(Kernels.BatchNormRelu);

        _input = accelerator.Allocate1D<float>(batchSize);
        _output = accelerator.Allocate1D<float>(batchSize);

        // Simulate activations from a neural network layer (normal-ish distribution)
        var rnd = new Random(42);
        var data = new float[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            // Box-Muller for approximate normal distribution
            double u1 = 1.0 - rnd.NextDouble();
            double u2 = rnd.NextDouble();
            data[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
        }
        _input.CopyFromCPU(data);
    }

    /// <inheritdoc />
    public void Execute()
    {
        // Typical batch norm parameters
        _kernel!(_batchSize, _input!.View, _output!.View,
            0.0f, 1.0f, 1.0f, 0.0f);
        _accelerator!.Synchronize();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _input?.Dispose();
        _output?.Dispose();
    }
}
