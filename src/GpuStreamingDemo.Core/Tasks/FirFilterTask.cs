using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that applies a Finite Impulse Response (FIR) filter on the GPU.
/// Represents signal processing and real-time filtering workloads.
/// </summary>
public sealed class FirFilterTask : IBenchmarkTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>? _kernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _input;
    private MemoryBuffer1D<float, Stride1D.Dense>? _coeffs;
    private MemoryBuffer1D<float, Stride1D.Dense>? _output;
    private int _batchSize;
    private const int NumTaps = 64;

    /// <inheritdoc />
    public string Name => "FIRFilter";

    /// <inheritdoc />
    public string Description => "64-tap FIR convolution filter — signal processing";

    /// <inheritdoc />
    public void Setup(Accelerator accelerator, int batchSize)
    {
        _accelerator = accelerator;
        _batchSize = batchSize;

        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(Kernels.FirFilter);

        _input = accelerator.Allocate1D<float>(batchSize);
        _coeffs = accelerator.Allocate1D<float>(NumTaps);
        _output = accelerator.Allocate1D<float>(batchSize);

        // Generate a synthetic signal
        var rnd = new Random(42);
        var signal = new float[batchSize];
        for (int i = 0; i < batchSize; i++)
            signal[i] = (float)(Math.Sin(2.0 * Math.PI * i / 100.0) + rnd.NextDouble() * 0.1);
        _input.CopyFromCPU(signal);

        // Low-pass filter coefficients (simple windowed sinc)
        var taps = new float[NumTaps];
        float sum = 0;
        for (int i = 0; i < NumTaps; i++)
        {
            float n = i - NumTaps / 2.0f;
            taps[i] = n == 0 ? 1.0f : (float)(Math.Sin(0.25 * Math.PI * n) / (Math.PI * n));
            sum += taps[i];
        }
        for (int i = 0; i < NumTaps; i++) taps[i] /= sum; // normalise
        _coeffs.CopyFromCPU(taps);
    }

    /// <inheritdoc />
    public void Execute()
    {
        _kernel!(_batchSize, _input!.View, _coeffs!.View, _output!.View, NumTaps);
        _accelerator!.Synchronize();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _input?.Dispose();
        _coeffs?.Dispose();
        _output?.Dispose();
    }
}
