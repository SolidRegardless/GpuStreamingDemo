using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core.Tasks;

/// <summary>
/// Benchmark task that runs a brute-force N-body gravitational simulation on the GPU.
/// Each thread computes forces from all other bodies — O(n²) per step.
/// Represents scientific computing and physics simulation workloads.
/// </summary>
public sealed class NBodyTask : IBenchmarkTask
{
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, float>? _kernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _posIn;
    private MemoryBuffer1D<float, Stride1D.Dense>? _velIn;
    private MemoryBuffer1D<float, Stride1D.Dense>? _posOut;
    private MemoryBuffer1D<float, Stride1D.Dense>? _velOut;
    private int _numBodies;

    /// <inheritdoc />
    public string Name => "NBody";

    /// <inheritdoc />
    public string Description => "Brute-force N-body gravitational simulation — scientific computing";

    /// <inheritdoc />
    public void Setup(Accelerator accelerator, int batchSize)
    {
        _accelerator = accelerator;
        // N-body is O(n²), so cap body count to keep runtime sane
        _numBodies = Math.Min(batchSize, 4096);

        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, float>(
            Kernels.NBody);

        _posIn = accelerator.Allocate1D<float>(_numBodies * 2);
        _velIn = accelerator.Allocate1D<float>(_numBodies * 2);
        _posOut = accelerator.Allocate1D<float>(_numBodies * 2);
        _velOut = accelerator.Allocate1D<float>(_numBodies * 2);

        // Random initial positions in [-1, 1], velocities near zero
        var rnd = new Random(42);
        var pos = new float[_numBodies * 2];
        var vel = new float[_numBodies * 2];
        for (int i = 0; i < _numBodies * 2; i++)
        {
            pos[i] = (float)(rnd.NextDouble() * 2.0 - 1.0);
            vel[i] = (float)(rnd.NextDouble() * 0.01 - 0.005);
        }
        _posIn.CopyFromCPU(pos);
        _velIn.CopyFromCPU(vel);
    }

    /// <inheritdoc />
    public void Execute()
    {
        _kernel!(_numBodies, _posIn!.View, _velIn!.View, _posOut!.View, _velOut!.View, _numBodies, 0.001f);
        _accelerator!.Synchronize();

        // Swap buffers for next iteration
        (_posIn, _posOut) = (_posOut, _posIn);
        (_velIn, _velOut) = (_velOut, _velIn);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _posIn?.Dispose();
        _velIn?.Dispose();
        _posOut?.Dispose();
        _velOut?.Dispose();
    }
}
