using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Encapsulates GPU compute engine functionality using ILGPU.
/// Manages accelerator lifecycle, memory allocation, and kernel execution.
/// </summary>
public sealed class GpuComputeEngine : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;

    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, float, float> _kernel;

    private MemoryBuffer1D<float, Stride1D.Dense>? _input;
    private MemoryBuffer1D<float, Stride1D.Dense>? _output;

    /// <summary>
    /// Gets the human-readable name of the accelerator.
    /// </summary>
    public string AcceleratorName => _accelerator.Name;

    /// <summary>
    /// Gets the type of the accelerator (CPU, CUDA, OpenCL, etc.).
    /// </summary>
    public AcceleratorType AcceleratorType => _accelerator.AcceleratorType;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuComputeEngine"/> class.
    /// </summary>
    /// <param name="kind">The type of accelerator to use.</param>
    public GpuComputeEngine(AcceleratorKind kind)
    {
        (_context, _accelerator) = AcceleratorFactory.Create(kind);

        _kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            float,
            float>(Kernels.AffineTransform);
    }

    /// <summary>
    /// Ensures that GPU buffers have sufficient capacity for the specified length.
    /// Reallocates buffers if necessary.
    /// </summary>
    /// <param name="length">The required capacity in elements.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when length is less than or equal to zero.</exception>
    private void EnsureCapacity(int length)
    {
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));

        if (_input is not null && _input.Length >= length)
            return;

        _input?.Dispose();
        _output?.Dispose();

        _input = _accelerator.Allocate1D<float>(length);
        _output = _accelerator.Allocate1D<float>(length);
    }

    /// <summary>
    /// Processes a batch of data on the GPU using an affine transform kernel.
    /// NOTE: Uses arrays to match ILGPU's buffer CopyFromCPU/CopyToCPU helpers cleanly.
    /// </summary>
    /// <param name="input">Input data array.</param>
    /// <param name="output">Output data array where results will be written.</param>
    /// <param name="length">The number of elements to process.</param>
    /// <param name="a">The scale factor for the affine transform.</param>
    /// <param name="b">The offset factor for the affine transform.</param>
    /// <exception cref="ArgumentNullException">Thrown when input or output is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when length is invalid or exceeds buffer bounds.</exception>
    public void ProcessBatch(float[] input, float[] output, int length, float a, float b)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (length <= 0 || length > input.Length || length > output.Length)
            throw new ArgumentOutOfRangeException(nameof(length));

        EnsureCapacity(length);

        _input!.CopyFromCPU(input); // copies full array; keep input == length-sized batch in this demo
        _kernel(length, _input!.View, _output!.View, a, b);
        _accelerator.Synchronize();
        _output.CopyToCPU(output);
    }

    /// <summary>
    /// Releases all GPU resources associated with this engine.
    /// </summary>
    public void Dispose()
    {
        _input?.Dispose();
        _output?.Dispose();
        _accelerator.Dispose();
        _context.Dispose();
    }
}
