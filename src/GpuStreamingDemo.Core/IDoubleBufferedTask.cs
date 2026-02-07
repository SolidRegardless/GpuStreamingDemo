using ILGPU.Runtime;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Extends <see cref="IBenchmarkTask"/> to support double-buffered execution.
/// Tasks implementing this interface decompose their work into separate
/// host-to-device copy, kernel execution, and device-to-host copy phases,
/// each bound to a specific ILGPU stream for async overlap.
/// </summary>
public interface IDoubleBufferedTask : IBenchmarkTask
{
    /// <summary>
    /// Sets up two sets of device buffers (A and B) for double-buffered execution.
    /// </summary>
    /// <param name="accelerator">The ILGPU accelerator to use.</param>
    /// <param name="batchSize">The number of elements per batch.</param>
    /// <param name="streamA">The first accelerator stream.</param>
    /// <param name="streamB">The second accelerator stream.</param>
    void SetupDoubleBuffered(Accelerator accelerator, int batchSize, AcceleratorStream streamA, AcceleratorStream streamB);

    /// <summary>
    /// Asynchronously copies input data from host to device for the specified buffer slot.
    /// </summary>
    /// <param name="slot">0 for buffer A, 1 for buffer B.</param>
    /// <param name="stream">The stream to use for the async copy.</param>
    void CopyToDeviceAsync(int slot, AcceleratorStream stream);

    /// <summary>
    /// Launches the kernel on the specified buffer slot using the given stream.
    /// Does not synchronise — the caller manages synchronisation.
    /// </summary>
    /// <param name="slot">0 for buffer A, 1 for buffer B.</param>
    /// <param name="stream">The stream to use for kernel launch.</param>
    void LaunchKernel(int slot, AcceleratorStream stream);

    /// <summary>
    /// Asynchronously copies results from device to host for the specified buffer slot.
    /// </summary>
    /// <param name="slot">0 for buffer A, 1 for buffer B.</param>
    /// <param name="stream">The stream to use for the async copy.</param>
    void CopyFromDeviceAsync(int slot, AcceleratorStream stream);

    /// <summary>
    /// Gets the host-side output data for the specified buffer slot (for verification).
    /// </summary>
    /// <param name="slot">0 for buffer A, 1 for buffer B.</param>
    /// <returns>The output array for the given slot.</returns>
    float[] GetHostOutput(int slot);
}
