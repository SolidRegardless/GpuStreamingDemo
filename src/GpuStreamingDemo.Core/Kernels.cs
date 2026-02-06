using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Contains GPU compute kernels for demonstration and benchmarking.
/// </summary>
public static class Kernels
{
    /// <summary>
    /// Simple affine transform kernel: y[i] = a*x[i] + b
    /// This is a placeholder kernel demonstrating kernel structure.
    /// Replace with your domain-specific math (filters, scoring, transforms, etc.).
    /// </summary>
    /// <param name="index">The 1D index of the current thread.</param>
    /// <param name="x">Input data array view.</param>
    /// <param name="y">Output data array view.</param>
    /// <param name="a">Scale factor.</param>
    /// <param name="b">Offset factor.</param>
    public static void AffineTransform(
        Index1D index,
        ArrayView<float> x,
        ArrayView<float> y,
        float a,
        float b)
    {
        if (index >= x.Length) return;
        y[index] = a * x[index] + b;
    }
}
