using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Specifies the type of GPU accelerator to use.
/// </summary>
public enum AcceleratorKind
{
    /// <summary>Automatically select the preferred GPU device.</summary>
    Auto,
    /// <summary>Use the CPU accelerator.</summary>
    Cpu,
    /// <summary>Use a CUDA-compatible GPU (NVIDIA).</summary>
    Cuda,
    /// <summary>Use an OpenCL-compatible device.</summary>
    OpenCL
}

/// <summary>
/// Factory for creating and configuring ILGPU compute accelerators.
/// </summary>
public static class AcceleratorFactory
{
    /// <summary>
    /// Creates an ILGPU context and accelerator of the specified kind.
    /// </summary>
    /// <param name="kind">The type of accelerator to create.</param>
    /// <returns>A tuple containing the created <see cref="Context"/> and <see cref="Accelerator"/>.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the requested accelerator type is not available on the system.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when an unknown accelerator kind is specified.</exception>
    public static (Context Context, Accelerator Accelerator) Create(AcceleratorKind kind)
    {
        var context = Context.CreateDefault();

        Accelerator accelerator = kind switch
        {
            AcceleratorKind.Auto   => context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context),

            // CPU accelerator creation is provided by ILGPU.Runtime.CPU
            AcceleratorKind.Cpu    => context.CreateCPUAccelerator(0),

            AcceleratorKind.Cuda   => CreateByType(context, AcceleratorType.Cuda),
            AcceleratorKind.OpenCL => CreateByType(context, AcceleratorType.OpenCL),

            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown accelerator kind.")
        };

        return (context, accelerator);
    }

    /// <summary>
    /// Creates an accelerator for a specific <see cref="AcceleratorType"/>.
    /// </summary>
    /// <param name="context">The ILGPU context to use.</param>
    /// <param name="type">The type of accelerator to create.</param>
    /// <returns>An <see cref="Accelerator"/> instance of the specified type.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no device of the specified type is found.</exception>
    private static Accelerator CreateByType(Context context, AcceleratorType type)
    {
        var device = context.Devices.FirstOrDefault(d => d.AcceleratorType == type)
            ?? throw new InvalidOperationException($"No {type} device found on this machine.");

        return device.CreateAccelerator(context);
    }
}
