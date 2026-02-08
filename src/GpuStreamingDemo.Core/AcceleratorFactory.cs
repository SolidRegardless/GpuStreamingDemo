using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

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
/// Describes an available compute device on the system.
/// </summary>
/// <param name="Index">Zero-based device index for CLI targeting.</param>
/// <param name="Name">Human-readable device name.</param>
/// <param name="Type">Accelerator type (CPU, CUDA, OpenCL).</param>
/// <param name="MemoryBytes">Total device memory in bytes (0 if unavailable).</param>
/// <param name="ComputeCapability">CUDA compute capability string, or null for non-CUDA devices.</param>
public sealed record DeviceInfo(
    int Index,
    string Name,
    AcceleratorType Type,
    long MemoryBytes,
    string? ComputeCapability
);

/// <summary>
/// Factory for creating and configuring ILGPU compute accelerators.
/// Supports device enumeration and targeting specific devices by index.
/// </summary>
public static class AcceleratorFactory
{
    /// <summary>
    /// Enumerates all available compute devices on the system.
    /// </summary>
    /// <returns>A list of <see cref="DeviceInfo"/> describing each available device.</returns>
    public static List<DeviceInfo> EnumerateDevices()
    {
        using var context = Context.CreateDefault();
        var devices = new List<DeviceInfo>();
        int index = 0;

        foreach (var device in context.Devices)
        {
            string? computeCap = null;
            long memoryBytes = 0;

            if (device is CudaDevice cudaDev)
            {
                computeCap = $"{cudaDev.Architecture}";
                memoryBytes = cudaDev.MemorySize;
            }
            else if (device.AcceleratorType != AcceleratorType.CPU)
            {
                memoryBytes = device.MemorySize;
            }

            devices.Add(new DeviceInfo(
                Index: index++,
                Name: device.Name,
                Type: device.AcceleratorType,
                MemoryBytes: memoryBytes,
                ComputeCapability: computeCap
            ));
        }

        return devices;
    }

    /// <summary>
    /// Creates an ILGPU context and accelerator of the specified kind.
    /// </summary>
    /// <param name="kind">The type of accelerator to create.</param>
    /// <returns>A tuple containing the created <see cref="Context"/> and <see cref="Accelerator"/>.</returns>
    public static (Context Context, Accelerator Accelerator) Create(AcceleratorKind kind)
    {
        var context = Context.CreateDefault();

        Accelerator accelerator = kind switch
        {
            AcceleratorKind.Auto   => context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context),
            AcceleratorKind.Cpu    => context.CreateCPUAccelerator(0),
            AcceleratorKind.Cuda   => CreateByType(context, AcceleratorType.Cuda),
            AcceleratorKind.OpenCL => CreateByType(context, AcceleratorType.OpenCL),

            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown accelerator kind.")
        };

        return (context, accelerator);
    }

    /// <summary>
    /// Creates an ILGPU context and accelerator for a specific device by index.
    /// The index corresponds to the order returned by <see cref="EnumerateDevices"/>.
    /// </summary>
    /// <param name="deviceIndex">Zero-based device index.</param>
    /// <returns>A tuple containing the created <see cref="Context"/>, <see cref="Accelerator"/>, and device name.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the device index is invalid.</exception>
    public static (Context Context, Accelerator Accelerator, string DeviceName) CreateByIndex(int deviceIndex)
    {
        var context = Context.CreateDefault();
        var allDevices = context.Devices.ToList();

        if (deviceIndex < 0 || deviceIndex >= allDevices.Count)
        {
            context.Dispose();
            throw new ArgumentOutOfRangeException(nameof(deviceIndex),
                $"Device index {deviceIndex} is out of range. Available devices: 0-{allDevices.Count - 1}");
        }

        var device = allDevices[deviceIndex];
        var accelerator = device.AcceleratorType == AcceleratorType.CPU
            ? context.CreateCPUAccelerator(0)
            : device.CreateAccelerator(context);

        return (context, accelerator, device.Name);
    }

    /// <summary>
    /// Creates contexts and accelerators for all available devices.
    /// Each entry must be disposed separately by the caller.
    /// </summary>
    /// <returns>A list of tuples containing context, accelerator, and device name.</returns>
    public static List<(Context Context, Accelerator Accelerator, string DeviceName)> CreateAll()
    {
        // We need a temporary context to discover devices, then create individual contexts
        // because ILGPU contexts can't be shared across some device types.
        var deviceInfos = EnumerateDevices();
        var results = new List<(Context, Accelerator, string)>();

        foreach (var info in deviceInfos)
        {
            try
            {
                var (ctx, accel, name) = CreateByIndex(info.Index);
                results.Add((ctx, accel, name));
            }
            catch
            {
                // Skip devices that fail to initialise
            }
        }

        return results;
    }

    private static Accelerator CreateByType(Context context, AcceleratorType type)
    {
        var device = context.Devices.FirstOrDefault(d => d.AcceleratorType == type)
            ?? throw new InvalidOperationException($"No {type} device found on this machine.");

        return device.CreateAccelerator(context);
    }
}
