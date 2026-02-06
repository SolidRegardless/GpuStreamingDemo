using GpuStreamingDemo.Core;

/// <summary>
/// Parses a command-line accelerator argument and returns the corresponding <see cref="AcceleratorKind"/>.
/// </summary>
/// <param name="s">The accelerator string (cpu, cuda, opencl, or all). Defaults to "all" if null.</param>
/// <returns>The parsed <see cref="AcceleratorKind"/>.</returns>
/// <exception cref="ArgumentException">Thrown when an invalid accelerator type is specified.</exception>
static AcceleratorKind ParseAccel(string? s) => (s ?? "all").ToLowerInvariant() switch
{
    "cpu" => AcceleratorKind.Cpu,
    "cuda" => AcceleratorKind.Cuda,
    "opencl" => AcceleratorKind.OpenCL,
    "all" => AcceleratorKind.Auto,
    _ => throw new ArgumentException("Use: cpu | cuda | opencl | all")
};

/// <summary>
/// Parses a command-line integer argument, returning a fallback value if parsing fails or the value is not positive.
/// </summary>
/// <param name="s">The string to parse as an integer.</param>
/// <param name="fallback">The value to return if parsing fails or the parsed value is not positive.</param>
/// <returns>The parsed integer if valid and positive; otherwise, the fallback value.</returns>
static int ParseInt(string? s, int fallback)
    => int.TryParse(s, out var v) && v > 0 ? v : fallback;

var argsDict = new Dictionary<string, string?>();
for (int i = 0; i < args.Length; i++)
{
    if (!args[i].StartsWith("--")) continue;
    var key = args[i][2..];
    var val = (i + 1 < args.Length && !args[i + 1].StartsWith("--")) ? args[i + 1] : null;
    argsDict[key] = val;
}

var accelArg = ParseAccel(argsDict.GetValueOrDefault("accel"));
var batchSize = ParseInt(argsDict.GetValueOrDefault("batch"), 1 << 20);
var iterations = ParseInt(argsDict.GetValueOrDefault("iters"), 50);
var warmup = ParseInt(argsDict.GetValueOrDefault("warmup"), 5);

var accelerators = accelArg == AcceleratorKind.Auto
    ? new[] { AcceleratorKind.Cpu, AcceleratorKind.Cuda, AcceleratorKind.OpenCL }
    : new[] { accelArg };

Console.WriteLine();
Console.WriteLine("=== GPU STREAMING BENCHMARK ===");
Console.WriteLine($"Batch size : {batchSize:N0} floats");
Console.WriteLine($"Iterations: {iterations}");
Console.WriteLine($"Warmup    : {warmup}");
Console.WriteLine();

var results = new List<BenchmarkResult>();

foreach (var accel in accelerators)
{
    try
    {
        Console.WriteLine($"Running benchmark: {accel}");
        var result = GpuBenchmarkRunner.Run(
            accel,
            batchSize,
            iterations,
            warmup
        );
        results.Add(result);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  {accel} unavailable ({ex.Message})");
    }
}

Console.WriteLine();
Console.WriteLine("=== RESULTS ===");
Console.WriteLine();

foreach (var r in results)
{
    Console.WriteLine($"{r.Accelerator,-8} | " +
                      $"Avg: {r.AvgLatencyMs,7:F2} ms | " +
                      $"P95: {r.P95LatencyMs,7:F2} ms | " +
                      $"Throughput: {r.ThroughputElementsPerSec / 1_000_000,8:F2} M elems/s");
}

Console.WriteLine();
Console.WriteLine("Done.");
