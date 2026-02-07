using GpuStreamingDemo.Core;
using GpuStreamingDemo.Core.Tasks;

// --- Argument parsing ---

var argsDict = new Dictionary<string, string?>();
for (int i = 0; i < args.Length; i++)
{
    if (!args[i].StartsWith("--")) continue;
    var key = args[i][2..];
    var val = (i + 1 < args.Length && !args[i + 1].StartsWith("--")) ? args[i + 1] : null;
    argsDict[key] = val;
}

static AcceleratorKind ParseAccel(string? s) => (s ?? "all").ToLowerInvariant() switch
{
    "cpu" => AcceleratorKind.Cpu,
    "cuda" => AcceleratorKind.Cuda,
    "opencl" => AcceleratorKind.OpenCL,
    "all" => AcceleratorKind.Auto,
    _ => throw new ArgumentException("Use: cpu | cuda | opencl | all")
};

static int ParseInt(string? s, int fallback)
    => int.TryParse(s, out var v) && v > 0 ? v : fallback;

var accelArg = ParseAccel(argsDict.GetValueOrDefault("accel"));
var batchSize = ParseInt(argsDict.GetValueOrDefault("batch"), 1 << 20);
var iterations = ParseInt(argsDict.GetValueOrDefault("iters"), 50);
var warmup = ParseInt(argsDict.GetValueOrDefault("warmup"), 5);
var taskFilter = argsDict.GetValueOrDefault("task")?.ToLowerInvariant();
var advanced = argsDict.ContainsKey("advanced");

// --- Available tasks ---

IBenchmarkTask[] CreateStandardTasks() =>
[
    new AffineTransformTask(),
    new VectorAddTask(),
    new SaxpyTask(),
    new Sha256Task(),
    new MandelbrotTask()
];

IBenchmarkTask[] CreateAdvancedTasks() =>
[
    new MonteCarloTask(),
    new FirFilterTask(),
    new GaussianBlurTask(),
    new BatchNormReluTask(),
    new NBodyTask(),
    new AesCryptoTask(),
    new JuliaSetTask()
];

IBenchmarkTask[] CreateAllTasks() => advanced ? CreateAdvancedTasks() : CreateStandardTasks();

var accelerators = accelArg == AcceleratorKind.Auto
    ? new[] { AcceleratorKind.Cpu, AcceleratorKind.Cuda, AcceleratorKind.OpenCL }
    : new[] { accelArg };

Console.WriteLine();
Console.WriteLine(advanced ? "=== GPU STREAMING BENCHMARK (ADVANCED) ===" : "=== GPU STREAMING BENCHMARK ===");
Console.WriteLine($"Batch size : {batchSize:N0} elements");
Console.WriteLine($"Iterations : {iterations}");
Console.WriteLine($"Warmup     : {warmup}");
if (taskFilter != null)
    Console.WriteLine($"Task filter: {taskFilter}");
Console.WriteLine();

var results = new List<BenchmarkResult>();

foreach (var accel in accelerators)
{
    foreach (var task in CreateAllTasks())
    {
        using (task)
        {
            if (taskFilter != null && !task.Name.Contains(taskFilter, StringComparison.OrdinalIgnoreCase))
                continue;

            try
            {
                Console.WriteLine($"Running {task.Name} on {accel}...");
                var result = GpuBenchmarkRunner.Run(task, accel, batchSize, iterations, warmup);
                results.Add(result);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  {task.Name}/{accel} failed: {ex.Message}");
            }
        }
    }
}

// --- Results table ---

Console.WriteLine();
Console.WriteLine("=== RESULTS ===");
Console.WriteLine();

var hdr = $"{"Task",-20} {"Accel",-8} {"Avg (ms)",10} {"P95 (ms)",10} {"Throughput",16}";
Console.WriteLine(hdr);
Console.WriteLine(new string('-', hdr.Length));

foreach (var r in results)
{
    Console.WriteLine(
        $"{r.TaskName,-20} {r.Accelerator,-8} {r.AvgLatencyMs,10:F2} {r.P95LatencyMs,10:F2} {r.ThroughputElementsPerSec / 1_000_000,13:F2} M/s");
}

Console.WriteLine();
Console.WriteLine("Done.");
