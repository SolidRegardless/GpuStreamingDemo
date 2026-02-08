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
var doubleBuffer = argsDict.ContainsKey("double-buffer");
var quick = argsDict.ContainsKey("quick");
var listDevices = argsDict.ContainsKey("list-devices");
var deviceIndex = argsDict.ContainsKey("device") ? int.Parse(argsDict["device"]!) : (int?)null;

// In quick mode, override to minimal settings for fast dev iteration
if (quick)
{
    batchSize = ParseInt(argsDict.GetValueOrDefault("batch"), 1 << 16); // 65,536 elements
    iterations = ParseInt(argsDict.GetValueOrDefault("iters"), 5);
    warmup = ParseInt(argsDict.GetValueOrDefault("warmup"), 1);
}

// --- Device enumeration ---

if (listDevices)
{
    Console.WriteLine();
    Console.WriteLine("=== AVAILABLE DEVICES ===");
    Console.WriteLine();

    var devices = AcceleratorFactory.EnumerateDevices();

    var devHdr = $"{"Index",-7} {"Type",-8} {"Name",-45} {"Memory",-14} {"Compute Cap",12}";
    Console.WriteLine(devHdr);
    Console.WriteLine(new string('-', devHdr.Length));

    foreach (var dev in devices)
    {
        var memStr = dev.MemoryBytes > 0
            ? $"{dev.MemoryBytes / (1024.0 * 1024.0):F0} MB"
            : "N/A";
        var ccStr = dev.ComputeCapability ?? "—";
        Console.WriteLine($"{dev.Index,-7} {dev.Type,-8} {dev.Name,-45} {memStr,-14} {ccStr,12}");
    }

    Console.WriteLine();
    Console.WriteLine($"Total: {devices.Count} device(s)");
    Console.WriteLine();
    Console.WriteLine("Use --device <index> to target a specific device.");
    Console.WriteLine();
    return;
}

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

IBenchmarkTask[] CreateQuickTask() =>
[
    new AffineTransformTask()
];

IBenchmarkTask[] CreateAllTasks() => quick ? CreateQuickTask() : advanced ? CreateAdvancedTasks() : CreateStandardTasks();

// --- Header ---

Console.WriteLine();
var modeLabel = doubleBuffer ? "=== GPU STREAMING BENCHMARK (DOUBLE-BUFFERED) ===" :
                quick ? "=== GPU STREAMING BENCHMARK (QUICK) ===" :
                advanced ? "=== GPU STREAMING BENCHMARK (ADVANCED) ===" :
                "=== GPU STREAMING BENCHMARK ===";
Console.WriteLine(modeLabel);
Console.WriteLine($"Batch size : {batchSize:N0} elements");
Console.WriteLine($"Iterations : {iterations}");
Console.WriteLine($"Warmup     : {warmup}");
if (taskFilter != null)
    Console.WriteLine($"Task filter: {taskFilter}");
if (doubleBuffer)
    Console.WriteLine("Mode       : Double-buffered (async overlap)");
if (deviceIndex.HasValue)
    Console.WriteLine($"Device     : #{deviceIndex.Value}");
Console.WriteLine();

// --- Determine which devices to run on ---

var results = new List<BenchmarkResult>();
var doubleBufferedResults = new List<DoubleBufferedResult>();

if (deviceIndex.HasValue)
{
    // Target a specific device by index
    RunOnDevice(deviceIndex.Value);
}
else if (accelArg == AcceleratorKind.Auto)
{
    // --accel all: enumerate and run on every device
    var devices = AcceleratorFactory.EnumerateDevices();
    foreach (var dev in devices)
    {
        RunOnDevice(dev.Index);
    }
}
else
{
    // Target a specific accelerator type (legacy behaviour)
    RunOnAccelKind(accelArg);
}

void RunOnDevice(int devIdx)
{
    foreach (var task in CreateAllTasks())
    {
        using (task)
        {
            if (taskFilter != null && !task.Name.Contains(taskFilter, StringComparison.OrdinalIgnoreCase))
                continue;

            try
            {
                Console.WriteLine($"Running {task.Name} on device #{devIdx}...");
                var result = GpuBenchmarkRunner.RunOnDevice(task, devIdx, batchSize, iterations, warmup);
                results.Add(result);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  {task.Name}/device#{devIdx} failed: {ex.Message}");
            }
        }
    }
}

void RunOnAccelKind(AcceleratorKind accel)
{
    foreach (var task in CreateAllTasks())
    {
        using (task)
        {
            if (taskFilter != null && !task.Name.Contains(taskFilter, StringComparison.OrdinalIgnoreCase))
                continue;

            try
            {
                if (doubleBuffer && task is IDoubleBufferedTask dbTask)
                {
                    Console.WriteLine($"Running {task.Name} on {accel} (double-buffered)...");
                    var result = DoubleBufferedBenchmarkRunner.Run(dbTask, accel, batchSize, iterations, warmup);
                    doubleBufferedResults.Add(result);
                }
                else
                {
                    Console.WriteLine($"Running {task.Name} on {accel}...");
                    var result = GpuBenchmarkRunner.Run(task, accel, batchSize, iterations, warmup);
                    results.Add(result);
                }
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

if (results.Count > 0)
{
    var hdr = $"{"Task",-20} {"Accel",-8} {"Device",-30} {"Avg (ms)",10} {"P95 (ms)",10} {"Throughput",16}";
    Console.WriteLine(hdr);
    Console.WriteLine(new string('-', hdr.Length));

    foreach (var r in results)
    {
        var shortName = TruncateDevice(r.DeviceName, 28);
        Console.WriteLine(
            $"{r.TaskName,-20} {r.Accelerator,-8} {shortName,-30} {r.AvgLatencyMs,10:F2} {r.P95LatencyMs,10:F2} {r.ThroughputElementsPerSec / 1_000_000,13:F2} M/s");
    }
}

if (doubleBufferedResults.Count > 0)
{
    Console.WriteLine();
    var dbHdr = $"{"Task",-25} {"Accel",-8} {"Device",-30} {"Avg (ms)",10} {"P95 (ms)",10} {"Throughput",16} {"1xBuf (ms)",12} {"Speedup",9} {"Overlap",9}";
    Console.WriteLine(dbHdr);
    Console.WriteLine(new string('-', dbHdr.Length));

    foreach (var r in doubleBufferedResults)
    {
        var shortName = TruncateDevice(r.DeviceName, 28);
        Console.WriteLine(
            $"{r.TaskName,-25} {r.Accelerator,-8} {shortName,-30} {r.AvgLatencyMs,10:F2} {r.P95LatencyMs,10:F2} {r.ThroughputElementsPerSec / 1_000_000,13:F2} M/s {r.SingleBufferAvgMs,10:F2}ms {r.SpeedupFactor,8:F2}x {r.OverlapPercentage,7:F1}%");
    }
}

Console.WriteLine();
Console.WriteLine("Done.");

static string TruncateDevice(string name, int max)
    => name.Length <= max ? name : name[..(max - 2)] + "..";
