using System.Diagnostics;
using System.Threading.Channels;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Represents a batch of data to be processed by the GPU pipeline.
/// </summary>
/// <param name="Input">Input data array for this batch.</param>
/// <param name="Output">Output data array for results.</param>
/// <param name="Length">The actual number of elements in this batch.</param>
/// <param name="Sequence">A sequence number for batch ordering and tracking.</param>
/// <param name="CreatedTicks">The timestamp (in ticks) when the batch was created.</param>
public sealed record Batch(float[] Input, float[] Output, int Length, long Sequence, long CreatedTicks);

/// <summary>
/// Configuration options for the streaming GPU pipeline.
/// </summary>
public sealed class StreamingPipelineOptions
{
    /// <summary>
    /// Gets or sets the batch size (number of float elements per batch).
    /// Default is 1,048,576 floats (~4 MB).
    /// </summary>
    public int BatchSize { get; init; } = 1 << 20; // 1,048,576 floats (~4 MB)

    /// <summary>
    /// Gets or sets the channel capacity for backpressure control.
    /// Default is 4.
    /// </summary>
    public int ChannelCapacity { get; init; } = 4; // backpressure

    /// <summary>
    /// Gets or sets the scale factor for the affine transform kernel.
    /// Default is 1.2345.
    /// </summary>
    public float A { get; init; } = 1.2345f;

    /// <summary>
    /// Gets or sets the offset factor for the affine transform kernel.
    /// Default is 0.9876.
    /// </summary>
    public float B { get; init; } = 0.9876f;
}

/// <summary>
/// Asynchronous streaming pipeline: producer -> GPU stage -> consumer.
/// Uses channels for clean backpressure and cancellation handling.
/// The pipeline processes batches of data concurrently across three tasks:
/// - Producer: generates or reads input data
/// - GPU Stage: processes data on the GPU
/// - Consumer: aggregates or publishes results
/// </summary>
public sealed class StreamingGpuPipeline : IAsyncDisposable
{
    private readonly StreamingPipelineOptions _opt;
    private readonly Channel<Batch> _in;
    private readonly Channel<Batch> _out;
    private readonly CancellationTokenSource _cts = new();

    private readonly GpuComputeEngine _engine;
    private Task? _producer;
    private Task? _gpuStage;
    private Task? _consumer;

    /// <summary>
    /// Occurs when the pipeline logs a message.
    /// </summary>
    public event Action<string>? OnLog;

    /// <summary>
    /// Occurs when a batch completes processing, providing the sequence number and elapsed time.
    /// </summary>
    public event Action<long, TimeSpan>? OnBatchCompleted;

    /// <summary>
    /// Initializes a new instance of the <see cref="StreamingGpuPipeline"/> class.
    /// </summary>
    /// <param name="opt">Pipeline configuration options.</param>
    /// <param name="accelKind">The type of accelerator to use.</param>
    public StreamingGpuPipeline(StreamingPipelineOptions opt, AcceleratorKind accelKind)
    {
        _opt = opt;
        _engine = new GpuComputeEngine(accelKind);

        _in = Channel.CreateBounded<Batch>(new BoundedChannelOptions(opt.ChannelCapacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        });

        _out = Channel.CreateBounded<Batch>(new BoundedChannelOptions(opt.ChannelCapacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        });
    }

    /// <summary>
    /// Gets the human-readable name of the accelerator.
    /// </summary>
    public string AcceleratorName => _engine.AcceleratorName;

    /// <summary>
    /// Gets the type of the accelerator as a string.
    /// </summary>
    public string AcceleratorType => _engine.AcceleratorType.ToString();

    /// <summary>
    /// Starts the pipeline, launching producer, GPU processing, and consumer tasks.
    /// </summary>
    public void Start()
    {
        var token = _cts.Token;
        _producer = Task.Run(() => ProducerLoop(token), token);
        _gpuStage = Task.Run(() => GpuLoop(token), token);
        _consumer = Task.Run(() => ConsumerLoop(token), token);

        Log($"Pipeline started on {AcceleratorName} ({AcceleratorType}).");
    }

    /// <summary>
    /// Asynchronously stops the pipeline and waits for all tasks to complete.
    /// </summary>
    /// <returns>A task representing the asynchronous stop operation.</returns>
    public async Task StopAsync()
    {
        _cts.Cancel();

        try
        {
            if (_producer != null) await _producer.ConfigureAwait(false);
        }
        catch (OperationCanceledException) { }

        _in.Writer.TryComplete();

        try
        {
            if (_gpuStage != null) await _gpuStage.ConfigureAwait(false);
        }
        catch (OperationCanceledException) { }

        _out.Writer.TryComplete();

        try
        {
            if (_consumer != null) await _consumer.ConfigureAwait(false);
        }
        catch (OperationCanceledException) { }

        Log("Pipeline stopped.");
    }

    /// <summary>
    /// Produces batches of data and sends them to the input channel.
    /// This is meant to be run as a long-lived task.
    /// </summary>
    /// <param name="token">Cancellation token for stopping the loop.</param>
    private async Task ProducerLoop(CancellationToken token)
    {
        var seq = 0L;
        var rnd = new Random(1234);

        while (!token.IsCancellationRequested)
        {
            var input = new float[_opt.BatchSize];
            var output = new float[_opt.BatchSize];

            // Simulated streaming data. Replace with your real feed.
            for (int i = 0; i < input.Length; i++)
                input[i] = (float)rnd.NextDouble();

            var batch = new Batch(
                Input: input,
                Output: output,
                Length: _opt.BatchSize,
                Sequence: seq++,
                CreatedTicks: Stopwatch.GetTimestamp());

            await _in.Writer.WriteAsync(batch, token).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Processes batches on the GPU and sends them to the output channel.
    /// This is meant to be run as a long-lived task.
    /// </summary>
    /// <param name="token">Cancellation token for stopping the loop.</param>
    private async Task GpuLoop(CancellationToken token)
    {
        await foreach (var batch in _in.Reader.ReadAllAsync(token).ConfigureAwait(false))
        {
            _engine.ProcessBatch(batch.Input, batch.Output, batch.Length, _opt.A, _opt.B);
            await _out.Writer.WriteAsync(batch, token).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Consumes processed batches from the output channel.
    /// Performs aggregation or publishing as needed.
    /// This is meant to be run as a long-lived task.
    /// </summary>
    /// <param name="token">Cancellation token for stopping the loop.</param>
    private async Task ConsumerLoop(CancellationToken token)
    {
        await foreach (var batch in _out.Reader.ReadAllAsync(token).ConfigureAwait(false))
        {
            // Do something useful: aggregate, publish, write to socket, etc.
            // We'll do a tiny reduction to prevent the JIT "optimising away" everything.
            double checksum = 0;
            for (int i = 0; i < 64 && i < batch.Length; i++)
                checksum += batch.Output[i];

            var elapsed = Stopwatch.GetElapsedTime(batch.CreatedTicks);
            OnBatchCompleted?.Invoke(batch.Sequence, elapsed);

            if ((batch.Sequence % 10) == 0)
                Log($"seq={batch.Sequence} latency={elapsed.TotalMilliseconds:F2}ms checksum={checksum:F4}");
        }
    }

    /// <summary>
    /// Logs a message with a timestamp by invoking the OnLog event.
    /// </summary>
    /// <param name="message">The message to log.</param>
    private void Log(string message) => OnLog?.Invoke($"[{DateTimeOffset.Now:HH:mm:ss}] {message}");

    /// <summary>
    /// Asynchronously disposes of the pipeline and releases all resources.
    /// </summary>
    /// <returns>A value task representing the asynchronous disposal operation.</returns>
    public async ValueTask DisposeAsync()
    {
        await StopAsync().ConfigureAwait(false);
        _engine.Dispose();
        _cts.Dispose();
    }
}
