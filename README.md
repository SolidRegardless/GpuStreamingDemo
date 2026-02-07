# 🔥 C# on the GPU: When Performance Stops Being About Code (and Starts Being About Physics)

> *Or: how I stopped micro-optimising loops, learned to love the GPU, and accidentally made my CPU cry.*

Full source code & benchmarks:
👉 **https://github.com/SolidRegardless/GpuStreamingDemo**

---

## 1. The Lie We All Tell Ourselves

At some point in every engineer's career, we tell ourselves a comforting lie:

> "If I just optimise this loop a bit more, everything will be fine."

So we:
- remove allocations
- unroll loops
- fight the branch predictor
- stare at flame graphs like they owe us money
- whisper sweet nothings to the JIT compiler

And sometimes… it works.

Until it doesn't.

Because there is a hard limit you eventually hit — and it isn't your skill. It's **physics**.

Your CPU is sitting there, sweating through a million elements one-by-one, while a GPU with thousands of cores is over in the corner filing its nails, bored out of its mind.

---

## 2. The Moment Performance Stops Being About Code

If your workload is:
- **large** (millions of elements)
- **predictable** (same operation repeated)
- **data-parallel** (each element independent)

Then the CPU is no longer your friend. It's more like that one colleague who insists on doing everything sequentially because "that's how we've always done it."

Not because CPUs are slow — they're incredible — but because they are optimised for:
- latency
- control flow
- branchy logic
- making you *feel* like you're in control

Not for chewing through gigabytes of identical maths like a wood chipper going through a forest.

That's not a programming problem. That's a *hardware mismatch*.

---

## 3. The Goal of This Experiment

I wanted to answer a very simple question:

> **Can modern C# orchestrate real GPU acceleration — with multiple workloads, cleanly, portably, and credibly?**

Not a toy example.
Not a CUDA-C microbenchmark where the author suspiciously omits memory transfer times.
Not a hand-waved blog post with numbers that smell like fiction.

But:
- real memory transfers (Host → Device → Host, no cheating)
- real kernels (five of them, from boring to brain-melting)
- real timing (warmup included, because we're not animals)
- real hardware

And numbers you can reproduce. Or cry about. Your choice.

---

## 4. The Rules (No Cheating)

I imposed some strict rules:

1. **Same batch size** everywhere (1,048,576 elements per task)
2. **Same memory path** (host → device → host)
3. **Same harness** — pluggable `IBenchmarkTask` interface
4. **Same codebase** — one project, all backends
5. **Only the execution backend may change**

If the numbers improved, it had to be because the **execution model changed** — not because I gamed the benchmark like a dodgy car salesman adjusting the odometer.

---

## 5. Architecture Overview

### High-Level Flow

```mermaid
flowchart LR
    subgraph Harness["🏗️ Benchmark Harness"]
        Runner[BenchmarkRunner]
    end

    subgraph Tasks["🔧 Pluggable Tasks"]
        T1[AffineTransform]
        T2[VectorAdd]
        T3[SAXPY]
        T4[SHA-256]
        T5[Mandelbrot]
    end

    subgraph Backends["⚡ Accelerators"]
        CPU[CPU]
        CUDA[CUDA]
        OCL[OpenCL]
    end

    Runner --> T1 & T2 & T3 & T4 & T5
    T1 & T2 & T3 & T4 & T5 --> CPU & CUDA & OCL
```

Key idea:

> **The harness orchestrates. The tasks define the work. The accelerator does the heavy lifting.**

Every task is self-contained. Every accelerator is interchangeable. The runner doesn't care what you're computing — it just times it and judges you silently.

### Pluggable Task Architecture

```mermaid
classDiagram
    class IBenchmarkTask {
        <<interface>>
        +Name: string
        +Description: string
        +Setup(accelerator, batchSize)
        +Execute()
        +Dispose()
    }

    class AffineTransformTask {
        y = a·x + b
    }

    class VectorAddTask {
        z = x + y
    }

    class SaxpyTask {
        y = a·x + y
    }

    class Sha256Task {
        SHA-256 compression
    }

    class MandelbrotTask {
        Escape iterations
    }

    class BenchmarkRunner {
        +Run(task, accel, batch, iters, warmup)
    }

    IBenchmarkTask <|.. AffineTransformTask
    IBenchmarkTask <|.. VectorAddTask
    IBenchmarkTask <|.. SaxpyTask
    IBenchmarkTask <|.. Sha256Task
    IBenchmarkTask <|.. MandelbrotTask
    BenchmarkRunner --> IBenchmarkTask : times
```

### Streaming Pipeline (Bonus Architecture)

For continuous workloads, there's also a full async streaming pipeline:

```mermaid
flowchart LR
    Producer["📡 Producer\n(generates batches)"]
    InCh["Channel\n(backpressure)"]
    GPU["⚡ GPU Stage\n(kernel execution)"]
    OutCh["Channel\n(backpressure)"]
    Consumer["📊 Consumer\n(aggregate/publish)"]

    Producer --> InCh --> GPU --> OutCh --> Consumer
```

---

## 6. The Five Horsemen of the GPU-ocalypse

### 🧮 AffineTransform — "The Classic"

```csharp
y[i] = a * x[i] + b
```

The Toyota Corolla of GPU kernels. Boring. Reliable. Gets the job done. Your CPU can do this too — just, you know... *slower*.

### ➕ VectorAdd — "The Hello World"

```csharp
z[i] = x[i] + y[i]
```

If GPU programming had a "Hello World", this would be it. Two arrays walk into a GPU. One array walks out. It's beautiful in its simplicity.

### ⚡ SAXPY — "The BLAS Legend"

```csharp
y[i] = a * x[i] + y[i]
```

Single-precision A·X Plus Y. The cornerstone of linear algebra. Every BLAS library benchmarks this. Every GPU vendor uses this to show off. We're using it because we're not above peer pressure.

### 🔐 SHA-256 — "The Brain Melter"

```csharp
// 64 rounds of SHA-256 compression per element
// Bit shifts, rotations, and existential dread
```

This is where things get spicy. Full SHA-256 single-block compression running *on the GPU*. 64 rounds of bit manipulation, 8 working variables, constant table lookups, and enough integer arithmetic to make your ALU question its life choices.

The CPU *really* doesn't enjoy this one.

### 🌀 Mandelbrot — "The Pretty One"

```csharp
// For each point: iterate z = z² + c until escape or max iterations
// Max 1000 iterations per element
```

The Mandelbrot set — where mathematics meets art and your GPU meets its purpose in life. Highly variable workload per element (some escape immediately, some spiral for 1000 iterations). This is the kind of divergent parallelism that separates GPUs from toys.

---

## 7. The `IBenchmarkTask` Interface (SOLID in Action)

```csharp
public interface IBenchmarkTask : IDisposable
{
    string Name { get; }
    string Description { get; }
    void Setup(Accelerator accelerator, int batchSize);
    void Execute();
}
```

Why this matters:

- **Single Responsibility** — each task owns its kernels, buffers, and logic
- **Open/Closed** — add new tasks without touching the runner
- **Liskov Substitution** — swap any task into the harness seamlessly
- **Interface Segregation** — clean, minimal contract
- **Dependency Inversion** — the runner depends on the abstraction, not concrete tasks

Want to add a new benchmark? One class. One file. Zero changes to the harness. Your future self will thank you.

```csharp
// Adding a new task is this easy:
IBenchmarkTask[] CreateAllTasks() =>
[
    new AffineTransformTask(),
    new VectorAddTask(),
    new SaxpyTask(),
    new Sha256Task(),
    new MandelbrotTask(),
    // new YourShinyNewTask(),  ← just add it here
];
```

---

## 8. Execution Model (What Actually Happens)

```mermaid
sequenceDiagram
    participant Host as 🖥️ CPU (Host)
    participant Device as 🎮 GPU (Device)

    Host->>Host: Allocate & fill input arrays
    Host->>Device: Copy batch (Host → Device)
    Host->>Device: Launch kernel
    Note over Device: 🔥 Thousands of threads<br/>execute in parallel
    Device-->>Host: Copy results (Device → Host)
    Host->>Host: Synchronise & measure
```

No magic. No zero-cost abstractions. No fairy dust.

Just silicon doing what it was designed to do — while your CPU watches from the sidelines, sipping coffee, wondering where it all went wrong.

---

## 9. Benchmark Methodology

### Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 1,048,576 elements (~4 MB) |
| Warmup iterations | 5 |
| Measured iterations | 20 |
| Tasks | 5 (all) |
| Backends | CPU, CUDA, OpenCL |

### Why Warmup Matters

- JIT compilation (first run penalty)
- Kernel compilation (ILGPU compiles on first use)
- Driver initialisation
- Memory allocation warm paths

If you skip warmup, you're benchmarking your compiler, not your code. Don't be that person.

### CLI Options

```bash
dotnet run -- --accel all --batch 1048576 --iters 50 --warmup 5

# Filter to a specific task
dotnet run -- --task sha256

# CPU only
dotnet run -- --accel cpu

# Go big
dotnet run -- --batch 4194304 --iters 100
```

---

## 10. 🏆 The Results (Real Numbers, Real Hardware)

### Full Results Table

```
Task                 Accel      Avg (ms)   P95 (ms)       Throughput
--------------------------------------------------------------------
AffineTransform      Cpu          177.40     237.18          5.91 M/s
VectorAdd            Cpu          126.38     184.06          8.30 M/s
Saxpy                Cpu          118.52     131.26          8.85 M/s
SHA256               Cpu          839.06    1151.37          1.25 M/s
Mandelbrot           Cpu          493.25     514.17          2.13 M/s
AffineTransform      Cuda           0.02       0.05      57065.36 M/s
VectorAdd            Cuda           0.02       0.07      54330.36 M/s
Saxpy                Cuda           0.37       0.39       2870.81 M/s
SHA256               Cuda           0.27       0.55       3901.68 M/s
Mandelbrot           Cuda           0.17       0.18       6048.02 M/s
```

### The Speedup (Brace Yourself)

| Task | CPU (ms) | CUDA (ms) | Speedup | Reaction |
|------|----------|-----------|---------|----------|
| AffineTransform | 177.40 | 0.02 | **~8,850×** | 😱 |
| VectorAdd | 126.38 | 0.02 | **~6,300×** | 🤯 |
| Saxpy | 118.52 | 0.37 | **~320×** | 😮 |
| SHA-256 | 839.06 | 0.27 | **~3,100×** | 💀 |
| Mandelbrot | 493.25 | 0.17 | **~2,900×** | 🔥 |

Let that sink in.

The CPU took **839 milliseconds** to do SHA-256 on a million elements.

The GPU did it in **0.27 milliseconds**.

That's not an optimisation. That's a *paradigm shift*.

### Visualised

```mermaid
xychart-beta
    title "Average Latency by Task (ms, log scale concept)"
    x-axis ["Affine", "VecAdd", "Saxpy", "SHA256", "Mandel"]
    y-axis "Avg Latency (ms)" 0 --> 900
    bar [177.40, 126.38, 118.52, 839.06, 493.25]
    bar [0.02, 0.02, 0.37, 0.27, 0.17]
```

*(The CUDA bars are so small they're basically invisible. That's the point.)*

---

## 11. Interpreting the Numbers (This Is the Important Bit)

### CPU
- 118–839 ms per batch depending on workload complexity
- High tail latency (P95 significantly above average)
- Throughput measured in **single-digit millions**
- SHA-256 and Mandelbrot are *punishing*

### CUDA
- 0.02–0.37 ms per batch
- Remarkably tight latency distribution
- Throughput measured in **thousands of millions**
- Even the "hard" tasks (SHA-256, Mandelbrot) complete in under a millisecond

This is not 10% faster. This is not 2× faster.

This is **two to four orders of magnitude**.

---

## 12. "But My CPU Has 32 Cores!"

Yes. And the GPU still wins.

Because:
- **Thousands** of execution lanes (not 32, not 64 — *thousands*)
- Memory coalescing (sequential access patterns → bandwidth efficiency)
- Predictable access patterns → hardware prefetching
- Hardware designed for throughput, not latency

This isn't about disrespecting CPUs. CPUs are phenomenal at what they do — complex branching, system orchestration, running your IDE, arguing with your build system.

But asking a CPU to crunch a million identical math operations is like asking a Formula 1 driver to deliver parcels. Technically possible. Architecturally absurd.

---

## 13. Task-by-Task Analysis

### AffineTransform & VectorAdd (~6,300–8,850× speedup)

These are memory-bandwidth-bound kernels. Minimal compute per element, maximum data movement. The GPU's massive memory bus absolutely dominates here. The CPU doesn't stand a chance — it's like bringing a garden hose to fight a fire hydrant.

### SAXPY (~320× speedup)

Classic BLAS operation. The "lower" speedup (still 320×!) is likely due to the in-place update pattern — reading and writing the same array adds a dependency the memory controller has to manage. Still embarrassing for the CPU.

### SHA-256 (~3,100× speedup)

This is the *revenge of the ALU*. SHA-256 is compute-heavy — 64 rounds of bit shifts, rotations, additions, and logical operations per element. The CPU has to chew through each one sequentially. The GPU throws thousands of threads at it and finishes before the CPU has processed the first batch.

### Mandelbrot (~2,900× speedup)

Mandelbrot is interesting because it has *divergent* workloads — some points escape in 2 iterations, some take 1000. GPUs handle this with warp/wavefront execution. Even with divergence, the sheer parallelism crushes sequential execution.

---

## 14. Cross-Platform Reality (No Marketing Lies)

Let's be explicit. No mumbling.

### Source-level Portability

✔ C# / .NET 8
✔ ILGPU abstraction layer
✔ Same codebase, all platforms

### Backend Availability

| Platform | CPU | CUDA | OpenCL |
|----------|-----|------|--------|
| Windows  | ✅  | ✅   | ✅     |
| Linux    | ✅  | ✅   | ✅     |
| macOS    | ✅  | ❌   | ⚠️ Deprecated |

> **Execution is portable. Acceleration is hardware-dependent.**

Your code compiles everywhere. It *flies* where the hardware supports it. It *walks* where it doesn't. Honest. No lies.

---

## 15. Project Structure

```
GpuStreamingDemo/
├── GpuStreamingDemo.sln
├── Directory.Build.props
├── README.md
└── src/
    ├── GpuStreamingDemo.App/
    │   ├── GpuStreamingDemo.App.csproj
    │   └── Program.cs                    # CLI entry point
    └── GpuStreamingDemo.Core/
        ├── GpuStreamingDemo.Core.csproj
        ├── AcceleratorFactory.cs          # Hardware abstraction
        ├── BenchmarkResult.cs             # Result record
        ├── BenchmarkRunner.cs             # Pluggable harness
        ├── GpuComputeEngine.cs            # Compute engine (streaming)
        ├── IBenchmarkTask.cs              # Task interface (SOLID)
        ├── Kernels.cs                     # All GPU kernels
        ├── StreamingGpuPipeline.cs        # Async streaming pipeline
        └── Tasks/
            ├── AffineTransformTask.cs     # y = a·x + b
            ├── VectorAddTask.cs           # z = x + y
            ├── SaxpyTask.cs              # y = a·x + y
            ├── Sha256Task.cs             # SHA-256 compression
            └── MandelbrotTask.cs         # Mandelbrot iterations
```

---

## 16. Getting Started

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- For CUDA: NVIDIA GPU + [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- For OpenCL: Compatible GPU + OpenCL runtime

### Build & Run

```bash
# Clone
git clone https://github.com/SolidRegardless/GpuStreamingDemo.git
cd GpuStreamingDemo

# Build
dotnet build

# Run all benchmarks
dotnet run --project src/GpuStreamingDemo.App

# Run with options
dotnet run --project src/GpuStreamingDemo.App -- --accel cuda --task sha256 --iters 100
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--accel` | `all` | Accelerator: `cpu`, `cuda`, `opencl`, `all` |
| `--batch` | `1048576` | Elements per batch |
| `--iters` | `50` | Measured iterations |
| `--warmup` | `5` | Warmup iterations |
| `--task` | *(all)* | Filter to specific task name |

---

## 17. The Architectural Lesson

The biggest performance gain came from:

❌ clever code
❌ micro-optimisations
❌ fighting the compiler
❌ adding more `Span<T>` and hoping for the best

✅ **Changing where the code runs**

This is the lesson people miss. You can spend weeks shaving microseconds off a CPU implementation. Or you can spend an afternoon moving it to the GPU and gain *milliseconds*.

---

## 18. What This Enables

This pattern applies directly to:

- 💰 Financial simulations & Monte-Carlo risk models
- 📡 Signal processing & real-time filtering
- 🖼️ Image pipelines & computer vision
- 🤖 AI feature extraction & inference
- 🔬 Scientific computing & simulations
- 🔐 Cryptographic operations at scale
- 🌀 Fractal generation (obviously)

Anywhere data dominates branches. Anywhere parallelism is free. Anywhere your CPU is sweating and your GPU is napping.

---

## 19. What This Is *Not*

Let's be clear:

❌ Not a synthetic CUDA demo with cherry-picked numbers
❌ Not vendor-locked (CPU fallback always works)
❌ Not a one-off benchmark (pluggable, extensible, SOLID)
❌ Not a "just trust me" blog post

This is a **production-shaped system** with real architecture, real patterns, and real numbers.

---

## 20. Future Work (Because Of Course)

Natural next steps:

- 🔄 Double-buffered async overlap (hide transfer latency)
- 📊 PCIe vs kernel timing breakdown
- 📈 Batch size sweep analysis
- 🖥️ Multi-GPU fan-out
- 📝 CSV / Markdown export
- 🧪 More tasks (FFT, matrix multiply, sorting networks)
- 🐧 Linux / cross-platform CI benchmarks

Each one will squeeze more performance — but none will match the first win of moving to the GPU.

---

## 21. The Real Takeaway

C# is not slow.

Your CPU is not slow.

They're just running the wrong workload on the wrong silicon.

---

## 22. Final Thought

> *At some point, performance tuning stops being about code and starts being about physics. The sooner you accept that, the sooner your benchmarks stop making you sad.*

Full source, architecture, and benchmarks:

👉 **https://github.com/SolidRegardless/GpuStreamingDemo**

---

*Built with 🐾 and an unreasonable amount of GPU cycles.*
