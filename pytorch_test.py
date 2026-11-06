
import time
import statistics

import torch
import torch.nn as nn
import torch_directml

print("PyTorch:", torch.__version__)
print("DirectML available:", torch_directml.is_available())
print("GPU:", torch_directml.device_name(1))

# ==========================================
# Configuration
# ==========================================

TOTAL_INPUTS = 10_000

BATCH_SIZES = [256, 512, 1024]

WIDTH = 4096
NUM_LAYERS = 4

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5

DTYPE = torch.float32


# ==========================================
# Device
# ==========================================

GPU_INDEX = 1

device = torch_directml.device(GPU_INDEX)

print("Selected GPU:", torch_directml.device_name(GPU_INDEX))

print("=" * 65)
print("FP32 AMD GPU Benchmark")
print("=" * 65)

print("Device:", device)
print("PyTorch:", torch.__version__)
print("Total inputs:", TOTAL_INPUTS)
print("Network:", [WIDTH] * (NUM_LAYERS + 1))
print("Activations: ReLU")
print("Precision: FP32")


# ==========================================
# Network
# ==========================================

class BenchmarkNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(WIDTH, WIDTH)
            for _ in range(NUM_LAYERS)
        ])

    def forward(self, x):

        for layer in self.layers:
            x = torch.relu(layer(x))

        return x


model = BenchmarkNetwork().to(device)
model.eval()


# ==========================================
# Generate all input data
# ==========================================

print("\nGenerating input data...")

all_data = torch.randn(
    TOTAL_INPUTS,
    WIDTH,
    dtype=DTYPE
)

all_data = all_data.to(device)

print("Input data ready.")


# ==========================================
# Benchmark function
# ==========================================

def benchmark(batch_size):

    batches = [
        all_data[i:i + batch_size]
        for i in range(0, TOTAL_INPUTS, batch_size)
    ]

    actual_inputs = sum(batch.shape[0] for batch in batches)

    # FLOPs per input:
    # 4 dense layers, each 4096 x 4096.
    #
    # 2 operations per multiply-add.
    #
    # Bias and ReLU excluded.

    flops_per_input = (
        2 * NUM_LAYERS * WIDTH * WIDTH
    )

    total_flops = flops_per_input * actual_inputs

    def run_pass():
        output = None

        for batch in batches:
            output = model(batch)

        return output

    print(f"\nBatch size: {batch_size}")
    print(f"Number of batches: {len(batches)}")

    # Warmup
    with torch.no_grad():

        for _ in range(WARMUP_RUNS):
            output = run_pass()
            output[0, 0].item()

    # Benchmark
    timings = []

    with torch.no_grad():

        for run in range(BENCHMARK_RUNS):

            start = time.perf_counter()

            output = run_pass()

            # Force GPU completion before stopping timer.
            output[0, 0].item()

            elapsed = time.perf_counter() - start

            timings.append(elapsed)

            print(
                f"  Run {run + 1}: "
                f"{elapsed * 1000:.2f} ms"
            )

    average = statistics.mean(timings)
    median = statistics.median(timings)
    best = min(timings)

    return {
        "batch": batch_size,
        "batches": len(batches),
        "average": average,
        "median": median,
        "best": best,
        "avg_tflops": total_flops / average / 1e12,
        "best_tflops": total_flops / best / 1e12,
    }


# ==========================================
# Run all tests
# ==========================================

results = []

for batch_size in BATCH_SIZES:

    result = benchmark(batch_size)
    results.append(result)


# ==========================================
# Results
# ==========================================

print("\n")
print("=" * 90)
print("FINAL RESULTS - FP32")
print("=" * 90)

print(
    f"{'Batch':>8} "
    f"{'Batches':>10} "
    f"{'Average ms':>14} "
    f"{'Best ms':>12} "
    f"{'Average TFLOPS':>18} "
    f"{'Best TFLOPS':>15}"
)

print("-" * 90)

for r in results:

    print(
        f"{r['batch']:>8} "
        f"{r['batches']:>10} "
        f"{r['average'] * 1000:>14.2f} "
        f"{r['best'] * 1000:>12.2f} "
        f"{r['avg_tflops']:>18.3f} "
        f"{r['best_tflops']:>15.3f}"
    )

print("=" * 90)
