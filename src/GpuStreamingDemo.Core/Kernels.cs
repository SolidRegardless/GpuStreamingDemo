using ILGPU;
using ILGPU.Runtime;

namespace GpuStreamingDemo.Core;

/// <summary>
/// Contains GPU compute kernels for demonstration and benchmarking.
/// </summary>
public static class Kernels
{
    /// <summary>
    /// Affine transform kernel: y[i] = a * x[i] + b.
    /// </summary>
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

    /// <summary>
    /// Vector addition kernel: z[i] = x[i] + y[i].
    /// </summary>
    public static void VectorAdd(
        Index1D index,
        ArrayView<float> x,
        ArrayView<float> y,
        ArrayView<float> z)
    {
        if (index >= x.Length) return;
        z[index] = x[index] + y[index];
    }

    /// <summary>
    /// SAXPY kernel: y[i] = a * x[i] + y[i].
    /// </summary>
    public static void Saxpy(
        Index1D index,
        ArrayView<float> x,
        ArrayView<float> y,
        float a)
    {
        if (index >= x.Length) return;
        y[index] = a * x[index] + y[index];
    }

    // SHA-256 round constants.
    private static readonly uint[] K =
    {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    /// <summary>
    /// Right-rotate helper for 32-bit integers.
    /// </summary>
    private static uint RotR(uint x, int n) => (x >> n) | (x << (32 - n));

    /// <summary>
    /// SHA-256 single-block compression kernel. Each thread hashes one 64-byte message
    /// (16 uint words from <paramref name="messages"/>) and writes 8 uint words to <paramref name="hashes"/>.
    /// </summary>
    /// <param name="index">Thread index.</param>
    /// <param name="messages">Flat array of message words, 16 per thread.</param>
    /// <param name="hashes">Flat array of hash words, 8 per thread.</param>
    /// <param name="k">The 64 SHA-256 round constants.</param>
    public static void Sha256Compress(
        Index1D index,
        ArrayView<uint> messages,
        ArrayView<uint> hashes,
        ArrayView<uint> k)
    {
        int msgOff = index * 16;
        int hashOff = index * 8;

        // Message schedule - use local variables since ILGPU doesn't support stackalloc.
        // We'll compute W on the fly with a small sliding window approach.
        // Actually, let's just use 64 local vars via an unrolled approach - but that's impractical.
        // Instead, store W in a local array (ILGPU supports fixed-size local arrays via LocalMemory).
        // For simplicity, we'll compute the full schedule step by step.

        // W array - we need 64 entries. ILGPU doesn't support large local arrays well,
        // so we'll use a two-pass approach: first expand, then compress.
        // Actually, ILGPU local memory works. Let's just do it simply with variables.

        // Simplified: we'll just do the 64-round compression reading W on the fly.
        uint w0  = messages[msgOff +  0], w1  = messages[msgOff +  1];
        uint w2  = messages[msgOff +  2], w3  = messages[msgOff +  3];
        uint w4  = messages[msgOff +  4], w5  = messages[msgOff +  5];
        uint w6  = messages[msgOff +  6], w7  = messages[msgOff +  7];
        uint w8  = messages[msgOff +  8], w9  = messages[msgOff +  9];
        uint w10 = messages[msgOff + 10], w11 = messages[msgOff + 11];
        uint w12 = messages[msgOff + 12], w13 = messages[msgOff + 13];
        uint w14 = messages[msgOff + 14], w15 = messages[msgOff + 15];

        // Initial hash values
        uint h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
        uint h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;

        uint a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;

        // We do 64 rounds. For rounds 0-15, W = w0..w15.
        // For rounds 16-63, we compute W on the fly and shift the window.
        for (int i = 0; i < 64; i++)
        {
            uint wi;
            if (i < 16)
            {
                wi = i switch
                {
                    0 => w0, 1 => w1, 2 => w2, 3 => w3,
                    4 => w4, 5 => w5, 6 => w6, 7 => w7,
                    8 => w8, 9 => w9, 10 => w10, 11 => w11,
                    12 => w12, 13 => w13, 14 => w14, _ => w15
                };
            }
            else
            {
                // sigma0(w[i-15]) + w[i-16] + sigma1(w[i-2]) + w[i-7]
                var s0val = RotR(w1, 7) ^ RotR(w1, 18) ^ (w1 >> 3);
                var s1val = RotR(w14, 17) ^ RotR(w14, 19) ^ (w14 >> 10);
                wi = w0 + s0val + w9 + s1val;

                // Shift window
                w0 = w1; w1 = w2; w2 = w3; w3 = w4;
                w4 = w5; w5 = w6; w6 = w7; w7 = w8;
                w8 = w9; w9 = w10; w10 = w11; w11 = w12;
                w12 = w13; w13 = w14; w14 = w15; w15 = wi;
            }

            var S1 = RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25);
            var ch = (e & f) ^ (~e & g);
            var temp1 = h + S1 + ch + k[i] + wi;
            var S0 = RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22);
            var maj = (a & b) ^ (a & c) ^ (b & c);
            var temp2 = S0 + maj;

            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }

        hashes[hashOff + 0] = h0 + a;
        hashes[hashOff + 1] = h1 + b;
        hashes[hashOff + 2] = h2 + c;
        hashes[hashOff + 3] = h3 + d;
        hashes[hashOff + 4] = h4 + e;
        hashes[hashOff + 5] = h5 + f;
        hashes[hashOff + 6] = h6 + g;
        hashes[hashOff + 7] = h7 + h;
    }

    /// <summary>
    /// Mandelbrot set kernel. Computes the escape iteration count for each pixel.
    /// Maps the index to a point in the complex plane [-2, 1] x [-1.5, 1.5].
    /// </summary>
    /// <param name="index">Thread index (pixel index).</param>
    /// <param name="output">Output array of iteration counts.</param>
    /// <param name="width">Image width in pixels.</param>
    /// <param name="maxIter">Maximum iteration count.</param>
    public static void Mandelbrot(
        Index1D index,
        ArrayView<int> output,
        int width,
        int maxIter)
    {
        int px = index % width;
        int py = index / width;
        int height = (int)(output.Length / width);

        float x0 = (px / (float)width) * 3.0f - 2.0f;     // [-2, 1]
        float y0 = (py / (float)height) * 3.0f - 1.5f;     // [-1.5, 1.5]

        float x = 0, y = 0;
        int iter = 0;

        while (x * x + y * y <= 4.0f && iter < maxIter)
        {
            float xtemp = x * x - y * y + x0;
            y = 2 * x * y + y0;
            x = xtemp;
            iter++;
        }

        output[index] = iter;
    }
}
