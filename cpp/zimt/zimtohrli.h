// Copyright 2024 The Zimtohrli Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CPP_ZIMT_ZIMTOHRLI_H_
#define CPP_ZIMT_ZIMTOHRLI_H_

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace zimtohrli {

// A non-owning view of a contiguous sequence of elements.
// Similar to std::span (C++20), but provided for compatibility with older C++
// standards. It allows functions to operate on sequences of data without
// needing to know the underlying container type (e.g., std::vector, C-style
// array).
template <typename T>
struct Span {
  // Default copy constructor.
  Span(const Span& other) = default;

  // Constructs a Span from a std::vector.
  // The Span does not own the vector's data. The vector must outlive the Span.
  Span(std::vector<T>& vec) : size(vec.size()), data(vec.data()) {}

  // Constructs a Span from a raw pointer and size.
  // The Span does not own the data. The caller is responsible for ensuring the
  // data remains valid for the lifetime of the Span.
  explicit Span(T* data, size_t size) : size(size), data(data) {}

  // Constructs a Span from a std::vector of a potentially different but
  // convertible type. For example, can construct a Span<const float> from a
  // std::vector<float>.
  template <typename U>
  Span(const std::vector<U>& vec) noexcept
      : data(vec.data()), size(vec.size()) {
    static_assert(std::is_convertible_v<U(*)[], T(*)[]>,
                  "Cannot construct Span from vector of incompatible type.");
  }

  // Constructs a Span from another Span of a potentially different but
  // convertible type.
  template <typename U>
  Span(const Span<U>& other) noexcept : data(other.data), size(other.size) {
    static_assert(std::is_convertible_v<U(*)[], T(*)[]>,
                  "Cannot construct Span from Span of incompatible type.");
  }

  // Default copy assignment operator.
  Span& operator=(const Span& other) = default;

  // Accesses the element at the specified index (const version).
  const T& operator[](size_t index) const { return data[index]; }

  // Accesses the element at the specified index (non-const version).
  T& operator[](size_t index) { return data[index]; }

  // The number of elements in the Span.
  size_t size;
  // A pointer to the first element of the Span.
  T* data;
};

namespace {

#define assert_eq(a, b)                                                        \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      throw std::runtime_error(std::string("Assertion failed: ") + #a + " (" + \
                               std::to_string(a) + ") == " #b + " (" +         \
                               std::to_string(b) + ") at " + __FILE__ + ":" +  \
                               std::to_string(__LINE__));                      \
    }                                                                          \
  } while (0)

constexpr int64_t kNumRotators = 128;

// Converts filterbank channel energies to a perceptual loudness scale (approximating dB).
// This function applies a logarithmic transformation and frequency-dependent
// weighting to the input channel energies.
//
// Args:
//   channels: A pointer to an array of kNumRotators float values representing
//             the energy in each filterbank channel. These values are modified
//             in-place.
inline void LoudnessDb(float* channels) {
  // kMul represents frequency-dependent multipliers, likely derived from
  // equal-loudness contours or other psychoacoustic models. These values adjust
  // the perceived loudness of different frequency bands.
  static const float kMul[128] = {
      0.69022, 0.68908, 0.69206, 0.68780, 0.68780, 0.68780, 0.68780, 0.68780,
      0.68780, 0.68780, 0.68780, 0.68913, 0.69045, 0.69310, 0.69575, 0.69565,
      0.69697, 0.70122, 0.72878, 0.79911, 0.85713, 0.88063, 0.88563, 0.87561,
      0.81948, 0.70435, 0.63479, 0.58382, 0.52065, 0.48390, 0.46452, 0.47952,
      0.52686, 0.63677, 0.75972, 0.89449, 0.97411, 1.01874, 1.01105, 0.99306,
      0.93613, 0.92825, 0.93149, 0.98687, 1.05782, 1.16461, 1.25028, 1.30768,
      1.31484, 1.28574, 1.23002, 1.15336, 1.08800, 1.01472, 0.94610, 0.91856,
      0.87797, 0.85825, 0.82836, 0.82198, 0.81394, 0.82724, 0.84235, 0.86009,
      0.88276, 0.89349, 0.92543, 0.94822, 0.98526, 0.99730, 1.02097, 1.04071,
      1.05254, 1.06462, 1.06872, 1.07382, 1.06739, 1.06331, 1.05118, 1.05002,
      1.04803, 1.06729, 1.09680, 1.15208, 1.22492, 1.32630, 1.42049, 1.50444,
      1.58735, 1.65199, 1.69488, 1.70748, 1.74525, 1.68760, 1.66818, 1.63401,
      1.55136, 1.49170, 1.42649, 1.33453, 1.28618, 1.26523, 1.24900, 1.24898,
      1.27864, 1.28723, 1.28455, 1.29777, 1.29637, 1.29687, 1.29853, 1.30319,
      1.30207, 1.26835, 1.25100, 1.24664, 1.24041, 1.24297, 1.07569, 0.97131,
      0.95906, 1.21035, 0.85762, 0.77298, 1.12289, 0.74092, 0.99662, 1.11603,
  };
  // kBaseNoise is a small constant added to the channel energy before taking
  // the logarithm. This prevents -infinity results for channels with zero
  // energy and can also be seen as a noise floor or a way to stabilize the
  // logarithm for very small values. Its specific value is likely empirically
  // derived.
  static const float kBaseNoise = 886018.44434708043;
  for (int k = 0; k < kNumRotators; ++k) {
    channels[k] = log(channels[k] + kBaseNoise) * kMul[k];
  }
}

// Models the mechanical response of the human ear, specifically the ear drum
// and other connected mass-spring systems. This is a non-linear process that
// introduces complex spectral shifting of energy, simulating how the ear
// processes incoming sound waves.
struct Resonator {
  // Accumulator 0, representing a displacement or velocity in the resonator system.
  float acc0 = 0;
  // Accumulator 1, representing another state variable in the resonator system
  // (e.g., related to acceleration or a different part of the system).
  float acc1 = 0;

  // Updates the state of the resonator based on an input signal sample.
  // This function simulates the resonance and attenuation characteristics of the
  // ear.
  //
  // Args:
  //   signal: The input audio signal sample.
  //
  // Returns:
  //   The current output of the resonator model (acc0).
  float Update(float signal) {  // Resonate and attenuate.
    // These parameters (kMul0, kMul1) are constants that define the dynamics
    // of the resonator. They are likely empirically derived to match the
    // response characteristics of a typical human ear or a population average.
    // kMul0 typically represents a damping/feedback factor for acc0.
    static const float kMul0 = 0.93913835617233998;
    // kMul1 typically represents a coupling factor from acc1 to acc0.
    static const float kMul1 = -0.040539506065308289;
    acc0 = kMul0 * acc0 + kMul1 * acc1 + signal;
    acc1 += acc0;
    return acc0;
  }
};

// Computes the dot product of two 32-element float arrays.
// This is a common operation in signal processing, often used for FIR filters
// or convolutions.
//
// Args:
//   a: Pointer to the first array of 32 floats.
//   b: Pointer to the second array of 32 floats.
//
// Returns:
//   The dot product of the two arrays.
inline float Dot32(const float* a, const float* b) {
  // -ffast-math compiler flag can be beneficial for performance here.
  // Modern compilers like clang can often auto-vectorize (SIMD) this loop.
  float sum = 0;
  for (int i = 0; i < 32; ++i) sum += a[i] * b[i];
  return sum;
}

// Returns the center frequency for a given filter bank channel index.
// The frequencies are pre-defined and stored in the `kFreq` array.
// The array includes `kNumRotators` (128) center frequencies, plus one
// additional frequency at each end, making it 130 elements long. These extra
// frequencies might be used for calculating bandwidths at the edges.
//
// Args:
//   i: The index of the filter bank channel. This should be less than
//      kNumRotators. The function accesses kFreq[i+1].
//
// Returns:
//   The center frequency in Hz for the specified channel.
float Freq(int i) {
  // Center frequencies of the filter bank in Hz. There are kNumRotators (128)
  // channels, and kFreq stores kNumRotators + 2 values. The extra values at
  // the beginning and end are likely used for bandwidth calculations for the
  // first and last actual channels. The function returns kFreq[i+1], so
  // Freq(0) returns kFreq[1], Freq(kNumRotators-1) returns kFreq[kNumRotators].
  static const float kFreq[130] = {
      17.858,  24.349,  33.199,  42.359,  51.839,  61.651,  71.805,  82.315,
      93.192,  104.449, 116.099, 128.157, 140.636, 153.552, 166.919, 180.754,
      195.072, 209.890, 225.227, 241.099, 257.527, 274.528, 292.124, 310.336,
      329.183, 348.690, 368.879, 389.773, 411.398, 433.778, 456.941, 480.914,
      505.725, 531.403, 557.979, 585.484, 613.950, 643.411, 673.902, 705.459,
      738.119, 771.921, 806.905, 843.111, 880.584, 919.366, 959.503, 1001.04,
      1044.03, 1088.53, 1134.58, 1182.24, 1231.57, 1282.62, 1335.46, 1390.14,
      1446.73, 1505.31, 1565.93, 1628.67, 1693.60, 1760.80, 1830.35, 1902.34,
      1976.84, 2053.94, 2133.74, 2216.33, 2301.81, 2390.27, 2481.83, 2576.58,
      2674.65, 2776.15, 2881.19, 2989.91, 3102.43, 3218.88, 3339.40, 3464.14,
      3593.23, 3726.84, 3865.12, 4008.23, 4156.35, 4309.64, 4468.30, 4632.49,
      4802.43, 4978.31, 5160.34, 5348.72, 5543.70, 5745.49, 5954.34, 6170.48,
      6394.18, 6625.70, 6865.32, 7113.31, 7369.97, 7635.61, 7910.53, 8195.06,
      8489.53, 8794.30, 9109.73, 9436.18, 9774.04, 10123.7, 10485.6, 10860.1,
      11247.8, 11648.9, 12064.2, 12493.9, 12938.7, 13399.0, 13875.3, 14368.4,
      14878.7, 15406.8, 15953.4, 16519.1, 17104.5, 17710.4, 18337.6, 18986.6,
      19658.3, 20352.7,
  };
  return kFreq[i + 1];
}

// Calculates the bandwidth of a filter bank channel in Hz.
// The bandwidth is calculated based on the center frequencies of the current
// channel (i), the next channel (i+1), and the previous channel (i-1).
// This specific formula appears to be related to geometric means of adjacent
// frequencies.
//
// Args:
//   i: The index of the filter bank channel for which to calculate bandwidth.
//      This index should be valid such that Freq(i+1), Freq(i), and Freq(i-1)
//      are accessible and meaningful (i.e., i should be >= 0 and <
//      kNumRotators). The Freq function uses i+1 internally, so Freq(i-1)
//      accesses kFreq[i], Freq(i) accesses kFreq[i+1], and Freq(i+1) accesses
//      kFreq[i+2].
//
// Returns:
//   The calculated bandwidth in Hz for the specified channel.
double CalculateBandwidthInHz(int i) {
  return std::sqrt(Freq(i + 1) * Freq(i)) - std::sqrt(Freq(i - 1) * Freq(i));
}

// The Rotators class implements a bank of complex filters (oscillators) used
// to analyze an audio signal and transform it into a time-frequency
// representation (spectrogram). It performs filtering and downsampling.
// The Rotators class implements a bank of complex filters (oscillators) used
// to analyze an audio signal and transform it into a time-frequency
// representation (spectrogram). It performs filtering and downsampling.
// Each rotator corresponds to a specific frequency band.
class Rotators {
 private:
  // `rot` stores parameters for the complex rotators (oscillators).
  // It's a 4x`kNumRotators` array. For each of the `kNumRotators` channels:
  //  - `rot[0][i]`: Real part of the complex exponential for rotation (cos(omega*T)).
  //  - `rot[1][i]`: Imaginary part of the complex exponential for rotation (-sin(omega*T)).
  //  - `rot[2][i]`: Real part of the current state of the frequency rotator,
  //                 initialized with sqrt(gain[i]). This is updated over time.
  //  - `rot[3][i]`: Imaginary part of the current state of the frequency rotator,
  //                 initialized to 0. This is updated over time.
  // The layout is designed for potential SIMD optimization.
  float rot[4][kNumRotators];

  // `accu` stores accumulator values for the leaky integrators associated with
  // each rotator. It's a 6x`kNumRotators` array. For each channel `i`:
  //  - `accu[0][i]`, `accu[1][i]`: Real and imaginary parts of the 1st leaking accumulator.
  //  - `accu[2][i]`, `accu[3][i]`: Real and imaginary parts of the 2nd leaking accumulator.
  //  - `accu[4][i]`, `accu[5][i]`: Real and imaginary parts of the 3rd leaking accumulator.
  // These accumulators are used to estimate the energy in each band.
  float accu[6][kNumRotators] = {0};

  // `window` stores per-channel windowing coefficients (leaking factors) for
  // the accumulators. Derived from bandwidth and `kWindow`.
  float window[kNumRotators];

  // `gain` stores per-channel gain factors, used to initialize `rot[2]` and
  // scale the input signal's contribution to the accumulators.
  float gain[kNumRotators];

  // Renormalizes the `rot[2]` and `rot[3]` components periodically to maintain
  // their magnitude close to `gain[i]`. This helps prevent numerical drift or
  // instability in the rotator updates.
  void OccasionallyRenormalize() {
    for (int i = 0; i < kNumRotators; ++i) {
      // Calculate the current magnitude of (rot[2][i], rot[3][i])
      float current_magnitude =
          std::sqrt(rot[2][i] * rot[2][i] + rot[3][i] * rot[3][i]);
      // Avoid division by zero if magnitude is very small
      if (current_magnitude < 1e-9) continue;
      // Calculate the scaling factor to restore the desired magnitude (gain[i])
      float norm = gain[i] / current_magnitude;
      rot[2][i] *= norm;
      rot[3][i] *= norm;
    }
  }

  // Processes a single input signal sample through the filter bank.
  // For each channel, it updates the leaky accumulators and rotates the
  // frequency rotator state.
  //
  // Args:
  //   signal: The input audio sample after resonator and kernel processing.
  void IncrementAll(float signal) {
    for (int i = 0; i < kNumRotators; i++) {  // This loop is a candidate for SIMD.
      const float w = window[i]; // Leaking factor for this channel.
      // Apply leaking to all accumulators.
      for (int k = 0; k < 6; ++k) accu[k][i] *= w;

      // Update accumulators. The specific order (2,3 then 4,5 then 0,1) is
      // noted as working best, possibly due to data dependencies or numerical
      // properties. This represents a cascade of leaky integrators.
      accu[2][i] += accu[0][i];
      accu[3][i] += accu[1][i];
      accu[4][i] += accu[2][i];
      accu[5][i] += accu[3][i];

      // Add the current signal (scaled by the rotator's current state) to the
      // first accumulator. This is where the signal energy enters the system
      // for this frequency band.
      accu[0][i] += rot[2][i] * signal; // Real part
      accu[1][i] += rot[3][i] * signal; // Imaginary part

      // Rotate the frequency rotator (rot[2], rot[3]) by multiplying with
      // (rot[0], rot[1]), which are (cos(f), -sin(f)).
      // This is a complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
      // Here, (rot[2]+j*rot[3]) is being multiplied by (rot[0]+j*rot[1]).
      // rot[0] is cos(f), rot[1] is -sin(f).
      // New real part: rot[0]*rot[2] - rot[1]*rot[3]
      // New imag part: rot[0]*rot[3] + rot[1]*rot[2]
      const float current_rot_real = rot[2][i];
      const float current_rot_imag = rot[3][i];
      rot[2][i] = rot[0][i] * current_rot_real - rot[1][i] * current_rot_imag;
      rot[3][i] = rot[0][i] * current_rot_imag + rot[1][i] * current_rot_real;
    }
  }

 public:
  // Processes an input audio signal to produce a spectrogram.
  // The input signal is filtered by the resonator and then by the bank of
  // rotators. The energy from the rotators is accumulated and downsampled to
  // produce the output spectrogram.
  //
  // Args:
  //   in: Pointer to the input audio signal data (array of floats).
  //   in_size: Number of samples in the input signal.
  //   out: Pointer to the output spectrogram data (array of floats). This buffer
  //        should be pre-allocated. The layout is [out_shape0, kNumRotators].
  //   out_shape0: The number of time steps in the output spectrogram.
  //   out_stride: The stride of the output spectrogram data (typically kNumRotators).
  //   downsample: The downsampling factor, determining how many input samples
  //               contribute to one output spectrogram time step.
  void FilterAndDownsample(const float* in, size_t in_size, float* out,
                           size_t out_shape0, size_t out_stride,
                           int downsample) {
    // Assumed sample rate of the input audio signal.
    static const float kSampleRate = 48000.0;
    // Conversion factor from frequency in Hz to radians per sample.
    static const float kHzToRad = 2.0f * M_PI / kSampleRate;
    // Base windowing factor for leaky integrators. Empirically derived.
    static const double kWindow = 0.9996028710680265;
    // Magic constant used in bandwidth-dependent window calculation. Empirically derived.
    static const double kBandwidthMagic = 0.7328516996032982;
    // A large scaling factor for normalization, likely empirically derived for
    // optimal performance or numerical stability.
    static const double kScale = 929900594411.23657;

    // Calculate overall gain factor based on kScale and downsampling rate.
    const float gainer = sqrt(kScale / downsample);

    // Initialize rotator parameters for each channel.
    for (int i = 0; i < kNumRotators; ++i) {
      float bandwidth = CalculateBandwidthInHz(i);  // Bandwidth for this channel.
      // Calculate the per-channel leaking factor based on its bandwidth.
      window[i] = std::pow(kWindow, bandwidth * kBandwidthMagic);
      float windowM1 = 1.0f - window[i]; // (1 - window factor)
      // Target frequency for this rotator in radians per sample.
      const float f_rad = Freq(i) * kHzToRad;
      // Calculate gain for this channel. This depends on the window factor,
      // frequency, and bandwidth.
      gain[i] = gainer * pow(windowM1, 3.0) * Freq(i) / bandwidth;
      // Set the complex rotation vector (cos(f_rad), -sin(f_rad)).
      rot[0][i] = float(std::cos(f_rad));
      rot[1][i] = float(-std::sin(f_rad));
      // Initialize the rotator's state: real part is gain, imag part is 0.
      rot[2][i] = gain[i];
      rot[3][i] = 0.0f;
    }

    // Initialize output spectrogram buffer to zeros.
    for (size_t zz = 0; zz < out_shape0; zz++) {
      for (int k = 0; k < kNumRotators; ++k) {
        out[zz * out_stride + k] = 0;
      }
    }

    // Create a downsampling window (Hann-like or sigmoid-based).
    // This window is used to weight samples when accumulating energy for an
    // output spectrogram frame.
    std::vector<float> downsample_window(downsample);
    for (int i = 0; i < downsample; ++i) {
      // Sigmoid-based window function.
      downsample_window[i] =
          1.0 / (1.0 + exp(7.9446 * ((2.0 / downsample) * (i + 0.5) - 1)));
    }

    Resonator resonator; // Ear model resonator.
    size_t out_ix = 0;   // Current output spectrogram time step index.

    // Size of the FIR kernels (resonator and linear).
    constexpr size_t kKernelSize = 32;
    // FIR kernel representing the resonator's impulse response characteristics.
    // Values are empirically derived.
    static const float reso_kernel[kKernelSize] = {
      -0.0075642284403770708, 0.0041328270786934662, -7.6269851290751061e-06, 0.0061764514689768733,
      -0.0028376753880472038, -1.1759452250705732e-05, -0.0065499115361845562, -0.0069727090984949783,
      0.0034584201864033401, 0.003329316161974918, -0.0029971240720728575, 0.0034898641766847685,
      0.0017717742743446263, -0.0015229487607625498, 0.0039309982613565655, 0.001278227701047937,
      -0.0116877416785343, -0.00039070521292690666, -0.0015923522740827827, -0.0082269584153230185,
      -0.0063814620315990021, -0.0008796390298788419, -0.0071855544224704287, 0.0034822736952680863,
      -0.00041538926556568181, 0.0001753900488857857, -0.0011326124605282573, 0.00095353008231245965,
      0.0073567454219722467, -0.0016601446765057634, -0.0069136302438569507, 0.010715105623693549,
    };
    // FIR kernel representing a linear path, processed in parallel with the resonator.
    // Values are empirically derived.
    static const float linear_kernel[kKernelSize] = {
      -0.30960591444509439, -0.079455203026254709, -0.14108618014504098, 0.070751037303552131,
      0.14104891038659864, -0.17036477880916376, 0.014288229833457814, 0.27147357420390988,
      0.17978692186268302, 0.065653189749429991, 0.014169704877201516, 0.18257259370291729,
      0.0021021318985668257, 0.065359875882277235, -0.015544998395038102, -0.049398120278478827,
      -0.064034911106614606, -0.57876116795333099, 0.57561220696398696, 0.40135227167310927,
      -0.33118848897270026, 0.17695279679195522, 1.0491938729586434, -0.58835602045486513,
      -1.4541325309560014, 0.071462019783188307, 0.72056751090553661, 1.2265425406909325,
      -0.72083484154250099, 0.84200784192262634, -0.10112736611558046, -0.44049413285605787,
    };

    // Main processing loop: iterate over input samples.
    // `dix` is the index within the current downsampling window.
    for (size_t in_ix = 0, dix = 0; in_ix + kKernelSize < in_size; ++in_ix) {
      // Get the weight from the downsampling window for the current sample.
      const float weight = downsample_window[dix];

      // Process the input signal chunk:
      // 1. Apply resonator model (after FIR filtering with reso_kernel).
      // 2. Apply linear path (FIR filtering with linear_kernel).
      // 3. Sum the outputs and pass to IncrementAll to update rotators.
      IncrementAll(resonator.Update(Dot32(&in[in_ix], &reso_kernel[0])) +
                   Dot32(&in[in_ix], &linear_kernel[0]));

      // Accumulate energy into the output spectrogram bins.
      // The energy from the 3rd accumulator (accu[4], accu[5]) is used.
      // It's distributed between the current (out_ix) and next (out_ix+1)
      // output frames based on the `weight`. This provides smoother transitions.
      if (out_ix + 1 < out_shape0) {
        for (int k = 0; k < kNumRotators; ++k) {
          // Energy is sum of squares of real and imaginary parts of 3rd accumulator.
          float energy = accu[4][k] * accu[4][k] + accu[5][k] * accu[5][k];
          out[(out_ix + 1) * out_stride + k] += (1.0 - weight) * energy;
          out[out_ix * out_stride + k] += weight * energy;
        }
      } else { // For the last output frame, add all energy to it.
        for (int k = 0; k < kNumRotators; ++k) {
          float energy = accu[4][k] * accu[4][k] + accu[5][k] * accu[5][k];
          out[out_ix * out_stride + k] += energy;
        }
      }

      // Check if a downsampling block is complete or if it's the end of input.
      if (++dix == downsample || in_ix + kKernelSize + 1 == in_size) {
        // Current output frame is complete, apply loudness scaling.
        LoudnessDb(&out[out_stride * out_ix]);
        // Move to the next output frame.
        if (++out_ix >= out_shape0) {
          break; // Stop if all output frames are filled.
        }
        dix = 0; // Reset index for downsampling window.
        // Periodically renormalize rotators to prevent numerical drift.
        OccasionallyRenormalize();
      }
    }
  }
};

// Represents a spectrogram, which is a time-frequency representation of audio.
// The data is stored as a 2D array where rows are time steps and columns are
// frequency dimensions (channels/bands).
//
// The `values` buffer stores the spectrogram data in a contiguous block of memory,
// row by row (time step by time step).
// For a spectrogram with `num_steps` time steps and `num_dims` frequency dimensions:
// `values` will contain `num_steps * num_dims` floats.
// The layout is:
//   [step0_dim0, step0_dim1, ..., step0_dim(num_dims-1)],  // First time step
//   [step1_dim0, step1_dim1, ..., step1_dim(num_dims-1)],  // Second time step
//   ...
//   [step(num_steps-1)_dim0, ..., step(num_steps-1)_dim(num_dims-1)] // Last time step
struct Spectrogram {
  // Default move constructor.
  Spectrogram(Spectrogram&& other) = default;

  // Constructs a Spectrogram with a given number of time steps and a default
  // number of frequency dimensions (`kNumRotators`).
  // Allocates memory for the spectrogram values.
  Spectrogram(size_t num_steps)
      : num_steps(num_steps),
        num_dims(kNumRotators),
        values(std::make_unique<float[]>(num_steps * kNumRotators)) {}

  // Constructs a Spectrogram with a specified number of time steps and
  // frequency dimensions. Allocates memory for the spectrogram values.
  Spectrogram(size_t num_steps, size_t num_dims)
      : num_steps(num_steps),
        num_dims(num_dims),
        values(std::make_unique<float[]>(num_steps * num_dims)) {}

  // Constructs a Spectrogram by taking ownership of an existing unique_ptr
  // to float data.
  Spectrogram(size_t num_steps, size_t num_dims,
              std::unique_ptr<float[]> values)
      : num_steps(num_steps), num_dims(num_dims), values(std::move(values)) {}

  // Constructs a Spectrogram from a std::vector<float>.
  // The data from the vector is copied into a new buffer owned by the Spectrogram.
  Spectrogram(size_t num_steps, size_t num_dims, std::vector<float> data)
      : num_steps(num_steps),
        num_dims(num_dims),
        values(std::make_unique<float[]>(data.size())) {
    assert_eq(num_steps * num_dims, data.size()); // Ensure dimensions match data size
    std::memcpy(values.get(), data.data(), data.size() * sizeof(float));
  }

  // Constructs a Spectrogram from a raw float pointer (non-owning).
  // WARNING: This constructor creates a Spectrogram that does NOT own the data.
  // The caller is responsible for ensuring the lifetime of `data` exceeds
  // that of the Spectrogram object. The `values` unique_ptr is initialized
  // with the provided pointer but without a deleter, effectively making it
  // non-owning. This is generally unsafe and should be used with extreme caution.
  // Consider if a Span-based approach or copying the data would be safer.
  Spectrogram(size_t num_steps, size_t num_dims, float* data)
      : num_steps(num_steps), num_dims(num_dims), values(data) {} // Potentially unsafe: values doesn't own data.

  // Default move assignment operator.
  Spectrogram& operator=(Spectrogram&& other) = default;

  // Provides access to a time step (row) of the spectrogram (const version).
  // Returns a Span viewing the data for the n-th time step.
  Span<const float> operator[](size_t n) const {
    return Span<const float>(values.get() + n * num_dims, num_dims);
  }

  // Provides access to a time step (row) of the spectrogram (non-const version).
  // Returns a Span viewing the data for the n-th time step.
  Span<float> operator[](size_t n) {
    return Span(values.get() + n * num_dims, num_dims);
  }

  // Calculates the maximum absolute value among all elements in the spectrogram.
  float max() const {
    float res = 0; // Initialize with 0, assumes non-negative, or that absolute values are taken.
                   // If values can be negative, should initialize with -infinity or first element.
                   // Current implementation correctly uses std::abs.
    for (size_t step_idx = 0; step_idx < num_steps; ++step_idx) {
      for (size_t dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
        res = std::max(res, std::abs(operator[](step_idx)[dim_idx]));
      }
    }
    return res;
  }

  // Rescales all spectrogram values by a given factor.
  void rescale(float f) {
    for (size_t step_idx = 0; step_idx < num_steps; ++step_idx) {
      for (size_t dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
        operator[](step_idx)[dim_idx] *= f;
      }
    }
  }

  // Returns the total number of float values in the spectrogram data.
  size_t size() const { return num_steps * num_dims; }

  // Number of time steps in the spectrogram.
  size_t num_steps;
  // Number of frequency dimensions (channels/bands) per time step.
  size_t num_dims;
  // Unique pointer owning the flat array of spectrogram values.
  // Note: One constructor allows this to be non-owning if a raw pointer is passed;
  // this is a potentially risky pattern.
  std::unique_ptr<float[]> values;
};

// Computes the mean of values within a 2D sliding window over input data.
// The input data is accessed via the `input_loader` lambda. The function
// efficiently calculates these windowed means using prefix sums (also known as
// summed-area tables).
//
// The process involves two main passes:
// 1. Calculate windowed sums along the 'step' (time) axis:
//    a. Compute prefix sums along the step axis for each channel.
//    b. Use these prefix sums to find the sum of values within each step_window.
// 2. Calculate windowed sums along the 'channel' (frequency) axis from the
//    results of pass 1:
//    a. For each time step, compute prefix sums along the channel axis using the
//       step-windowed sums.
//    b. Use these new prefix sums to find the sum of values within each
//       channel_window. This result is now summed over both step_window and
//       channel_window.
// 3. Divide the 2D windowed sums by (step_window * channel_window) to get the mean.
//
// Args:
//   num_steps: The total number of time steps in the input data.
//   num_channels: The total number of frequency channels in the input data.
//   step_window: The size of the sliding window along the time step axis.
//   channel_window: The size of the sliding window along the channel axis.
//   input_loader: A callable (e.g., lambda) that takes `(size_t step_index,
//                 size_t channel_index)` and returns the float value at that
//                 position in the source data.
//
// Returns:
//   A Spectrogram object where each element `(s, c)` contains the mean of a
//   `step_window` x `channel_window` region of the input data, ending at
//   `(s, c)`. For elements near the beginning where a full window is not
//   available, the sum is over the available part of the window, but the
//   division is still by the full `step_window * channel_window`.
template <typename T>
Spectrogram WindowMean(size_t num_steps, size_t num_channels,
                       size_t step_window, size_t channel_window,
                       T input_loader) {
  // tmp_a will store the final result and intermediate sums.
  Spectrogram tmp_a(num_steps, num_channels);
  // tmp_b is used to store prefix sums.
  Spectrogram tmp_b(num_steps, num_channels);

  // Pass 1a: Populate tmp_b with prefix sums across the step axis (columns).
  // For each channel, tmp_b[s][c] = sum(input_loader(0..s, c)).
  {
    // Initialize the first row (step_index = 0) of prefix sums.
    Span<float> first_step_prefix_sum_data = tmp_b[0];
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      first_step_prefix_sum_data[channel_index] = input_loader(0, channel_index);
    }
  }
  // Calculate prefix sums for subsequent steps.
  for (size_t step_index = 1; step_index < num_steps; ++step_index) {
    Span<float> current_step_prefix_sum_data = tmp_b[step_index];
    Span<const float> prev_step_prefix_sum_data = tmp_b[step_index - 1];
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      current_step_prefix_sum_data[channel_index] =
          input_loader(step_index, channel_index) +
          prev_step_prefix_sum_data[channel_index];
    }
  }

  // Pass 1b: Populate tmp_a with windowed sums across the step axis using the
  // prefix sums in tmp_b.
  // tmp_a[s][c] = sum(input_loader(s-step_window+1 .. s, c)).
  // For s < step_window, it's sum(input_loader(0 .. s, c)).
  // 1: Copy the first `step_window` rows from tmp_b to tmp_a. For these rows,
  // the windowed sum is simply the prefix sum up to that row.
  if (step_window > 0 && num_steps > 0) {
    std::memcpy(tmp_a.values.get(), tmp_b.values.get(),
                std::min(step_window, num_steps) * num_channels * sizeof(float));
  }
  // 2: Compute windowed sums for the remaining rows by subtracting prefix sums.
  // sum(i-W+1 .. i) = prefix_sum(i) - prefix_sum(i-W).
  for (size_t step_index = step_window; step_index < num_steps; ++step_index) {
    Span<const float> full_prefix_sum_at_step = tmp_b[step_index];
    Span<const float> prefix_sum_before_window = tmp_b[step_index - step_window];
    Span<float> step_windowed_sum_output = tmp_a[step_index];
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      step_windowed_sum_output[channel_index] =
          full_prefix_sum_at_step[channel_index] -
          prefix_sum_before_window[channel_index];
    }
  }

  // Pass 2: Now, tmp_a contains sums over `step_window` for each channel.
  // We will reuse tmp_b and tmp_a to calculate 2D windowed sums.
  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    // Pass 2a: Populate tmp_b[step_index] with prefix sums across the channel
    // axis of the step-windowed sums currently in tmp_a[step_index].
    // tmp_b[s][c] = sum(tmp_a[s][0..c]).
    {
      Span<const float> current_step_windowed_sums = tmp_a[step_index]; // Input for this pass
      Span<float> channel_prefix_sum_storage = tmp_b[step_index]; // Output for this pass
      if (num_channels > 0) {
        channel_prefix_sum_storage[0] = current_step_windowed_sums[0];
        for (size_t channel_index = 1; channel_index < num_channels;
            ++channel_index) {
          channel_prefix_sum_storage[channel_index] =
              channel_prefix_sum_storage[channel_index - 1] +
              current_step_windowed_sums[channel_index];
        }
      }
    }
    // Pass 2b: Populate tmp_a[step_index] (final sum destination for this step)
    // with windowed sums across the channel axis, using prefix sums from tmp_b.
    // tmp_a[s][c] = sum(tmp_a_prev_pass[s][c-channel_window+1 .. c]).
    {
      Span<const float> channel_prefix_sums = tmp_b[step_index]; // Input for this pass
      Span<float> final_2d_sum_output = tmp_a[step_index]; // Output for this pass (and the function)

      // Copy the first `channel_window` elements.
      if (channel_window > 0 && num_channels > 0) {
        std::memcpy(final_2d_sum_output.data, channel_prefix_sums.data,
                    std::min(channel_window, num_channels) * sizeof(float));
      }
      // Compute windowed sums for remaining channels.
      for (size_t channel_index = channel_window; channel_index < num_channels;
           ++channel_index) {
        final_2d_sum_output[channel_index] =
            channel_prefix_sums[channel_index] -
            channel_prefix_sums[channel_index - channel_window];
      }
    }
  }

  // Pass 3: Divide all 2D windowed sums by (step_window * channel_window)
  // to get the mean values.
  // Handle cases where window sizes are zero to avoid division by zero.
  if (step_window == 0 || channel_window == 0) {
    // If either window dimension is zero, the mean is ill-defined or should be zero.
    // Setting all to zero.
    for (size_t i = 0; i < num_steps * num_channels; ++i) {
      tmp_a.values[i] = 0.0f;
    }
  } else {
    const float reciprocal_window_area = 1.0f / (static_cast<float>(step_window) * static_cast<float>(channel_window));
    for (size_t step_index = 0; step_index < num_steps; ++step_index) {
      Span<float> result_data_for_step = tmp_a[step_index];
      for (size_t channel_index = 0; channel_index < num_channels;
          ++channel_index) {
        result_data_for_step[channel_index] *= reciprocal_window_area;
      }
    }
  }

  return tmp_a;
}

// Calculates a modified version of the Neural Structural Similarity (NSIM)
// metric between two spectrograms `a` and `b`. NSIM aims to compare structural
// similarity by looking at local intensity (mean), contrast (standard deviation),
// and structure (covariance) components.
//
// This implementation uses time-aligned spectrogram data based on `time_pairs`
// obtained from Dynamic Time Warping (DTW). It computes local statistics (mean,
// variance, covariance) using `WindowMean` over specified window sizes.
//
// The core NSIM formula is typically:
//   NSIM(x,y) = I(x,y) * C(x,y) * S(x,y)
//   I(x,y) = (2*mu_x*mu_y + C1) / (mu_x^2 + mu_y^2 + C1) (Intensity comparison)
//   C(x,y) = (2*sigma_x*sigma_y + C2) / (sigma_x^2 + sigma_y^2 + C2) (Contrast comparison)
//   S(x,y) = (sigma_xy + C3) / (sigma_x*sigma_y + C3) (Structure comparison)
// This implementation has some deviations and ad-hoc modifications as noted
// in the code comments (e.g., clamping, L1 diff addition).
//
// Reference for original NSIM concepts:
// The paper linked (https://doi.org/10.1016/j.specom.2011.09.004) seems more
// related to feature extraction for ASR rather than being the canonical SSIM/NSIM
// source. Standard SSIM/MS-SSIM references might be more appropriate for the
// general concept, though this is a "nonstandard version".
// A common SSIM reference: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli,
// E. P. (2004). Image quality assessment: from error visibility to structural
// similarity. IEEE transactions on image processing, 13(4), 600-612.
//
// Args:
//   a: The first spectrogram.
//   b: The second spectrogram.
//   time_pairs: A vector of pairs `(index_a, index_b)` mapping time steps from
//               spectrogram `a` to spectrogram `b` as determined by DTW.
//               The NSIM calculation will be performed on these aligned pairs.
//   step_window: The size of the sliding window along the time step axis for
//                calculating local statistics (mean, variance, covariance).
//   channel_window: The size of the sliding window along the channel (frequency)
//                   axis for calculating local statistics.
//
// Returns:
//   A float value between 0.0 and 1.0 representing the similarity, where 1.0
//   indicates high similarity.
float NSIM(const Spectrogram& a, const Spectrogram& b,
           const std::vector<std::pair<size_t, size_t>>& time_pairs,
           size_t step_window, size_t channel_window) {
  assert_eq(a.num_dims, b.num_dims); // Spectrograms must have same number of frequency channels.
  const size_t num_channels = a.num_dims;
  // Number of steps to compare is determined by the length of the DTW path.
  const size_t num_steps = time_pairs.size();

  // Calculate local means for spectrogram 'a' along the DTW path.
  const Spectrogram mean_a =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   // `step_index` here refers to an index along the `time_pairs` path.
                   // `time_pairs[step_index].first` gives the actual step in spectrogram `a`.
                   return a[time_pairs[step_index].first][channel_index];
                 });

  // Calculate local means for spectrogram 'b' along the DTW path.
  const Spectrogram mean_b =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   return b[time_pairs[step_index].second][channel_index];
                 });

  // Calculate local variances for spectrogram 'a'.
  // Note: This computes (value - mean_window_at_value)^2. The mean used for
  // each point `(s,c)` is `mean_a[s][c]`, which is the mean of the window
  // *ending* at `(s,c)`.
  const Spectrogram var_a = WindowMean(
      num_steps, num_channels, step_window, channel_window,
      [&](size_t step_index, size_t channel_index) {
        const float delta = a[time_pairs[step_index].first][channel_index] -
                            mean_a[step_index][channel_index]; // mean_a[step_index] is window mean at this point
        return delta * delta;
      });

  // Calculate local variances for spectrogram 'b'.
  const Spectrogram var_b = WindowMean(
      num_steps, num_channels, step_window, channel_window,
      [&](size_t step_index, size_t channel_index) {
        const float delta = b[time_pairs[step_index].second][channel_index] -
                            mean_b[step_index][channel_index];
        return delta * delta;
      });

  // Calculate local covariances between 'a' and 'b'.
  const Spectrogram cov = WindowMean(
      num_steps, num_channels, step_window, channel_window,
      [&](size_t step_index, size_t channel_index) {
        const float delta_a = a[time_pairs[step_index].first][channel_index] -
                              mean_a[step_index][channel_index];
        const float delta_b = b[time_pairs[step_index].second][channel_index] -
                              mean_b[step_index][channel_index];
        return delta_a * delta_b;
      });

  // These constants are part of the "nsim-inspired ad hoc aggregation".
  // Their specific values and roles are empirically derived to optimize
  // performance on a specific test set.
  // C1, C3 are stabilization constants, similar to those in standard SSIM.
  // C4, C5, C6, C7, C8 and P0, P1, P2 are for ad-hoc modifications:
  // - Clamping of structure value (using C8)
  // - Power functions (P0, P1, P2) for tuning sensitivity
  // - Adding a scaled L1 difference component (using C7)
  // These deviate from standard SSIM/NSIM formulations.
  const float C1 = 28.341082593304403;    // For intensity component, similar to (K1*L)^2 in SSIM.
  const float C3 = 1.6705576583956854;    // For structure/covariance component, similar to C3 in SSIM, or (K2*L)^2 / 2.
  const float C4 = 5.5778917823818053e-05; // Offset for structure component before power.
  const float C5 = 2.5568733818058373e-07; // Offset for structure component after first power, before second power.
  const float C6 = 3.510912492638396e-08; // Final offset for structure component.
  const float C7 = 2.4720299934548813e-07; // Scaling factor for the L1 difference term.
  const float C8 = 0.63782947152876834;    // Clamping threshold for the base structure value.
  const float P0 = 0.84007774751632736;    // Exponent for the intensity component.
  const float P1 = 1.7336006381611897;    // First exponent for the structure component.
  const float P2 = 0.19488367705873288;    // Second exponent for the structure component.

  double nsim_sum = 0.0; // Use double for sum to maintain precision before final division.
  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      const float ma = mean_a[step_index][channel_index];
      const float mb = mean_b[step_index][channel_index];
      // Standard deviation is sqrt of variance. Add small epsilon to prevent sqrt(0) or issues with tiny variances.
      const float std_a = std::sqrt(std::max(0.0f, var_a[step_index][channel_index]));
      const float std_b = std::sqrt(std::max(0.0f, var_b[step_index][channel_index]));
      const float cab = cov[step_index][channel_index];

      // Intensity component: (2 * sqrt(ma*mb) + C1) / (|ma| + |mb| + C1)
      // Note: Original SSIM uses (2*ma*mb + C1) / (ma^2 + mb^2 + C1).
      // This version uses sqrt(ma*mb) and |ma|+|mb|, which is different.
      // It also applies a power P0.
      const float intensity_numerator = 2 * std::sqrt(std::max(0.0f, ma * mb)) + C1;
      const float intensity_denominator = std::abs(ma) + std::abs(mb) + C1;
      const float intensity_val = intensity_denominator < 1e-9f ? 1.0f : intensity_numerator / intensity_denominator;
      const float intensity_component = std::pow(std::max(0.0f, intensity_val), P0);

      // Structure component: (cov_ab + C3) / (std_a * std_b + C3)
      // This is similar to the SSIM structure component.
      const float structure_numerator = cab + C3;
      const float structure_denominator = std_a * std_b + C3;
      const float structure_base = structure_denominator < 1e-9f ? 1.0f : structure_numerator / structure_denominator;

      // Ad-hoc modification: clamping and multiple power transformations.
      const float structure_clamped = structure_base < C8 ? C8 : structure_base;
      const float structure_powered_1 = std::pow(structure_clamped + C4, P1);
      const float structure_component = std::pow(std::max(0.0f, structure_powered_1 + C5), P2) + C6;

      // Combined NSIM for this point (intensity * structure).
      // Note: Standard SSIM often includes a contrast component as well.
      // Here, contrast seems implicitly handled or combined differently.
      const float nsim_local = intensity_component * structure_component;

      // Ad-hoc modification: adding a scaled L1 difference.
      const float val_a = a[time_pairs[step_index].first][channel_index];
      const float val_b = b[time_pairs[step_index].second][channel_index];
      const float l1_diff_term = C7 * std::abs(val_a - val_b);
      const float nsim_modified_local = nsim_local + l1_diff_term;

      nsim_sum += nsim_modified_local;
    }
  }
  // Average the sum over all points and clamp to [0, 1].
  if (num_steps == 0 || num_channels == 0) return 0.0f; // Avoid division by zero.
  return std::clamp<float>(
      static_cast<float>(nsim_sum) / static_cast<float>(num_steps * num_channels), 0.0f, 1.0f);
}

// Represents the accumulated cost matrix used in Dynamic Time Warping (DTW).
// The matrix stores the minimum cumulative cost of aligning subsequences of
// two time series (spectrograms in this context).
// `values[i][j]` (conceptually, stored flat) holds the cost of the optimal
// alignment between the first `i` elements of sequence A and the first `j`
// elements of sequence B.
//
// The matrix is initialized with infinity for all cells except for `cost[0][0] = 0`,
// representing zero cost to align two empty sequences. The DTW algorithm then
// fills this matrix.
struct CostMatrix {
  // Retrieves the cost at a given pair of steps from the flattened `values` vector.
  // Args:
  //   step_a: Index for the first spectrogram's time step.
  //   step_b: Index for the second spectrogram's time step.
  // Returns: The accumulated cost up to (step_a, step_b).
  const double get(size_t step_a, size_t step_b) {
    return values[step_a * steps_b + step_b];
  }

  // Sets the cost at a given pair of steps in the flattened `values` vector.
  // Args:
  //   step_a: Index for the first spectrogram's time step.
  //   step_b: Index for the second spectrogram's time step.
  //   value: The accumulated cost to set.
  void set(size_t step_a, size_t step_b, double value) {
    values[step_a * steps_b + step_b] = value;
  }

  // Constructor for CostMatrix.
  // Initializes a matrix of size `steps_a` x `steps_b` with all values set to
  // positive infinity, except for `cost[0][0]`, which is set to 0.
  // Args:
  //   steps_a: The number of time steps in the first spectrogram.
  //   steps_b: The number of time steps in the second spectrogram.
  CostMatrix(size_t steps_a, size_t steps_b)
      : steps_a(steps_a),
        steps_b(steps_b),
        values(std::vector<double>(steps_a * steps_b, // Flat storage
                                   std::numeric_limits<double>::max())) {
    if (steps_a > 0 && steps_b > 0) { // Ensure valid access to (0,0)
      set(0, 0, 0); // Cost of aligning two empty prefixes is 0.
    }
  }

  // Number of steps in the first sequence/spectrogram.
  size_t steps_a;
  // Number of steps in the second sequence/spectrogram.
  size_t steps_b;
  // Flattened vector storing the cost values. Access using `get` and `set`,
  // or manually via `values[row * num_cols + col]`.
  std::vector<double> values;
};

// Computes a distance metric between two spectrogram frames (time steps).
// This function calculates the sum of squared differences between the feature
// vectors (frequency channels) of two spectrogram frames (`a[step_a]` and
// `b[step_b]`) and then raises this sum to a power (0.23289...).
// This powered Euclidean distance (or a variation thereof) serves as the local
// cost `d(i, j)` for aligning frame `i` of spectrogram `a` with frame `j` of
// spectrogram `b` in the Dynamic Time Warping (DTW) algorithm.
//
// Args:
//   a: The first spectrogram.
//   b: The second spectrogram.
//   step_a: The time step index in the first spectrogram.
//   step_b: The time step index in the second spectrogram.
//
// Returns:
//   The computed distance (local cost) between the two specified spectrogram frames.
double delta_norm(const Spectrogram& a, const Spectrogram& b, size_t step_a,
                  size_t step_b) {
  Span<const float> dims_a = a[step_a]; // Feature vector for frame step_a from spectrogram a
  Span<const float> dims_b = b[step_b]; // Feature vector for frame step_b from spectrogram b
  assert_eq(dims_a.size, dims_b.size); // Ensure feature vectors have the same dimensionality.

  double sum_sq_diff = 0;
  for (size_t index = 0; index < dims_a.size; index++) {
    float delta = dims_a[index] - dims_b[index];
    sum_sq_diff += delta * delta;
  }
  // The exponent 0.23289... is unusual. Standard Euclidean distance would be sqrt(sum_sq_diff)
  // (i.e., power 0.5). This specific power might be empirically chosen to tune
  // the sensitivity of the distance metric for this particular application.
  // It makes the metric less sensitive to large differences than Euclidean distance,
  // and more sensitive than Manhattan distance (power 1 on absolute differences).
  return std::pow(sum_sq_diff, 0.23289303544689094);
}

// Computes Dynamic Time Warping (DTW) between two spectrograms `spec_a` and
// `spec_b` to find the optimal alignment path between them.
// DTW is an algorithm for measuring similarity between two temporal sequences
// that may vary in speed or timing.
// See: https://en.wikipedia.org/wiki/Dynamic_time_warping
//
// The function performs two main steps:
// 1. Filling the cost matrix: It iteratively computes the minimum accumulated
//    cost to align subsequences of `spec_a` and `spec_b`. The local cost for
//    aligning `spec_a[i]` and `spec_b[j]` is given by `delta_norm`.
//    The recurrence relation is:
//    `cost[i][j] = local_cost(i,j) + min(cost[i-1][j-1] * kMul00_weight,  // Match/Substitution (weighted)
//                                       cost[i-1][j],                     // Deletion in spec_b (gap in spec_a)
//                                       cost[i][j-1])`                    // Insertion in spec_b (gap in spec_b)
//    Note: The `kMul00` factor weights the diagonal match/substitution step.
// 2. Backtracking: It traces back the path of minimum cost from `cost[N-1][M-1]`
//    to `cost[0][0]` to find the sequence of aligned time step pairs.
//
// Args:
//   spec_a: The first spectrogram.
//   spec_b: The second spectrogram.
//
// Returns:
//   A vector of pairs `(index_a, index_b)`, representing the optimal alignment
//   path. `index_a` is a time step index in `spec_a`, and `index_b` is its
//   corresponding aligned time step index in `spec_b`.
std::vector<std::pair<size_t, size_t>> DTW(const Spectrogram& spec_a,
                                           const Spectrogram& spec_b) {
  // Ensure both spectrograms have the same number of frequency dimensions.
  assert_eq(spec_a.num_dims, spec_b.num_dims);

  if (spec_a.num_steps == 0 || spec_b.num_steps == 0) {
    return {}; // Cannot compute DTW for empty spectrograms.
  }

  CostMatrix cost_matrix(spec_a.num_steps, spec_b.num_steps);

  // Empirically derived multiplier for the cost of a diagonal step (match/substitution).
  // Values less than 1.0 penalize diagonal moves less than insertions/deletions,
  // encouraging matches.
  static const double kMul00 = 0.98585952515276176;

  // Fill the cost matrix.
  // Start from (1,1) because (0,0) is already 0, and rows/columns starting
  // with 0 are boundary conditions (aligning with an empty prefix).
  // Standard DTW often initializes the first row and column considering cumulative
  // local costs. This implementation implicitly handles this by starting loops at 1
  // and relying on cost_matrix.get() for out-of-bounds-like access if not careful,
  // but `CostMatrix` is initialized with max values.
  // The `cost_matrix.get(0,0)` is 0. Other `get(0,j)` or `get(i,0)` will be infinity
  // unless explicitly set. Let's verify the boundary conditions.
  // For i=0: cost_matrix(0, j) = cost_matrix(0, j-1) + delta_norm(0,j)
  // For j=0: cost_matrix(i, 0) = cost_matrix(i-1, 0) + delta_norm(i,0)
  // This is typically how the first row/column are filled.

  // Initialize first row
  for (size_t j = 1; j < spec_b.num_steps; ++j) {
    cost_matrix.set(0, j, cost_matrix.get(0, j - 1) + delta_norm(spec_a, spec_b, 0, j));
  }
  // Initialize first column
  for (size_t i = 1; i < spec_a.num_steps; ++i) {
    cost_matrix.set(i, 0, cost_matrix.get(i - 1, 0) + delta_norm(spec_a, spec_b, i, 0));
  }

  // Fill the rest of the cost matrix.
  for (size_t spec_a_index = 1; spec_a_index < spec_a.num_steps;
       ++spec_a_index) {
    for (size_t spec_b_index = 1; spec_b_index < spec_b.num_steps;
         ++spec_b_index) {
      const double local_cost =
          delta_norm(spec_a, spec_b, spec_a_index, spec_b_index);

      // Cost from a diagonal step (match/substitution)
      const double match_cost =
          cost_matrix.get(spec_a_index - 1, spec_b_index - 1);
      // Cost from a vertical step (deletion in spec_b / gap in spec_a)
      const double deletion_cost = cost_matrix.get(spec_a_index - 1, spec_b_index);
      // Cost from a horizontal step (insertion in spec_b / gap in spec_b)
      const double insertion_cost = cost_matrix.get(spec_a_index, spec_b_index - 1);

      // The cost of moving diagonally is weighted by kMul00.
      // The cost of moving horizontally or vertically is the full local_cost.
      const double min_prev_cost =
          std::min({match_cost * kMul00, deletion_cost, insertion_cost});

      cost_matrix.set(spec_a_index, spec_b_index, local_cost + min_prev_cost);
    }
  }

  // Backtrack from the end to the beginning to find the optimal path.
  std::vector<std::pair<size_t, size_t>> path;
  size_t current_a = spec_a.num_steps - 1;
  size_t current_b = spec_b.num_steps - 1;
  path.push_back({current_a, current_b});

  while (current_a > 0 || current_b > 0) {
    if (current_a == 0) { // Can only move left
      current_b--;
    } else if (current_b == 0) { // Can only move up
      current_a--;
    } else {
      const double match_cost = cost_matrix.get(current_a - 1, current_b - 1);
      const double deletion_cost = cost_matrix.get(current_a - 1, current_b); // Move up
      const double insertion_cost = cost_matrix.get(current_a, current_b - 1); // Move left

      // Choose the step that led to the minimum cost at (current_a, current_b).
      // Note: The kMul00 factor is applied during matrix filling, not here in backtracking.
      // We are looking for the predecessor that has the minimum *accumulated* cost.
      if (match_cost <= deletion_cost && match_cost <= insertion_cost) {
        current_a--;
        current_b--;
      } else if (deletion_cost < match_cost && deletion_cost < insertion_cost) { // Corrected logic for min
        current_a--;
      } else if (insertion_cost < match_cost && insertion_cost < deletion_cost) {
        current_b--;
      } else {
        // Handle ties: prefer diagonal, then up, then left (arbitrary but consistent)
        if (match_cost <= deletion_cost && match_cost <= insertion_cost) {
             current_a--; current_b--;
        } else if (deletion_cost <= insertion_cost) {
            current_a--;
        } else {
            current_b--;
        }
      }
    }
    path.push_back({current_a, current_b});
  }
  std::reverse(path.begin(), path.end()); // Path was built backwards.
  return path;
}

// Expected sample rate for input audio signals processed by Zimtohrli.
constexpr float kSampleRate = 48000;

// The Zimtohrli struct encapsulates the entire process for computing
// perceptual audio similarity. It includes methods for:
// 1. Analyzing an audio signal to produce a Zimtohrli spectrogram.
// 2. Calculating the perceptual distance between two such spectrograms.
//
// The process involves several stages:
// - Modeling of the human ear's response (Resonator).
// - Time-frequency analysis using a filterbank of Rotators.
// - Conversion to a loudness scale.
// - Dynamic Time Warping (DTW) to align spectrograms.
// - Neural Structural Similarity (NSIM) to compare aligned spectrograms.
//
// Parameters within this struct control aspects of this pipeline, such as
// window sizes for NSIM and rates related to perceptual processing.
struct Zimtohrli {
  // Analyzes an input audio signal and populates a pre-allocated Spectrogram.
  //
  // Args:
  //   signal: A Span of floats representing the input audio signal samples.
  //           The signal is expected to be mono and at `kSampleRate`.
  //   spectrogram: A pre-allocated Spectrogram object that will be filled with
  //                the analysis results. Its `num_dims` must match `kNumRotators`.
  //                Its `num_steps` determines the temporal resolution of the analysis.
  void Analyze(Span<const float> signal, Spectrogram& spectrogram) const {
    assert_eq(spectrogram.num_dims, kNumRotators); // Ensure output dimensions match filterbank.
    Rotators rots; // Create a Rotators instance for filterbank processing.
    // Calculate the downsampling factor. This is the number of input samples
    // that will contribute to a single time step in the spectrogram.
    // It ensures that the entire signal is processed to fill the spectrogram's num_steps.
    // Note: integer division truncates; if signal.size is not a multiple of num_steps,
    // the last few samples might be handled differently by FilterAndDownsample depending
    // on its exact boundary logic.
    size_t downsample_factor = signal.size / spectrogram.num_steps;
    if (downsample_factor == 0 && signal.size > 0) { // Avoid division by zero if num_steps is very large
        downsample_factor = 1; // Process at least one sample per step if possible
    } else if (signal.size == 0 && spectrogram.num_steps > 0) {
        // If signal is empty but spectrogram steps are requested, this will result in issues.
        // FilterAndDownsample might try to read from empty signal or produce empty output.
        // Consider returning early or throwing an error for empty signal if num_steps > 0.
        // For now, let FilterAndDownsample handle it (it might zero-fill or behave unexpectedly).
    }


    rots.FilterAndDownsample(signal.data, signal.size, spectrogram.values.get(),
                             spectrogram.num_steps, spectrogram.num_dims,
                             downsample_factor);
  }

  // Analyzes an input audio signal and returns a new Spectrogram.
  // This is a convenience overload that calculates the required number of
  // spectrogram steps and allocates the Spectrogram object.
  //
  // Args:
  //   signal: A Span of floats representing the input audio signal samples.
  //
  // Returns:
  //   A Spectrogram object containing the analysis results.
  Spectrogram Analyze(Span<const float> signal) const {
    // Determine the number of time steps for the spectrogram based on the signal length
    // and the perceptual sample rate.
    Spectrogram spec(SpectrogramSteps(signal.size), kNumRotators);
    Analyze(signal, spec); // Call the other Analyze method to fill the spectrogram.
    return spec;
  }

  // Calculates the number of time steps a spectrogram should have for a given
  // number of input audio samples. This is based on the desired
  // `perceptual_sample_rate`.
  //
  // Args:
  //   num_samples: The number of samples in the input audio signal.
  //
  // Returns:
  //   The calculated number of spectrogram time steps.
  size_t SpectrogramSteps(size_t num_samples) const {
    if (kSampleRate == 0) return 0; // Avoid division by zero
    // The number of steps is (num_samples / kSampleRate) * perceptual_sample_rate, rounded up.
    return static_cast<size_t>(std::ceil(static_cast<float>(num_samples) *
                                         perceptual_sample_rate / kSampleRate));
  }

  // Calculates the perceptual distance between two Zimtohrli spectrograms.
  // The distance is 1.0 - NSIM_score, where NSIM_score is between 0 and 1.
  // Thus, distance is also between 0 (identical) and 1 (very different).
  // Spectrogram `b` is rescaled to match the maximum absolute value of `a`
  // before comparison. This is a form of loudness normalization.
  //
  // Args:
  //   spectrogram_a: The first spectrogram.
  //   spectrogram_b: The second spectrogram. This spectrogram may be modified
  //                  (rescaled) by this function.
  //
  // Returns:
  //   A float representing the distance (0.0 to 1.0).
  float Distance(const Spectrogram& spectrogram_a,
                 Spectrogram& spectrogram_b) const {
    assert_eq(spectrogram_a.num_dims, spectrogram_b.num_dims); // Must have same feature dimensions.

    const float max_a = spectrogram_a.max();
    const float max_b = spectrogram_b.max();

    // Rescale spectrogram_b to match the dynamic range of spectrogram_a.
    // This helps to normalize for overall loudness differences.
    // Avoid division by zero if max_b is zero.
    if (max_b != 0 && std::abs(max_a - max_b) > 1e-9f) { // Check if rescaling is needed and safe
      spectrogram_b.rescale(max_a / max_b);
    } else if (max_b == 0 && max_a != 0) {
      // If b is all zeros and a is not, they are different.
      // NSIM might handle this, but direct rescaling is problematic.
      // The NSIM will likely be low.
    }


    // Align the spectrograms in time using Dynamic Time Warping.
    std::vector<std::pair<size_t, size_t>> time_pairs;
    time_pairs = DTW(spectrogram_a, spectrogram_b);

    // If DTW returns an empty path (e.g., one spectrogram was empty),
    // similarity is zero, so distance is one.
    if (time_pairs.empty()) {
        return 1.0f;
    }

    // Compute the Neural Structural Similarity (NSIM) on the time-aligned spectrograms.
    float nsim_score = NSIM(spectrogram_a, spectrogram_b, time_pairs,
                            nsim_step_window, nsim_channel_window);

    // Distance is 1 - similarity.
    return 1.0f - nsim_score;
  }

  // Window size in time steps (along the perceptual_sample_rate axis) used for
  // calculating local statistics (mean, variance, covariance) in the NSIM function.
  size_t nsim_step_window = 6;

  // Window size in frequency channels (bands) used for calculating local
  // statistics in the NSIM function.
  size_t nsim_channel_window = 5;

  // "The clock frequency of the brain?!" - A comment from the original code.
  // This likely refers to a characteristic frequency related to auditory processing
  // or perception, possibly related to the rate of auditory "snapshots".
  // Used to determine `samples_per_perceptual_block`.
  float high_gamma_band = 84.0; // Hz

  // Number of audio samples at `kSampleRate` that correspond to one "perceptual block"
  // or "auditory snapshot", derived from `high_gamma_band`.
  // E.g., 48000 Hz / 84 Hz approx 571 samples.
  int samples_per_perceptual_block = int(kSampleRate / high_gamma_band);

  // The effective sample rate at which the auditory system is considered to
  // process distinct temporal events. This determines the time resolution of
  // the Zimtohrli spectrogram before DTW.
  // It's `kSampleRate / samples_per_perceptual_block`, which simplifies to `high_gamma_band`.
  float perceptual_sample_rate = kSampleRate / samples_per_perceptual_block; // Effectively high_gamma_band

  // The reference Sound Pressure Level (dB SPL) for a sinusoidal signal with an
  // amplitude of 1.0 (full scale). This is used for calibration or interpretation
  // of loudness levels, though not directly used in the core distance calculation path shown.
  float full_scale_sine_db = 78.3;
};

}  // namespace

}  // namespace zimtohrli

#endif  // CPP_ZIMT_ZIMTOHRLI_H_
