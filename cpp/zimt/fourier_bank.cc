#include "fourier_bank.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_split.h"
#include "sndfile.hh"

namespace tabuli {

float SimpleDb(float energy) {
  // ideally 78.3 db
  static const float full_scale_sine_db = 75.27901963526045;
  static const float exp_full_scale_sine_db = exp(full_scale_sine_db);
  // epsilon, but the biggest one you saw (~4.95e23)
  static const float epsilon = 1.0033294789821357e-09 * exp_full_scale_sine_db;
  // kMul allows faster log instead of log10 below, incorporating multiplying by 10 for decibel.
  constexpr float kMul = 10.0/log(10);
  return kMul * log(energy + epsilon);
}

void FinalizeDb(hwy::AlignedNDArray<float, 2>& channels, float mul,
                size_t out_ix) {
  double masker = 0.0;
  static const double octaves_in_20_to_20000 = log(20000/20.)/log(2);
  static const double octaves_per_rot =
      octaves_in_20_to_20000 / float(kNumRotators - 1);
  static const double masker_step_per_octave_up_0 = 15.892019717473835;
  static const double masker_step_per_octave_up_1 = 21.852019717473834;
  static const double masker_step_per_octave_up_2 = 20.79201971747383;
  static const double masker_step_per_rot_up_0 = octaves_per_rot * masker_step_per_octave_up_0;
  static const double masker_step_per_rot_up_1 = octaves_per_rot * masker_step_per_octave_up_1;
  static const double masker_step_per_rot_up_2 = octaves_per_rot * masker_step_per_octave_up_2;
  static const double masker_gap_up = 19.140338374861235;
  static const float maskingStrengthUp = 0.1252262923615547;
  static const float up_blur = 0.8738593591692092;
  static const float fraction_up = 1.02;

  static const double masker_step_per_octave_down = 42.33972783112732;
  static const double masker_step_per_rot_down = octaves_per_rot * masker_step_per_octave_down;
  static const double masker_gap_down = 19.66099875393617;
  static const float maskingStrengthDown = 0.19329999999999992;
  static const float down_blur = 0.714425315233319;

  static const float min_limit = -11.397341001787765;
  static const float fraction_down = 1.02;
  // Scan frequencies from bottom to top, let lower frequencies to mask higher frequencies.
  // 'masker' maintains the masking envelope from one bin to next.
  for (int k = 0; k < kNumRotators; ++k) {
    float v = SimpleDb(mul * channels[{out_ix}][k]);
    if (v < min_limit) {
      v = min_limit;
    }
    float v2 = (1 - up_blur) * v2 + up_blur * v;
    if (k == 0) {
      v2 = v;
    }
    if (masker < v2) {
      masker = v2;
    }
    float mask = fraction_up * masker - masker_gap_up;
    if (v < mask) {
      v = maskingStrengthUp * mask + (1.0 - maskingStrengthUp) * v;
    }
    channels[{out_ix}][k] = v;
    if (3 * k < kNumRotators) {
      masker -= masker_step_per_rot_up_0;
    } else if (3 * k < 2 * kNumRotators) {
      masker -= masker_step_per_rot_up_1;
    } else {
      masker -= masker_step_per_rot_up_2;
    }
  }
  // Scan frequencies from top to bottom, let higher frequencies to mask lower frequencies.
  // 'masker' maintains the masking envelope from one bin to next.
  masker = 0.0;
  for (int k = kNumRotators - 1; k >= 0; --k) {
    float v = channels[{out_ix}][k];
    float v2 = (1 - down_blur) * v2 + down_blur * v;
    if (k == kNumRotators - 1) {
      v2 = v;
    }
    if (masker < v) {
      masker = v;
    }
    float mask = fraction_down * masker - masker_gap_down;
    if (v < mask) {
      v = maskingStrengthDown * mask + (1.0 - maskingStrengthDown) * v;
    }
    channels[{out_ix}][k] = v;
    masker -= masker_step_per_rot_down;
  }
}

void Rotators::FilterAndDownsample(hwy::Span<const float> signal,
                                   hwy::AlignedNDArray<float, 2>& channels,
                                   int downsampling) {
  float scaling_for_downsampling = 1.0f / downsampling;
  size_t out_ix = 0;
  for (int64_t ii = 0; ii < signal.size(); ii += downsampling) {
    OccasionallyRenormalize();
    for (int64_t zz = 0; zz < downsampling; ++zz) {
      int64_t input_ix = ii + zz;
      if (input_ix >= signal.size()) {
        if (out_ix < channels.shape()[0]) {
          FinalizeDb(channels, scaling_for_downsampling, out_ix);
        }
        if (out_ix != channels.shape()[0] - 1) {
          fprintf(stderr,
                  "strange thing #9831021 happened in FilterAndDownsample\n");
          abort();
        }
        return;
      }
      IncrementAll(signal[input_ix]);
      if (zz == 0) {
        for (int k = 0; k < kNumRotators; ++k) {
          float energy =
              channel[0].accu[4][k] * channel[0].accu[4][k] +
              channel[0].accu[5][k] * channel[0].accu[5][k];
          channels[{out_ix}][k] = energy;
        }
      } else {
        for (int k = 0; k < kNumRotators; ++k) {
          float energy =
              channel[0].accu[4][k] * channel[0].accu[4][k] +
              channel[0].accu[5][k] * channel[0].accu[5][k];
          channels[{out_ix}][k] += energy;
        }
      }
    }
    FinalizeDb(channels, scaling_for_downsampling, out_ix);
    ++out_ix;
    if (out_ix >= channels.shape()[0]) {
      return;
    }
  }
}

double CalculateBandwidth(double low, double mid, double high) {
  const double geo_mean_low = std::sqrt(low * mid);
  const double geo_mean_high = std::sqrt(mid * high);
  return std::abs(geo_mean_high - mid) + std::abs(mid - geo_mean_low);
}

Rotators::Rotators(int num_channels, std::vector<float> frequency,
                   std::vector<float> filter_gains, const float sample_rate) {
  channel.resize(num_channels);
  static const float kWindow = 0.9996028710680265;
  static const double kBandwidthMagic = 0.7328516996032982;
  for (int i = 0; i < kNumRotators; ++i) {
    // The parameter relates to the frequency shape overlap and window length
    // of triple leaking integrator.
    float bw = CalculateBandwidth(
        i == 0 ? frequency[1] : frequency[i - 1], frequency[i],
        i + 1 == kNumRotators ? frequency[i - 1] : frequency[i + 1]);
    window[i] = std::pow(kWindow, bw * kBandwidthMagic);
    float windowM1 = 1.0f - window[i];
    float f = frequency[i] * 2.0f * M_PI / sample_rate;
    static const float full_scale_sine_db = exp(75.27901963526045);
    const float gainer = 2.0f * sqrt(full_scale_sine_db);
    gain[i] = gainer * filter_gains[i] * pow(windowM1, 3.0);
    rot[0][i] = float(std::cos(f));
    rot[1][i] = float(-std::sin(f));
    rot[2][i] = gain[i];
    rot[3][i] = 0.0f;
  }
  rotator_frequency = frequency;
}

void Rotators::OccasionallyRenormalize() {
  for (int i = 0; i < kNumRotators; ++i) {
    float norm = gain[i] / sqrt(rot[2][i] * rot[2][i] + rot[3][i] * rot[3][i]);
    rot[2][i] *= norm;
    rot[3][i] *= norm;
  }
}

void Rotators::IncrementAll(float signal) {
  for (int i = 0; i < kNumRotators; i++) {
    const float tr = rot[0][i] * rot[2][i] - rot[1][i] * rot[3][i];
    const float tc = rot[0][i] * rot[3][i] + rot[1][i] * rot[2][i];
    rot[2][i] = tr;
    rot[3][i] = tc;
    const float w = window[i];
    for (int c = 0; c < 1; ++c) {
      channel[c].accu[0][i] *= w;
      channel[c].accu[1][i] *= w;
      channel[c].accu[2][i] *= w;
      channel[c].accu[3][i] *= w;
      channel[c].accu[4][i] *= w;
      channel[c].accu[5][i] *= w;
      channel[c].accu[2][i] += channel[c].accu[0][i];
      channel[c].accu[3][i] += channel[c].accu[1][i];
      channel[c].accu[4][i] += channel[c].accu[2][i];
      channel[c].accu[5][i] += channel[c].accu[3][i];
      channel[c].accu[0][i] += rot[2][i] * signal;
      channel[c].accu[1][i] += rot[3][i] * signal;
    }
  }
}

}  // namespace tabuli
