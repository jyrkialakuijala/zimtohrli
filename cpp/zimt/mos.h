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

#ifndef CPP_ZIMT_MOS_H_
#define CPP_ZIMT_MOS_H_

namespace zimtohrli {

// Returns a _very_approximate_ mean opinion score based on the
// provided Zimtohrli distance.
// This is calibrated using default settings of v0.1.5, with a
// minimum channel bandwidth (zimtohrli::Cam.minimum_bandwidth_hz)
// of 5Hz and perceptual sample rate
// (zimtohrli::Distance(..., perceptual_sample_rate, ...) of 100Hz.
float MOSFromZimtohrli(float zimtohrli_distance);

}  // namespace zimtohrli

#endif  // CPP_ZIMT_MOS_H_