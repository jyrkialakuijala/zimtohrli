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

#include "visqol_model.h"

#include "absl/types/span.h"
#include "libsvm_nu_svr_model.h"

namespace zimtohrli {

absl::Span<const char> ViSQOLModel() {
  return absl::Span<char>(reinterpret_cast<char*>(visqol_model_bytes),
                          visqol_model_bytes_len);
}

}  // namespace zimtohrli