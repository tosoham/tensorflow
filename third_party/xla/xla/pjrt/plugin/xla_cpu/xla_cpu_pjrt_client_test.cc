/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"

#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {

TEST(XlaCpuPjrtClientTest, GetXlaPjrtCpuClient) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetXlaPjrtCpuClient(CpuClientOptions()));
  EXPECT_EQ(client->platform_name(), "cpu");
}

}  // namespace xla
