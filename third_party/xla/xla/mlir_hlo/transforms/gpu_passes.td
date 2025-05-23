/* Copyright 2022 The OpenXLA Authors.

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

#ifndef MLIR_HLO_TRANSFORMS_GPU_PASSES
#define MLIR_HLO_TRANSFORMS_GPU_PASSES

include "mlir/Pass/PassBase.td"

def GpuKernelToNVVMPass : Pass<"gpu-kernel-to-nvvm", "gpu::GPUModuleOp"> {
  let summary = "Pass to transform a gpu module to nvvm.";
  let dependentDialects = ["LLVM::LLVMDialect", "NVVM::NVVMDialect"];
  let options = [
      Option<"useBarePtrCallConv", "use-bare-ptr-call-conv", "bool",
             /*default=*/"false", "Use bare pointer memref to llvm lowering">,
  ];
  let constructor = "createGpuKernelToNvvmPass()";
}

def GpuKernelToROCDLPass : Pass<"gpu-kernel-to-rocdl", "gpu::GPUModuleOp"> {
  let summary = "Pass to transform a gpu module to rocdl.";
  let dependentDialects = ["LLVM::LLVMDialect", "ROCDL::ROCDLDialect"];
  let constructor = "createGpuKernelToRocdlPass()";
  let options = [
    Option<"chipset", "chipset", "std::string",
           /*default=*/"\"gfx000\"",
           "Chipset that these operations will run on">,
  ];
}

#endif // MLIR_HLO_TRANSFORMS_GPU_PASSES
