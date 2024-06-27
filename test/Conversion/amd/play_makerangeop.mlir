// triton-opt -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm

#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  tt.func @global_store_vec1(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<16x16xf16, #mma>, %arg2: i32 {tt.divisibility = 16 : i32}) {
    %14 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %93 = tt.expand_dims %14 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<16x1xi32, #mma>
    %96 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>, #mma>
    %97 = tt.addptr %96, %93 : tensor<16x1x!tt.ptr<f16>, #mma>, tensor<16x1xi32, #mma>
    %99 = tt.broadcast %97 : tensor<16x1x!tt.ptr<f16>, #mma> -> tensor<16x16x!tt.ptr<f16>, #mma>
    tt.store %99, %arg1 : tensor<16x16x!tt.ptr<f16>, #mma>
    tt.return
  }
}
