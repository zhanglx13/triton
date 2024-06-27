
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  tt.func @global_store_vec1(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<16x16xf16, #mma>, %arg2: i32 {tt.divisibility = 16 : i32}) {
    // offs_am[:, None]
    %14 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    // offs_an[None, :]
    %15 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>

    %93 = tt.expand_dims %14 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<16x1xi32, #mma>

    %94 = tt.splat %arg2 : i32 -> tensor<16x1xi32, #mma>
    %95 = arith.muli %94, %93 : tensor<16x1xi32, #mma>

    %96 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>, #mma>
    %97 = tt.addptr %96, %95 : tensor<16x1x!tt.ptr<f16>, #mma>, tensor<16x1xi32, #mma>

    %98 = tt.expand_dims %15 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x16xi32, #mma>
    %99 = tt.broadcast %97 : tensor<16x1x!tt.ptr<f16>, #mma> -> tensor<16x16x!tt.ptr<f16>, #mma>
    %100 = tt.broadcast %98 : tensor<1x16xi32, #mma> -> tensor<16x16xi32, #mma>
    %101 = tt.addptr %99, %100 : tensor<16x16x!tt.ptr<f16>, #mma>, tensor<16x16xi32, #mma>
    tt.store %101, %arg1 : tensor<16x16x!tt.ptr<f16>, #mma>
    tt.return
  }
}
