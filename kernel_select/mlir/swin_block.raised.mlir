module {
  func.func @swin_block(%arg0: tensor<96xbf16>, %arg1: tensor<96xbf16>, %arg2: tensor<288x96xbf16>, %arg3: tensor<288xbf16>, %arg4: tensor<96x96xbf16>, %arg5: tensor<96xbf16>, %arg6: tensor<96xbf16>, %arg7: tensor<96xbf16>, %arg8: tensor<384x96xbf16>, %arg9: tensor<384xbf16>, %arg10: tensor<96x384xbf16>, %arg11: tensor<96xbf16>, %arg12: tensor<1x56x56x96xbf16>) -> tensor<1x56x56x96xbf16> {
    %0 = ue.layer_norm %arg12, %arg0, %arg1 : (tensor<1x56x56x96xbf16>, tensor<96xbf16>, tensor<96xbf16>) -> tensor<1x56x56x96xbf16>
    %1 = ue.window_partition %0 {window_size = 7 : i64} : (tensor<1x56x56x96xbf16>) -> tensor<64x49x96xbf16>
    %2 = ue.matmul %1, %arg2 : (tensor<64x49x96xbf16>, tensor<288x96xbf16>) -> tensor<64x49x288xbf16>
    %3 = ue.reshape %2 : (tensor<64x49x288xbf16>) -> tensor<64x49x3x3x32xbf16>
    %4 = ue.permute %3 {perm = [2, 0, 3, 1, 4]} : (tensor<64x49x3x3x32xbf16>) -> tensor<3x64x3x49x32xbf16>
    %5 = ue.select %4 {dim = 0 : i64, index = 0 : i64} : (tensor<3x64x3x49x32xbf16>) -> tensor<64x3x49x32xbf16>
    %6 = ue.select %4 {dim = 0 : i64, index = 1 : i64} : (tensor<3x64x3x49x32xbf16>) -> tensor<64x3x49x32xbf16>
    %7 = ue.select %4 {dim = 0 : i64, index = 2 : i64} : (tensor<3x64x3x49x32xbf16>) -> tensor<64x3x49x32xbf16>
    %8 = ue.flash_attn_pf %5 {heads = 3 : i64} : (tensor<64x3x49x32xbf16>) -> tensor<64x3x49x32xbf16>
    %9 = ue.permute %8 {perm = [0, 2, 1, 3]} : (tensor<64x3x49x32xbf16>) -> tensor<64x49x3x32xbf16>
    %10 = ue.reshape %9 : (tensor<64x49x3x32xbf16>) -> tensor<64x49x96xbf16>
    %11 = ue.matmul %10, %arg4 : (tensor<64x49x96xbf16>, tensor<96x96xbf16>) -> tensor<64x49x96xbf16>
    %12 = ue.window_reverse %11 {window_size = 7 : i64} : (tensor<64x49x96xbf16>) -> tensor<1x56x56x96xbf16>
    %13 = ue.eltwise %arg12, %12 : (tensor<1x56x56x96xbf16>, tensor<1x56x56x96xbf16>) -> tensor<1x56x56x96xbf16>
    %14 = ue.layer_norm %13, %arg6, %arg7 : (tensor<1x56x56x96xbf16>, tensor<96xbf16>, tensor<96xbf16>) -> tensor<1x56x56x96xbf16>
    %15 = ue.matmul %14, %arg8 {act = "gelu"} : (tensor<1x56x56x96xbf16>, tensor<384x96xbf16>) -> tensor<1x56x56x384xbf16>
    %16 = ue.matmul %15, %arg10 : (tensor<1x56x56x384xbf16>, tensor<96x384xbf16>) -> tensor<1x56x56x96xbf16>
    %17 = ue.eltwise %13, %16 : (tensor<1x56x56x96xbf16>, tensor<1x56x56x96xbf16>) -> tensor<1x56x56x96xbf16>
    return %17 : tensor<1x56x56x96xbf16>
  }
}

