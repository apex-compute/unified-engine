module {
  func.func @swin_block(%0: tensor<96xbf16>, %1: tensor<96xbf16>, %2: tensor<288x96xbf16>, %3: tensor<288xbf16>, %4: tensor<96x96xbf16>, %5: tensor<96xbf16>, %6: tensor<96xbf16>, %7: tensor<96xbf16>, %8: tensor<384x96xbf16>, %9: tensor<384xbf16>, %10: tensor<96x384xbf16>, %11: tensor<96xbf16>, %12: tensor<1x56x56x96xbf16>) -> tensor<1x56x56x96xbf16> {
    %13 = ue.layer_norm %12, %0, %1 : (tensor<1x56x56x96xbf16>, tensor<96xbf16>, tensor<96xbf16>) -> tensor<1x56x56x96xbf16>
    %14 = ue.reshape %13 : (tensor<1x56x56x96xbf16>) -> tensor<1x8x7x8x7x96xbf16>
    %15 = ue.permute %14 {perm = [0, 1, 3, 2, 4, 5]} : (tensor<1x8x7x8x7x96xbf16>) -> tensor<1x8x8x7x7x96xbf16>
    %16 = ue.reshape %15 : (tensor<1x8x8x7x7x96xbf16>) -> tensor<64x49x96xbf16>
    %17 = ue.matmul %16, %2 : (tensor<64x49x96xbf16>, tensor<288x96xbf16>) -> tensor<64x49x288xbf16>
    %18 = ue.reshape %17 : (tensor<64x49x288xbf16>) -> tensor<64x49x3x3x32xbf16>
    %19 = ue.permute %18 {perm = [2, 0, 3, 1, 4]} : (tensor<64x49x3x3x32xbf16>) -> tensor<3x64x3x49x32xbf16>
    %20 = ue.select %19 {dim = 0 : i64, index = 0 : i64} : (tensor<3x64x3x49x32xbf16>) -> tensor<64x3x49x32xbf16>
    %21 = ue.select %19 {dim = 0 : i64, index = 1 : i64} : (tensor<3x64x3x49x32xbf16>) -> tensor<64x3x49x32xbf16>
    %22 = ue.select %19 {dim = 0 : i64, index = 2 : i64} : (tensor<3x64x3x49x32xbf16>) -> tensor<64x3x49x32xbf16>
    %23 = ue.flash_attn_pf %20 {causal = false, heads = 3 : i64} : (tensor<64x3x49x32xbf16>) -> tensor<64x3x49x32xbf16>
    %24 = ue.permute %23 {perm = [0, 2, 1, 3]} : (tensor<64x3x49x32xbf16>) -> tensor<64x49x3x32xbf16>
    %25 = ue.reshape %24 : (tensor<64x49x3x32xbf16>) -> tensor<64x49x96xbf16>
    %26 = ue.matmul %25, %4 : (tensor<64x49x96xbf16>, tensor<96x96xbf16>) -> tensor<64x49x96xbf16>
    %27 = ue.reshape %26 : (tensor<64x49x96xbf16>) -> tensor<1x8x8x7x7x96xbf16>
    %28 = ue.permute %27 {perm = [0, 1, 3, 2, 4, 5]} : (tensor<1x8x8x7x7x96xbf16>) -> tensor<1x8x7x8x7x96xbf16>
    %29 = ue.reshape %28 : (tensor<1x8x7x8x7x96xbf16>) -> tensor<1x56x56x96xbf16>
    %30 = ue.eltwise %12, %29 {mode = "add"} : (tensor<1x56x56x96xbf16>, tensor<1x56x56x96xbf16>) -> tensor<1x56x56x96xbf16>
    %31 = ue.layer_norm %30, %6, %7 : (tensor<1x56x56x96xbf16>, tensor<96xbf16>, tensor<96xbf16>) -> tensor<1x56x56x96xbf16>
    %32 = ue.matmul %31, %8 {act = "gelu"} : (tensor<1x56x56x96xbf16>, tensor<384x96xbf16>) -> tensor<1x56x56x384xbf16>
    %33 = ue.matmul %32, %10 : (tensor<1x56x56x384xbf16>, tensor<96x384xbf16>) -> tensor<1x56x56x96xbf16>
    %34 = ue.eltwise %30, %33 {mode = "add"} : (tensor<1x56x56x96xbf16>, tensor<1x56x56x96xbf16>) -> tensor<1x56x56x96xbf16>
    return %34 : tensor<1x56x56x96xbf16>
  }
}
