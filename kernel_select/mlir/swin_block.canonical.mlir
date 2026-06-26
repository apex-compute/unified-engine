module {
  func.func @swin_block(%arg0: tensor<3136x96xbf16>, %arg1: tensor<96xbf16>, %arg2: tensor<96xbf16>, %arg3: tensor<288x96xbf16>, %arg4: tensor<96x96xbf16>, %arg5: tensor<96xbf16>, %arg6: tensor<96xbf16>, %arg7: tensor<384x96xbf16>, %arg8: tensor<96x384xbf16>) -> tensor<3136x96xbf16> {
    %0 = ue.layer_norm %arg0, %arg1, %arg2 : (tensor<3136x96xbf16>, tensor<96xbf16>, tensor<96xbf16>) -> tensor<3136x96xbf16>
    %1 = ue.window_partition %0 {window_size = 7 : i64} : (tensor<3136x96xbf16>) -> tensor<3136x96xbf16>
    %2 = ue.matmul %1, %arg3 : (tensor<3136x96xbf16>, tensor<288x96xbf16>) -> tensor<3136x288xbf16>
    %3 = ue.flash_attn_pf %2 {heads = 3 : i64} : (tensor<3136x288xbf16>) -> tensor<3136x96xbf16>
    %4 = ue.matmul %3, %arg4 : (tensor<3136x96xbf16>, tensor<96x96xbf16>) -> tensor<3136x96xbf16>
    %5 = ue.window_reverse %4 {window_size = 7 : i64} : (tensor<3136x96xbf16>) -> tensor<3136x96xbf16>
    %6 = ue.layer_norm_post_add %arg0, %5, %arg5, %arg6 : (tensor<3136x96xbf16>, tensor<3136x96xbf16>, tensor<96xbf16>, tensor<96xbf16>) -> tensor<3136x96xbf16>
    %7 = ue.matmul %6, %arg7 {act = "gelu"} : (tensor<3136x96xbf16>, tensor<384x96xbf16>) -> tensor<3136x384xbf16>
    %8 = ue.matmul %7, %arg8 : (tensor<3136x384xbf16>, tensor<96x384xbf16>) -> tensor<3136x96xbf16>
    %9 = ue.eltwise %6, %8 : (tensor<3136x96xbf16>, tensor<3136x96xbf16>) -> tensor<3136x96xbf16>
    return %9 : tensor<3136x96xbf16>
  }
}

