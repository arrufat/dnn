#pragma once

#include "blocks.h"

namespace dnn
{
    // resnet 18
    template<typename SUBNET>
    using resnet18_level1 = resblock512<resblock_down<512, SUBNET>>;
    template<typename SUBNET>
    using resnet18_level2 = resblock256<resblock_down<256, SUBNET>>;
    template<typename SUBNET>
    using resnet18_level3 = resblock128<resblock_down<128, SUBNET>>;
    template<typename SUBNET>
    using resnet18_level4 = repeat<2, resblock64, SUBNET>;
    // the resnet 18 backbone
    template<typename INPUT>
    using resnet18_backbone = avg_pool_everything<
        resnet18_level1<
        resnet18_level2<
        resnet18_level3<
        resnet18_level4<
        max_pool<3, 2, 1, relu<bn_con<conv<64, 7, 2, 1,
        INPUT
        >>>>>>>>>;
    using resnet18_t = loss_multiclass_log<fc<1000, resnet18_backbone<input_rgb_image>>>;

    // resnet 34
    template<typename SUBNET>
    using resnet34_level1 = repeat<2, resblock512, resblock_down<512, SUBNET>>;
    template<typename SUBNET>
    using resnet34_level2 = repeat<5, resblock256, resblock_down<256, SUBNET>>;
    template<typename SUBNET>
    using resnet34_level3 = repeat<3, resblock128, resblock_down<128, SUBNET>>;
    template<typename SUBNET>
    using resnet34_level4 = repeat<3, resblock64, SUBNET>;
    // the resnet 34 backbone
    template<typename INPUT>
    using resnet34_backbone = avg_pool_everything<
        resnet34_level1<
        resnet34_level2<
        resnet34_level3<
        resnet34_level4<
        max_pool<3, 2, 1, relu<bn_con<conv<64, 7, 2, 1,
        INPUT
        >>>>>>>>>;
    using resnet34_t = loss_multiclass_log<fc<1000, resnet34_backbone<input_rgb_image>>>;

    // resnet 50
    template<typename SUBNET>
    using resnet50_level1 = repeat<2, resbottleneck512, resbottleneck_down<512, SUBNET>>;
    template<typename SUBNET>
    using resnet50_level2 = repeat<5, resbottleneck256, resbottleneck_down<256, SUBNET>>;
    template<typename SUBNET>
    using resnet50_level3 = repeat<3, resbottleneck128, resbottleneck_down<128, SUBNET>>;
    template<typename SUBNET>
    using resnet50_level4 = repeat<3, resbottleneck64, SUBNET>;
    // the resnet 50 backbone
    template<typename INPUT>
    using resnet50_backbone = avg_pool_everything<
        resnet50_level1<
        resnet50_level2<
        resnet50_level3<
        resnet50_level4<
        max_pool<3, 2, 1, relu<bn_con<conv<64, 7, 2, 1,
        INPUT
        >>>>>>>>>;
    using resnet50_t = loss_multiclass_log<fc<1000, resnet50_backbone<input_rgb_image>>>;

    // resnet 101
    template<typename SUBNET>
    using resnet101_level1 = repeat<2, resbottleneck512, resbottleneck_down<512, SUBNET>>;
    template<typename SUBNET>
    using resnet101_level2 = repeat<22, resbottleneck256, resbottleneck_down<256, SUBNET>>;
    template<typename SUBNET>
    using resnet101_level3 = repeat<3, resbottleneck128, resbottleneck_down<128, SUBNET>>;
    template<typename SUBNET>
    using resnet101_level4 = repeat<3, resbottleneck64, SUBNET>;
    // the resnet 101 backbone
    template<typename INPUT>
    using resnet101_backbone = avg_pool_everything<
        resnet101_level1<
        resnet101_level2<
        resnet101_level3<
        resnet101_level4<
        max_pool<3, 2, 1, relu<bn_con<conv<64, 7, 2, 1,
        INPUT
        >>>>>>>>>;
    using resnet101_t = loss_multiclass_log<fc<1000, resnet101_backbone<input_rgb_image>>>;

    // resnet 152
    template<typename SUBNET>
    using resnet152_level1 = repeat<2, resbottleneck512, resbottleneck_down<512, SUBNET>>;
    template<typename SUBNET>
    using resnet152_level2 = repeat<35, resbottleneck256, resbottleneck_down<256, SUBNET>>;
    template<typename SUBNET>
    using resnet152_level3 = repeat<7, resbottleneck128, resbottleneck_down<128, SUBNET>>;
    template<typename SUBNET>
    using resnet152_level4 = repeat<3, resbottleneck64, SUBNET>;
    // the resnet 152 backbone
    template<typename INPUT>
    using resnet152_backbone = avg_pool_everything<
        resnet152_level1<
        resnet152_level2<
        resnet152_level3<
        resnet152_level4<
        max_pool<3, 2, 1, relu<bn_con<conv<64, 7, 2, 1,
        INPUT
        >>>>>>>>>;
    using resnet152_t = loss_multiclass_log<fc<1000, resnet152_backbone<input_rgb_image>>>;
}
