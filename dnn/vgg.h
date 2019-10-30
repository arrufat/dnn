#pragma once

#include "layers.h"

namespace dnn::vgg
{
    // the main vgg building block
    template<long num_filters, template<typename> class BN, typename SUBNET>
    using block = relu<BN<con<num_filters, 3, 3, 1, 1, SUBNET>>>;
    template<typename SUBNET>
    using final_fc = fc<1000, fc<4095, fc<4096, SUBNET>>>;

    namespace bn::train
    {
        template<typename SUBNET> using regularization = bn_con<SUBNET>;
        template<typename SUBNET> using block512 = block<512, regularization, SUBNET>;
        template<typename SUBNET> using block256 = block<256, regularization, SUBNET>;
        template<typename SUBNET> using block128 = block<128, regularization, SUBNET>;
        template<typename SUBNET> using block64 = block<64, regularization, SUBNET>;

        template<long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
        using backbone = final_fc<
            max_pool<2, 2, 2, 2, repeat<nb_512, block512,
            max_pool<2, 2, 2, 2, repeat<nb_512, block512,
            max_pool<2, 2, 2, 2, repeat<nb_256, block256,
            max_pool<2, 2, 2, 2, repeat<nb_128, block128,
            max_pool<2, 2, 2, 2, repeat<nb_64, block64,
            INPUT>>>>>>>>>>>;
        using _11 = backbone<2, 2, 1, 1, input_rgb_image>;
        using _13 = backbone<2, 2, 2, 2, input_rgb_image>;
        using _16 = backbone<3, 3, 2, 2, input_rgb_image>;
        using _19 = backbone<4, 4, 2, 2, input_rgb_image>;
    }
    namespace bn::infer
    {
        template<typename SUBNET> using regularization = affine<SUBNET>;
        template<typename SUBNET> using block512 = block<512, regularization, SUBNET>;
        template<typename SUBNET> using block256 = block<256, regularization, SUBNET>;
        template<typename SUBNET> using block128 = block<128, regularization, SUBNET>;
        template<typename SUBNET> using block64 = block<64, regularization, SUBNET>;

        template<long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
        using backbone = final_fc<
            max_pool<2, 2, 2, 2, repeat<nb_512, block512,
            max_pool<2, 2, 2, 2, repeat<nb_512, block512,
            max_pool<2, 2, 2, 2, repeat<nb_256, block256,
            max_pool<2, 2, 2, 2, repeat<nb_128, block128,
            max_pool<2, 2, 2, 2, repeat<nb_64, block64,
            INPUT>>>>>>>>>>>;
        using _11 = backbone<2, 2, 1, 1, input_rgb_image>;
        using _13 = backbone<2, 2, 2, 2, input_rgb_image>;
        using _16 = backbone<3, 3, 2, 2, input_rgb_image>;
        using _19 = backbone<4, 4, 2, 2, input_rgb_image>;
    }
}
