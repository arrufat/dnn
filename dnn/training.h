#include <dlib/dnn.h>

namespace dnn
{
    using dlib::dnn_trainer;
    using dlib::set_dnn_prefer_fastest_algorithms;
    using dlib::set_dnn_prefer_smallest_algorithms;
    using dlib::sgd, dlib::adam;
    using dlib::serialize, dlib::deserialize;
}
