#include "src/model.h"
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include "npy.h"

using keras2cpp::Model;
using keras2cpp::Tensor;


int main(int argc, char *argv[]) {
            
    // Initialize model.
    auto model = Model::load(argv[1]);

    // Load data
    bool fortran_order;
    std::vector<unsigned long> shape;
    std::vector<double> data;
    npy::LoadArrayFromNumpy(argv[2], shape, fortran_order, data);
    
    size_t count = 0;
    for (auto&& dim : shape) {
        //printf("%zu ", shape[count]);
        ++count;
    }
    
    // Create Tensor
    auto in = [&]() {
        if (count == 1) return Tensor{shape[0]};
        if (count == 2) return Tensor{shape[0], shape[1]};
        if (count == 3) return Tensor{shape[0], shape[1], shape[2]};
        if (count == 4) return Tensor{shape[0], shape[1], shape[2], shape[3]};
        return Tensor{1};
    }();
    
    in.data_ = std::vector<float>(data.begin(), data.end());
    
    // Run prediction.
    Tensor out = model(in);
    out.print();
       
    return 0;
}
