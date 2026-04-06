#include <iostream>
#include "TensorLib.h"

using namespace OwnTensor;

int main() {
    Tensor t(Shape{{1, 10}}, TensorOptions().with_dtype(Dtype::Int64));
    std::cout << "Tensor data: ";
    t.display();
    return 0;
}
