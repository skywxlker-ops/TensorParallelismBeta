#include "../tensor/dtensor.h"
#include "../bridge/tensor_ops_bridge.h"
#include "../process_group/ProcessGroupNCCL.h"
#include <iostream>

extern "C" void __force_link_dtensor_symbols() {
    std::cerr << "[DTensor] Forcing symbol linkage (no runtime init)..." << std::endl;

    // Reference TensorOpsBridge::add (returns Tensor, not void)
    auto bridge_fn = &TensorOpsBridge::add;
    auto pg_fn = &ProcessGroup::getRank;

    (void)bridge_fn;
    (void)pg_fn;

    // Force RTTI so linker keeps vtables and typeinfo
    (void)typeid(DTensor);
    (void)typeid(ProcessGroup);

    std::cerr << "[DTensor] Symbols linked (safe & static )" << std::endl;
}
