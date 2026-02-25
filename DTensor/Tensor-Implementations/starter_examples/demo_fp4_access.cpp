#include <iostream>
#include <vector>
#include <iomanip>
#include "core/Tensor.h"
#include "dtype/Types.h"
#include "dtype/FP4Converters.h"

using namespace OwnTensor;

// Demo 1: Unpacked FP4 (1 value per byte, lower 4 bits used)
void demo_unpacked_fp4() {
    std::cout << "\n=== Demo 1: Unpacked FP4 (Float4_e2m1) ===\n";
    std::cout << "Storage: 1 byte per value (lower 4 bits used)\n";

    // 1. Create Tensor
    Tensor t(Shape{{4}}, Dtype::Float4_e2m1);
    
    // 2. Prepare Data
    std::vector<float4_e2m1_t> data = {
        float4_e2m1_t(1.0f), float4_e2m1_t(2.0f),
        float4_e2m1_t(3.0f), float4_e2m1_t(4.0f)
    };
    t.set_data(data);

    // 3. Access Data Pointer
    // We use the specific struct type for the pointer
    float4_e2m1_t* ptr = t.data<float4_e2m1_t>();

    // 4. Iterate and Read
    std::cout << "Values in Tensor: ";
    for (size_t i = 0; i < t.numel(); ++i) {
        // Implicit conversion to float for printing
        float val = static_cast<float>(ptr[i]);
        std::cout << val << " ";
    }
    std::cout << "\n";

    // 5. Modify via Pointer
    std::cout << "Modifying index 0 to 6.0 via pointer...\n";
    ptr[0] = float4_e2m1_t(6.0f);
    std::cout << "New value at index 0: " << static_cast<float>(ptr[0]) << "\n";
}

// Demo 2: Packed FP4 (2 values per byte)
void demo_packed_fp4() {
    std::cout << "\n=== Demo 2: Packed FP4 (Float4_e2m1_2x) ===\n";
    std::cout << "Storage: 1 byte per 2 values (High nibble = val1, Low nibble = val0)\n";

    // 1. Create Tensor
    // Shape is logical elements? Or packed elements?
    // Usually Tensor shape is logical. But for packed types, the storage is compressed.
    // If we allocate Shape{{4}}, and Dtype is Float4_e2m1_2x, does it mean 4 packed bytes (8 values) or 4 logical values (2 bytes)?
    // The current implementation likely treats 'Float4_e2m1_2x' as a distinct type of size 1 byte.
    // So Shape{{2}} of Float4_e2m1_2x would hold 2 bytes = 4 logical FP4 values.
    // Let's assume we want to store 4 logical values. That requires 2 packed elements.
    Tensor t(Shape{{2}}, Dtype::Float4_e2m1_2x); 
    
    // 2. Prepare Data (Packed)
    // We need to pack pairs of values into float4_e2m1_2x_t
    // Pair 1: 1.0, 2.0
    float4_e2m1_t v0(1.0f);
    float4_e2m1_t v1(2.0f);
    float4_e2m1_2x_t packed1(v0, v1); // Low=1.0, High=2.0

    // Pair 2: 3.0, 4.0
    float4_e2m1_t v2(3.0f);
    float4_e2m1_t v3(4.0f);
    float4_e2m1_2x_t packed2(v2, v3); // Low=3.0, High=4.0

    std::vector<float4_e2m1_2x_t> data = {packed1, packed2};
    t.set_data(data);

    // 3. Access Data Pointer
    float4_e2m1_2x_t* ptr = t.data<float4_e2m1_2x_t>();

    // 4. Iterate and Read
    std::cout << "Packed Values in Tensor (2 elements, 4 logical values):\n";
    for (size_t i = 0; i < t.numel(); ++i) {
        float4_e2m1_2x_t packed = ptr[i];
        
        // Unpack
        float val_low = static_cast<float>(packed.get_low());
        float val_high = static_cast<float>(packed.get_high());
        
        std::cout << "  Byte " << i << ": Low=" << val_low << ", High=" << val_high << "\n";
    }

    // 5. Modify via Pointer
    std::cout << "Modifying Byte 1 High nibble to 6.0...\n";
    // Current: Low=3.0, High=4.0. Want: Low=3.0, High=6.0
    float4_e2m1_2x_t current = ptr[1];
    current.set_high(float4_e2m1_t(6.0f));
    ptr[1] = current;
    
    std::cout << "New Byte 1 High: " << static_cast<float>(ptr[1].get_high()) << "\n";
}

// Demo 3: Conversions using FP4Converters.h
void demo_conversions() {
    std::cout << "\n=== Demo 3: Conversions ===\n";
    
    // 1. Unpacked Conversions
    float4_e2m1_t val(3.0f);
    std::cout << "Original FP4: " << static_cast<float>(val) << "\n";
    
    float f32 = fp4_to_fp32(val);
    float16_t f16 = fp4_to_fp16(val);
    float4_e2m1_t f4 = fp4_to_fp4(val);
    
    std::cout << "  -> FP32: " << f32 << "\n";
    std::cout << "  -> FP16: " << static_cast<float>(f16) << "\n";
    std::cout << "  -> FP4:  " << static_cast<float>(f4) << "\n";

    // 2. Packed Conversions
    float4_e2m1_t v0(1.0f);
    float4_e2m1_t v1(6.0f);
    float4_e2m1_2x_t packed(v0, v1);
    
    std::cout << "Packed FP4 (Low=1.0, High=6.0):\n";
    
    float out_f32[2];
    packed_fp4_to_fp32(packed, out_f32);
    std::cout << "  -> FP32: [" << out_f32[0] << ", " << out_f32[1] << "]\n";
    
    float16_t out_f16[2];
    packed_fp4_to_fp16(packed, out_f16);
    std::cout << "  -> FP16: [" << static_cast<float>(out_f16[0]) << ", " << static_cast<float>(out_f16[1]) << "]\n";
    
    float4_e2m1_t out_f4[2];
    packed_fp4_to_fp4(packed, out_f4);
    std::cout << "  -> FP4:  [" << static_cast<float>(out_f4[0]) << ", " << static_cast<float>(out_f4[1]) << "]\n";
}

int main() {
    try {
        demo_unpacked_fp4();
        demo_packed_fp4();
        demo_conversions();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
