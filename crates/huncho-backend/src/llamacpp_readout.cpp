// Exact signatures from pinned llama.cpp src/llama-ext.h at
// 26394b4e6749a41c3633db040e0987500a5f7013. The staging API uses C++ linkage;
// expose only these two opaque-context functions to Rust through a C ABI.
// No private object layouts, model mutations, sampling or device APIs.
#include <cstdint>
struct llama_context;
void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked);
float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i);

extern "C" void huncho_llama_set_masked_hidden(llama_context * ctx) {
    llama_set_embeddings_nextn(ctx, true, true);
}
extern "C" float * huncho_llama_get_masked_hidden(llama_context * ctx, int32_t i) {
    return llama_get_embeddings_nextn_ith(ctx, i);
}
