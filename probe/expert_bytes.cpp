// Read-only GGUF geometry audit. Does not load tensor data or use a GPU.
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <cstdint>
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    ggml_context* tensors=nullptr;
    gguf_init_params params{true,&tensors};
    gguf_context* file=gguf_init_from_file(argv[1],params);
    if(!file || !tensors) return 2;
    int active=0;
    for(int64_t i=0;i<gguf_get_n_kv(file);++i) {
        const std::string key=gguf_get_key(file,i);
        if(key.find(".expert_used_count")!=std::string::npos) {
            active=int(gguf_get_val_u32(file,i));
            std::printf("%s=%d\n",key.c_str(),active);
        }
    }
    std::map<std::string,int> counts;
    uint64_t bytes=0,delta=0,down=0;
    for(int64_t i=0;i<gguf_get_n_tensors(file);++i) {
        const char* name=gguf_get_tensor_name(file,i);
        if(!std::strstr(name,"_exps.weight")) continue;
        ggml_tensor* t=ggml_get_tensor(tensors,name);
        if(!t || t->ne[2]<1) return 3;
        const bool is_down=std::strstr(name,"ffn_down_exps.weight");
        const uint64_t per_expert=ggml_nbytes(t)/t->ne[2];
        const std::string role=is_down?"down":(std::strstr(name,"gate")?"gate":"up");
        const std::string geometry=role+" "+ggml_type_name(t->type)+" "+std::to_string(t->ne[0])+"x"+std::to_string(t->ne[1]);
        ++counts[geometry]; bytes+=per_expert;
        if(is_down) {
            down+=per_expert;
            if(t->type==GGML_TYPE_Q6_K) {
                if(t->ne[0]%ggml_blck_size(GGML_TYPE_Q4_K)) return 4;
                const uint64_t q4=(t->ne[0]/ggml_blck_size(GGML_TYPE_Q4_K))*ggml_type_size(GGML_TYPE_Q4_K)*t->ne[1];
                delta+=per_expert-q4;
            }
        }
    }
    for(const auto& [key,count]:counts) std::printf("%s: %d tensors\n",key.c_str(),count);
    std::printf("Q4_K block %lld values/%zu bytes; Q6_K block %lld/%zu\n",
                (long long)ggml_blck_size(GGML_TYPE_Q4_K),ggml_type_size(GGML_TYPE_Q4_K),
                (long long)ggml_blck_size(GGML_TYPE_Q6_K),ggml_type_size(GGML_TYPE_Q6_K));
    std::printf("one expert/layer summed: total=%llu down=%llu potential_Q6_to_Q4_saving=%llu bytes\n",
                (unsigned long long)bytes,(unsigned long long)down,(unsigned long long)delta);
    std::printf("active=%d => expert bytes/token=%llu down bytes/token=%llu saving=%llu (%.6f%%)\n",
                active,(unsigned long long)(bytes*active),(unsigned long long)(down*active),
                (unsigned long long)(delta*active),100.*double(delta)/bytes);
    gguf_free(file);ggml_free(tensors);
    return active>0?0:5;
}
