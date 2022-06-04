#include "Enclave.h"

void printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
}

template <typename F>
inline void compute(F func, float* first, float* second, float* result, size_t dim){
    for(size_t i = 0; i < dim; i += 8){
        const __m256 a = _mm256_load_ps(&first[i]);
        const __m256 b = _mm256_load_ps(&second[i]);
        const __m256 res = func(a, b);
        _mm256_stream_ps(&result[i], res);
    }
}
void ecall_add(float* first, float* second, float* result, size_t dim){
    //for(size_t i = 0;i < dim;i++)result[i] = first[i] + second[i];
    if(dim % 8){
        for(size_t i = 0;i < dim;i++)result[i] = first[i] + second[i];
    }
    else{
        auto add_func = [] (__m256 a,__m256 b) {return _mm256_add_ps(a,b);};
        compute(add_func, first, second, result, dim);
    }
}
void ecall_sub(float* first, float* second, float* result, size_t dim){
    //for(size_t i = 0;i < dim;i++)result[i] = first[i] - second[i];
    if(dim % 8){
        for(size_t i = 0;i < dim;i++)result[i] = first[i] - second[i];
    }
    else{
        auto add_func = [] (__m256 a,__m256 b) {return _mm256_sub_ps(a,b);};
        compute(add_func, first, second, result, dim);
    }
}