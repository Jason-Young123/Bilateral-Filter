//基于cuda的双边滤波器
#pragma once

#include <algorithm>
#include <numeric>
#include <filesystem>
#include <utils.h>
#include <myPixel.h>
#include <cuda_fp16.h>


//格式化输出
#define RESET       "\033[0m"
#define BOLD        "\033[1m"
#define UNDERLINE   "\033[4m"
#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define BLUE        "\033[34m"

#define EMSG        "\033[31m???????\033[0m"


//常量内存(lut)声明
static __constant__ float spatial_lut_data[6][6];
static __constant__ float color_lut_data[768];

__device__ inline float spatial_lut(int delta_y, int delta_x){
    return spatial_lut_data[delta_y][delta_x];
}


__device__ inline float color_lut(int L1_diff){
    return color_lut_data[L1_diff];
}


//主体函数: 基于cuda的双边滤波, T = uint8_t / myPixel
template <typename T>
__global__ void bilateralFilter(T* src, T* dst, int radius, int width, int height, float sigma_spatial_sq, float sigma_color_sq){
    int x_dst = blockIdx.x * blockDim.x + threadIdx.x;//小图dst的x坐标
    int y_dst = blockIdx.y * blockDim.y + threadIdx.y;//小图dst的y坐标
    int x_src = x_dst + radius;//大图src的x坐标
    int y_src = y_dst + radius;//大图src的y坐标

    //一些宏定义, 计算常量和累加器
    int width_plus_radius = width + 2 * radius;
    #define SRC_AT(y, x)        src[(y) * width_plus_radius + (x)]
    #define DST_AT(y, x)        dst[(y) * width + (x)]
    float weight_sum = 0;
    float3 RGB_product_sum = make_float3(0.0f, 0.0f, 0.0f);
    float Gray_product_sum = 0.0f;


    //主要处理逻辑
    if(x_dst < width && y_dst < height){//如果在范围内才计算(考虑边界block里面的部分thread可能越界),注意是针对被生成对象(小图dst)而言
        T p = SRC_AT(y_src, x_src);//获取center_pixel

        for(int i = - radius; i <= radius; i += 1){//行
            //for(int j = - radius; j < radius; j+=2){//列
            for(int j = - radius; j <= radius; j += 1){
                //step1: 计算weight_spatial
                int dist_sq = i * i + j * j;
                //int dist_sq1 = i * i + (j + 1) * (j + 1);
                if(dist_sq > radius * radius){
                    continue;
                }

                //step2: 计算weight_color/
                T q = SRC_AT(i + y_src, j + x_src);//获取neighbor_pixel
                float diff_color = q - p;
                float w = __expf( - dist_sq / sigma_spatial_sq - diff_color * diff_color / sigma_color_sq);

                //T q1 = SRC_AT(i + y_src, j + x_src + 1);
                //float diff_color1 = q1 - p;
                //float w1 = __expf( - dist_sq1 / sigma_spatial_sq - diff_color1 * diff_color1 / sigma_color_sq);

                weight_sum += w;
                //weight_sum += (w + w1);

                if constexpr(std::is_same_v<T, myPixel>){//RGB
                    uint32_t q_val = *(uint32_t*)(&q);
                    RGB_product_sum.x += (q_val & 0xff) * w; RGB_product_sum.y += ((q_val >> 8) & 0xFF) * w; RGB_product_sum.z += ((q_val >> 16) & 0xFF) * w;
                
                    //uint32_t q_val1 = *(uint32_t*)(&q1);
                    //RGB_product_sum.x += (q_val1 & 0xff) * w1; RGB_product_sum.y += ((q_val1 >> 8) & 0xFF) * w1; RGB_product_sum.z += ((q_val1 >> 16) & 0xFF) * w1;

                
                }
                else{//灰度
                    //Gray_product_sum += (q * w + q1 * w1);
                    Gray_product_sum += q * w;
                }
            }
        }

        float weight_inv = 1.0f / weight_sum;
        if constexpr(std::is_same_v<T, myPixel>){
            DST_AT(y_dst, x_dst) = myPixel(
                (uint8_t)__float2uint_rn(RGB_product_sum.x * weight_inv),
                (uint8_t)__float2uint_rn(RGB_product_sum.y * weight_inv),
                (uint8_t)__float2uint_rn(RGB_product_sum.z * weight_inv)
            );
        }
        else{
            DST_AT(y_dst, x_dst) = (uint8_t)__float2uint_rn(Gray_product_sum * weight_inv);
        }

    }
    
    #undef SRC_AT
    #undef DST_AT
}





//主体函数: 基于cuda的双边滤波, T = uint8_t / myPixel
/*template <typename T>
__global__ void bilateralFilter(T* src, T* dst, int radius, int width, int height, float sigma_spatial_sq, float sigma_color_sq){
    int x_dst = blockIdx.x * blockDim.x + threadIdx.x;//小图dst的x坐标
    int y_dst = blockIdx.y * blockDim.y + threadIdx.y;//小图dst的y坐标
    int x_src = x_dst + radius;//大图src的x坐标
    int y_src = y_dst + radius;//大图src的y坐标

    //一些宏定义, 计算常量和累加器
    int width_plus_radius = width + 2 * radius;
    #define SRC_AT(y, x)        src[(y) * width_plus_radius + (x)]
    #define DST_AT(y, x)        dst[(y) * width + (x)]
    float weight_sum = 0;
    float3 RGB_product_sum = make_float3(0.0f, 0.0f, 0.0f);
    float Gray_product_sum = 0.0f;


    //主要处理逻辑
    if(x_dst < width && y_dst < height){//如果在范围内才计算(考虑边界block里面的部分thread可能越界),注意是针对被生成对象(小图dst)而言
        T p = SRC_AT(y_src, x_src);//获取center_pixel

        for(int i = - radius; i <= radius; i += 1){//行
            for(int j = - radius; j < radius; j+=2){//列
                //step1: 计算weight_spatial
                int dist_sq = i * i + j * j;
                int dist_sq1 = i * i + (j + 1) * (j + 1);
                if(dist_sq > radius * radius){
                    continue;
                }

                //step2: 计算weight_color/
                T q = SRC_AT(i + y_src, j + x_src);//获取neighbor_pixel
                float diff_color = q - p;
                float w = __expf( - dist_sq / sigma_spatial_sq - diff_color * diff_color / sigma_color_sq);

                T q1 = SRC_AT(i + y_src, j + x_src + 1);
                float diff_color1 = q1 - p;
                float w1 = __expf( - dist_sq1 / sigma_spatial_sq - diff_color1 * diff_color1 / sigma_color_sq);

                //weight_sum += w;
                weight_sum += (w + w1);

                if constexpr(std::is_same_v<T, myPixel>){//RGB
                    uint32_t q_val = *(uint32_t*)(&q);
                    RGB_product_sum.x += (q_val & 0xff) * w; RGB_product_sum.y += ((q_val >> 8) & 0xFF) * w; RGB_product_sum.z += ((q_val >> 16) & 0xFF) * w;
                
                    uint32_t q_val1 = *(uint32_t*)(&q1);
                    RGB_product_sum.x += (q_val1 & 0xff) * w1; RGB_product_sum.y += ((q_val1 >> 8) & 0xFF) * w1; RGB_product_sum.z += ((q_val1 >> 16) & 0xFF) * w1;

                
                }
                else{//灰度
                    Gray_product_sum += (q * w + q1 * w1);
                    //Gray_product_sum += q * w;
                }
            }
        }

        float weight_inv = 1.0f / weight_sum;
        if constexpr(std::is_same_v<T, myPixel>){
            DST_AT(y_dst, x_dst) = myPixel(
                (uint8_t)__float2uint_rn(RGB_product_sum.x * weight_inv),
                (uint8_t)__float2uint_rn(RGB_product_sum.y * weight_inv),
                (uint8_t)__float2uint_rn(RGB_product_sum.z * weight_inv)
            );
        }
        else{
            DST_AT(y_dst, x_dst) = (uint8_t)__float2uint_rn(Gray_product_sum * weight_inv);
        }

    }
    
    #undef SRC_AT
    #undef DST_AT
}*/




/*template <typename T>
__global__ void bilateralFilter(T* src, T* dst, int radius, int width, int height, float sigma_spatial_sq, float sigma_color_sq){
    int x_dst = blockIdx.x * blockDim.x + threadIdx.x;//小图dst的x坐标
    int y_dst = blockIdx.y * blockDim.y + threadIdx.y;//小图dst的y坐标
    int x_src = x_dst + radius;//大图src的x坐标
    int y_src = y_dst + radius;//大图src的y坐标

    //一些宏定义, 计算常量和累加器
    int width_plus_radius = width + 2 * radius;
    #define SRC_AT(y, x)        src[(y) * width_plus_radius + (x)]
    #define DST_AT(y, x)        dst[(y) * width + (x)]
    float weight_sum = 0;
    float3 RGB_product_sum = make_float3(0.0f, 0.0f, 0.0f);
    float Gray_product_sum = 0.0f;


    //主要处理逻辑
    if(x_dst < width && y_dst < height){//如果在范围内才计算(考虑边界block里面的部分thread可能越界),注意是针对被生成对象(小图dst)而言
        T p = SRC_AT(y_src, x_src);//获取center_pixel

        for(int i = - radius; i <= radius; i += 1){//行
            for(int j = - radius; j < radius; j+=4){//列
                //step1: 计算weight_spatial
                int dist_sq = i * i + j * j;
                int dist_sq1 = i * i + (j + 1) * (j + 1);
                int dist_sq2 = i * i + (j + 2) * (j + 2);
                int dist_sq3 = i * i + (j + 3) * (j + 3);
                if(dist_sq > radius * radius){
                    continue;
                }

                //step2: 计算weight_color/
                T q = SRC_AT(i + y_src, j + x_src);//获取neighbor_pixel
                float diff_color = q - p;
                float w = __expf( - dist_sq / sigma_spatial_sq - diff_color * diff_color / sigma_color_sq);

                T q1 = SRC_AT(i + y_src, j + x_src + 1);
                float diff_color1 = q1 - p;
                float w1 = __expf( - dist_sq1 / sigma_spatial_sq - diff_color1 * diff_color1 / sigma_color_sq);

                T q = SRC_AT(i + y_src, j + x_src + 2);//获取neighbor_pixel
                float diff_color = q - p;
                float w = __expf( - dist_sq / sigma_spatial_sq - diff_color * diff_color / sigma_color_sq);

                T q1 = SRC_AT(i + y_src, j + x_src + 1);
                float diff_color1 = q1 - p;
                float w1 = __expf( - dist_sq1 / sigma_spatial_sq - diff_color1 * diff_color1 / sigma_color_sq);

                //weight_sum += w;
                weight_sum += (w + w1);

                if constexpr(std::is_same_v<T, myPixel>){//RGB
                    uint32_t q_val = *(uint32_t*)(&q);
                    RGB_product_sum.x += (q_val & 0xff) * w; RGB_product_sum.y += ((q_val >> 8) & 0xFF) * w; RGB_product_sum.z += ((q_val >> 16) & 0xFF) * w;
                
                    uint32_t q_val1 = *(uint32_t*)(&q1);
                    RGB_product_sum.x += (q_val1 & 0xff) * w1; RGB_product_sum.y += ((q_val1 >> 8) & 0xFF) * w1; RGB_product_sum.z += ((q_val1 >> 16) & 0xFF) * w1;

                
                }
                else{//灰度
                    Gray_product_sum += (q * w + q1 * w1);
                    //Gray_product_sum += q * w;
                }
            }
        }

        float weight_inv = 1.0f / weight_sum;
        if constexpr(std::is_same_v<T, myPixel>){
            DST_AT(y_dst, x_dst) = myPixel(
                (uint8_t)__float2uint_rn(RGB_product_sum.x * weight_inv),
                (uint8_t)__float2uint_rn(RGB_product_sum.y * weight_inv),
                (uint8_t)__float2uint_rn(RGB_product_sum.z * weight_inv)
            );
        }
        else{
            DST_AT(y_dst, x_dst) = (uint8_t)__float2uint_rn(Gray_product_sum * weight_inv);
        }

    }
    
    #undef SRC_AT
    #undef DST_AT
}
*/




//不包含malloc和free的纯粹版本
template <typename T>
void runFilterPure(const T* h_src, T* d_src, T* d_dst, T* h_dst, int radius, int width, int height, float sigma_spatial_sq, float sigma_color_sq, size_t src_size, size_t dst_size, cudaStream_t stream){
    if(h_src == nullptr){
        return;
    }

    //copy from host to device
    RUNTIME_CHECK(cudaMemcpyAsync(d_src, h_src, src_size, cudaMemcpyHostToDevice, stream));

    //kernel
    dim3 block_dim(16, 16);//blockDim固定为32 x 32
    dim3 grid_dim((width + 15) / 16, (height + 15) / 16);//gridDim根据img尺寸适配
    bilateralFilter<T><<<grid_dim, block_dim, 0, stream>>>(d_src, d_dst, radius, width, height, sigma_spatial_sq, sigma_color_sq);

    //copy from device to host
    RUNTIME_CHECK(cudaMemcpyAsync(h_dst, d_dst, dst_size, cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
}



//非模板函数声明
void runSingleCase(const std::filesystem::path& casePath, int radius, float sigma_spatial, float sigma_color, int warmup_round, int test_round);

void runAll(const std::string testerPath, int warmup_round, int test_round);

void runTester(const std::string& testerPath, int warmup_round, int test_round, bool only4K = true);