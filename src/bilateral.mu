#include <bilateral.h>
#include <auxiliary.h>
#include <myopencv.h>




//掐头去尾求平均用时
float ave_time(const std::vector<float>& time_ms, size_t size) {
    if (size < 3 || time_ms.size() < size) {
        float sum = std::accumulate(time_ms.begin(), time_ms.begin() + std::min(size, time_ms.size()), 0.0f);
        return sum / std::min(size, time_ms.size());
    }

    auto it_begin = time_ms.begin();
    auto it_end = time_ms.begin() + size;

    //遍历寻找最大最小值
    auto [min_it, max_it] = std::minmax_element(it_begin, it_end);
    float min_val = *min_it;
    float max_val = *max_it;

    float sum = std::accumulate(it_begin, it_end, 0.0f);

    return (sum - min_val - max_val) / (float)(size - 2);
}






void runSingleCase(const std::filesystem::path& casePath, int radius, float sigma_spatial, float sigma_color, int warmup_round, int test_round){
    std::string fileName = casePath.filename().string();//文件名,如cameraman.bin
    std::string filePath = casePath.string();//文件路径
    //std::string outPath = "result/gpu/" + fileName;
    //std::string refPath = "result/opencv/" + fileName;

    int width, height;
    uint8_t* src_gray = nullptr;
    myPixel* src_rgb = nullptr;
    getBin(filePath, width, height, src_gray, src_rgb);
    int channels = src_gray ? 1 : 3;
    float sigma_color_sq = 2 * sigma_color * sigma_color;
    float sigma_spatial_sq = 2 * sigma_spatial * sigma_spatial;


    std::string outPath = (channels == 1) ? ("result/gpu/gray/" + fileName) : ("result/gpu/rgb/" + fileName);
    std::string refPath = (channels == 1) ? ("result/opencv/gray/" + fileName) : ("result/opencv/rgb/" + fileName);


    std::cout << BLUE << " [ " << casePath.stem().string() << " ] " << ":";
    if (width == 3840 && height == 2160) {
        std::cout << " 4K(3840 * 2160 * " << channels << ")" << RESET << std::endl;
    } else {
        std::cout << " " << width << " * " << height << " * " << channels << RESET << std::endl;
    }

    auto runCase = [&](auto* h_src){
        using Type = std::remove_pointer_t<decltype(h_src)>;
        size_t src_size = static_cast<size_t>(width + 2 * radius) * (height + 2 * radius) * sizeof(Type);
        size_t dst_size = static_cast<size_t>(width) * height * sizeof(Type);
        Type* h_src_r101 = nullptr, *h_dst = new Type[width * height];
        h_src_r101 = Reflect101<Type>(h_src, width, height, radius);

        //gpu版本
        RUNTIME_CHECK(musaHostRegister(h_src_r101, src_size, musaHostRegisterDefault));
        RUNTIME_CHECK(musaHostRegister(h_dst, dst_size, musaHostRegisterDefault));

        //预分配显存, 只需进行一次
        Type *d_src, *d_dst;
        musaStream_t stream;
        RUNTIME_CHECK(musaMalloc(&d_src, src_size));
        RUNTIME_CHECK(musaMalloc(&d_dst, dst_size));
        RUNTIME_CHECK(musaStreamCreate(&stream));
        
        for(int i = 0; i < warmup_round; ++i){//warmup
            runFilterPure<Type>(h_src_r101, d_src, d_dst, h_dst, radius, width, height, sigma_spatial_sq, sigma_color_sq, src_size, dst_size, stream);
        }
        std::vector<float> time_ms;
        for(int i = 0; i < test_round; ++i){//test
            auto start = std::chrono::high_resolution_clock::now();
            runFilterPure<Type>(h_src_r101, d_src, d_dst, h_dst, radius, width, height, sigma_spatial_sq, sigma_color_sq, src_size, dst_size, stream);
            auto end = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration<float, std::milli>(end - start).count();
            time_ms.push_back(duration);
        }
        
        //清理资源, 只需进行一次
        RUNTIME_CHECK(musaHostUnregister(h_src_r101));
        RUNTIME_CHECK(musaHostUnregister(h_dst));
        RUNTIME_CHECK(musaFree(d_src));
        RUNTIME_CHECK(musaFree(d_dst));
        RUNTIME_CHECK(musaStreamDestroy(stream));

        float time_ms_gpu = ave_time(time_ms, test_round);
        float throughput_gpu = (width * height / 1000000.0f) / (time_ms_gpu / 1000.0f);
        genBin<Type>(outPath, h_dst, width, height);
        printf("gpu( musa ): Time: %8.3f ms | Throughput: %8.2f MP/s \n", time_ms_gpu, throughput_gpu);

#ifdef HAS_CV
        std::vector<float> time_ms1;
        for(int i = 0; i < test_round; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            runFilter_cv<Type>(h_src, radius, width, height, sigma_spatial, sigma_color, h_dst);
            auto end = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration<float, std::milli>(end - start).count();
            time_ms1.push_back(duration);
        }
        float time_ms_cv = ave_time(time_ms1, test_round);
        float throughput_cv = (width * height / 1000000.0f) / (time_ms_cv / 1000.0f);
        genBin<Type>(refPath, h_dst, width, height);

        printf("cpu(opencv): Time: %8.3f ms | Throughput: %8.2f MP/s \n", time_ms_cv, throughput_cv);
        
        std::cout << "Acceleration Ratio: " << throughput_gpu / throughput_cv << std::endl;
        
        float MAE = binDiff(refPath, outPath);
        if (MAE >= 1) {
            std::cout << "MAE: " << RED << MAE << " ( failed )" << RESET << std::endl << std::endl;
        } else {
            std::cout << "MAE: " << GREEN << MAE << " ( passed )" << RESET << std::endl << std::endl;
        }
#endif

        delete[] h_src_r101; delete[] h_dst;
    };

    if (channels == 1) {
        runCase(src_gray);
        delete[] src_gray;
    } else {
        runCase(src_rgb);
        delete[] src_rgb;
    }

}





void runAll(const std::string testerPath, int warmup_round, int test_round){
    //step0: preparation, get configuration info
    std::string cfgPath = testerPath + "/config.txt";//tester/config.txt
    int radius = 5;
    float sigma_color = 30.0f;
    float sigma_spatial = 3.0f;
    getCfg(cfgPath, radius, sigma_spatial, sigma_color);
    float sigma_color_sq = 2 * sigma_color * sigma_color;
    float sigma_spatial_sq = 2 * sigma_spatial * sigma_spatial;
    std::system("mkdir -p result/gpu/gray result/gpu/rgb result/opencv/gray result/opencv/rgb");//存储结果

    //init spatial_lut
    float h_spatial_lut_data[6][6];
    for(int i = 0; i < 6; ++i){
        for(int j = 0; j < 6; ++j){
            h_spatial_lut_data[i][j] = expf(-(float)(i * i + j * j) / (2.0f * sigma_spatial * sigma_spatial));
        }
    }
    musaMemcpyToSymbol(spatial_lut_data, h_spatial_lut_data, sizeof(h_spatial_lut_data));
    
    //init color_lut
    float h_color_lut_data[768];
    for(int i = 0; i < 768; ++i){
        h_color_lut_data[i] = expf(-(float)(i * i) / (2.0f * sigma_color * sigma_color));
    }
    musaMemcpyToSymbol(color_lut_data, h_color_lut_data, sizeof(h_color_lut_data));


    //step1: 灰度测试
    std::cout << "\n" << BOLD << BLUE << " ### Tester of gray images: ### " << RESET << std::endl << std::endl;
    std::string grayPath = testerPath + "/gray/4K";
    for(const auto& entry : std::filesystem::recursive_directory_iterator(grayPath)){
        if(entry.is_regular_file()){
            runSingleCase(entry.path(), radius, sigma_spatial, sigma_color, warmup_round, test_round);
        }
    }


    //step2: RGB测试
    std::cout << "\n" << BOLD << BLUE << " ### Tester of RGB images: ### " << RESET << std::endl << std::endl;
    std::string rgbPath = testerPath + "/rgb/4K";
    for(const auto& entry : std::filesystem::recursive_directory_iterator(rgbPath)){
        if(entry.is_regular_file()){
            runSingleCase(entry.path(), radius, sigma_spatial, sigma_color, warmup_round, test_round);
        }
    }

}




//统一测试, testerPath = "tester"
/*void runTester(const std::string& testerPath, int warmup_round, int test_round, bool only4K){
    //step0: preparation, get configuration info
    std::string cfgPath = testerPath + "/config.txt";//tester/config.txt
    int radius = 5;
    float sigma_color = 30.0f;
    float sigma_spatial = 3.0f;
    getCfg(cfgPath, radius, sigma_spatial, sigma_color);
    float sigma_color_sq = 2 * sigma_color * sigma_color;
    float sigma_spatial_sq = 2 * sigma_spatial * sigma_spatial;
    std::system("mkdir -p result/gpu/gray result/gpu/rgb result/opencv/gray result/opencv/rgb");//存储结果

    //init spatial_lut
    float h_spatial_lut_data[6][6];
    for(int i = 0; i < 6; ++i){
        for(int j = 0; j < 6; ++j){
            h_spatial_lut_data[i][j] = expf(-(float)(i * i + j * j) / (2.0f * sigma_spatial * sigma_spatial));
        }
    }
    musaMemcpyToSymbol(spatial_lut_data, h_spatial_lut_data, sizeof(h_spatial_lut_data));
    
    //init color_lut
    float h_color_lut_data[768];
    for(int i = 0; i < 768; ++i){
        h_color_lut_data[i] = expf(-(float)(i * i) / (2.0f * sigma_color * sigma_color));
    }
    musaMemcpyToSymbol(color_lut_data, h_color_lut_data, sizeof(h_color_lut_data));


    //step1: 灰度测试
    std::cout << "\n" << BOLD << BLUE << " ### Tester of gray images: ### " << RESET << std::endl << std::endl;
    std::string grayPath = only4K ? (testerPath + "/gray/4K") : (testerPath + "/gray");
    for(const auto& entry : std::filesystem::recursive_directory_iterator(grayPath)){
        if (!entry.is_regular_file()) continue;
        std::string fileName = entry.path().filename().string();//文件名,如cameraman.bin
        std::string filePath = entry.path().string();//文件路径
        std::string outPath = "result/gpu/gray/" + fileName;
        std::string refPath = "result/opencv/gray/" + fileName;

        int width, height;
        uint8_t* src_gray = nullptr;
        myPixel* src_rgb = nullptr;
        getBin(filePath, width, height, src_gray, src_rgb);
        size_t src_size = (width + 2 * radius) * (height + 2 * radius) * sizeof(uint8_t);
        size_t dst_size = width * height * sizeof(uint8_t);
        //reflect101
        uint8_t* src_gray_r101 = nullptr, *dst_gray = new uint8_t[width * height];
        src_gray_r101 = Reflect101<uint8_t>(src_gray, width, height, radius);
        
        //gpu版本bF,统计数据并输出bin
        //注册pinned memory
        RUNTIME_CHECK(musaHostRegister(src_gray_r101, src_size, musaHostRegisterDefault));
        RUNTIME_CHECK(musaHostRegister(dst_gray, dst_size, musaHostRegisterDefault));

        //预分配显存, 只需进行一次
        uint8_t *d_src, *d_dst;
        musaStream_t stream;
        RUNTIME_CHECK(musaMalloc(&d_src, src_size));
        RUNTIME_CHECK(musaMalloc(&d_dst, dst_size));
        RUNTIME_CHECK(musaStreamCreate(&stream));
        
        //warm-up
        for(int i = 0; i < warmup_round; ++i){//warmup
            runFilterPure(src_gray_r101, d_src, d_dst, dst_gray, radius, width, height, sigma_spatial_sq, sigma_color_sq, src_size, dst_size, stream);
        }

        //正式测试 + 计时
        std::vector<float> time_ms;
        for(int i = 0; i < test_round; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            runFilterPure(src_gray_r101, d_src, d_dst, dst_gray, radius, width, height, sigma_spatial_sq, sigma_color_sq, src_size, dst_size, stream);
            auto end = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration<float, std::milli>(end - start).count();
            time_ms.push_back(duration);
        }
        
        //清理资源, 只需进行一次
        RUNTIME_CHECK(musaHostUnregister(src_gray_r101));
        RUNTIME_CHECK(musaHostUnregister(dst_gray));
        RUNTIME_CHECK(musaFree(d_src));
        RUNTIME_CHECK(musaFree(d_dst));
        RUNTIME_CHECK(musaStreamDestroy(stream));

        float time_ms_gpu = ave_time(time_ms, test_round);
        float throughput_gpu = (width * height / 1000000.0f) / (time_ms_gpu / 1000.0f);
        genBin<uint8_t>(outPath, dst_gray, width, height);


        //opencv版本bF
#ifdef HAS_CV
        std::vector<float> time_ms1;
        for(int i = 0; i < test_round; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            runFilter_cv<uint8_t>(src_gray, radius, width, height, sigma_spatial, sigma_color, dst_gray);
            auto end = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration<float, std::milli>(end - start).count();
            time_ms1.push_back(duration);
        }
        float time_ms_cv = ave_time(time_ms1, test_round);
        float throughput_cv = (width * height / 1000000.0f) / (time_ms_cv / 1000.0f);
        genBin<uint8_t>(refPath, dst_gray, width, height);
#else
        float time_ms_cv = -1.0f;
        float throughput_cv = -1.0f;
#endif

        //结果对比
        float MAE = binDiff(refPath, outPath);
        std::cout << BLUE << " [ " << entry.path().stem().string() << " ] " << ":";
        if (width == 3840 && height == 2160) {
            std::cout << " 4K(3840 * 2160)" << std::endl;
        } else {
            std::cout << " " << width << " * " << height << std::endl;
        }
        std:: cout << RESET;
        printf("gpu( musa ): Time: %8.3f ms | Throughput: %8.2f MP/s \n", time_ms_gpu, throughput_gpu);
        printf("cpu(opencv): Time: %8.3f ms | Throughput: %8.2f MP/s \n", time_ms_cv, throughput_cv);
        //printf("cpu(opencv): Time: %s ms | Throughput: %s MP/s \n", EMSG, EMSG);
        std::cout << "Acceleration Ratio: " << throughput_gpu / throughput_cv << std::endl;
        if (MAE >= 1) {
            std::cout << "MAE: " << RED << MAE << " ( failed )" << RESET << std::endl << std::endl;
        } else {
            std::cout << "MAE: " << GREEN << MAE << " ( passed )" << RESET << std::endl << std::endl;
        }

        delete[] src_gray; delete[] src_gray_r101; delete[] dst_gray;
    }


    //step2: rgb测试
    std::cout << "\n" << BOLD << BLUE << " ### Tester of RGB images: ### " << RESET << std::endl << std::endl;
    std::string rgbPath = only4K ? (testerPath + "/rgb/4K") : (testerPath + "/rgb");
    for(const auto& entry : std::filesystem::recursive_directory_iterator(rgbPath)){
        if (!entry.is_regular_file()) continue;
        std::string fileName = entry.path().filename().string();//文件名,如lena.bin
        std::string filePath = entry.path().string();//文件路径
        std::string outPath = "result/gpu/rgb/" + fileName;
        std::string refPath = "result/opencv/rgb/" + fileName;
        
        int width, height;
        uint8_t* src_gray = nullptr;
        myPixel* src_rgb = nullptr;
        getBin(filePath, width, height, src_gray, src_rgb);
        size_t src_size = (width + 2 * radius) * (height + 2 * radius) * sizeof(myPixel);
        size_t dst_size = width * height * sizeof(myPixel);
        //reflect101
        myPixel* src_rgb_r101 = nullptr, *dst_rgb = new myPixel[width * height];
        src_rgb_r101 = Reflect101<myPixel>(src_rgb, width, height, radius);

        //gpu版本bF
        //注册pinned memory
        musaHostRegister(src_rgb_r101, src_size, musaHostRegisterDefault);
        musaHostRegister(dst_rgb, dst_size, musaHostRegisterDefault);

        //预分配显存, 只需进行一次
        myPixel *d_src, *d_dst;
        musaStream_t stream;
        RUNTIME_CHECK(musaMalloc(&d_src, src_size));
        RUNTIME_CHECK(musaMalloc(&d_dst, dst_size));
        RUNTIME_CHECK(musaStreamCreate(&stream));

        //warm-up, 3轮
        for(int i = 0; i < warmup_round; ++i){//warmup
            runFilterPure(src_rgb_r101, d_src, d_dst, dst_rgb, radius, width, height, sigma_spatial_sq, sigma_color_sq, src_size, dst_size, stream);
        }

        //正式测试 + 计时, 10轮
        std::vector<float> time_ms;
        for(int i = 0; i < test_round; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            runFilterPure(src_rgb_r101, d_src, d_dst, dst_rgb, radius, width, height, sigma_spatial_sq, sigma_color_sq, src_size, dst_size, stream);
            auto end = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration<float, std::milli>(end - start).count();
            time_ms.push_back(duration);
        }

        //清理资源, 只需进行一次
        RUNTIME_CHECK(musaHostUnregister(src_rgb_r101));
        RUNTIME_CHECK(musaHostUnregister(dst_rgb));
        RUNTIME_CHECK(musaFree(d_src));
        RUNTIME_CHECK(musaFree(d_dst));
        RUNTIME_CHECK(musaStreamDestroy(stream));

        float time_ms_gpu = ave_time(time_ms, test_round);
        float throughput_gpu = (width * height / 1000000.0f) / (time_ms_gpu / 1000.0f);
        genBin<myPixel>(outPath, dst_rgb, width, height);
        //delete[] dst_rgb;

        //opencv版本bF
#ifdef HAS_CV
        std::vector<float> time_ms1;
        for(int i = 0; i < test_round; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            runFilter_cv<myPixel>(src_rgb, radius, width, height, sigma_spatial, sigma_color, dst_rgb);
            auto end = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration<float, std::milli>(end - start).count();
            time_ms1.push_back(duration);
        }
        float time_ms_cv = ave_time(time_ms1, test_round);
        float throughput_cv = (width * height / 1000000.0f) / (time_ms_cv / 1000.0f);
        genBin<myPixel>(refPath, dst_rgb, width, height);
#else
        float time_ms_cv = -1.0f;
        float throughput_cv = -1.0f;
#endif

        //结果对比
        float MAE = binDiff(refPath, outPath);
        std::cout << BLUE << " [ " << entry.path().stem().string() << " ] " << ":";
        if (width == 3840 && height == 2160) {
            std::cout << " 4K(3840 * 2160)" << std::endl;
        } else {
            std::cout << " " << width << " * " << height << std::endl;
        }
        std::cout << RESET;
        printf("gpu( musa ): Time: %8.3f ms | Throughput: %8.2f MP/s \n", time_ms_gpu, throughput_gpu);
        printf("cpu(opencv): Time: %8.3f ms | Throughput: %8.2f MP/s \n", time_ms_cv, throughput_cv);
        std::cout << "Acceleration Ratio: " << throughput_gpu / throughput_cv << std::endl;
        if (MAE >= 1) {
            std::cout << "MAE: " << RED << MAE << " ( failed )" << RESET << std::endl << std::endl;
        } else {
            std::cout << "MAE: " << GREEN << MAE << " ( passed )" << RESET << std::endl << std::endl;
        }

        delete[] src_rgb; delete[] src_rgb_r101; delete[] dst_rgb;
    }

    std::cout << "\nSuccessfully run all testers." << std::endl;
}*/









