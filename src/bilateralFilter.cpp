#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>


//key1: opencv并非采用正方形窗口,而是圆
//key2: opencl计算diff_color时并非采用L2范数,而是L1范数的平方


/************************************ 辅助函数 *************************************/
//解析bin文件
bool getBin(const char* binPath, cv::Mat& img){
    std::ifstream file(binPath, std::ios::binary);
    if(!file){
        std::cerr << "Error: Could not open file: " << binPath << std::endl;
        return false;
    }

    //step1: 获取头信息
    int width, height, channels;
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));
    file.read(reinterpret_cast<char*>(&channels), sizeof(int));
    int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
    
    //step2: 获取像素信息
    img.create(height, width, type);
    size_t dataSize = static_cast<size_t>(width) * height * channels;
    file.read(reinterpret_cast<char*>(img.data), dataSize);

    file.close();
    std::cout << "Successfully getBin: " << binPath << std::endl;
    return true;
}



//解析cfg文件
bool getCfg(const char* cfgPath, int& radius, float& sigma_spatial, float& sigma_color) {
    std::ifstream file(cfgPath);
    if(!file){
        std::cerr << "Error: Could not open file: " << cfgPath << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("radius") != std::string::npos) 
            sscanf(line.c_str(), " radius = %d", &radius);
        else if (line.find("sigma_spatial") != std::string::npos)
            sscanf(line.c_str(), " sigma_spatial = %f", &sigma_spatial);
        else if (line.find("sigma_color") != std::string::npos)
            sscanf(line.c_str(), " sigma_color = %f", &sigma_color);
    }
    file.close();
    std::cout << "Successfully getCfg: " << cfgPath << std::endl;
    return true;
}


//生成bin文件, RGB还是BGR取决于img自身
bool genBin(const char* outBinPath, const cv::Mat& img){
    std::ofstream file(outBinPath, std::ios::binary);
    if(!file){
        std::cerr << "Error: Could not open file: " << outBinPath << std::endl;
        return false;
    }
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    file.write(reinterpret_cast<char*>(&width), sizeof(int));
    file.write(reinterpret_cast<char*>(&height), sizeof(int));
    file.write(reinterpret_cast<char*>(&channels), sizeof(int));
    file.write(reinterpret_cast<char*>(img.data), size_t(width * height * channels));

    file.close();
    std::cout << "Successfully genBin: " << outBinPath << std::endl;
    return true;
}


//生成图片
bool genImg(const char* outImgPath, const cv::Mat& img, bool isRGB){
    cv::Mat finalImg;
    if(img.channels() == 3 && isRGB){
        cv::cvtColor(img, finalImg, cv::COLOR_RGB2BGR);
    }
    else{
        finalImg = img;
    }
    cv::imwrite(outImgPath, finalImg);
    
    std::cout << "Successfully genImg: " << outImgPath << std::endl;
    return true;
}


namespace fs = std::filesystem;
void genTester() {
    fs::create_directories("tester/gray");
    fs::create_directories("tester/rgb");

    struct Config {
        std::string srcPath;
        std::string dstPath;
        bool isRGB;
    };

    Config configs[] = {
        {"resource/gray", "tester/gray", false},
        {"resource/rgb", "tester/rgb", true}
    };

    for (const auto& cfg : configs) {//每一个目录,rgb or gray
        if (!fs::exists(cfg.srcPath)) {
            std::cout << "Directory not found: " << cfg.srcPath << std::endl;
            continue;
        }

        for (const auto& entry : fs::directory_iterator(cfg.srcPath)){//每一个文件
            if (!entry.is_regular_file()){
                continue;
            }
            fs::path filePath = entry.path();

            cv::Mat img = cv::imread(filePath.string(), cfg.isRGB ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);

            if (img.empty()) {
                continue;
            }

            if (cfg.isRGB) {
                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            }

            std::string outName = filePath.stem().string() + ".bin";
            fs::path outPath = fs::path(cfg.dstPath) / outName;
            if (genBin(outPath.string().c_str(), img)) {
                std::cout << "[Tester] Converted: " << filePath.filename() << " -> " << outPath.string() << std::endl;
            }
        }
    }
}


/*********************************** 基于opencv的bF *************************************/
bool bilateralFilter(const char* binPath, const char* cfgPath, const char* outBinPath, const char* outImgPath){
    //step1: 获取图像, 默认RGB
    cv::Mat inImg;
    bool flag1 = getBin(binPath, inImg);

    //step2: 获取cfg
    int radius = 5;
    float sigma_spatial = 3.0f;
    float sigma_color = 30.0f;
    bool flag2 = getCfg(cfgPath, radius, sigma_spatial, sigma_color);

    //step3: 双边滤波
    cv::Mat outImg;
    if(flag1 && flag2){
        int d = 2 * radius + 1;
        cv::bilateralFilter(inImg, outImg, d, sigma_color, sigma_spatial);
    }

    //step4: 输出bin和图像
    genBin(outBinPath, outImg);
    genImg(outImgPath, outImg, true);

    return true;
}





/*********************************** 手动实现的bF *************************************/
//大图坐标相对小图坐标而言,取值范围在-radius ~ len + radius - 1
inline int mapReflect101(int p, int len){
    if(p < 0){//左越界
        return -p;
    }
    else if(p >= len){//右越界
        return 2 * (len - 1) - p;
    }
    else{//不越界
        return p;
    }
}


//将输入图像以reflect101模式进行边缘延拓
cv::Mat Reflect101(const cv::Mat& img, int radius){
    cv::Mat outImg;
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
    
    outImg.create(height + 2 * radius, width + 2 * radius, type);

    for(int i = 0; i < height + 2 * radius; ++i){//行
        for(int j = 0; j < width + 2 * radius; ++j){//列
            //计算大图中每个像素点相对于小图的位置(-radius, len + radius)
            int relative_i = i - radius;
            int relative_j = j - radius;

            int src_y = mapReflect101(relative_i, height);//纵坐标
            int src_x = mapReflect101(relative_j, width);//横坐标

            if (channels == 3) {
                outImg.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(src_y, src_x);
            } else {
                outImg.at<uchar>(i, j) = img.at<uchar>(src_y, src_x);
            }
        }
    }
    return outImg;
}


//逐像素点进行bF滤波, T = cv::Vec3b或uchar, 但是顺序默认RGB
template <typename T>
T bilateralFilterPixel(int y, int x, int radius, const cv::Mat& img, float sigma_spatial, float sigma_color){//(y, x)为中心点坐标
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
    assert((y >= radius) && (y <= height - radius - 1));
    assert((x >= radius) && (x <= width - radius - 1));

    float weight_sum = 0;
    float R_product_sum = 0, G_product_sum = 0, B_product_sum = 0, Gray_product_sum = 0;

    float spatial_coeff = -1.0f / (2 * sigma_spatial * sigma_spatial);
    float color_coeff = -1.0f / (2 * sigma_color * sigma_color);

    T center_pixel = img.at<T>(y, x);//中心像素点(y, x)

    for(int i = y - radius; i <= y + radius; ++i){//行
        for(int j = x - radius; j <= x + radius; ++j){//列
            
            float diff_spatial = (float)((i - y) * (i - y) + (j - x) * (j - x));
            float weight_spatial = exp(diff_spatial * spatial_coeff);

            //key1: opencv并非采用正方形窗口,而是圆
            if(diff_spatial > radius * radius){
                continue;
            }

            T neighbor_pixel = img.at<T>(i, j);//临近像素点(i, j)
            if constexpr(std::is_same_v<T, cv::Vec3b>){//RGB图像
                cv::Vec3b p = reinterpret_cast<const cv::Vec3b&>(center_pixel);
                cv::Vec3b q = reinterpret_cast<const cv::Vec3b&>(neighbor_pixel);
                //key2: opencl计算diff_color时并非采用L2范数,而是L1范数的平方
                //float diff_color = (float)((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
                float diff_color = float(std::abs(int(p[0]) - int(q[0])) + std::abs(int(p[1]) - int(q[1])) + std::abs(int(p[2]) - int(q[2])));
                float weight_color = exp(diff_color * diff_color * color_coeff);
                
                float w = weight_spatial * weight_color;
                weight_sum += w;
                R_product_sum += q[0] * w;
                G_product_sum += q[1] * w;
                B_product_sum += q[2] * w;
            }
            else{//灰度图像
                float diff_color = (float)((center_pixel - neighbor_pixel) * (center_pixel - neighbor_pixel));
                float weight_color = exp(diff_color * color_coeff);

                float w = weight_spatial * weight_color;
                weight_sum += w;
                Gray_product_sum += neighbor_pixel * w;
            }
        }
    }

    if constexpr(std::is_same_v<T, cv::Vec3b>){
        return cv::Vec3b(
            cv::saturate_cast<uchar>(R_product_sum / weight_sum),
            cv::saturate_cast<uchar>(G_product_sum / weight_sum),
            cv::saturate_cast<uchar>(B_product_sum / weight_sum)
        );
    }
    else{
        return cv::saturate_cast<uchar>(Gray_product_sum / weight_sum);
    }

}

//手动实现cpu版本的bF
bool bilateralFilter_cpu(const char* binPath, const char* cfgPath, const char* outBinPath, const char* outImgPath){
    //step1: 获取图像, 默认RGB
    cv::Mat inImg;
    bool flag1 = getBin(binPath, inImg);

    //step2: 获取cfg
    int radius = 5;
    float sigma_spatial = 3.0f;
    float sigma_color = 30.0f;
    bool flag2 = getCfg(cfgPath, radius, sigma_spatial, sigma_color);

    if(!(flag1 && flag2)){
        return false;
    }

    int width = inImg.cols;
    int height = inImg.rows;

    //step3: 手动双边滤波
    cv::Mat inImg_reflected101 = Reflect101(inImg, radius);//获取reflect101之后的大图
    cv::Mat outImg = cv::Mat::zeros(height, width, inImg.type());
    for(int i = 0; i < height; ++i){//行
        for(int j = 0; j < width; ++j){//列
            int relative_i = i + radius;//映射到大图上的行坐标
            int relative_j = j + radius;//映射到大图上的列坐标
            if(inImg.channels() == 3){//RGB图像
                outImg.at<cv::Vec3b>(i, j) = bilateralFilterPixel<cv::Vec3b>(relative_i, relative_j, radius, inImg_reflected101, sigma_spatial, sigma_color);
            }
            else{//灰度图像
                outImg.at<uchar>(i, j) = bilateralFilterPixel<uchar>(relative_i, relative_j, radius, inImg_reflected101, sigma_spatial, sigma_color);
            }
        }
    }

    //step4: 输出bin和图像
    genBin(outBinPath, outImg);
    genImg(outImgPath, outImg, true);

    return true;
}





float diff(const char* binPath_ref, const char* binPath_test){
    std::ifstream file_ref(binPath_ref, std::ios::binary);
    std::ifstream file_test(binPath_test, std::ios::binary);

    if(!file_ref || !file_test){
        std::cerr << "Error: Could not open file for comparison." << std::endl;
        return -1.0f;
    }

    int w1, h1, c1;
    int w2, h2, c2;
    file_ref.read(reinterpret_cast<char*>(&w1), sizeof(int));
    file_ref.read(reinterpret_cast<char*>(&h1), sizeof(int));
    file_ref.read(reinterpret_cast<char*>(&c1), sizeof(int));
    file_test.read(reinterpret_cast<char*>(&w2), sizeof(int));
    file_test.read(reinterpret_cast<char*>(&h2), sizeof(int));
    file_test.read(reinterpret_cast<char*>(&c2), sizeof(int));
    if(w1 != w2 || h1 != h2 || c1 != c2){
        std::cerr << "Error: Image size not match." << std::endl;
        return -1.0f;
    }

    size_t totalPixels = static_cast<size_t>(w1) * h1 * c1;
    std::vector<unsigned char> data_ref(totalPixels);
    std::vector<unsigned char> data_test(totalPixels);
    file_ref.read(reinterpret_cast<char*>(data_ref.data()), totalPixels);
    file_test.read(reinterpret_cast<char*>(data_test.data()), totalPixels);

    file_ref.close();
    file_test.close();

    int totalError = 0;
    for (size_t i = 0; i < totalPixels; ++i) {
        totalError += std::abs(int(data_ref[i]) - int(data_test[i]));
    }

    float MAE = float(totalError) / float(totalPixels);
    return MAE;
}








int main(){
    //opencv实现bF
    //bilateralFilter("resource/lena.bin", "resource/config.txt", "resource/lena_out_opencv.bin", "resource/lena_out_opencv.png");
    //bilateralFilter("resource/mandrill.bin", "resource/config.txt", "resource/mandrill_out_opencv.bin", "resource/mandrill_out_opencv.png");

    //手动实现bF
    //bilateralFilter_cpu("resource/lena.bin", "resource/config.txt", "resource/lena_out_cpu.bin", "resource/lena_out_cpu.png");
    //bilateralFilter_cpu("resource/mandrill.bin", "resource/config.txt", "resource/mandrill_out_cpu.bin", "resource/mandrill_out_cpu.png");

    //比较误差
    //float MAE = diff("resource/lena_out_gpu.bin", "resource/lena_out_cpu.bin");
    //float MAE = diff("resource/mandrill_out_opencv.bin", "resource/mandrill_out_cpu.bin");
    //std::cout << "MAE: " << MAE << std::endl;


    genTester();


    return 0;
}