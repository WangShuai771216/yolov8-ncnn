#include "AI_property.h"

// 将版本字符串分割成三个整数（主版本号、次版本号和修订号）
static std::vector<int> parseVersion(const std::string& version) {
    std::vector<int> versionParts;
    std::stringstream ss(version);
    std::string item;
    while (std::getline(ss, item, '.')) {
        versionParts.push_back(std::stoi(item));
    }
    return versionParts;
}

// 比较两个版本号，返回true如果v1 < v2
static bool isVersionLess(const std::vector<int>& v1, const std::vector<int>& v2) {
    for (size_t i = 0; i < v1.size(); ++i) {
        if (v1[i] < v2[i]) {
            return true;
        } else if (v1[i] > v2[i]) {
            return false;
        }
    }
    return false;
}

// 判断版本号是否在范围内
static bool isVersionInRange(const std::string& deviceVersion, const std::string& minVersion, const std::string& maxVersion) {
    std::vector<int> device = parseVersion(deviceVersion);
    std::vector<int> minVer = parseVersion(minVersion);
    std::vector<int> maxVer = parseVersion(maxVersion);
    
    return !isVersionLess(device, minVer) && !isVersionLess(maxVer, device);
}
// 递归创建目录的函数定义
static bool createDirectories(const std::string& path) {
    size_t pos = 0;
    std::string delimiter = "/";
    std::string currentPath;

    // 在Linux/Unix系统中，路径通常以"/"开头
    if (path[0] == '/') {
        pos = 1; // 跳过第一个字符
    }
    while ((pos = path.find(delimiter, pos)) != std::string::npos) {
        currentPath = path.substr(0, pos);
        if (!currentPath.empty() && access(currentPath.c_str(), F_OK) != 0) {
            if (mkdir(currentPath.c_str(), 0777) != 0 && errno != EEXIST) {
                std::cerr << "Error creating directory: " << currentPath << " - " << strerror(errno) << std::endl;
                return false;
            }
        }
        pos++;
    }
    // 处理最后一级目录
    if (!path.empty() && access(path.c_str(), F_OK) != 0) {
        if (mkdir(path.c_str(), 0777) != 0 && errno != EEXIST) {
            std::cerr << "Error creating directory: " << path << " - " << strerror(errno) << std::endl;
            return false;
        }
    }
    return true;
}
//写入结果
static void createAndWriteFile(const std::string& filePath, const std::string& content) {
    // 找到最后一个斜杠，确定目录路径
    size_t lastSlashPos = filePath.find_last_of("/");
    if (lastSlashPos == std::string::npos) {
        std::cerr << "Invalid file path: " << filePath << std::endl;
        return;
    }
    std::string dirPath = filePath.substr(0, lastSlashPos);
    // 创建目录（如果不存在）
    if (!createDirectories(dirPath)) {
        return;
    }
    // 创建并打开文件
    std::ofstream outFile(filePath);
    // 检查文件是否成功打开
    if (!outFile) {
        std::cerr << "Error creating file: " << filePath << std::endl;
        return;
    }
    // 向文件写入内容
    outFile << content;
    // 关闭文件
    outFile.close();
    std::cout << "File created and content written successfully at: " << filePath << std::endl;
}

static void createAndWriteFile_json(const std::string& filePath, json_object* jsonContent) {
#ifndef CROSS_COMPILE
    size_t lastSlashPos = filePath.find_last_of("/");
    if (lastSlashPos == std::string::npos) {
        std::cerr << "Invalid file path: " << filePath << std::endl;
        return;
    }
    std::string dirPath = filePath.substr(0, lastSlashPos);
    if (!createDirectories(dirPath)) {
        return;
    }
#endif
   
    // std::ofstream outFile(filePath);
    // if (!outFile) {
    //     std::cerr << "Error creating file: " << filePath << std::endl;
    //     return;
    // }
    // const char* jsonString = json_object_to_json_string_ext(jsonContent, JSON_C_TO_STRING_PRETTY);
    // outFile <<jsonString <<"\n";
    // outFile.close();
    // std::cout <<"File created and JSON content written successfully at: "<< filePath << std::endl;

    // Convert the JSON object to a string without pretty formatting (no new lines)
    const char *json_str = json_object_to_json_string_ext(jsonContent, JSON_C_TO_STRING_PLAIN);
    // 创建一个新的字符串来保存未转义的JSON字符串
    char *unescaped_json_str = strdup(json_str);
    for (char *p = unescaped_json_str; *p; ++p) {
        if (*p == '\\' && *(p+1) == '/') {
            memmove(p, p+1, strlen(p));
        }
    }
    printf("%s\n", unescaped_json_str);
    // Write the JSON string to a file
    FILE *file = fopen(filePath.c_str(), "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing\n");
        // json_object_put(jsonContent);
        return;
    }

    fprintf(file, "%s", unescaped_json_str);
    fclose(file);

    // Clean up
    // json_object_put(jsonContent);
}
// 解析JSON数据
static void parseJSON(const std::string& jsonStr, DeviceConfig& deviceConfig) {
    json_tokener* tok = json_tokener_new();
    json_object* jobj = json_tokener_parse_ex(tok, jsonStr.c_str(), jsonStr.length());
    json_object* jDevice = nullptr;
    json_object* jVersion = nullptr;
    json_object* jModels = nullptr;
    json_object* jjson_path = nullptr;

    // 获取顶层对象
    json_object_object_foreach(jobj, key, val) {
        if (strcmp(key, "device") == 0) {
            jDevice = val;
        } else if (strcmp(key, "version") == 0) {
            jVersion = val;
        } else if (strcmp(key, "json_path") == 0) {
            jjson_path = val;
        }else if (strcmp(key, "models") == 0) {
            jModels = val;
        }
    }

    // 获取设备名称和版本号
    if (jDevice) {
        deviceConfig.device = json_object_get_string(jDevice);
    }
    if (jVersion) {
        // 分割版本范围字符串
        std::string version_range = json_object_get_string(jVersion);
        size_t dashPos = version_range.find('-');
        deviceConfig.min_version = version_range.substr(0, dashPos);
        deviceConfig.max_version = version_range.substr(dashPos + 1);
    }
    if (jjson_path) {
        deviceConfig.json_path = json_object_get_string(jjson_path);
    }
    // 获取模型信息
    if (jModels) {
        int arraylen = json_object_array_length(jModels);
        for (int i = 0; i < arraylen; i++) {
            json_object* jModel = json_object_array_get_idx(jModels, i);
            Model model_;
            json_object_object_foreach(jModel, key, val) {
                if (strcmp(key, "name") == 0) {
                    model_.name = json_object_get_string(val);
                } else if (strcmp(key, "labels") == 0) {
                    int labelsLen = json_object_array_length(val);
                    for (int j = 0; j < labelsLen; ++j) {
                        json_object* jLabel = json_object_array_get_idx(val, j);
                        Label label;
                        json_object_object_foreach(jLabel, labelKey, labelVal) {
                            if (strcmp(labelKey, "confinence") == 0) {
                                json_object* jConfidence = json_object_array_get_idx(labelVal, 0);
                                label.confidence[0] = json_object_get_double(jConfidence);
                                jConfidence = json_object_array_get_idx(labelVal, 1);
                                label.confidence[1] = json_object_get_double(jConfidence);
                            } else if (strcmp(labelKey, "cls_name") == 0) {
                                label.cls_name = json_object_get_string(labelVal);
                            }
                        }
                        model_.labels.push_back(label);
                    }
                }
            }
            deviceConfig.models.push_back(model_);
        }
    }

    json_object_put(jobj);
    json_tokener_free(tok);
}

static bool AI_property_parse(std::string jsonStr, DeviceConfig& deviceConfig){
    //  // 读取JSON文件内容
    // std::ifstream file(json_file);
    // if (!file.is_open()) {
    //     std::cerr << "Error opening file." << std::endl;
    //     return false;
    // }

    // std::stringstream buffer;
    // buffer << file.rdbuf();
    // std::string jsonStr = buffer.str();

    // 解析JSON数据
    parseJSON(jsonStr, deviceConfig);

    // 打印设备配置信息
    std::cout << "Device: " << deviceConfig.device << std::endl;
    std::cout << "Version: " << deviceConfig.min_version <<" " << deviceConfig.max_version << std::endl;
    std::cout << "Models:" << std::endl;
    for (const auto& model : deviceConfig.models) {
        std::cout << "  Name: " << model.name << std::endl;
        for (const auto& label : model.labels) {
            std::cout << "  labels.cls_name: " << label.cls_name << std::endl;
            std::cout << "  labels.Confidence: [" << label.confidence[0] << ", " << label.confidence[1] << "]" << std::endl;

        }
        std::cout << std::endl;
    }

    return true;
}

static bool AI_data_return_decide(const char* model,DeviceConfig& deviceConfig,
                            std::vector<Object> objects,float& obj_prob){
    std::vector<const char*> class_names; // 使用 std::vector 替代 C 数组
    if (std::string(model).find("ywjc") != std::string::npos) {
        class_names = {"foreign_object"}; // 初始化数组
    }
    else if (std::string(model).find("zwjc") != std::string::npos) {
        class_names = {"dirty"}; // 初始化数组

    }else if (std::string(model).find("blob") != std::string::npos) {
         class_names = {"nozzle-all","nozzle-blob"}; // 初始化数组 
    }
    else if (std::string(model).find("warp") != std::string::npos) {
         class_names = {"warping"}; // 初始化数组 
    }else {
        class_names = {"nozzle-blob", "spaghetti", "stringing"}; // 初始化数组
    }

    bool model_found = false;   
    Model model_info;                          
    for (const auto& model_ : deviceConfig.models) {
        if (strcmp(model_.name.c_str(), model) == 0) {
            model_found = true;
            model_info = model_;
            break;
        }
    }
    if (model_found) {
        for(const auto& label_info : model_info.labels){
            for (const auto& obj : objects) {
                // std::cout << "model found:" << label_info.cls_name<<" "<< class_names[obj.lable] << std::endl;
                // std::cout<<"confidence:"<<obj.prob<<" "<<label_info.confidence[0]<<" "<<label_info.confidence[1]<<std::endl;
                if(class_names[obj.label] ==label_info.cls_name && 
                    obj.prob > label_info.confidence[0] && 
                    obj.prob <label_info.confidence[1]){
                        obj_prob = obj.prob;
                        // std::cout << "detect " << obj.lable << " " << obj.prob << std::endl;
                        return true;
                }
            }
                
        }
    }
    return false;
}

// 回调函数，用于处理接收到的响应数据
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t total_size = size * nmemb;
    response->append((char*)contents, total_size);
    return total_size;
}
//请求URL
static int curl_ai_property_data(std::string device,std::string& response){
    CURL* curl;
    CURLcode res;
    // 初始化 cURL
    curl = curl_easy_init();
    if (curl) {
        // 构建请求的 URL
        std::string url = "http://172.23.88.143:9002/api/ai_config_get?device=" + device;
        
        // 设置请求的 URL
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // 设置回调函数，用于接收响应数据
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        // 发送请求
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "cURL request failed: " << curl_easy_strerror(res) << std::endl;
        }

        // 输出响应结果
        std::cout << "Response: " << response << std::endl;
        
        // 清理 cURL
        curl_easy_cleanup(curl);
        std::string response_err = R"({"error":"File not found"})";
        // std::cout<<"Response: " << response_err << std::endl;
        static int curl_post_times = 0;
        
        if(response != response_err){
            return 0;
        }
        return curl_post_times--;
    }
    return -1;
}

void ai_inference_result_decide(const std::string device,const std::string device_vision,const char* mode,std::vector<Object> objects,cv::Mat m){
    std::string response;
    static bool stop_flag = false;
    int response_suc = 0;
    if(!stop_flag){
        response_suc = curl_ai_property_data(device,response);
        std::cout<<"response_suc = " <<response_suc<<std::endl;
        if(response_suc < -3){
            stop_flag = true;
            return;
        }
    }else{
        return;
    }
    
    
    if(!response_suc){
        DeviceConfig deviceConfig;
        parseJSON(response, deviceConfig);

        
        //判断数据是否需要回传
        if(isVersionInRange(device_vision, deviceConfig.min_version, deviceConfig.max_version) && objects.size()>0){
            float obj_prob = 0.0;
            if(AI_data_return_decide(mode,deviceConfig,objects,obj_prob)){
                std::cout << "AI data return success" << std::endl;
                time_t nowtime = time(NULL);
                struct tm *p;
                p = localtime(&nowtime);
                // std::cout<<"time:"<<p<<std::endl;
#ifndef CROSS_COMPILE
                std::string path = "./data/ai_image/ai_data.json"; // 你可以在这里指定路径
                std::string filename = "./data/ai_image/ai_property/" + device + "-" + mode + "-"
                 + std::to_string(p->tm_year + 1900) + "_" + std::to_string(p->tm_mon + 1) 
                 + "_" + std::to_string(p->tm_mday) + "_" + std::to_string(p->tm_hour) 
                 + "_" + std::to_string(p->tm_min) + "_" + std::to_string(p->tm_sec) + ".jpg";
#else
                std::string path = deviceConfig.json_path; // 你可以在这里指定路径
                std::string filename = "/usr/data/ai_image/ai_property/" + device + "-" + mode + "-" 
                + std::to_string(p->tm_year + 1900) + "_" + std::to_string(p->tm_mon + 1) 
                + "_" + std::to_string(p->tm_mday) + "_" + std::to_string(p->tm_hour) 
                + "_" + std::to_string(p->tm_min) + "_" + std::to_string(p->tm_sec) + ".jpg";

#endif
                std::string con_data_ai = std::to_string(obj_prob) +"|" + "print_id" + "|" + filename + "\n";
                // createAndWriteFile(path, content);
                
                
                std::chrono::milliseconds reqId_ms = std::chrono::duration_cast< std::chrono::milliseconds >(
                    std::chrono::system_clock::now().time_since_epoch());

                std::cout << reqId_ms.count() << std::endl;
                json_object* content = json_object_new_object();
                
                std::string reqId_ms_str = std::to_string(reqId_ms.count());
                json_object_object_add(content, "reqId", json_object_new_string(reqId_ms_str.c_str()));
                json_object_object_add(content, "dn", json_object_new_string("00000000000000"));
                json_object_object_add(content, "code", json_object_new_string("key609"));
                json_object_object_add(content, "data", json_object_new_string(con_data_ai.c_str())); 
                // json_object_object_add(content, "confinence", json_object_new_double(obj_prob));
                createAndWriteFile_json(path, content);
                json_object_put(content); // Free the JSON object
               
                size_t lastSlashPos = filename.find_last_of("/");
                if (lastSlashPos == std::string::npos) {
                    std::cerr << "Invalid file path: " << filename << std::endl;
                    return;
                }
                std::string dirPath = filename.substr(0, lastSlashPos);
                if (!createDirectories(dirPath)) {
                    return;
                }
                cv::imwrite(filename,m);
            }
        }
    }

}




