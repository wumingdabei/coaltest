// Copyright 2022 lb
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/** @brief Returns the number of installed CUDA-enabled devices.
Use this function before any other CUDA functions calls. If OpenCV is compiled without CUDA support,
this function returns 0. If the CUDA driver is not installed, or is incompatible, this function
returns -1.
 */
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
using namespace std;
using namespace cv;
using namespace cv::cuda;
int main()
{
    int num_devices = getCudaEnabledDeviceCount();
    if (num_devices == 0 )
    {
        std::cout << "OpenCV is compiled without CUDA support" << endl;
        return -1;
    }
    else if (num_devices == -1)
    {
        std::cout << "CUDA driver is not installed" << endl;
        return -1;
    }
    else if (num_devices >= 1)
    {
        std::cout << "CUDA-Opencv can be used and the number of GPU is :" << num_devices << endl;
        return -1;
    }
    return 0;
}