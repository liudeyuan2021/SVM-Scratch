#include "hog.h"
// #include "sr_common_inc.h"
// #include "ocl_runtime.h"


#define PI (std::acos(-1))

// #if 1
// #undef PIP_PROFILE_TAG
// #undef PIP_PROFILE_TAG_CPU

// #define PIP_PROFILE_TAG(x)
// #define PIP_PROFILE_TAG_CPU(x)
// #endif

void _hog_normalize_block(cv::Mat block, cv::Mat &dst, std::string method, float eps=1e-5)
{
    if(method == "L1")
    {
        printf("Not support L1\n");
        // out = block / (np.sum(np.abs(block)) + eps)
    }
    else if(method == "L1-sqrt")
    {
        printf("Not support L1-sqrt\n");
        // out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    }
    else if(method == "L2")
    {
        printf("Not support L2\n");
        // out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    }
    else if(method == "L2-Hys")
    {
        // out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        // out = np.minimum(out, 0.2)
        // out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
        cv::Mat block_tmp = block.clone();
        cv::multiply(block, block, block_tmp);
        cv::Scalar block_sum = cv::sum(block_tmp);
        float eps_squar = eps*eps;
        float block_sq = block_sum[0] + eps_squar;
        float block_sqrt = std::sqrt(block_sq);
        for(uint32_t i=0;i<block.total();i++)
        {
            float* tmpdata = (float*)block.data+i;
            *tmpdata = std::min((*tmpdata/block_sqrt), 0.2f);
            *((float*)block.data+i) = *tmpdata;
        }

        cv::Mat out = block.clone();
        cv::multiply(block, block, out);
        cv::Scalar out_sum = cv::sum(out);
        float out_sq = out_sum[0] + eps_squar;
        float out_sqrt = std::sqrt(out_sq);
        for(uint32_t i=0;i<out.total();i++)
        {
            float* tmpdata = (float*)block.data+i;
            *tmpdata = (*tmpdata)/out_sqrt;
            *((float*)out.data+i) = *tmpdata;
        }

        dst = out.clone();
    }
    else
    {
        printf("Selected block normalization method is invalid.\n");
    }
    return;
}

float _cell_hog(cv::Mat magnitude, cv::Mat orientation, float orientation_start, float orientation_end,
                int cell_columns, int cell_rows, int column_index, int row_index, int size_columns, int size_rows,
                int range_rows_start, int range_rows_stop, int range_columns_start, int range_columns_stop
                )
{
    float total = 0.0;
    for(int cell_row=range_rows_start; cell_row<range_rows_stop; cell_row++)
    {
        int cell_row_index = row_index + cell_row;
        if(cell_row_index < 0 || cell_row_index >= size_rows)
        {
            continue;
        }
        for(int cell_column=range_columns_start; cell_column<range_columns_stop; cell_column++)
        {
            int cell_column_index = column_index + cell_column;
            if(cell_column_index < 0 || 
                cell_column_index >= size_columns ||
                orientation.at<float>(cell_row_index, cell_column_index) >= orientation_start ||
                orientation.at<float>(cell_row_index, cell_column_index) < orientation_end)
            {
                continue;
            }
            total += magnitude.at<float>(cell_row_index, cell_column_index);
        } 
    }
    return (total/(cell_rows * cell_columns));
}


// void get_orientation_magnitude(StMat gradient_row, StMat gradient_col, StMat &dst_orientation, StMat &dst_magnitude)
// {
//     OclRuntime *ocl = GetOclRuntime();
//     cl_command_queue cmdq = ocl->getCommandQueue();
//     cl_mem clMemSrc0 = gradient_row.gpuData();
//     cl_mem clMemSrc1 = gradient_col.gpuData();
//     cl_mem clMemDst0 = dst_orientation.gpuData();
//     cl_mem clMemDst1 = dst_magnitude.gpuData();

//     int width = gradient_row.cols();
//     int height = gradient_row.rows();

//     cl_kernel kernel;
//     kernel = (*ocl)["get_orientation_magnitude_kernel"];
     
//     cl_int err = CL_SUCCESS;
//     int i = 0;
//     clSetKernelArg(kernel, i++, sizeof(cl_mem), &(clMemSrc0));  CHECK_ERR(err);
//     clSetKernelArg(kernel, i++, sizeof(cl_mem), &(clMemSrc1));  CHECK_ERR(err);
//     clSetKernelArg(kernel, i++, sizeof(cl_mem), &(clMemDst0));  CHECK_ERR(err);
//     clSetKernelArg(kernel, i++, sizeof(cl_mem), &(clMemDst1));  CHECK_ERR(err);
//     clSetKernelArg(kernel, i++, sizeof(int), &(width));

//     size_t gs[2] = {(size_t)width, (size_t)(height)};
//     err = ocl->oclEnqueueNDRangeKernel(kernel, 2, NULL, gs, NULL, 0, NULL, NULL);    CHECK_ERR(err);
// }

void _hog_histograms(cv::Mat gradient_columns, cv::Mat gradient_rows, int cell_columns, int cell_rows, 
                    int size_columns, int size_rows, int number_of_cells_columns, int number_of_cells_rows,
                    int number_of_orientations, cv::Mat &orientation_histogram)    
{
    // PIP_PROFILE_BEGIN;
    // magnitude = np.hypot(gradient_columns, gradient_rows)
    // hypot 操作： hypot(x,y) = sqrt(x*x+y*y)
    cv::Mat g_col_tmp;
    cv::Mat g_row_tmp;
    cv::Mat g_tmp;
    cv::Mat magnitude;
    cv::multiply(gradient_columns, gradient_columns, g_col_tmp);
    cv::multiply(gradient_rows, gradient_rows, g_row_tmp);
    g_tmp = g_col_tmp + g_row_tmp;
    cv::sqrt(g_tmp, magnitude);
    
    // PIP_PROFILE_TAG("_hog_histograms 01 finish ");

    // orientation = np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % 180
    // np.arctan2(gradient_rows, gradient_columns) 操作：计算 矩阵对应位上的 反正切值
    int imgH = gradient_rows.rows;
    int imgW = gradient_rows.cols;
    cv::Mat orientation = cv::Mat::zeros(imgH, imgW, CV_32FC1);
    for(int r=0; r<imgH; r++)
    {
        for(int c=0; c<imgW; c++)
        {
            float a = gradient_rows.at<float>(r, c);
            float b = gradient_columns.at<float>(r, c);
            float res_tmp = atan2(a, b) *180/PI;
            /***
             * Need Fix : float precision error
             ***/
            if (a == 0) res_tmp = 0.0f;
            if (a != 0 && b == 0) res_tmp=90.0f;
            // float res = 0.0;
            // if(res_tmp == 0 || res_tmp == 180)
            //     res = 0.0;
            // else if(res_tmp < 0)
            // {
            //     res = res_tmp + 180;
            // }
            // else
            // {
            //     res = res_tmp;
            // }
            float res = res_tmp < 0.0f ? res_tmp + 180.0f : res_tmp;
            orientation.at<float>(r, c) = res; 
        }
    }
    // PIP_PROFILE_TAG("_hog_histograms 02 finish ");
    // display<float>("orientation", orientation);

    {
        std::cout << "------------------------" << std::endl;
        std::cout << "Orientation: " << std::endl;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                std::cout << orientation.at<float>(i, j) << " ";    
            }
            std::cout << std::endl;
        }
        std::cout << "------------------------" << std::endl  << std::endl ;
    }

    int r_0 = int(cell_rows / 2);
    int c_0 = int(cell_columns / 2);
    int cc = cell_rows * number_of_cells_rows;
    int cr = cell_columns * number_of_cells_columns;
    int range_rows_stop = int((cell_rows + 1) / 2);
    int range_rows_start = int(-(cell_rows / 2));
    int range_columns_stop = int((cell_columns + 1) / 2);
    int range_columns_start = int(-(cell_columns / 2));
    /***
     * Need Fix : set float to int
     ***/
    float number_of_orientations_per_180 = float(int(180. / number_of_orientations));

    
    // compute orientations integral images
    for(int i=0; i<number_of_orientations; i++)
    {
        // isolate orientations in this range
        float orientation_start = number_of_orientations_per_180 * (i + 1);
        float orientation_end = number_of_orientations_per_180 * i;
        int c = c_0;
        int r = r_0;
        int r_i = 0;
        int c_i = 0;

        while(r < cc)
        {
            c_i = 0;
            c = c_0;
            while(c < cr)
            {
                float tmp = _cell_hog(magnitude, orientation,
                                        orientation_start, orientation_end,
                                        cell_columns, cell_rows, c, r,
                                        size_columns, size_rows,
                                        range_rows_start, range_rows_stop,
                                        range_columns_start, range_columns_stop);
                // Mat 多维数据访问利用 Mat.data数据块的索引来读写数据
                uint32_t index = r_i*orientation_histogram.step1(0) + c_i*orientation_histogram.step1(1) + i*orientation_histogram.step1(2);
                float *data = (float*)orientation_histogram.data+index;
                *data = tmp;
                c_i += 1;
                c += cell_columns;
            }
            r_i += 1;
            r += cell_rows;
        }
    }
    // PIP_PROFILE_TAG("_hog_histograms 03 finish ");

    // printf("=============debug info========================\n")
    // uint32_t index = 0;
    // for(int a=0; a<orientation_histogram.size[0]; a++)
    // {
    //     for(int b=0; b<orientation_histogram.size[1]; b++)
    //     {
    //         for(int c=0; c<orientation_histogram.size[2]; c++)
    //         {
    //             index = a*orientation_histogram.step1(0) + b*orientation_histogram.step1(1) + c*orientation_histogram.step1(2);
    //             float* data = (float*)orientation_histogram.data + index;
    //             printf("%.8f,", *(data));
    //         }
    //         printf("|\n");
    //     }
    //     printf("\n");
    //     break;
    // }
    // test the display Mat with 4 dims
    // int size1D[1] = {5};
    // int size2D[2] = {5, 4};
    // int size3D[3] = {5, 4, 3};
    // int size4D[4] = {5, 4, 3, 2};
    // int size5D[5] = {5, 4, 3, 2, 2};
    // cv::Mat tmp1(1, size1D, CV_32FC1, cv::Scalar(1.2));
    // cv::Mat tmp2(2, size2D, CV_32FC1, cv::Scalar(1.2));
    // cv::Mat tmp3(3, size3D, CV_32FC1, cv::Scalar(1.6));
    // cv::Mat tmp4(4, size4D, CV_32FC1, cv::Scalar(1.4));
    // cv::Mat tmp5(5, size5D, CV_32FC1, cv::Scalar(1.5));
    // display<float>("tmp1", tmp1);
    // display<float>("tmp2", tmp2);
    // display<float>("tmp3", tmp3);
    // display<float>("tmp4", tmp4);
    // display<float>("tmp5", tmp5);
    // printf("=============debug info========================\n")
}

// void _hog_histograms_gpu(StMat gradient_columns, StMat gradient_rows, int cell_columns, int cell_rows, 
//                          int size_columns, int size_rows, int number_of_cells_columns, int number_of_cells_rows,
//                          int number_of_orientations, cv::Mat &orientation_histogram)    
// {
//     PIP_PROFILE_BEGIN;

//     // magnitude = np.hypot(gradient_columns, gradient_rows)
//     // hypot 操作： hypot(x,y) = sqrt(x*x+y*y)

//     // orientation = np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % 180
//     // np.arctan2(gradient_rows, gradient_columns) 操作：计算 矩阵对应位上的 反正切值
//     int imgH = gradient_rows.rows();
//     int imgW = gradient_rows.cols();

//     StMat orientation_gpu = StMat(StMat::GPU_BUF, imgH, imgW, gradient_rows.type());
//     StMat magnitude_gpu = StMat(StMat::GPU_BUF, imgH, imgW, gradient_rows.type());
//     get_orientation_magnitude(gradient_rows, gradient_columns, orientation_gpu, magnitude_gpu);

//     cv::Mat magnitude = cv::Mat::zeros(imgH, imgW, CV_32FC1);
//     cv::Mat orientation = cv::Mat::zeros(imgH, imgW, CV_32FC1);

//     // orientation_gpu.asBuffer(StMat::CPU).refMat().convertTo(orientation, orientation.type());
//     // magnitude_gpu.asBuffer(StMat::CPU).refMat().convertTo(magnitude, magnitude.type());
//     orientation_gpu.asBuffer(StMat::CPU).refMat().copyTo(orientation);
//     magnitude_gpu.asBuffer(StMat::CPU).refMat().copyTo(magnitude);

//     PIP_PROFILE_TAG("_hog_histograms 02 finish ");
//     // display<float>("orientation", orientation);

//     int r_0 = int(cell_rows / 2);
//     int c_0 = int(cell_columns / 2);
//     int cc = cell_rows * number_of_cells_rows;
//     int cr = cell_columns * number_of_cells_columns;
//     int range_rows_stop = int((cell_rows + 1) / 2);
//     int range_rows_start = int(-(cell_rows / 2));
//     int range_columns_stop = int((cell_columns + 1) / 2);
//     int range_columns_start = int(-(cell_columns / 2));
//     int number_of_orientations_per_180 = int(180. / number_of_orientations);
    
//     // compute orientations integral images
//     for(int i=0; i<number_of_orientations; i++)
//     {
//         // isolate orientations in this range
//         int orientation_start = number_of_orientations_per_180 * (i + 1);
//         int orientation_end = number_of_orientations_per_180 * i;
//         int c = c_0;
//         int r = r_0;
//         int r_i = 0;
//         int c_i = 0;

//         while(r < cc)
//         {
//             c_i = 0;
//             c = c_0;
//             while(c < cr)
//             {
//                 float tmp = _cell_hog(magnitude, orientation,
//                                         orientation_start, orientation_end,
//                                         cell_columns, cell_rows, c, r,
//                                         size_columns, size_rows,
//                                         range_rows_start, range_rows_stop,
//                                         range_columns_start, range_columns_stop);
//                 // Mat 多维数据访问利用 Mat.data数据块的索引来读写数据
//                 uint32_t index = r_i*orientation_histogram.step1(0) + c_i*orientation_histogram.step1(1) + i*orientation_histogram.step1(2);
//                 float *data = (float*)orientation_histogram.data+index;
//                 *data = tmp;
//                 c_i += 1;
//                 c += cell_columns;
//             }
//             r_i += 1;
//             r += cell_rows;
//         }
//     }
//     PIP_PROFILE_TAG("_hog_histograms 03 finish ");

// }

// void get_g_row_col_mat(StMat src, StMat& dst_row, StMat &dst_col)
// {
//     OclRuntime *ocl = GetOclRuntime();
//     cl_command_queue cmdq = ocl->getCommandQueue();
//     cl_mem clMemSrc = src.gpuData();
//     cl_mem clMemDst_row = dst_row.gpuData();
//     cl_mem clMemDst_col = dst_col.gpuData();

//     int width = src.cols();
//     int height = src.rows();
//     int gs_w = width - 2;
//     int gs_h = height;

//     cl_kernel kernel;
//     kernel = (*ocl)["get_g_col_mat_kernel"];
     
//     cl_int err = CL_SUCCESS;
//     int i = 0;
//     clSetKernelArg(kernel, i++, sizeof(cl_mem), &(clMemSrc));       CHECK_ERR(err);
//     clSetKernelArg(kernel, i++, sizeof(cl_mem), &(clMemDst_col));   CHECK_ERR(err);
//     clSetKernelArg(kernel, i++, sizeof(int), &(width));

//     size_t gs[2] = {(size_t)gs_w, (size_t)(gs_h)};
//     err = ocl->oclEnqueueNDRangeKernel(kernel, 2, NULL, gs, NULL, 0, NULL, NULL);    CHECK_ERR(err);
//     ocl->oclFinish();

//     kernel = (*ocl)["get_g_row_mat_kernel"];
//     gs_w = width;
//     gs_h = height - 2;
     
//     i = 0;
//     clSetKernelArg(kernel, i++, sizeof(cl_mem), &(clMemSrc));       CHECK_ERR(err);
//     clSetKernelArg(kernel, i++, sizeof(cl_mem), &(clMemDst_row));   CHECK_ERR(err);
//     clSetKernelArg(kernel, i++, sizeof(int), &(width));

//     gs[0] = (size_t)gs_w;
//     gs[1] = (size_t)gs_h;
//     err = ocl->oclEnqueueNDRangeKernel(kernel, 2, NULL, gs, NULL, 0, NULL, NULL);    CHECK_ERR(err);
//     ocl->oclFinish();
// }

void _hog_channel_gradient(cv::Mat channel, cv::Mat &g_row, cv::Mat &g_col)
{
    // Compute unnormalized gradient image along `row` and `col` axes.
    // Parameters
    // ----------
    // channel : (M, N) ndarray
    //     Grayscale image or one of image channel.
    // Returns
    // -------
    // g_row, g_col : channel gradient along `row` and `col` axes correspondingly.
    int imgH = channel.rows;
    int imgW = channel.cols;
    g_row = cv::Mat::zeros(imgH, imgW, CV_32FC1);
    g_col = cv::Mat::zeros(imgH, imgW, CV_32FC1);

    cv::Mat r_rang0 = channel.rowRange(cv::Range(2, imgH));
    cv::Mat r_rang1 = channel.rowRange(cv::Range(0, imgH-2));
    cv::Mat tmp_r = r_rang0 - r_rang1;
    tmp_r.copyTo(g_row.rowRange(1, imgH-1));
    
    cv::Mat c_rang0 = channel.colRange(cv::Range(2, imgW));
    cv::Mat c_rang1 = channel.colRange(cv::Range(0, imgW-2));
    cv::Mat tmp_c = c_rang0 - c_rang1;
    tmp_c.copyTo(g_col.colRange(1, imgW-1));
}

// void _hog_channel_gradient_gpu(StMat channel, StMat &g_row, StMat &g_col)
// {
//     // Compute unnormalized gradient image along `row` and `col` axes.
//     // Parameters
//     // ----------
//     // channel : (M, N) ndarray
//     //     Grayscale image or one of image channel.
//     // Returns
//     // -------
//     // g_row, g_col : channel gradient along `row` and `col` axes correspondingly.
//     int imgH = channel.rows();
//     int imgW = channel.cols();
//     g_row = StMat::zeros(StMat::GPU_BUF, imgH, imgW, CV_32FC1);
//     g_col = StMat::zeros(StMat::GPU_BUF, imgH, imgW, CV_32FC1);

//     get_g_row_col_mat(channel, g_row, g_col);
// }

void getHogFeature(cv::Mat image, cv::Mat &imgfeature)
{
    // PIP_PROFILE_BEGIN;

    int orientations       = 8;
    int pixels_per_cell[2] = {16, 16};
    int cells_per_block[2] = {1, 1};
    std::string block_norm = "L2-Hys";
    bool transform_sqrt    = false;
    bool multichannel      = false;
    bool visualize         = false;     // unused
    bool feature_vector    = true;      // unused

    imgfeature.release();
    multichannel = image.dims > 2 ? true:false;
    int ndim_spatial = image.dims;
    if(ndim_spatial != 2)
    {
        printf("Only images with 2 spatial dimensions are supported. If using with color/multichannel images, specify `multichannel=True`.\n");
        return;
    }

    if(transform_sqrt)
    {
       cv::sqrt(image,image);
    }

    cv::Mat g_row;
    cv::Mat g_col;
    if(multichannel)
    {
        // 目前只实现单通道图片
        printf("Only supported singal channel\n");
        return;
    }
    else
    {
        _hog_channel_gradient(image, g_row, g_col);
    }
    // PIP_PROFILE_TAG_CPU("_hog_channel_gradient");

    int s_row = image.rows;
    int s_col = image.cols;
    int c_row = pixels_per_cell[0];
    int c_col = pixels_per_cell[1];
    int b_row = cells_per_block[0];
    int b_col = cells_per_block[1];

    int n_cells_row = int(s_row / c_row);
    int n_cells_col = int(s_col / c_col);

    // compute orientations integral images
    int ori_his3D[3] = {n_cells_row, n_cells_col, orientations};
    cv::Mat orientation_histogram(3, ori_his3D, CV_32FC1, cv::Scalar(0));
    _hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row, n_cells_col, n_cells_row, orientations, orientation_histogram);
    // PIP_PROFILE_TAG_CPU("_hog_histograms");
    
    {
        std::cout << "------------------------" << std::endl;
        std::cout << "Orientation Histogram: " << std::endl;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                std::cout << orientation_histogram.at<float>(i, j, 0) << " ";    
            }
            std::cout << std::endl;
        }
        std::cout << "------------------------" << std::endl  << std::endl ;
    }

    if(visualize)
    {
        // 用于可视化结果，暂不实现
    }

    int n_blocks_row = (n_cells_row - b_row) + 1;
    int n_blocks_col = (n_cells_col - b_col) + 1;
    int normblock_5D[5] = {n_blocks_row, n_blocks_col, b_row, b_col, orientations};
    cv::Mat normalized_blocks(5, normblock_5D, CV_32FC1, cv::Scalar(0));

    int block_3D[3] = {b_row, b_col, orientations};
    cv::Mat block(3, block_3D, CV_32FC1, cv::Scalar(0));

    for(int r=0; r<n_blocks_row; r++)
    {
        for(int c=0; c<n_blocks_col; c++)
        {
            // 构造 block 并把数据正确取到
            cv::Mat block(3, block_3D, CV_32FC1, cv::Scalar(0));
            uint32_t index = 0;
            uint32_t set_index = 0;
            for(int a0=r; a0<r+b_row; a0++)
            {
                for(int a1=c; a1<c+b_col; a1++)
                {
                    for(int a2=0; a2<orientations; a2++)
                    {
                        index = a0*orientation_histogram.step1(0) + a1*orientation_histogram.step1(1) + a2*orientation_histogram.step1(2);
                        float *input_data = (float*)orientation_histogram.data + index;
                        set_index = (a0-r)*block.step1(0) + (a1-c)*block.step1(1) + a2*block.step1(2);
                        float *set_data = (float*)block.data + set_index;
                        *set_data = *input_data;
                    }
                }
            }
            // display<float>("block", block, 2);
        
            cv::Mat nor_block;
            _hog_normalize_block(block, nor_block, block_norm);
            index = 0;
            set_index = 0;
            for(int a0=0; a0<b_row; a0++)
            {
                for(int a1=0; a1<b_col; a1++)
                {
                    for(int a2=0; a2<orientations; a2++)
                    {
                        index = a0*nor_block.step1(0) + a1*nor_block.step1(1) + a2*nor_block.step1(2);
                        float *input_data = (float*)nor_block.data + index;
                        set_index = r*normalized_blocks.step1(0) + c*normalized_blocks.step1(1) + a0*normalized_blocks.step1(2) +
                                    a1*normalized_blocks.step1(3) + a2*normalized_blocks.step1(4);
                        float *set_data = (float*)normalized_blocks.data + set_index;
                        *set_data = *input_data;
                    }
                }
            }
        }
    }
    // PIP_PROFILE_TAG_CPU("_hog_normalize_block");

    // display<float>("normalized_blocks", normalized_blocks, 1);
    // data 的数据应该不需要专门做 ravel 的操作，可以直接顺序访问（存储就是按顺序存的）
    // for(uint32_t i=0; i<normalized_blocks.total(); i++)
    // {
    //     if(i%8==0)
    //         printf("\n");
    //     printf("%.8f, ", *((float*)normalized_blocks.data + i));
    //     if (i> 32)
    //         break;
    // }
    imgfeature = normalized_blocks.clone();
    return;
}

// void getHogFeature_gpu(StMat image, cv::Mat &imgfeature)
// {
//     PIP_PROFILE_BEGIN;

//     int orientations       = 8;
//     int pixels_per_cell[2] = {16, 16};
//     int cells_per_block[2] = {1, 1};
//     std::string block_norm = "L2-Hys";
//     bool transform_sqrt    = false;
//     bool multichannel      = false;
//     bool visualize         = false;     // unused
//     bool feature_vector    = true;      // unused

//     imgfeature.release();
//     multichannel = image.channels() > 1 ? true:false;

//     if(transform_sqrt)
//     {
//         // cv::sqrt(image,image);
//     }

//     StMat g_row;
//     StMat g_col;
//     if(multichannel)
//     {
//         // 目前只实现单通道图片
//         printf("Only supported singal channel\n");
//         return;
//     }
//     else
//     {
//         _hog_channel_gradient_gpu(image, g_row, g_col);
//     }
//     PIP_PROFILE_TAG_CPU("_hog_channel_gradient");

//     int s_row = image.rows();
//     int s_col = image.cols();
//     int c_row = pixels_per_cell[0];
//     int c_col = pixels_per_cell[1];
//     int b_row = cells_per_block[0];
//     int b_col = cells_per_block[1];

//     int n_cells_row = int(s_row / c_row);
//     int n_cells_col = int(s_col / c_col);

//     // compute orientations integral images
//     int ori_his3D[3] = {n_cells_row, n_cells_col, orientations};
//     cv::Mat orientation_histogram(3, ori_his3D, CV_32FC1, cv::Scalar(0));
//     _hog_histograms_gpu(g_col, g_row, c_col, c_row, s_col, s_row, n_cells_col, n_cells_row, orientations, orientation_histogram);
//     PIP_PROFILE_TAG_CPU("_hog_histograms");
    
//     if(visualize)
//     {
//         // 用于可视化结果，暂不实现
//     }

//     int n_blocks_row = (n_cells_row - b_row) + 1;
//     int n_blocks_col = (n_cells_col - b_col) + 1;
//     int normblock_5D[5] = {n_blocks_row, n_blocks_col, b_row, b_col, orientations};
//     cv::Mat normalized_blocks(5, normblock_5D, CV_32FC1, cv::Scalar(0));

//     int block_3D[3] = {b_row, b_col, orientations};
//     cv::Mat block(3, block_3D, CV_32FC1, cv::Scalar(0));

//     for(int r=0; r<n_blocks_row; r++)
//     {
//         for(int c=0; c<n_blocks_col; c++)
//         {
//             // 构造 block 并把数据正确取到
//             cv::Mat block(3, block_3D, CV_32FC1, cv::Scalar(0));
//             uint32_t index = 0;
//             uint32_t set_index = 0;
//             for(int a0=r; a0<r+b_row; a0++)
//             {
//                 for(int a1=c; a1<c+b_col; a1++)
//                 {
//                     for(int a2=0; a2<orientations; a2++)
//                     {
//                         index = a0*orientation_histogram.step1(0) + a1*orientation_histogram.step1(1) + a2*orientation_histogram.step1(2);
//                         float *input_data = (float*)orientation_histogram.data + index;
//                         set_index = (a0-r)*block.step1(0) + (a1-c)*block.step1(1) + a2*block.step1(2);
//                         float *set_data = (float*)block.data + set_index;
//                         *set_data = *input_data;
//                     }
//                 }
//             }
//             // display<float>("block", block, 2);
        
//             cv::Mat nor_block;
//             _hog_normalize_block(block, nor_block, block_norm);
//             index = 0;
//             set_index = 0;
//             for(int a0=0; a0<b_row; a0++)
//             {
//                 for(int a1=0; a1<b_col; a1++)
//                 {
//                     for(int a2=0; a2<orientations; a2++)
//                     {
//                         index = a0*nor_block.step1(0) + a1*nor_block.step1(1) + a2*nor_block.step1(2);
//                         float *input_data = (float*)nor_block.data + index;
//                         set_index = r*normalized_blocks.step1(0) + c*normalized_blocks.step1(1) + a0*normalized_blocks.step1(2) +
//                                     a1*normalized_blocks.step1(3) + a2*normalized_blocks.step1(4);
//                         float *set_data = (float*)normalized_blocks.data + set_index;
//                         *set_data = *input_data;
//                     }
//                 }
//             }
//         }
//     }
//     PIP_PROFILE_TAG_CPU("_hog_normalize_block");

//     // display<float>("normalized_blocks", normalized_blocks, 1);
//     // data 的数据应该不需要专门做 ravel 的操作，可以直接顺序访问（存储就是按顺序存的）
//     // for(uint32_t i=0; i<normalized_blocks.total(); i++)
//     // {
//     //     if(i%8==0)
//     //         printf("\n");
//     //     printf("%.8f, ", *((float*)normalized_blocks.data + i));
//     //     if (i> 32)
//     //         break;
//     // }
//     imgfeature = normalized_blocks.clone();
//     return;
// }
