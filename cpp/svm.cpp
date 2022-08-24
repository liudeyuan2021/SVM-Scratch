#include "svm.h"
// #include <arm_neon.h>
// #include "sr_common_inc.h"

// #if 1
// #undef PIP_PROFILE_TAG
// #undef PIP_PROFILE_TAG_CPU

// #define PIP_PROFILE_TAG(x)
// #define PIP_PROFILE_TAG_CPU(x)
// #endif

float k_function(float* x_value, int x_dim, float* y_value, int y_dim, float gamma)
{
    float sum=0;
    int dim = x_dim > y_dim ? y_dim : x_dim;
    for(int i=0; i<dim; i++)
    {
        sum += (x_value[i]-y_value[i]) * (x_value[i]-y_value[i]);
    }

    for(int i=dim; i<x_dim; i++)
    {
        sum += x_value[i] * x_value[i];
    }
    for(int i=dim; i<y_dim; i++)
    {
        sum += y_value[i] * y_value[i];
    }

    return std::exp(-gamma * sum);
}

int svm_predict(float* feature, int32_t feature_len, float* SV, int32_t SV_len, float gamma, 
                    int nr_class, int l, int32_t* nSV, float* sv_coef, int32_t sv_coef_len, 
                    float rho, std::vector<int> label)
{
    // PIP_PROFILE_BEGIN;

    std::vector<float> kvalue(l);
    for(int i=0; i<l; i++)
    {
        int index = i*feature_len;
        // 这里两个 dim的长度都为3600 即feature_len 的长度
        kvalue[i] = k_function(feature, feature_len, SV+index, feature_len, gamma);
    }
    // PIP_PROFILE_TAG_CPU("k_function");
    // printf("%.8f,%.8f,%.8f,%.8f,%.8f \n", kvalue[0],kvalue[1],kvalue[2],kvalue[3],kvalue[4]);

    std::vector<int> start(nr_class);
    start[0] = 0;
    for(int i=1; i<nr_class; i++)
    {
        start[i] = start[i-1] + nSV[i-1];
    }
    // PIP_PROFILE_TAG_CPU("start");
    // printf("%d, %d\n", nSV[0], nSV[1]);
    // printf("%d, %d\n", start[0], start[1]);

    std::vector<int> vote(nr_class, 0);
    for(int i=0, j=i+1; j<nr_class; j++)
    {
        float sum=0;
        int si = start[i];
        int sj = start[j];
        int ci = nSV[i];
        int cj = nSV[j];

        std::vector<float> coef1(sv_coef_len);
        std::vector<float> coef2(sv_coef_len);
        uint32_t coef1_index = 0;
        uint32_t coef2_index = 0;
        coef1_index = (j-1) * sv_coef_len;
        coef2_index = (i) * sv_coef_len;
        std::cout << sv_coef_len << std::endl;
        for(int n=0; n<sv_coef_len; n++)
        {
            coef1[n] = sv_coef[coef1_index+n];
            coef2[n] = sv_coef[coef2_index+n];
        }
        // printf("coef1 is %.8f, %.8f, %.8f, %.8f \n", coef1[1], coef1[2], coef1[3], coef1[4]);
        // printf("coef2 is %.8f, %.8f, %.8f, %.8f \n", coef2[1], coef2[2], coef2[3], coef2[4]);

        for(int k=0; k<ci; k++)
        {
            sum += coef1[int(si+k)] * kvalue[int(si+k)];
        }
        for(int k=0; k<cj; k++)
        {
            sum += coef2[int(sj+k)] * kvalue[int(sj+k)];
        }
        // printf("sum is %.8f\n", sum);

        sum -= rho;

        if(sum > 0)
        {
            vote[i] += 1;
        }
        else
        {
            vote[j] += 1;
        }
        // printf("sum is %.8f\n", sum);
        // printf("vote[i] is %d\n", vote[i]);
        // printf("vote[j] is %d\n", vote[j]);
    }
    // PIP_PROFILE_TAG_CPU("vote");

    int vote_max_id = 0;
    for(int i=1; i<nr_class; i++)
    {
        if(vote[i] > vote[vote_max_id])
            vote_max_id = i;
    }
    // PIP_PROFILE_TAG_CPU("vote_max_id");


    // printf("res is %d\n",label[vote_max_id]);
    return label[vote_max_id];
}


void getSvmPredict(float* feature, int feature_len, int &dec_values)
{
    /***
     * Need Fix : error set uint32_t to uint8_t
     ***/
    uint32_t support_len    = (uint32_t)(sizeof(support_int32_bin)        /  sizeof(int32_t));
    uint32_t SV_len         = (uint32_t)(sizeof(SV_float32_bin)           /  sizeof(float));
    uint32_t nSV_len        = (uint32_t)(sizeof(nSV_int32_bin)            /  sizeof(int32_t));
    uint32_t svCoef_len     = (uint32_t)(sizeof(sv_coef_float32_bin)      /  sizeof(float));
    // uint8_t intercept_len  = sizeof(intercept_float32_bin)    /sizeof(float);
    // uint8_t svmType_len    = sizeof(svm_type_float32_bin)     /sizeof(float);
    // uint8_t kernel_len     = sizeof(kernel_int32_string_bin)  /sizeof(int32_t);
    // uint8_t degree_len     = sizeof(degree_int32_bin)         /sizeof(int32_t);
    // uint8_t gamma_len      = sizeof(gamma_float32_bin)        /sizeof(float);
    // uint8_t coef0_len      = sizeof(coef0_float32_bin)        /sizeof(float);

    int nSV_shape0 = nSV_len;
    float rho = -(*(float*)intercept_float32_bin);
    std::vector<int> label(nSV_len);
    for(uint8_t i=0; i<nSV_len; i++)
    {
        label[i] = i;
    }

    dec_values = svm_predict(feature, feature_len, (float*)SV_float32_bin, SV_len, (*(float*)gamma_float32_bin),
                             nSV_shape0, support_len, (int32_t*)nSV_int32_bin, (float*)sv_coef_float32_bin, 
                             svCoef_len, rho, label);
}