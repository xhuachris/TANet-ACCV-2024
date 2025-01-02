# TANet: Triplet Attention Network for All-In-One Adverse Weather Image Restoration (ACCV-2024)
Hsing-Hua Wang, Fu-Jen Tsai, Yen-Yu Lin, Chia-Wen Lin

> **Abstract.** Adverse weather image restoration aims to remove unwanted degraded artifacts, such as haze, rain, and snow, caused by adverse weather conditions. Existing methods achieve remarkable results for addressing single-weather conditions. However, they face challenges when encountering unpredictable weather conditions, which often happen in real-world scenarios. Although different weather conditions exhibit different degradation patterns, they share common characteristics that are highly related and complementary, such as occlusions caused by degradation patterns, color distortion, and contrast attenuation due to the scattering of atmospheric particles. Therefore, we focus on leveraging common knowledge across multiple weather conditions to restore images in a unified manner. In this paper, we propose a Triplet Attention Network (TANet) to efficiently and effectively address all-in-one adverse weather image restoration. TANet consists of Triplet Attention Block (TAB) that incorporates three types of attention mechanisms: Local Pixel-wise Attention (LPA) and Global Strip-wise Attention (GSA) to address occlusions caused by non-uniform degradation patterns, and Global Distribution Attention (GDA) to address color distortion and contrast attenuation caused by atmospheric phenomena. By leveraging common knowledge shared across different weather conditions, TANet successfully addresses multiple weather conditions in a unified manner. Experimental results show that TANet efficiently and effectively achieves state-of-the-art performance in all-in-one adverse weather image restoration.

## Framework

![image](https://github.com/user-attachments/assets/c9457c72-087c-4c5b-bab1-e8dcdff0e837)


## Training & Testing Data

```plaintext
└── dataset
    ├── train
        ├── haze
            ├── in
            └── gt
        ├── rain
            ├── in
            └── gt
        └── snow
            ├── in
            └── gt
    └──  test
        ├── haze
            ├── in
            └── gt
        ├── rain
            ├── in
            └── gt
        └── snow
            ├── in
            └── gt

data_path = {your data path}/dataset/
