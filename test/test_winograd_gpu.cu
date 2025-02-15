#include "gtest/gtest.h"
#include "../src/winograd/winograd.cuh"
#include "../src/shared/utils.hpp"

TEST(WinogradGPUTest, Winograd16x16RandomTest) {
    int width = 16;
    int height = 16;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* output = new float[o_size];
    float* filter = new float[9]{
        -1.0f, -1.f, -1.f,
        0.0f, 0.f, 0.f,
        1.0f, 1.f, 1.f
    };
    float *input = new float[i_size]{
        -0.5350976, -0.5162272, -0.4421986, 0.8904997, 1.3258739, 1.9706974, 1.0145639, -0.95220184, -0.97428346, 0.04679603, -1.4407685, -0.3263395, 0.19722596, 0.53707594, -1.1247728, -1.8083316,
        1.1048017, -0.31682682, -0.3861142, -1.0213436, 0.4369808, -0.12138513, 0.9701048, -0.6529229, 0.6646843, -1.3402871, 2.3575554, -0.4567878, 0.35450226, -0.63233155, -0.5867306, -0.63946813,
        -0.10249648, -0.7354589, 0.00668383, 0.4711692, -2.1716573, -0.09946217, -0.68491095, 0.18848616, -0.6526224, 2.9957654, 1.6437014, -0.6570906, -0.04611707, 0.79069203, -0.55495507, -0.55911916,
        -0.8375089, -1.8012884, 0.85582983, -1.1194319, 0.39701924, 0.05773385, -0.06619139, -0.12465051, -1.5484633, 0.572171, 2.0378442, 0.1985576, -0.9130602, 1.0940013, 1.5274712, -0.00564122,
        -1.334098, -2.0258439, 0.5659482, -0.9595648, -0.35880005, 0.3781668, 0.02351619, 0.6105718, -2.409095, 0.26268584, -0.7681236, 0.1306608, 1.0470053, 0.68470716, -0.17977868, 0.8958127,
        0.3724931, 0.4302194, -0.10011721, 0.21198513, -0.50267553, 0.56807137, -0.49818277, 0.54216695, -0.57486755, 1.4969674, -0.71904385, -0.1621428, -0.14271654, 0.34327796, -0.1183044, 1.3763658,
        1.0993363, 0.18509866, 0.15291801, -0.01571355, 0.26842582, 1.1697806, -0.12944265, 0.4319386, -1.2965109, 0.04096505, 1.3339938, 1.1023108, 1.6553675, 0.6421887, 0.3448147, -0.3870128,
        -1.1007198, 0.2549909, 1.4506998, -0.05383182, 0.14044647, -1.3123842, 0.63701, 0.09667397, 0.6728563, -0.46123496, -1.4114279, 0.68959785, -0.00467085, 1.9965749, -2.0551648, -0.5938545,
        1.4083165, 0.7263961, 0.25710163, -0.07328705, 0.06610007, 1.3062121, 0.9612808, -0.36544847, 0.4501592, 0.452522, -0.5675108, 0.75954837, 0.06879467, -0.50186753, 0.2214549, 0.35033318,
        -1.5676223, -0.43263105, 0.4755949, -2.933593, -1.3408656, 0.11497255, -0.09229068, 0.37689874, 0.6975029, 0.5381443, 0.135462, -0.5486209, 2.181916, -1.0404798, 0.3702275, 0.8548304,
        -0.54710346, 1.3784856, -0.41743618, 1.0120428, 0.5977921, 0.35826036, 0.760523, -1.204926, 0.32725173, -0.9072339, 1.0569718, 1.354367, -0.5449049, -0.13203046, 0.04077841, 0.9341047,
        0.6173598, 2.6862183, -0.64694184, -0.9140651, 0.12596616, -0.64018524, -0.40749133, -1.2274076, 0.47251287, 0.30050537, -0.299991, -0.476304, -0.34612674, 1.2715071, 0.16330566, 1.4280875,
        -0.41111878, 0.96190333, 0.9988569, 1.3345702, 0.6940598, -0.86650795, 0.59717995, -1.6848304, -1.150636, -0.43661347, -1.0741702, 0.85703546, -1.1398683, 0.45116454, 0.70429456, -1.0354427,
        1.3094063, -0.5147296, -2.4437058, 0.17292948, -0.5825417, -1.0101184, 0.76460916, 1.4952679, 0.6876886, -0.21921411, -0.646878, 1.1040075, 0.6188062, -0.27701074, -0.9813094, 0.03928494,
        0.5826639, -0.9622782, -0.03354212, -0.37532848, 0.871826, 0.49240977, 0.25349027, -0.02522762, 0.32819486, -0.1826999, -0.77371436, -0.5816155, 0.3703585, 0.9748316, 0.20551129, 1.0648986,
        0.02514829, -0.6458998, 1.2394371, 0.5311768, 0.3176219, 1.6934156, 1.0432411, 0.3328851, 0.3343556, -0.42710662, 2.2918043, 1.1266874, -0.64352643, 1.4844886, 0.43759418, 0.09948757
    };

    
    // Initialize the input with values from 0 to i_size
    // for (int i = 0; i < i_size; i++) {
    //     input[i] = matrix[i];
    // }

    winograd_host(output, input, filter, width, height, 3, 3, 1, 0);

    float *expected = new float[o_size]{
         0.6622519f, -0.18967983f, -3.4679792f, -5.987021f, -7.2671657f, -2.6289468f, -0.23712581f, 4.4113183f, 6.3551006f, 5.702688f, 2.5103757f, -0.32047802f, 0.5800907f, 2.0726464f ,
        -2.1848283f, -0.34060574f, 1.1038942f, 0.04106914f, -0.89713883f, -0.32890478f, -2.7211714f, 0.22758293f, -0.62040067f, 2.2480924f, -0.93192816f, 1.1141158f, 2.5729723f, 4.4743614f ,
        -1.9627221f, -2.1618547f, 0.94138765f, 0.8597522f, 2.9989133f, 1.6081417f, -0.6259599f, -4.0674667f, -6.9013777f, -4.357153f, -0.5309514f, 1.7748889f, 1.3623139f, 1.7241232f ,
         2.4855628f, 2.6069777f, -0.5242248f, 0.9420597f, -0.82134867f, 0.74516356f, 1.208422f, 2.5652099f, -0.8584957f, -2.192792f, -2.347245f, -0.3410801f, -1.6261553f, -1.0144919f ,
         4.2313466f, 2.7417636f, 1.158047f, 2.362691f, 1.2658808f, 0.46002185f, 0.78099203f, 0.7122302f, 2.992981f, 2.8520465f, 3.6821294f, 1.5374937f, 1.0904372f, -0.8007505f ,
        -0.09762442f, 1.1097716f, 1.928122f, -1.5031506f, -0.10214084f, -1.1907558f, 1.9374237f, -1.1559715f, -1.4028625f, -1.7988458f, 0.29740238f, 2.643083f, -0.14551783f, -2.2537837f ,
         0.95446134f, 0.5879075f, -0.15571564f, -0.12346768f, 1.0248293f, 0.42976797f, 2.0400066f, 1.3608401f, 0.2567225f, -1.83271f, -3.8308394f, -3.0733914f, -2.853989f, -0.53007007f ,
        -2.1296291f, -4.542488f, -5.3361783f, -2.9337163f, -0.78325593f, 0.9782809f, -0.42442912f, 1.3042507f, 2.570916f, 1.3080504f, 2.4952579f, -2.0886865f, 1.5749245f, 0.83702254f ,
        -1.9778683f, 1.0628815f, 0.942484f, 0.6690701f, -0.6170173f, -1.9881868f, -1.1631429f, -2.322141f, 0.14181918f, 0.85954523f, 1.6056015f, 0.35095617f, -0.42453897f, 0.77293205f ,
         4.181295f, 4.0158405f, 2.3638227f, 2.7312021f, 0.3964733f, -2.6746647f, -2.144497f, -2.066935f, -0.898082f, -0.600775f, -2.8911788f, -0.14373887f, -0.4229777f, 2.6783223f ,
         1.1356956f, 1.3222383f, 1.8350883f, -0.8059733f, -1.2918437f, -1.8680158f, -2.1211352f, -1.4871716f, -3.1384091f, -2.1578531f, -3.223437f, -0.50909996f, 0.65174776f, -0.72283626f ,
        -4.3056655f, -3.9107172f, -1.4182773f, 0.00855362f, 0.09365946f, 3.5248427f, 4.109952f, 2.4181316f, -0.6514307f, 0.713705f, 2.1983573f, 0.99672645f, -1.7282001f, -4.081936f ,
        -1.9627979f, -4.6664796f, -2.5645316f, -0.1732148f, 1.1929942f, 2.674831f, 2.794744f, 3.3923473f, 2.0332003f, -0.8842816f, 0.3720316f, 0.59524286f, 1.5351106f, 2.125225f ,
         2.2677147f, 3.9102201f, 4.9415536f, 3.961945f, 3.8823295f, 1.8197832f, -1.2370837f, -1.7236083f, 2.3774567f, 2.7534697f, 1.6990296f, 0.52184665f, 1.9180703f, 3.2406054f 
    };

    EXPECT_TRUE(compare_arrays(output, expected, o_width, 1e-5)) << "16x16 Random input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}



TEST(WinogradGPUTest, Winograd7x7Test) {
    int width = 7;
    int height = 7;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };
    
    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, o_width, 1e-5)) << "7x7 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, Winograd10x10Test) {
    int width = 10;
    int height = 10;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };
    
    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, 8, 1e-5)) << "10x10 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, Winograd16x16Test) {
    int width = 16;
    int height = 16;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, 14, 1e-5)) << "16x16 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, Winograd32x32Test) {
    int width = 32;
    int height = 32;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, 30, 1e-5)) << "32x32 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, Winograd64x64Test) {
    int width = 64;
    int height = 64;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, 62, 1e-5)) << "64x64 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, Winograd128x128Test) {
    int width = 128;
    int height = 128;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, 126, 1e-5)) << "128x128 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

// RGB Channel same kernel
TEST(WinogradGPUTest, RGBWinograd10x10Test) {
    int width = 10;
    int height = 10;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;


    float* input = new float[i_size * nb_channel];
    float* output = new float[o_size * nb_channel];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };
    
    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++){
        for (int i = 0; i < i_size; i++) {
            input[i + c*i_size] = static_cast<float>(i);
        }
    }

    winograd_host(output, input, filter, width, height, 3, 3, nb_channel, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    for (int c=0; c<nb_channel; c++){
        EXPECT_TRUE(compare_arrays(output + c*o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, RGBWinograd16x16Test) {
    int width = 16;
    int height = 16;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float* input = new float[i_size * nb_channel];
    float* output = new float[o_size * nb_channel];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++){
        for (int i = 0; i < i_size; i++) {
            input[i + c*i_size] = static_cast<float>(i);
        }
    }

    winograd_host(output, input, filter, width, height, 3, 3, nb_channel, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    for (int c=0; c<nb_channel; c++){
        EXPECT_TRUE(compare_arrays(output + c*o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, RGBWinograd32x32Test) {
    int width = 32;
    int height = 32;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float* input = new float[i_size * nb_channel];
    float* output = new float[o_size * nb_channel];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++){
        for (int i = 0; i < i_size; i++) {
            input[i + c*i_size] = static_cast<float>(i);
        }
    }

    winograd_host(output, input, filter, width, height, 3, 3, nb_channel, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    for (int c=0; c<nb_channel; c++){
        EXPECT_TRUE(compare_arrays(output + c*o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, RGBWinograd64x64Test) {
    int width = 64;
    int height = 64;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float* input = new float[i_size * nb_channel];
    float* output = new float[o_size * nb_channel];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++){
        for (int i = 0; i < i_size; i++) {
            input[i + c*i_size] = static_cast<float>(i);
        }
    }

    winograd_host(output, input, filter, width, height, 3, 3, nb_channel, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    for (int c=0; c<nb_channel; c++){
        EXPECT_TRUE(compare_arrays(output + c*o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, RGBWinograd128x128Test) {
    int width = 128;
    int height = 128;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float* input = new float[i_size * nb_channel];
    float* output = new float[o_size * nb_channel];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++){
        for (int i = 0; i < i_size; i++) {
            input[i + c*i_size] = static_cast<float>(i);
        }
    }

    winograd_host(output, input, filter, width, height, 3, 3, nb_channel, 0);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    for (int c=0; c<nb_channel; c++){
        EXPECT_TRUE(compare_arrays(output + c*o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}


int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
