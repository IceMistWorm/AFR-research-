/* 
 * Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "tutorial.h"
#include <optixu/optixu_aabb_namespace.h>

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type , , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(float, number_of_parent_tiles, , ); //Stores the number of parent tiles. 


rtBuffer<unsigned int, 2>        rnd_seeds;

// add more buffers for drawing tiles
rtBuffer<int, 2>                 busy_buffer;
rtBuffer<int, 2>				 stencil_buffer;         //1 if should not be reprojected. 0 if it should.
rtBuffer<uchar4, 2>				 color_buffer;           //Stores the color value of the pixel before tranparency

rtBuffer<uint2, 1>				leaf_tile_indices;       //Stores the indices of the leaf tiles
rtBuffer<unsigned int, 1>		leaf_tile_sizes;         //Stores the size of the corresponding leaf tile

rtBuffer<uint2, 1>              parent_tile_indices;
rtBuffer<unsigned int, 1>		parent_tile_sizes;

rtBuffer<float, 1>              variance_buffer;         //Stores the variance
rtBuffer<float, 1>				parent_variance_buffer;  //Stores the variance of the parent tiles
rtBuffer<uint, 2>               pixel_at_tile_buffer;    //Store the pixel belongs to which tile

rtBuffer<float, 1>              render_elapse_time_buffer;
rtBuffer<int, 2>                mini_tile_buffer;
rtBuffer<int, 2>                random_map_buffer;

rtBuffer<float, 2>              tile_gradient_buffer;    //Stores gradient sum in the tile
rtBuffer<float3, 2>             extent_buffer;           //Stores extent

//
// Pinhole camera implementation
//
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtBuffer<uchar4, 2>              output_buffer;
rtBuffer<int, 1>				 ray_per_sec_buffer;
rtBuffer<int, 1>				 raycounting_buffer;
rtBuffer<float3, 2>              float_color_buffer;
rtBuffer<float3, 2>              float_temp_buffer1;
rtBuffer<float3, 2>              float_temp_buffer2;
rtBuffer<float3, 2>              float_temp_buffer3;
rtBuffer<uchar4, 2>				 temp_buffer1;
rtBuffer<uchar4, 2>				 temp_buffer2;
rtBuffer<uchar4, 2>				 temp_buffer3;
rtBuffer<float3, 2>              crosshair_buffer, crosshair_buffer1, crosshair_buffer2, crosshair_buffer3;
rtBuffer<float, 2>               sample_time_buffer, sample_time_temp_buffer1, sample_time_temp_buffer2, sample_time_temp_buffer3;
rtBuffer<int, 1>                 screen_size_buffer;

rtBuffer<float3, 2>				 extent_buffer1;           //Stores extent
rtBuffer<float3, 2>				 extent_buffer2;           //Stores extent
rtBuffer<float3, 2>				 extent_buffer3;           //Stores extent
rtBuffer<float3, 1>              extent_at_tile;
rtBuffer<unsigned int, 1>       show_variance_buffer;
rtBuffer<unsigned int, 1>       show_gradient_buffer;
rtBuffer<unsigned int, 1>       show_tile_buffer;
rtBuffer<unsigned int, 1>       is_moving_buffer;

#define PI 3.1415926
// define weight matrix for Gaussian Filter (old, toooooo blurry)

__device__ float gaussian_1[1] = { 1.000000 };
__device__ float gaussian_3[3] = { 0.285472, 0.429056, 0.285472 };
__device__ float gaussian_5[5] = { 0.142483, 0.225851, 0.263331, 0.225851, 0.142483 };
__device__ float gaussian_7[7] = { 0.092665, 0.137812, 0.174867, 0.189313, 0.174867, 0.137812, 0.092665 };
__device__ float gaussian_9[9] = { 0.068208, 0.095616, 0.121705, 0.140662, 0.147616, 0.140662, 0.121705, 0.095616, 0.068208 };
__device__ float gaussian_11[11] = { 0.053809, 0.072025, 0.090358, 0.106246, 0.11709, 0.120945, 0.11709, 0.106246, 0.090358, 0.072025, 0.053809 };
__device__ float gaussian_13[13] = { 0.044386, 0.057305, 0.070626, 0.083094, 0.093325, 0.100059, 0.10241, 0.100059, 0.093325, 0.083094, 0.070626, 0.057305, 0.044386 };
__device__ float gaussian_15[15] = { 0.037739, 0.047357, 0.057387, 0.067154, 0.075886, 0.08281, 0.087265, 0.088802, 0.087265, 0.08281, 0.075886, 0.067154, 0.057387, 0.047357, 0.037739 };
__device__ float gaussian_17[17] = { 0.032818, 0.040246, 0.048031, 0.055783, 0.063047, 0.069345, 0.074226, 0.077317, 0.078376, 0.077317, 0.074226, 0.069345, 0.063047, 0.055783, 0.048031, 0.040246, 0.032818 };
__device__ float gaussian_19[19] = { 0.02902, 0.034926, 0.041127, 0.047385, 0.053419, 0.058922, 0.063592, 0.067153, 0.069384, 0.070144, 0.069384, 0.067153, 0.063592, 0.058922, 0.053419, 0.047385, 0.041127, 0.034926, 0.02902 };
__device__ float gaussian_21[21] = { 0.026011, 0.030815, 0.035862, 0.040996, 0.046037, 0.050784, 0.055029, 0.058575, 0.061247, 0.062908, 0.063472, 0.062908, 0.061247, 0.058575, 0.055029, 0.050784, 0.046037, 0.040996, 0.035862, 0.030815, 0.026011 };
__device__ float gaussian_23[23] = { 0.000093, 0.00033, 0.001039, 0.002903, 0.007184, 0.015758, 0.030627, 0.052751, 0.080517, 0.108910, 0.130549, 0.138679, 0.130549, 0.108910, 0.080517, 0.052751, 0.030627, 0.015758, 0.007184, 0.002903, 0.001039, 0.00033, 0.000093 };
__device__ float gaussian_25[25] = { 0.000044, 0.000159, 0.000511, 0.001470, 0.003785, 0.008718, 0.017966, 0.033126, 0.054647, 0.080660, 0.106520, 0.125863, 0.133062, 0.125863, 0.106520, 0.080660, 0.054647, 0.033126, 0.017966, 0.008718, 0.003785, 0.001470, 0.000511, 0.000159, 0.000044 };
__device__ float gaussian_27[27] = { 0.000021, 0.000077, 0.000251, 0.000740, 0.001970, 0.004731, 0.010250, 0.020031, 0.035313, 0.056153, 0.080548, 0.104224, 0.121650, 0.128084, 0.121650, 0.104224, 0.080548, 0.056153, 0.035313, 0.020031, 0.010250, 0.004731, 0.001970, 0.000740, 0.000251, 0.000077, 0.000021 };
__device__ float gaussian_29[29] = { 0.000010, 0.000037, 0.000123, 0.000371, 0.001016, 0.002530, 0.005723, 0.011760, 0.021952, 0.037225, 0.057344, 0.080249, 0.102022, 0.117826, 0.123621, 0.117826, 0.102022, 0.080249, 0.057344, 0.037225, 0.021952, 0.011760, 0.005723, 0.002530, 0.001016, 0.000371, 0.000123, 0.000037, 0.000010 };
__device__ float gaussian_31[31] = { 0.000005, 0.000018, 0.000060, 0.000185, 0.000520, 0.001337, 0.003140, 0.006740, 0.013226, 0.023722, 0.038889, 0.058274, 0.079816, 0.099924, 0.114345, 0.119600, 0.114345, 0.099924, 0.079816, 0.058274, 0.038889, 0.023722, 0.013226, 0.006740, 0.003140, 0.001337, 0.000520, 0.000185, 0.000060, 0.000018, 0.000005 };
__device__ float gaussian_33[33] = { 0.000002, 0.000009, 0.000029, 0.000092, 0.000265, 0.000700, 0.001699, 0.003790, 0.007770, 0.014640, 0.025350, 0.040339, 0.058992, 0.079283, 0.097922, 0.111148, 0.115943, 0.111148, 0.097922, 0.079283, 0.058992, 0.040339, 0.025350, 0.014640, 0.007770, 0.003790, 0.001699, 0.000700, 0.000265, 0.000092, 0.000029, 0.000009, 0.000002 };
//__device__ float gaussian_35[35] ={};
//__device__ float gaussian_37[37] = 
//__device__ float gaussian_39[39] = 
//__device__ float gaussian_41[41] = 

/*
// define weight matrix for Gaussian Filter (new!)

__device__ float gaussian_1[1] = { 1.000000 };
__device__ float gaussian_3[3] = { 0.282332, 0.435336, 0.282332};
__device__ float gaussian_5[5] = { 0.066576, 0.244739, 0.37737, 0.244739, 0.066576 };
__device__ float gaussian_7[7] = { 0.044533, 0.117512, 0.210298, 0.255313, 0.210298, 0.117512, 0.044533 };
__device__ float gaussian_9[9] = { 0.019538, 0.056663, 0.121206, 0.191261, 0.222666, 0.191261, 0.121206, 0.056663, 0.019538};
__device__ float gaussian_11[11] = { 0.00881, 0.027144, 0.065119, 0.121654, 0.176995, 0.200556, 0.176995, 0.121654, 0.065119, 0.027144, 0.00881};
__device__ float gaussian_13[13] = { 0.004046, 0.012996, 0.033758, 0.070923, 0.120521, 0.165661, 0.184193, 0.165661, 0.120521, 0.070923, 0.033758, 0.012996, 0.004046};
__device__ float gaussian_15[15] = { 0.001873, 0.006209, 0.017117, 0.039241, 0.074812, 0.118611, 0.156392, 0.171491, 0.156392, 0.118611, 0.074812,0.039241, 0.017117, 0.006209, 0.001873};
__device__ float gaussian_17[17] = { 0.000874, 0.00297, 0.008571, 0.021008, 0.043747, 0.077393, 0.116323, 0.148539, 0.161151, 0.148539, 0.116323, 0.077393, 0.043747, 0.021008, 0.008571, 0.00297, 0.000874};
__device__ float gaussian_19[19] = { 0.000414, 0.00143, 0.004271, 0.011027, 0.024604, 0.04744, 0.079054, 0.113849, 0.1417, 0.152423, 0.1417, 0.113849, 0.079054, 0.04744, 0.024604, 0.011027, 0.004271, 0.00143, 0.000414};
__device__ float gaussian_21[21] = { 0.000195, 0.000685, 0.002109, 0.005686, 0.01343, 0.027789, 0.05038, 0.080024, 0.111368, 0.135795, 0.145075, 0.135795, 0.111368, 0.080024, 0.05038, 0.027789, 0.01343, 0.005686, 0.002109, 0.000685,0.000195};
*/
/*

// test other gaussian filter weights sig = 1
__device__ float gaussian_1[1] = { 1.000000 };
__device__ float gaussian_3[3] = { 0.27901, 0.44198, 0.27901 };
__device__ float gaussian_5[5] = { 0.06136, 0.24477, 0.38774, 0.24477, 0.06136};
__device__ float gaussian_7[7] = { 0.00598, 0.060626,0.241843, 0.383103, 0.241843, 0.060626, 0.00598};
__device__ float gaussian_9[9] = { 0.000229, 0.005977, 0.060598, 0.241732, 0.382928, 0.241732, 0.060598, 0.005977, 0.000229};
__device__ float gaussian_11[11] = {0.000003, 0.000229, 0.005977, 0.060598, 0.24173, 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003};
__device__ float gaussian_13[13] = { 0, 0.000003, 0.000229, 0.005977, 0.060598, 0.24173, 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0};
__device__ float gaussian_15[15] = { 0, 0, 0.000003, 0.000229, 0.005977, 0.060598, 0.24173, 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0};
__device__ float gaussian_17[17] = { 0, 0, 0, 0.000003, 0.000229, 0.005977, 0.060598, 0.24173, 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0};
__device__ float gaussian_19[19] = { 0, 0, 0, 0, 0.000003, 0.000229, 0.005977, 0.060598, 0.24173, 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0};
__device__ float gaussian_21[21] = { 0, 0, 0, 0, 0, 0.000003, 0.000229, 0.005977, 0.060598, 0.24173, 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0};
*/

/*
// test other gaussian filter weights sig = 3
__device__ float gaussian_1[1] = { 1.000000 };
__device__ float gaussian_3[3] = { 0.327162, 0.345675, 0.327162};
__device__ float gaussian_5[5] = { 0.1784, 0.210431, 0.222338, 0.210431, 0.1784};
__device__ float gaussian_7[7] = { 0.106595, 0.140367, 0.165569, 0.174938, 0.165569, 0.140367, 0.106595};
__device__ float gaussian_9[9] = { 0.063327, 0.093095, 0.122589, 0.144599, 0.152781, 0.144599, 0.122589, 0.093095, 0.063327};
__device__ float gaussian_11[11] = { 0.035822, 0.05879, 0.086425, 0.113806, 0.13424, 0.141836, 0.13424, 0.113806, 0.086425, 0.05879, 0.035822};
__device__ float gaussian_13[13] = { 0.018816, 0.034474, 0.056577, 0.083173, 0.109523, 0.129188, 0.136498, 0.129188, 0.109523, 0.083173, 0.056577, 0.034474, 0.018816};
__device__ float gaussian_15[15] = { 0.009033, 0.018476, 0.033851, 0.055555, 0.08167, 0.107545, 0.126854, 0.134032, 0.126854, 0.107545, 0.08167, 0.055555, 0.033851, 0.018476, 0.009033};
__device__ float gaussian_17[17] = { 0.003924, 0.008962, 0.018331, 0.033585, 0.055119, 0.081029, 0.106701, 0.125858, 0.13298, 0.125858, 0.106701, 0.081029, 0.055119, 0.033585, 0.018331, 0.008962, 0.003924};
__device__ float gaussian_19[19] = { 0.001535, 0.003912, 0.008934, 0.018275, 0.033482, 0.05495, 0.08078, 0.106373, 0.125472, 0.132572, 0.125472, 0.106373, 0.08078, 0.05495, 0.033482, 0.018275, 0.008934, 0.003912, 0.001535};
__device__ float gaussian_21[21] = { 0.000539, 0.001533, 0.003908, 0.008925, 0.018255, 0.033446, 0.054891, 0.080693, 0.106259, 0.125337, 0.132429, 0.125337, 0.106259, 0.080693, 0.054891, 0.033446, 0.018255, 0.008925, 0.003908, 0.001533, 0.000539};


// test other gaussian filter weights sig = 7 (wayyyy to blurrrr)
__device__ float gaussian_1[1] = { 1.000000 };
__device__ float gaussian_3[3] = { 0.3322, 0.335601, 0.3322 };
__device__ float gaussian_5[5] = { 0.195938, 0.202018, 0.204087, 0.202018, 0.195938 };
__device__ float gaussian_7[7] = { 0.135679, 0.142769, 0.147199, 0.148706, 0.147199, 0.142769, 0.135679 };
__device__ float gaussian_9[9] = { 0.100856, 0.108311, 0.11397, 0.117507, 0.11871, 0.117507, 0.11397, 0.108311, 0.100856 };
__device__ float gaussian_11[11] = { 0.077718, 0.08518, 0.091475, 0.096255, 0.099242, 0.100258, 0.099242, 0.096255, 0.091475, 0.08518, 0.077718 };
__device__ float gaussian_13[13] = { 0.061002, 0.068236, 0.074787, 0.080315, 0.084512, 0.087134, 0.088026, 0.087134, 0.084512, 0.080315, 0.074787, 0.068236, 0.061002 };
__device__ float gaussian_15[15] = { 0.048277, 0.055112, 0.061647, 0.067566, 0.07256, 0.076352, 0.078721, 0.079527, 0.078721, 0.076352, 0.07256, 0.067566, 0.061647, 0.055112, 0.048277 };
__device__ float gaussian_17[17] = { 0.038265, 0.044582, 0.050895, 0.05693, 0.062396, 0.067007, 0.070509, 0.072697, 0.073441, 0.072697, 0.070509, 0.067007, 0.062396, 0.05693, 0.050895, 0.044582, 0.038265 };
__device__ float gaussian_19[19] = { 0.030234, 0.035951, 0.041886, 0.047817, 0.053487, 0.058623, 0.062955, 0.066245, 0.068301, 0.069, 0.068301, 0.066245, 0.062955, 0.058623, 0.053487, 0.047817, 0.041886, 0.035951, 0.030234 };
__device__ float gaussian_21[21] = { 0.023732, 0.028799, 0.034245, 0.039898, 0.045548, 0.050948, 0.05584, 0.059967, 0.063101, 0.065059, 0.065725, 0.065059, 0.063101, 0.059967, 0.05584, 0.050948, 0.045548, 0.039898, 0.034245, 0.028799, 0.023732 };
 */


 // temporal filter (sig = 1 )
 /*
 __device__ float gaussianT_1[1] = { 1.000000 };
 __device__ float gaussianT_3[2] = { 0.44198, 0.27901 };
 __device__ float gaussianT_5[3] = { 0.38774, 0.24477, 0.06136 };
 __device__ float gaussianT_7[4] = { 0.383103, 0.241843, 0.060626, 0.00598 };
 __device__ float gaussianT_9[5] = { 0.382928, 0.241732, 0.060598, 0.005977, 0.000229 };
 __device__ float gaussianT_11[6] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003 };
 __device__ float gaussianT_13[7] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0 };
 __device__ float gaussianT_15[8] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0 };
 __device__ float gaussianT_17[9] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0 };
 __device__ float gaussianT_19[10] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0 };
 __device__ float gaussianT_21[11] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0 };
 __device__ float gaussianT_23[12] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0 };
 __device__ float gaussianT_25[13] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0 };
 __device__ float gaussianT_27[14] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0 };
 __device__ float gaussianT_29[15] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
 __device__ float gaussianT_31[16] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
 __device__ float gaussianT_33[17] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 __device__ float gaussianT_35[18] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 __device__ float gaussianT_37[19] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 __device__ float gaussianT_39[20] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 __device__ float gaussianT_41[21] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
 __device__ float gaussianT_43[23] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 __device__ float gaussianT_45[24] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
 __device__ float gaussianT_47[25] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 __device__ float gaussianT_49[26] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 */

 // temporal filter (new!)

__device__ float gaussianT_1[1] = { 1.000000 };
__device__ float gaussianT_3[2] = { 0.435336, 0.282332 };
__device__ float gaussianT_5[3] = { 0.37737, 0.244739, 0.066576 };
__device__ float gaussianT_7[4] = { 0.255313, 0.210298, 0.117512, 0.044533 };
__device__ float gaussianT_9[5] = { 0.222666, 0.191261, 0.121206, 0.056663, 0.019538 };
__device__ float gaussianT_11[6] = { 0.200556, 0.176995, 0.121654, 0.065119, 0.027144, 0.00881 };
__device__ float gaussianT_13[7] = { 0.184193, 0.165661, 0.120521, 0.070923, 0.033758, 0.012996, 0.004046 };
__device__ float gaussianT_15[8] = { 0.171491, 0.156392, 0.118611, 0.074812,0.039241, 0.017117, 0.006209, 0.001873 };
__device__ float gaussianT_17[9] = { 0.161151, 0.148539, 0.116323, 0.077393, 0.043747, 0.021008, 0.008571, 0.00297, 0.000874 };
__device__ float gaussianT_19[10] = { 0.152423, 0.1417, 0.113849, 0.079054, 0.04744, 0.024604, 0.011027, 0.004271, 0.00143, 0.000414 };
__device__ float gaussianT_21[11] = { 0.145075, 0.135795, 0.111368, 0.080024, 0.050380, 0.027789, 0.01343, 0.005686, 0.002109, 0.000685, 0.000195 };
__device__ float gaussianT_23[12] = { 0.138679, 0.130549, 0.108910, 0.080517, 0.052751, 0.030627, 0.015758, 0.007184, 0.002903, 0.001039, 0.00033, 0.000093 };
__device__ float gaussianT_25[13] = { 0.133062, 0.125863, 0.106520, 0.080660, 0.054647, 0.033126, 0.017966, 0.008718, 0.003785, 0.001470, 0.000511, 0.000159, 0.000044 };
__device__ float gaussianT_27[14] = { 0.128084, 0.121650, 0.104224, 0.080548, 0.056153, 0.035313, 0.020031, 0.010250, 0.004731, 0.001970, 0.000740, 0.000251, 0.000077, 0.000021 };
__device__ float gaussianT_29[15] = { 0.123621, 0.117826, 0.102022, 0.080249, 0.057344, 0.037225, 0.021952, 0.011760, 0.005723, 0.002530, 0.001016, 0.000371, 0.000123, 0.000037, 0.000010 };
__device__ float gaussianT_31[16] = { 0.119600, 0.114345, 0.099924, 0.079816, 0.058274, 0.038889, 0.023722, 0.013226, 0.006740, 0.003140, 0.001337, 0.000520, 0.000185, 0.000060, 0.000018, 0.000005 };
__device__ float gaussianT_33[17] = { 0.115943, 0.111148, 0.097922, 0.079283, 0.058992, 0.040339, 0.025350, 0.014640, 0.007770, 0.003790, 0.001699, 0.000700, 0.000265, 0.000092, 0.000029, 0.000009, 0.000002 };
__device__ float gaussianT_35[18] = { 0.112606, 0.108209, 0.096020, 0.078679, 0.059533, 0.041596, 0.026838, 0.015989, 0.008797, 0.004469, 0.002096, 0.000908, 0.000363, 0.000134, 0.000046, 0.000014, 0.000004, 0.000001 };
__device__ float gaussianT_37[19] = { 0.109538, 0.105486, 0.094207, 0.078025, 0.059929, 0.042688, 0.028198, 0.017274, 0.009814, 0.005171, 0.002526, 0.001145, 0.000481, 0.000187, 0.000068, 0.000023, 0.000007, 0.000002, 0.000001 };
__device__ float gaussianT_39[20] = { 0.106711, 0.102961, 0.092484, 0.077336, 0.060204, 0.043631, 0.029437, 0.018489, 0.010811, 0.005885, 0.002982, 0.001407, 0.000618, 0.000253, 0.000096, 0.000034, 0.000011, 0.000003, 0.000001, 0 };
__device__ float gaussianT_41[21] = { 0.104090, 0.100606, 0.090840, 0.076624, 0.060379, 0.044447, 0.030566, 0.019636, 0.011785, 0.006607, 0.003460, 0.001693, 0.000774, 0.000330, 0.000132, 0.000049, 0.000017, 0.000006, 0.000002, 0, 0 };
__device__ float gaussianT_43[23] = { 0.101655, 0.098408, 0.089276, 0.075899, 0.060471, 0.045149, 0.031591, 0.020714, 0.012729, 0.007330, 0.003956, 0.002000, 0.000948, 0.000421, 0.000175, 0.000068, 0.000025, 0.000009, 0.000003, 0.000001, 0, 0 };
__device__ float gaussianT_45[24] = { 0.099381, 0.096345, 0.087781, 0.075167, 0.060492, 0.045753, 0.032523, 0.021728, 0.013642, 0.008050, 0.004464, 0.002327, 0.001140, 0.000525, 0.000227, 0.000092, 0.000035, 0.000013, 0.000004, 0.000001, 0, 0, 0 };
__device__ float gaussianT_47[25] = { 0.097255, 0.094408, 0.086356, 0.074434, 0.060455, 0.046269, 0.033368, 0.022676, 0.014521, 0.008762, 0.004982, 0.002669, 0.001348, 0.000641, 0.000287, 0.000121, 0.000048, 0.000018, 0.000006, 0.000002, 0.000001, 0, 0, 0 };
__device__ float gaussianT_49[26] = { 0.095258, 0.092581, 0.084992, 0.073702, 0.060369, 0.046708, 0.034135, 0.023564, 0.015366, 0.009464, 0.005506, 0.003026, 0.001571, 0.000770, 0.000357, 0.000156, 0.000064, 0.000025, 0.000009, 0.000003, 0.000001, 0, 0, 0, 0 };



// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
	const unsigned int LCG_A = 1664525u;
	const unsigned int LCG_C = 1013904223u;
	prev = (LCG_A * prev + LCG_C);
	return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
	return ((float)lcg(prev) / (float)0x01000000);
}


static __device__ __inline__ void frameless_rendering()
{
	size_t2 screen = output_buffer.size();

	volatile unsigned int seed = rnd_seeds[launch_index]; // volatile workaround for cuda 2.0 bug
	unsigned int new_seed = seed;
	float anytorandom = rnd(new_seed);
	float anytorandom2 = rnd(new_seed);
	rnd_seeds[launch_index] = new_seed;

	float x_offset = rnd(new_seed);
	float y_offset = rnd(new_seed);

	uint2 start_pixel;
	start_pixel.x = 0;
	start_pixel.y = 0;

	x_offset *= screen_size_buffer[0];
	y_offset *= screen_size_buffer[1];

	uint2 total_offset = make_uint2(x_offset, y_offset);
	uint2 rand_pixel = start_pixel + total_offset;

	//Ray creation
	float2 d = (make_float2(rand_pixel)) / make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;

	rtTrace(top_object, ray, prd);

	if (busy_buffer[rand_pixel] == 0) //If pixel is free
	{
		atomicExch(&busy_buffer[rand_pixel], 1); //Set pixel to busy

		//output_buffer[rand_pixel] = make_color(prd.result);
		color_buffer[rand_pixel] = make_color(prd.result);
		stencil_buffer[rand_pixel] = 1;
		atomicExch(&busy_buffer[rand_pixel], 0); //Set pixel to free.
	}

}

// shoot 3 rays simultaneously, and calculate the gradient and add to the tile gradient buffer in this thread 
static __device__ __inline__ void shoot_3rays_new()
{
	// record the start time from rendering

	float start_to_render_time = render_elapse_time_buffer[0];
	clock_t shoot_ray_start_time = clock();

	size_t2 screen = output_buffer.size();

	volatile unsigned int seed = rnd_seeds[launch_index]; // volatile workaround for cuda 2.0 bug
	unsigned int new_seed = seed;
	float anytorandom = rnd(new_seed);
	float anytorandom2 = rnd(new_seed);
	rnd_seeds[launch_index] = new_seed;
	uint which_grid_calculate = launch_index.x % leaf_tile_indices.size();
	uint2 centre_pixel = make_uint2(leaf_tile_indices[which_grid_calculate].x, leaf_tile_indices[which_grid_calculate].y);

	//rtPrintf("launch index = %u  %u\n", launch_index.x, launch_index.y);

	for (int i = 0; i < 1; i++)
	{
		float x_offset = rnd(new_seed);
		float y_offset = rnd(new_seed);

		unsigned int tilesize = leaf_tile_sizes[launch_index.x];
		uint2 start_pixel;
		start_pixel.x = centre_pixel.x - tilesize / 2u;
		start_pixel.y = centre_pixel.y - tilesize / 2u;

		x_offset *= tilesize;
		y_offset *= tilesize;

		uint2 total_offset = make_uint2(x_offset, y_offset);
		uint2 rand_pixel = start_pixel + total_offset;
		float addrandomness = rnd_seeds[launch_index] * 0.000000000000039;

		/*
		if (busy_buffer[rand_pixel] == 1) // if pixel is busy
		{
		x_offset = rnd(new_seed);
		y_offset = rnd(new_seed);
		tilesize = leaf_tile_sizes[launch_index.x];
		start_pixel.x = centre_pixel.x - tilesize / 2u;
		start_pixel.y = centre_pixel.y - tilesize / 2u;

		x_offset *= tilesize;
		y_offset *= tilesize;
		total_offset = make_uint2(x_offset, y_offset);
		rand_pixel = start_pixel + total_offset;
		}*/


		//Right pixel
		uint2 right_pixel = rand_pixel + make_uint2(1.0f, 0.0f);
		//Top pixel
		uint2 top_pixel = rand_pixel + make_uint2(0.0f, 1.0f);


		//Ray creation (Center)
		float2 d = (make_float2(rand_pixel)) / make_float2(screen) * 2.f - 1.f;
		
		d.x += addrandomness;

		anytorandom = rnd(new_seed);
		anytorandom2 = rnd(new_seed);
		rnd_seeds[launch_index] = new_seed;
		addrandomness = rnd_seeds[launch_index] * 0.00000000000007;

		d.y += addrandomness;

		float3 ray_origin = eye;
		float3 ray_direction_center = normalize(d.x*U + d.y*V + W);
		
		anytorandom = rnd(new_seed);
		anytorandom2 = rnd(new_seed);
		rnd_seeds[launch_index] = new_seed;
		addrandomness = rnd_seeds[launch_index] * 0.00000000000007;


		optix::Ray ray = optix::make_Ray(ray_origin, ray_direction_center, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

		PerRayData_radiance prd;
		prd.importance = 1.f;
		prd.depth = 0;

		//Ray creation (Top)
		float2 d_top = (make_float2(top_pixel)) / make_float2(screen) * 2.f - 1.f;
		float3 ray_direction_top = normalize(d_top.x*U + d_top.y*V + W);
		
		d_top.x += addrandomness;

		anytorandom = rnd(new_seed);
		anytorandom2 = rnd(new_seed);
		rnd_seeds[launch_index] = new_seed;
		addrandomness = rnd_seeds[launch_index] * 0.00000000000007;

		d_top.y += addrandomness;

		anytorandom = rnd(new_seed);
		anytorandom2 = rnd(new_seed);
		rnd_seeds[launch_index] = new_seed;
		addrandomness = rnd_seeds[launch_index] * 0.00000000000007;

		optix::Ray ray_top = optix::make_Ray(ray_origin, ray_direction_top, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

		PerRayData_radiance prd_top;
		prd_top.importance = 1.f;
		prd_top.depth = 0;

		//Ray creation (Right)
		float2 d_right = (make_float2(right_pixel)) / make_float2(screen) * 2.f - 1.f;
		float3 ray_direction_right = normalize(d_right.x*U + d_right.y*V + W);
		
		d_right.x += addrandomness;

		anytorandom = rnd(new_seed);
		anytorandom2 = rnd(new_seed);
		rnd_seeds[launch_index] = new_seed;
		addrandomness = rnd_seeds[launch_index] * 0.00000000000007;

		d_right.y += addrandomness;

		anytorandom = rnd(new_seed);
		anytorandom2 = rnd(new_seed);
		rnd_seeds[launch_index] = new_seed;
		addrandomness = rnd_seeds[launch_index] * 0.00000000000007;
		
		optix::Ray ray_right = optix::make_Ray(ray_origin, ray_direction_right, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
		

		PerRayData_radiance prd_right;
		prd_right.importance = 1.f;
		prd_right.depth = 0;


		rtTrace(top_object, ray, prd);
		rtTrace(top_object, ray_top, prd_top);
		rtTrace(top_object, ray_right, prd_right);

		float lum_centre = luminance(prd.result);
		float lum_right = luminance(prd_right.result);
		float lum_top = luminance(prd_top.result);

		uchar4 old_color_char = color_buffer[rand_pixel];
		float3 old_color = make_float3(old_color_char.z, old_color_char.y, old_color_char.x)*make_float3(1.0f / 255.99f);;

		//old_color = float_color_buffer[rand_pixel];

		float lum_temporal = luminance(old_color);

		clock_t shoot_ray_end_time = clock();
		float sample_elapse_time = (float)(((shoot_ray_end_time - shoot_ray_start_time) / 100000000.0f));

		//if (launch_index.x == 100u && launch_index.y == 100u)
		//	rtPrintf("srt = %f, set = %f\n", start_to_render_time, sample_elapse_time);

		float finish_sample_time_in_system = start_to_render_time + sample_elapse_time;

		finish_sample_time_in_system = start_to_render_time;


		float old_time = sample_time_buffer[rand_pixel];
		float new_time = finish_sample_time_in_system;
		float elapse_time = (new_time - old_time);

		//elapse_time /= 1000000000;


		if (elapse_time < 0)
			elapse_time *= -1;



		float YIQ_standard_center, YIQ_standard_right, YIQ_standard_top, YIQ_standard_old;

		YIQ_standard_center = prd.result.x * 0.299 + prd.result.y * 0.587 + prd.result.z * 0.114;
		YIQ_standard_right = prd_right.result.x * 0.299 + prd_right.result.y * 0.587 + prd_right.result.z * 0.114;
		YIQ_standard_top = prd_top.result.x * 0.299 + prd_top.result.y * 0.587 + prd_top.result.z * 0.114;
		YIQ_standard_old = old_color.x * 0.299 + old_color.y * 0.587 + old_color.z * 0.114;

		float_temp_buffer3[rand_pixel] = float_temp_buffer2[rand_pixel];
		float_temp_buffer2[rand_pixel] = float_temp_buffer1[rand_pixel];
		float_temp_buffer1[rand_pixel] = float_color_buffer[rand_pixel];
		float_color_buffer[rand_pixel] = prd.result;


		float g_x = lum_centre - lum_right;
		float g_y = lum_centre - lum_top;
		float g_t = lum_temporal - lum_centre;
		/*
		float g_x = YIQ_standard_center - YIQ_standard_right;
		float g_y = YIQ_standard_center - YIQ_standard_top;
		float g_t = YIQ_standard_old - YIQ_standard_center;
		*/

		if (g_x < 0)
			g_x *= -1;
		if (g_y < 0)
			g_y *= -1;
		if (g_t < 0)
			g_t *= -1;

		if (elapse_time == 0.0f) {
			elapse_time = 0.00001f;
			g_t = 0;
		}

		g_t /= elapse_time;
		//g_t *= exp(-3.27* elapse_time);


		//sample_time_buffer[rand_pixel] = new_time;

		// add crosshair to the crosshair_buffer

		float3 crosshair = make_float3(g_x, g_y, g_t);
		crosshair_buffer[rand_pixel] = crosshair;

		// calculate color and show gradient?
		uchar4 char_color = make_color(prd.result);
		float3 color = make_float3(char_color.z, char_color.y, char_color.x)*make_float3(1.0f / 255.99f);

		// extent per tile calculation 
		float3 extent;
		extent.x = extent.y = extent.z = 0.0f;
		//float vs = 0.04;
		//float rl = 100.0 / ((float(tilesize) * (float)tilesize) * (float)0.07);
		//vs = 1 / rl;
		float basic_tile_size = 64.0f;
		float vs = (ray_per_sec_buffer[0] / (256 * basic_tile_size))*(float(tilesize) * (float)tilesize) / basic_tile_size;
		//vs = (20 *(float(tilesize) * (float)tilesize) )/ basic_tile_size;;

		uint2 gradient_buffer_index_x = make_uint2(launch_index.x, 0.0f);
		uint2 gradient_buffer_index_y = make_uint2(launch_index.x, 1.0f);
		uint2 gradient_buffer_index_t = make_uint2(launch_index.x, 2.0f);
		float gradient_x_at_the_tile = tile_gradient_buffer[gradient_buffer_index_x];
		float gradient_y_at_the_tile = tile_gradient_buffer[gradient_buffer_index_y];
		float gradient_t_at_the_tile = tile_gradient_buffer[gradient_buffer_index_t];

		bool leaveLoop = false;
		while (!leaveLoop) {
			//if (busy_buffer[rand_pixel] == 0) //If pixel is free
			if (atomicExch(&(busy_buffer[rand_pixel]), 1u) == 0u) //If pixel is free
			{
				crosshair_buffer[rand_pixel] = crosshair;
				extent_buffer[rand_pixel] = extent;

				temp_buffer3[rand_pixel] = temp_buffer2[rand_pixel];
				temp_buffer2[rand_pixel] = temp_buffer1[rand_pixel];
				temp_buffer1[rand_pixel] = color_buffer[rand_pixel];

				color_buffer[rand_pixel] = make_color(prd.result);

				sample_time_temp_buffer3[rand_pixel] = sample_time_temp_buffer2[rand_pixel];
				sample_time_temp_buffer2[rand_pixel] = sample_time_temp_buffer1[rand_pixel];
				sample_time_temp_buffer1[rand_pixel] = sample_time_buffer[rand_pixel];
				sample_time_buffer[rand_pixel] = finish_sample_time_in_system;
				raycounting_buffer[0] += 1;
				leaveLoop = true;
				atomicExch(&busy_buffer[rand_pixel], 0); //Set pixel to free.
			}
			else {
				leaveLoop = true;
				//rayrejecting_buffer[0] += 1;
			}
		}
		leaveLoop = false;
		/**/
		while (!leaveLoop) {
			if (top_pixel.x < screen_size_buffer[0] && top_pixel.y < screen_size_buffer[1]) {
				if (atomicExch(&(busy_buffer[top_pixel]), 1u) == 0u) //If pixel is free
				{
					sample_time_temp_buffer3[top_pixel] = sample_time_temp_buffer2[top_pixel];
					sample_time_temp_buffer2[top_pixel] = sample_time_temp_buffer1[top_pixel];
					sample_time_temp_buffer1[top_pixel] = sample_time_buffer[top_pixel];
					color_buffer[top_pixel] = make_color(prd_top.result);

					sample_time_buffer[top_pixel] = finish_sample_time_in_system;
					temp_buffer3[top_pixel] = temp_buffer2[top_pixel];
					temp_buffer2[top_pixel] = temp_buffer1[top_pixel];
					temp_buffer1[top_pixel] = color_buffer[top_pixel];
					raycounting_buffer[0] += 1;
					leaveLoop = true;
					atomicExch(&busy_buffer[top_pixel], 0); //Set pixel to free.
				}
				else {
					leaveLoop = true;
					//rayrejecting_buffer[0] += 1;
				}
			}
			else {
				leaveLoop = true;
				//rayrejecting_buffer[0] += 1;
			}
		}

		leaveLoop = false;
		while (!leaveLoop) {
			if (right_pixel.x < screen_size_buffer[0] && right_pixel.y < screen_size_buffer[1]) {
				if (atomicExch(&(busy_buffer[right_pixel]), 1u) == 0u) //If pixel is free
				{
					temp_buffer3[right_pixel] = temp_buffer2[right_pixel];
					temp_buffer2[right_pixel] = temp_buffer1[right_pixel];
					temp_buffer1[right_pixel] = color_buffer[right_pixel];
					color_buffer[right_pixel] = make_color(prd_right.result);

					sample_time_temp_buffer3[right_pixel] = sample_time_temp_buffer2[right_pixel];
					sample_time_temp_buffer2[right_pixel] = sample_time_temp_buffer1[right_pixel];
					sample_time_temp_buffer1[right_pixel] = sample_time_buffer[right_pixel];
					sample_time_buffer[right_pixel] = finish_sample_time_in_system;
					raycounting_buffer[0] += 1;
					leaveLoop = true;
					atomicExch(&busy_buffer[right_pixel], 0); //Set pixel to free.
				}
				else {
					leaveLoop = true;
					//rayrejecting_buffer[0] += 1;
				}
			}
			else {
				leaveLoop = true;
				//rayrejecting_buffer[0] += 1;
			}
		}
	}
}


/*
shoot 3 rays simultaneously, and calculate the gradient and add to the tile gradient buffer in this thread
combine the random map method and our tiling
*/
static __device__ __inline__ void shoot_3rays_use_random_map_tile()
{
	// record the start time from rendering

	float start_to_render_time = render_elapse_time_buffer[0];
	clock_t shoot_ray_start_time = clock();

	size_t2 screen = output_buffer.size();

	// assign a thread to a certain tile by its launch index x, this will distribute threads equally to every tile

	uint which_grid_calculate = launch_index.x % leaf_tile_indices.size();
	uint2 centre_pixel = make_uint2(leaf_tile_indices[which_grid_calculate].x, leaf_tile_indices[which_grid_calculate].y);
	uint tilesize = leaf_tile_sizes[launch_index.x];

	uint count_mini_tile_quantity_length = tilesize / 8u;
	uint2 start_pixel;
	start_pixel.x = centre_pixel.x - tilesize / 2u;
	start_pixel.y = centre_pixel.y - tilesize / 2u;



	// start to pick a mini tile base on tile info, now randomly choose a mini-tile inside the tile

	uint2 start_mini_tile;
	start_mini_tile.x = start_pixel.x / 8u;
	start_mini_tile.y = start_pixel.y / 8u;

	uint2 offset_mini_tile;
	volatile unsigned int seed = rnd_seeds[launch_index]; // volatile workaround for cuda 2.0 bug
	unsigned int new_seed = seed;
	float anytorandom = rnd(new_seed);
	float anytorandom2 = rnd(new_seed);
	rnd_seeds[launch_index] = new_seed;

	offset_mini_tile.x = uint(rnd(new_seed) * 100) % count_mini_tile_quantity_length;
	offset_mini_tile.y = uint(rnd(new_seed) * 100) % count_mini_tile_quantity_length;

	uint2 which_mini_tile;

	which_mini_tile.x = start_mini_tile.x + offset_mini_tile.x;
	which_mini_tile.y = start_mini_tile.y + offset_mini_tile.y;

	if (which_mini_tile.x > 64u) {
		which_mini_tile.x = 0u;
	}
	if (which_mini_tile.y > 64u) {
		which_mini_tile.y = 0u;
	}

	start_pixel.x = which_mini_tile.x * 8u;
	start_pixel.y = which_mini_tile.y * 8u;

	uint mini_tile_pixel_lookup = mini_tile_buffer[which_mini_tile];
	mini_tile_buffer[which_mini_tile] = mini_tile_pixel_lookup + 1;
	if (mini_tile_buffer[which_mini_tile] > 64) {
		mini_tile_buffer[which_mini_tile] = 0u;
	}

	uint x_lookup = mini_tile_pixel_lookup / 8u;
	uint y_lookup = mini_tile_pixel_lookup % 8u;
	uint2 total_lookup = make_uint2(x_lookup, y_lookup);

	uint lookup_location = random_map_buffer[total_lookup];

	uint x_offset = lookup_location / 8u;
	uint y_offset = lookup_location % 8u;
	uint2 total_offset = make_uint2(x_offset, y_offset);
	uint2 rand_pixel = start_pixel + total_offset;


	//pixel_at_tile_buffer[rand_pixel] = which_grid_calculate;

	//Right pixel
	uint2 right_pixel = rand_pixel + make_uint2(1.0f, 0.0f);
	//Top pixel
	uint2 top_pixel = rand_pixel + make_uint2(0.0f, 1.0f);


	//Ray creation (Center)
	float2 d = (make_float2(rand_pixel)) / make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction_center = normalize(d.x*U + d.y*V + W);

	optix::Ray ray = optix::make_Ray(ray_origin, ray_direction_center, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;

	//Ray creation (Top)
	float2 d_top = (make_float2(top_pixel)) / make_float2(screen) * 2.f - 1.f;
	float3 ray_direction_top = normalize(d_top.x*U + d_top.y*V + W);
	optix::Ray ray_top = optix::make_Ray(ray_origin, ray_direction_top, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_radiance prd_top;
	prd_top.importance = 1.f;
	prd_top.depth = 0;

	//Ray creation (Right)
	float2 d_right = (make_float2(right_pixel)) / make_float2(screen) * 2.f - 1.f;
	float3 ray_direction_right = normalize(d_right.x*U + d_right.y*V + W);
	optix::Ray ray_right = optix::make_Ray(ray_origin, ray_direction_right, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_radiance prd_right;
	prd_right.importance = 1.f;
	prd_right.depth = 0;


	rtTrace(top_object, ray, prd);
	rtTrace(top_object, ray_top, prd_top);
	rtTrace(top_object, ray_right, prd_right);

	float lum_centre = luminance(prd.result);
	float lum_right = luminance(prd_right.result);
	float lum_top = luminance(prd_top.result);

	uchar4 old_color_char = color_buffer[rand_pixel];
	float3 old_color = make_float3(old_color_char.z, old_color_char.y, old_color_char.x)*make_float3(1.0f / 255.99f);;

	old_color = float_color_buffer[rand_pixel];

	float lum_temporal = luminance(old_color);

	clock_t shoot_ray_end_time = clock();
	float sample_elapse_time = (float)(((shoot_ray_end_time - shoot_ray_start_time) / 1000000000.0f));

	//if (launch_index.x == 100u && launch_index.y == 100u)
	//	rtPrintf("srt = %f, set = %f\n", start_to_render_time, sample_elapse_time);

	float finish_sample_time_in_system = start_to_render_time + sample_elapse_time;

	finish_sample_time_in_system = start_to_render_time;


	float old_time = sample_time_buffer[rand_pixel];
	float new_time = finish_sample_time_in_system;
	float elapse_time = (new_time - old_time);

	if (launch_index.x == 100u && launch_index.y == 100u) {
		//rtPrintf("el time = %f\n", elapse_time);
	}

	//elapse_time /= 1000000000;

	if (launch_index.x == 100u && launch_index.y == 100u)
		//rtPrintf("elapse_time = %f\n", elapse_time);


		if (elapse_time < 0) {
			//rtPrintf("nagative issue really here!!!!!!!!!");
			elapse_time *= -1;
		}



	float YIQ_standard_center, YIQ_standard_right, YIQ_standard_top, YIQ_standard_old;

	YIQ_standard_center = prd.result.x * 0.299 + prd.result.y * 0.587 + prd.result.z * 0.114;
	YIQ_standard_right = prd_right.result.x * 0.299 + prd_right.result.y * 0.587 + prd_right.result.z * 0.114;
	YIQ_standard_top = prd_top.result.x * 0.299 + prd_top.result.y * 0.587 + prd_top.result.z * 0.114;
	YIQ_standard_old = old_color.x * 0.299 + old_color.y * 0.587 + old_color.z * 0.114;

	float_temp_buffer3[rand_pixel] = float_temp_buffer2[rand_pixel];
	float_temp_buffer2[rand_pixel] = float_temp_buffer1[rand_pixel];
	float_temp_buffer1[rand_pixel] = float_color_buffer[rand_pixel];
	float_color_buffer[rand_pixel] = prd.result;


	float g_x = lum_centre - lum_right;
	float g_y = lum_centre - lum_top;
	float g_t = lum_temporal - lum_centre;
	/*
	float g_x = YIQ_standard_center - YIQ_standard_right;
	float g_y = YIQ_standard_center - YIQ_standard_top;
	float g_t = YIQ_standard_old - YIQ_standard_center;
	*/

	if (g_x < 0)
		g_x *= -1;
	if (g_y < 0)
		g_y *= -1;
	if (g_t < 0)
		g_t *= -1;


	if (elapse_time == 0.0f) {
		if (launch_index.x == 100u && launch_index.y == 100u) {
			//rtPrintf("old t = %f, start_to_render_time = %f,sample_elapse_time = %f, el time = %f\n", old_time,start_to_render_time, sample_elapse_time,elapse_time);
		}
		//g_t = crosshair_buffer[rand_pixel].z;
		g_t /= 0.000001f;
	}
	else {
		g_t /= elapse_time;
	}
	g_t /= 2;

	//g_t /= elapse_time;

	//g_t *= exp(-3.27* elapse_time);





	//sample_time_buffer[rand_pixel] = new_time;

	// add crosshair to the crosshair_buffer

	float3 crosshair = make_float3(g_x, g_y, g_t);


	// calculate color and show gradient?
	uchar4 char_color = make_color(prd.result);
	float3 color = make_float3(char_color.z, char_color.y, char_color.x)*make_float3(1.0f / 255.99f);

	// extent per tile calculation 
	float3 extent;
	extent.x = extent.y = extent.z = 0.0f;
	//float vs = 0.04;
	//float rl = 100.0 / ((float(tilesize) * (float)tilesize) * (float)0.07);
	//vs = 1 / rl;
	float basic_tile_size = 64.0f;
	float vs = (ray_per_sec_buffer[0] / (256 * basic_tile_size))*(float(tilesize) * (float)tilesize) / basic_tile_size;
	//vs = (20 *(float(tilesize) * (float)tilesize) )/ basic_tile_size;;

	uint2 gradient_buffer_index_x = make_uint2(launch_index.x, 0.0f);
	uint2 gradient_buffer_index_y = make_uint2(launch_index.x, 1.0f);
	uint2 gradient_buffer_index_t = make_uint2(launch_index.x, 2.0f);
	float gradient_x_at_the_tile = tile_gradient_buffer[gradient_buffer_index_x];
	float gradient_y_at_the_tile = tile_gradient_buffer[gradient_buffer_index_y];
	float gradient_t_at_the_tile = tile_gradient_buffer[gradient_buffer_index_t];

	bool leaveLoop = false;
	while (!leaveLoop) {
		//if (busy_buffer[rand_pixel] == 0) //If pixel is free
		if (atomicExch(&(busy_buffer[rand_pixel]), 1u) == 0u) //If pixel is free
		{
			crosshair_buffer[rand_pixel] = crosshair;
			extent_buffer[rand_pixel] = extent;

			temp_buffer3[rand_pixel] = temp_buffer2[rand_pixel];
			temp_buffer2[rand_pixel] = temp_buffer1[rand_pixel];
			temp_buffer1[rand_pixel] = color_buffer[rand_pixel];

			color_buffer[rand_pixel] = make_color(prd.result);

			sample_time_temp_buffer3[rand_pixel] = sample_time_temp_buffer2[rand_pixel];
			sample_time_temp_buffer2[rand_pixel] = sample_time_temp_buffer1[rand_pixel];
			sample_time_temp_buffer1[rand_pixel] = sample_time_buffer[rand_pixel];
			sample_time_buffer[rand_pixel] = finish_sample_time_in_system;
			raycounting_buffer[0] += 1;
			leaveLoop = true;
			atomicExch(&busy_buffer[rand_pixel], 0); //Set pixel to free.
		}
		else {
			leaveLoop = true;
			//rayrejecting_buffer[0] += 1;
		}
	}
	leaveLoop = false;
	/**/
	while (!leaveLoop) {
		if (atomicExch(&(busy_buffer[top_pixel]), 1u) == 0u) //If pixel is free
		{
			sample_time_temp_buffer3[top_pixel] = sample_time_temp_buffer2[top_pixel];
			sample_time_temp_buffer2[top_pixel] = sample_time_temp_buffer1[top_pixel];
			sample_time_temp_buffer1[top_pixel] = sample_time_buffer[top_pixel];
			color_buffer[top_pixel] = make_color(prd_top.result);

			sample_time_buffer[top_pixel] = finish_sample_time_in_system;
			temp_buffer3[top_pixel] = temp_buffer2[top_pixel];
			temp_buffer2[top_pixel] = temp_buffer1[top_pixel];
			temp_buffer1[top_pixel] = color_buffer[top_pixel];
			raycounting_buffer[0] += 1;
			leaveLoop = true;
			atomicExch(&busy_buffer[top_pixel], 0); //Set pixel to free.
		}
		else {
			leaveLoop = true;
			//rayrejecting_buffer[0] += 1;
		}
	}

	leaveLoop = false;
	while (!leaveLoop) {
		if (atomicExch(&(busy_buffer[right_pixel]), 1u) == 0u) //If pixel is free
		{
			temp_buffer3[right_pixel] = temp_buffer2[right_pixel];
			temp_buffer2[right_pixel] = temp_buffer1[right_pixel];
			temp_buffer1[right_pixel] = color_buffer[right_pixel];
			color_buffer[right_pixel] = make_color(prd_right.result);

			sample_time_temp_buffer3[right_pixel] = sample_time_temp_buffer2[right_pixel];
			sample_time_temp_buffer2[right_pixel] = sample_time_temp_buffer1[right_pixel];
			sample_time_temp_buffer1[right_pixel] = sample_time_buffer[right_pixel];
			sample_time_buffer[right_pixel] = finish_sample_time_in_system;
			raycounting_buffer[0] += 1;
			leaveLoop = true;
			atomicExch(&busy_buffer[right_pixel], 0); //Set pixel to free.
		}
		else {
			leaveLoop = true;
			//rayrejecting_buffer[0] += 1;
		}
	}

}



/*
shoot 3 rays simultaneously, and calculate the gradient and add to the tile gradient buffer in this thread
combine the random map method and our tiling
*/
static __device__ __inline__ void shoot_3rays_use_random_map_tile_test()
{
	// record the start time from rendering

	float start_to_render_time = render_elapse_time_buffer[0];
	clock_t shoot_ray_start_time = clock();

	size_t2 screen = output_buffer.size();

	// assign a thread to a certain tile by its launch index x, this will distribute threads equally to every tile

	uint which_grid_calculate = launch_index.x % leaf_tile_indices.size();
	uint2 centre_pixel = make_uint2(leaf_tile_indices[which_grid_calculate].x, leaf_tile_indices[which_grid_calculate].y);
	uint tilesize = leaf_tile_sizes[launch_index.x];

	uint count_mini_tile_quantity_length = tilesize / 8u;
	uint2 start_pixel;
	start_pixel.x = centre_pixel.x - tilesize / 2u;
	start_pixel.y = centre_pixel.y - tilesize / 2u;



	// start to pick a mini tile base on tile info, now randomly choose a mini-tile inside the tile

	uint2 start_mini_tile;
	start_mini_tile.x = start_pixel.x / 8u;
	start_mini_tile.y = start_pixel.y / 8u;

	uint2 offset_mini_tile;
	volatile unsigned int seed = rnd_seeds[launch_index]; // volatile workaround for cuda 2.0 bug
	unsigned int new_seed = seed;
	float anytorandom = rnd(new_seed);
	float anytorandom2 = rnd(new_seed);
	rnd_seeds[launch_index] = new_seed;

	offset_mini_tile.x = uint(rnd(new_seed) * 100) % count_mini_tile_quantity_length;
	offset_mini_tile.y = uint(rnd(new_seed) * 100) % count_mini_tile_quantity_length;

	uint2 which_mini_tile;

	which_mini_tile.x = start_mini_tile.x + offset_mini_tile.x;
	which_mini_tile.y = start_mini_tile.y + offset_mini_tile.y;

	if (which_mini_tile.x > 64u) {
		which_mini_tile.x %= 64u;
	}
	if (which_mini_tile.y > 64u) {
		which_mini_tile.y %= 64u;
	}

	start_pixel.x = which_mini_tile.x * 8u;
	start_pixel.y = which_mini_tile.y * 8u;

	uint mini_tile_pixel_lookup = mini_tile_buffer[which_mini_tile];
	mini_tile_buffer[which_mini_tile] = mini_tile_pixel_lookup + 1;
	if (mini_tile_buffer[which_mini_tile] > 64) {
		mini_tile_buffer[which_mini_tile] = 0u;
	}

	uint x_lookup = mini_tile_pixel_lookup / 8u;
	uint y_lookup = mini_tile_pixel_lookup % 8u;
	uint2 total_lookup = make_uint2(x_lookup, y_lookup);

	uint lookup_location = random_map_buffer[total_lookup];

	uint x_offset = lookup_location / 8u;
	uint y_offset = lookup_location % 8u;
	uint2 total_offset = make_uint2(x_offset, y_offset);
	uint2 rand_pixel = start_pixel + total_offset;


	//pixel_at_tile_buffer[rand_pixel] = which_grid_calculate;

	//Right pixel
	uint2 right_pixel = rand_pixel + make_uint2(1.0f, 0.0f);
	//Top pixel
	uint2 top_pixel = rand_pixel + make_uint2(0.0f, 1.0f);


	//Ray creation (Center)
	float2 d = (make_float2(rand_pixel)) / make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction_center = normalize(d.x*U + d.y*V + W);

	optix::Ray ray = optix::make_Ray(ray_origin, ray_direction_center, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;

	//Ray creation (Top)
	float2 d_top = (make_float2(top_pixel)) / make_float2(screen) * 2.f - 1.f;
	float3 ray_direction_top = normalize(d_top.x*U + d_top.y*V + W);
	optix::Ray ray_top = optix::make_Ray(ray_origin, ray_direction_top, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_radiance prd_top;
	prd_top.importance = 1.f;
	prd_top.depth = 0;

	//Ray creation (Right)
	float2 d_right = (make_float2(right_pixel)) / make_float2(screen) * 2.f - 1.f;
	float3 ray_direction_right = normalize(d_right.x*U + d_right.y*V + W);
	optix::Ray ray_right = optix::make_Ray(ray_origin, ray_direction_right, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_radiance prd_right;
	prd_right.importance = 1.f;
	prd_right.depth = 0;


	rtTrace(top_object, ray, prd);
	rtTrace(top_object, ray_top, prd_top);
	rtTrace(top_object, ray_right, prd_right);

	float lum_centre = luminance(prd.result);
	float lum_right = luminance(prd_right.result);
	float lum_top = luminance(prd_top.result);

	uchar4 old_color_char = color_buffer[rand_pixel];
	float3 old_color = make_float3(old_color_char.z, old_color_char.y, old_color_char.x)*make_float3(1.0f / 255.99f);;

	old_color = float_color_buffer[rand_pixel];

	float lum_temporal = luminance(old_color);

	clock_t shoot_ray_end_time = clock();
	float sample_elapse_time = (float)(((shoot_ray_end_time - shoot_ray_start_time) / 1000000000.0f));

	//if (launch_index.x == 100u && launch_index.y == 100u)
	//	rtPrintf("srt = %f, set = %f\n", start_to_render_time, sample_elapse_time);

	float finish_sample_time_in_system = start_to_render_time + sample_elapse_time;

	//finish_sample_time_in_system = start_to_render_time;


	float old_time = sample_time_buffer[rand_pixel];
	float new_time = finish_sample_time_in_system;
	float elapse_time = (new_time - old_time);

	if (launch_index.x == 100u && launch_index.y == 100u) {
		//rtPrintf("el time = %f\n", elapse_time);
	}

	//elapse_time /= 1000000000;

	if (launch_index.x == 100u && launch_index.y == 100u)
		//rtPrintf("elapse_time = %f\n", elapse_time);


		if (elapse_time < 0) {
			//rtPrintf("nagative issue really here!!!!!!!!!");
			elapse_time *= -1;
		}



	float YIQ_standard_center, YIQ_standard_right, YIQ_standard_top, YIQ_standard_old;

	YIQ_standard_center = prd.result.x * 0.299 + prd.result.y * 0.587 + prd.result.z * 0.114;
	YIQ_standard_right = prd_right.result.x * 0.299 + prd_right.result.y * 0.587 + prd_right.result.z * 0.114;
	YIQ_standard_top = prd_top.result.x * 0.299 + prd_top.result.y * 0.587 + prd_top.result.z * 0.114;
	YIQ_standard_old = old_color.x * 0.299 + old_color.y * 0.587 + old_color.z * 0.114;

	float_temp_buffer3[rand_pixel] = float_temp_buffer2[rand_pixel];
	float_temp_buffer2[rand_pixel] = float_temp_buffer1[rand_pixel];
	float_temp_buffer1[rand_pixel] = float_color_buffer[rand_pixel];
	float_color_buffer[rand_pixel] = prd.result;


	float g_x = lum_centre - lum_right;
	float g_y = lum_centre - lum_top;
	float g_t = lum_temporal - lum_centre;
	/*
	float g_x = YIQ_standard_center - YIQ_standard_right;
	float g_y = YIQ_standard_center - YIQ_standard_top;
	float g_t = YIQ_standard_old - YIQ_standard_center;
	*/

	if (g_x < 0)
		g_x *= -1;
	if (g_y < 0)
		g_y *= -1;
	if (g_t < 0)
		g_t *= -1;

	if (launch_index.x == 100u && launch_index.y == 100u) {
		//rtPrintf("tile size = %u\n", tilesize);
		//rtPrintf("old t = %f, start_to_render_time = %f,sample_elapse_time = %f, el time = %f\n", old_time, start_to_render_time, sample_elapse_time, elapse_time);
	}


	if (elapse_time == 0.0f) {
		
		//g_t = crosshair_buffer[rand_pixel].z;
		g_t /= 0.000001f;
		g_t = 0;
	}
	else {
		g_t /= elapse_time;
	}
	g_t /= 2;

	//g_t /= elapse_time;

	//g_t *= exp(-3.27* elapse_time);





	//sample_time_buffer[rand_pixel] = new_time;

	// add crosshair to the crosshair_buffer

	float3 crosshair = make_float3(g_x, g_y, g_t);


	// calculate color and show gradient?
	uchar4 char_color = make_color(prd.result);
	float3 color = make_float3(char_color.z, char_color.y, char_color.x)*make_float3(1.0f / 255.99f);

	// extent per tile calculation 
	float3 extent;
	extent.x = extent.y = extent.z = 0.0f;
	//float vs = 0.04;
	//float rl = 100.0 / ((float(tilesize) * (float)tilesize) * (float)0.07);
	//vs = 1 / rl;
	float basic_tile_size = 64.0f;
	float vs = (ray_per_sec_buffer[0] / (256 * basic_tile_size))*(float(tilesize) * (float)tilesize) / basic_tile_size;
	//vs = (20 *(float(tilesize) * (float)tilesize) )/ basic_tile_size;;

	uint2 gradient_buffer_index_x = make_uint2(launch_index.x, 0.0f);
	uint2 gradient_buffer_index_y = make_uint2(launch_index.x, 1.0f);
	uint2 gradient_buffer_index_t = make_uint2(launch_index.x, 2.0f);
	float gradient_x_at_the_tile = tile_gradient_buffer[gradient_buffer_index_x];
	float gradient_y_at_the_tile = tile_gradient_buffer[gradient_buffer_index_y];
	float gradient_t_at_the_tile = tile_gradient_buffer[gradient_buffer_index_t];

	if (launch_index.x == 100u && launch_index.y == 100u) {
		//rtPrintf("%f %f %f \n" , gradient_x_at_the_tile, gradient_y_at_the_tile, gradient_t_at_the_tile);
		//rtPrintf("%f %f %f \n", crosshair.x, crosshair.y, crosshair.z);
	}

	bool leaveLoop = false;
	while (!leaveLoop) {
		//if (busy_buffer[rand_pixel] == 0) //If pixel is free
		if (atomicExch(&(busy_buffer[rand_pixel]), 1u) == 0u) //If pixel is free
		{
			crosshair_buffer[rand_pixel] = crosshair;
			extent_buffer[rand_pixel] = extent;

			temp_buffer3[rand_pixel] = temp_buffer2[rand_pixel];
			temp_buffer2[rand_pixel] = temp_buffer1[rand_pixel];
			temp_buffer1[rand_pixel] = color_buffer[rand_pixel];

			color_buffer[rand_pixel] = make_color(prd.result);

			sample_time_temp_buffer3[rand_pixel] = sample_time_temp_buffer2[rand_pixel];
			sample_time_temp_buffer2[rand_pixel] = sample_time_temp_buffer1[rand_pixel];
			sample_time_temp_buffer1[rand_pixel] = sample_time_buffer[rand_pixel];
			sample_time_buffer[rand_pixel] = finish_sample_time_in_system;
			raycounting_buffer[0] += 1;
			leaveLoop = true;
			atomicExch(&busy_buffer[rand_pixel], 0); //Set pixel to free.
		}
		else {
			leaveLoop = true;
			//rayrejecting_buffer[0] += 1;
		}
	}
	leaveLoop = false;
	/**/
	while (!leaveLoop) {
		if (top_pixel.x < screen_size_buffer[0] && top_pixel.y < screen_size_buffer[0]) {
			if (atomicExch(&(busy_buffer[top_pixel]), 1u) == 0u) //If pixel is free
			{
				sample_time_temp_buffer3[top_pixel] = sample_time_temp_buffer2[top_pixel];
				sample_time_temp_buffer2[top_pixel] = sample_time_temp_buffer1[top_pixel];
				sample_time_temp_buffer1[top_pixel] = sample_time_buffer[top_pixel];
				color_buffer[top_pixel] = make_color(prd_top.result);

				sample_time_buffer[top_pixel] = finish_sample_time_in_system;
				temp_buffer3[top_pixel] = temp_buffer2[top_pixel];
				temp_buffer2[top_pixel] = temp_buffer1[top_pixel];
				temp_buffer1[top_pixel] = color_buffer[top_pixel];
				raycounting_buffer[0] += 1;
				leaveLoop = true;
				atomicExch(&busy_buffer[top_pixel], 0); //Set pixel to free.
			}
			else {
				leaveLoop = true;
				//rayrejecting_buffer[0] += 1;
			}
		}
		else{
			leaveLoop = true;
		//rayrejecting_buffer[0] += 1;
		}
	}

	leaveLoop = false;
	while (!leaveLoop) {
		if (right_pixel.x < screen_size_buffer[0] && right_pixel.y < screen_size_buffer[0]) {
			if (atomicExch(&(busy_buffer[right_pixel]), 1u) == 0u) //If pixel is free
			{
				temp_buffer3[right_pixel] = temp_buffer2[right_pixel];
				temp_buffer2[right_pixel] = temp_buffer1[right_pixel];
				temp_buffer1[right_pixel] = color_buffer[right_pixel];
				color_buffer[right_pixel] = make_color(prd_right.result);

				sample_time_temp_buffer3[right_pixel] = sample_time_temp_buffer2[right_pixel];
				sample_time_temp_buffer2[right_pixel] = sample_time_temp_buffer1[right_pixel];
				sample_time_temp_buffer1[right_pixel] = sample_time_buffer[right_pixel];
				sample_time_buffer[right_pixel] = finish_sample_time_in_system;
				raycounting_buffer[0] += 1;
				leaveLoop = true;
				atomicExch(&busy_buffer[right_pixel], 0); //Set pixel to free.
			}
			else {
				leaveLoop = true;
				//rayrejecting_buffer[0] += 1;
			}
		}
		else {
			leaveLoop = true;
			//rayrejecting_buffer[0] += 1;
		}
	}

}



static __device__ __inline__ void new_calculate_variance(bool leaf)
{
	// find the center pixel of a tile
	uint2 centre_pixel;
	unsigned int tile_size;
	if (leaf)
	{
		centre_pixel = make_uint2(leaf_tile_indices[launch_index.x].x, leaf_tile_indices[launch_index.x].y);
		tile_size = leaf_tile_sizes[launch_index.x];
	}
	else
	{
		centre_pixel = make_uint2(parent_tile_indices[launch_index.x].x, parent_tile_indices[launch_index.x].y);
		tile_size = parent_tile_sizes[launch_index.x];
	}
	//rtPrintf("tilesize = %u\n", tile_size);
	// find the start pixel of a tile (left to right, buttom to top)
	uint2 start_pixel;
	start_pixel.x = centre_pixel.x - tile_size / 2u;
	start_pixel.y = centre_pixel.y - tile_size / 2u;

	// find the end pixel of a tile
	uint2 end_pixel;
	end_pixel.x = centre_pixel.x + tile_size / 2u;
	end_pixel.y = centre_pixel.y + tile_size / 2u;

	// indicate the current pixel to iterate
	uint2 index_pixel = start_pixel;

	// initiate variables
	uchar4 mean_color;
	uchar4 char_color, char_color2, char_color3, char_color4;
	float3 color, color2, color3, color4;
	float3 color_square;
	float n = 0.0;
	float3 mean; mean.x = mean.y = mean.z = 0.0;
	float3 m2; m2.x = m2.y = m2.z = 0.0;
	float3 variance;
	float YIQ_standard_Y;


	float cross_hair_sum_x = 0.0f, cross_hair_sum_y = 0.0f, cross_hair_sum_t = 0.0f;
	float temp_sum_x = 0.0f, temp_sum_y = 0.0f, temp_sum_z = 0.0f;

	for (unsigned int i = 0; i < tile_size; i++)
	{
		for (unsigned int j = 0; j < tile_size; j++)
		{
			n++;
			//n = n + 4;
			uint2 offset = make_uint2(i, j);
			uint2 index = start_pixel + offset;

			char_color = color_buffer[index];
			char_color2 = temp_buffer1[index];
			char_color3 = temp_buffer2[index];
			char_color4 = temp_buffer3[index];
			color = make_float3(char_color.x, char_color.y, char_color.z)*make_float3(1.0f / 255.99f);
			color2 = make_float3(char_color2.x, char_color2.y, char_color2.z)*make_float3(1.0f / 255.99f);
			color3 = make_float3(char_color3.x, char_color3.y, char_color3.z)*make_float3(1.0f / 255.99f);
			color4 = make_float3(char_color4.x, char_color4.y, char_color4.z)*make_float3(1.0f / 255.99f);


			float3 delta = color - mean;

			mean += delta / n;
			m2 += delta * (color - mean);

			for (int k = 0; k < 4; k++) {
				if (k == 0) {
					float3 gradient = crosshair_buffer[index];
					cross_hair_sum_x += gradient.x;
					cross_hair_sum_y += gradient.y;
					cross_hair_sum_t += gradient.z;
				}
				else if (k == 1) {
					float3 gradient = crosshair_buffer1[index];
					gradient *= exp(-3.47);
					cross_hair_sum_x += gradient.x;
					cross_hair_sum_y += gradient.y;
					cross_hair_sum_t += gradient.z;
				}
				else if (k == 2) {
					float3 gradient = crosshair_buffer2[index];
					gradient *= exp(-3.47 * 2);
					cross_hair_sum_x += gradient.x;
					cross_hair_sum_y += gradient.y;
					cross_hair_sum_t += gradient.z;
				}
				else if (k == 3) {
					float3 gradient = crosshair_buffer3[index];
					gradient *= exp(-3.47 * 3);
					cross_hair_sum_x += gradient.x;
					cross_hair_sum_y += gradient.y;
					cross_hair_sum_t += gradient.z;
				}
			}
		}
	}

	variance = m2 / (n - 1);
	//rtPrintf("variance_show= %u\n", show_variance_buffer[launch_index.x]);

	float v = luminance(variance);
	YIQ_standard_Y = variance.x * 0.299 + variance.y * 0.587 + variance.z * 0.114;

	if (leaf)
	{
		//variance_buffer[launch_index.x] = v;
		variance_buffer[launch_index.x] = YIQ_standard_Y;

		uint2 gradient_buffer_index_x = make_uint2(launch_index.x, 0.0f);
		uint2 gradient_buffer_index_y = make_uint2(launch_index.x, 1.0f);
		uint2 gradient_buffer_index_t = make_uint2(launch_index.x, 2.0f);
		//rtPrintf(" %f %f \n", cross_hair_sum_x, cross_hair_sum_y);
		tile_gradient_buffer[gradient_buffer_index_x] = cross_hair_sum_x;
		tile_gradient_buffer[gradient_buffer_index_y] = cross_hair_sum_y;
		tile_gradient_buffer[gradient_buffer_index_t] = cross_hair_sum_t;
	}
	else
	{
		//parent_variance_buffer[launch_index.x] = v;
		parent_variance_buffer[launch_index.x] = YIQ_standard_Y;
	}

	if (show_variance_buffer[0] > 0) {
		for (unsigned int i = 0; i < tile_size; i++)
		{
			for (unsigned int j = 0; j < tile_size; j++)
			{
				uint2 offset = make_uint2(i, j);
				uint2 index = start_pixel + offset;
				char_color = color_buffer[index];
				color = make_float3(char_color.z, char_color.y, char_color.x)*make_float3(1.0f / 255.99f);
				YIQ_standard_Y = variance.x * 0.299 + variance.y * 0.587 + variance.z * 0.114;

				color.x = color.x + YIQ_standard_Y * 10;
				output_buffer[index] = make_color(color);
			}
		}
	}

}

static __device__ __inline__ void calculate_extent() {

	// find the center pixel of a tile
	uint2 centre_pixel;
	unsigned int tile_size = leaf_tile_sizes[launch_index.x];

	// find the start pixel of a tile (left to right, buttom to top)
	uint2 start_pixel;
	start_pixel.x = centre_pixel.x - tile_size / 2u;
	start_pixel.y = centre_pixel.y - tile_size / 2u;

	// indicate the current pixel to iterate
	uint2 index_pixel = start_pixel;

	// tempory change to gradient per pixel
	float basic_tile_size = 64.0f;
	float vs = (ray_per_sec_buffer[0] / (256 * basic_tile_size))*(float(tile_size) * (float)tile_size) / basic_tile_size;

	vs = 1/(((float)ray_per_sec_buffer[0] / 256.0f) / (float(tile_size) * (float)tile_size));

	//vs = ((float)ray_per_sec_buffer[0]/256.0f)();

	uint2 gradient_buffer_index_x = make_uint2(launch_index.x, 0.0f);
	uint2 gradient_buffer_index_y = make_uint2(launch_index.x, 1.0f);
	uint2 gradient_buffer_index_t = make_uint2(launch_index.x, 2.0f);

	float gradient_x_at_the_tile = tile_gradient_buffer[gradient_buffer_index_x];
	float gradient_y_at_the_tile = tile_gradient_buffer[gradient_buffer_index_y];
	float gradient_t_at_the_tile = tile_gradient_buffer[gradient_buffer_index_t];

	// some sanity check
	if (gradient_x_at_the_tile <= 0)
		gradient_x_at_the_tile *= -1;
	if (gradient_y_at_the_tile <= 0)
		gradient_y_at_the_tile *= -1;
	if (gradient_t_at_the_tile <= 0)
		gradient_t_at_the_tile *= -1;

	if (launch_index.x == 100u) {
		//rtPrintf("vs = %f, x = %f, y = %f, z = %f\n", vs, gradient_x_at_the_tile, gradient_y_at_the_tile, gradient_t_at_the_tile);
		//rtPrintf("tilesize = %u\n", tile_size);
	}


	float3 extent;

	float maxfiltersize = 100.0f, maxfiltersize_t = 25.0f;

	if (gradient_x_at_the_tile == 0.0f && gradient_y_at_the_tile == 0.0f && gradient_t_at_the_tile == 0.0f)
	{
		extent.x = extent.y = maxfiltersize;
		extent.z = maxfiltersize_t;
	}
	else if (gradient_x_at_the_tile == 0.0f && gradient_y_at_the_tile == 0.0f) {
		extent.x = maxfiltersize;
		extent.y = maxfiltersize;
		extent.z = 0.0f;
	}
	else if (gradient_y_at_the_tile == 0.0f && gradient_t_at_the_tile == 0.0f) {
		extent.x = 0.0f;
		extent.y = maxfiltersize;
		extent.z = maxfiltersize_t;
	}
	else if (gradient_x_at_the_tile == 0.0f && gradient_t_at_the_tile == 0.0f) {
		extent.x = maxfiltersize;
		extent.y = 0.0f;
		extent.z = maxfiltersize_t;
	}
	else if (gradient_x_at_the_tile == 0.0f) {
		extent.x = maxfiltersize;
		extent.y = 0.0f;
		extent.z = 0.0f;
	}
	else if (gradient_y_at_the_tile == 0.0f) {
		extent.x = 0.0f;
		extent.y = maxfiltersize;
		extent.z = 0.0f;
	}
	else if (gradient_t_at_the_tile == 0.0f) {
		extent.x = 0.0f;
		extent.y = 0.0f;
		extent.z = maxfiltersize_t;
	}
	else
	{

		extent.x = pow(((gradient_y_at_the_tile * gradient_t_at_the_tile * vs) / (gradient_x_at_the_tile * gradient_x_at_the_tile)), 0.33f);
		extent.y = pow(((gradient_x_at_the_tile * gradient_t_at_the_tile * vs) / (gradient_y_at_the_tile * gradient_y_at_the_tile)), 0.33f);
		extent.z = pow(((gradient_x_at_the_tile * gradient_y_at_the_tile * vs) / (gradient_t_at_the_tile * gradient_t_at_the_tile)), 0.33f);
	}
	extent_at_tile[launch_index.x] = extent;

	if (launch_index.x == 133u) {
		
		//rtPrintf(" In calculate extent: x = %f, y = %f, z = %f\n", extent.x, extent.y, extent.z);
		
		//rtPrintf("ray_per_sec_buffer = %d \n", ray_per_sec_buffer[0]);
		//rtPrintf("vs = %f, x = %f, y = %f, z = %f\n",vs, gradient_x_at_the_tile, gradient_y_at_the_tile, gradient_t_at_the_tile);
		//rtPrintf("screen_size_buffer0 = %d screen_size_buffer1 = %d\n", screen_size_buffer[0], screen_size_buffer[1]);
	}
}




static  __device__ __inline__ void no_reconstruction() {
	uint2 index = make_uint2(launch_index.x, launch_index.y);
	uchar4 final_color = color_buffer[index];
	output_buffer[index] = final_color;
}


static __device__ __inline__ void gaussian_filter_to_whole_image_gather()
{
	uint2 index = make_uint2(launch_index.x, launch_index.y);
	uchar4 final_color1 = color_buffer[index];
	uint which_grid_calculate = pixel_at_tile_buffer[index];
	
	if (which_grid_calculate > 256) {
		// assign a tile if the call is illegal, should not happen after every pixel is updated at least once
		//rtPrintf("not reach here\n, which_grid_calculate = %u", which_grid_calculate);

		which_grid_calculate = launch_index.x;
	}
	uint2 centre_pixel = make_uint2(leaf_tile_indices[which_grid_calculate].x, leaf_tile_indices[which_grid_calculate].y);
	unsigned int tilesize = leaf_tile_sizes[which_grid_calculate];

	float sig = 10.0f;
	float3 extents = extent_buffer[index];
	uint gaussian_filter_size = 21u;
	/*
	if(extents.y!=0)
	rtPrintf("search for y = %u \n", extents.y);
	*/
	float total_weight = 0.0f;
	uchar4 total_color, final_color;
	float3 tot_color;
	tot_color.x = tot_color.y = tot_color.z = 0.0f;

	uint gaussian_start_index_x = launch_index.x - gaussian_filter_size / 2 + 1;
	uint gaussian_start_index_y = launch_index.y - gaussian_filter_size / 2 + 1;
	uint2 guassian_start_index = make_uint2(gaussian_start_index_x, gaussian_start_index_y);

	for (unsigned int i = 0u; i < gaussian_filter_size; i++)
	{
		for (unsigned int j = 0u; j < gaussian_filter_size; j++)
		{
			float N = (1.0f) / (2.0f*PI*sig*sig);
			int x = (int)i;
			int y = (int)j;
			float E = exp(((x*x + y * y) / (2.0f*sig*sig))* (-1.0f));
			N *= E;
			uint color_index_x = guassian_start_index.x + i;
			uint color_index_y = guassian_start_index.y + j;

			if (color_index_x < 0 || color_index_x > screen_size_buffer[0] || color_index_y < 0 || color_index_y > screen_size_buffer[1]) {
				continue;
			}
			uint2 color_index = make_uint2(color_index_x, color_index_y);
			uchar4 color_info = color_buffer[color_index];
			float3 color_info_float = make_float3(color_info.z, color_info.y, color_info.x)*make_float3(1.0f / 255.99f);
			color_info_float *= make_float3(N);
			tot_color += color_info_float;
			total_weight += N;
		}
	}//for loop ends

	tot_color *= make_float3(1.0f / total_weight);
	final_color = make_color(tot_color);

	// end gather process

	// draw the tiles & extent

	if (show_gradient_buffer[0] > 0)
	{
		float3 going_to_add_gradient_color = crosshair_buffer[index];
		float3 final_color_make = make_float3(final_color.z, final_color.y, final_color.x)*make_float3(1.0f / 255.99f);

		final_color_make.x += going_to_add_gradient_color.x;
		final_color_make.y += going_to_add_gradient_color.y;
		final_color_make.z += going_to_add_gradient_color.z;

		final_color = make_color(final_color_make);
		output_buffer[index] = final_color;
	}
	else
	{
		output_buffer[index] = final_color;
	}


	// draw the tiles & extent
	if (show_tile_buffer[0] > 0) {
		//if (rand_pixel.x == centre_pixel.x - tilesize / 2u || rand_pixel.y == centre_pixel.y - tilesize / 2u) {
		if (index.x == centre_pixel.x - tilesize / 2u || index.x == centre_pixel.x + tilesize / 2u || index.y == centre_pixel.y - tilesize / 2u || index.y == centre_pixel.y + tilesize / 2u) {
			float3 whiteline = make_float3(255.0f, 255.0f, 255.0f);
			output_buffer[index] = make_color(whiteline);
		}
		else {
			output_buffer[index] = final_color;
		}

	}
	else {
		output_buffer[index] = final_color;
	}
	
}

// implement 3D Gaussian filter using convolution
static __device__ __inline__ void convolution_3D_to_tile_gather()
{
	// record the start time from rendering

	float start_to_reconstruction_time = render_elapse_time_buffer[0];
	clock_t reconstruction_start_time = clock();

	// define the input index as thread index
	uint2 index = make_uint2(launch_index.x, launch_index.y);
	// using the input index to find the tile (tile = grid in this code)
	uint which_grid_calculate = pixel_at_tile_buffer[index];

	// assign a tile if the call is illegal, should not happen after every pixel is updated at least once
	if (which_grid_calculate > 256) {
		which_grid_calculate = launch_index.x;
	}
	// the center pixel of the tile
	uint2 centre_pixel = make_uint2(leaf_tile_indices[which_grid_calculate].x, leaf_tile_indices[which_grid_calculate].y);
	// tile size of the tile
	unsigned int tilesize = leaf_tile_sizes[which_grid_calculate];

	// find the calculated extent(filter size) in this tile
	float3 extents = extent_buffer[index];
	uint3 filter_size = make_uint3((unsigned int)extents.x, (unsigned int)extents.y, (unsigned int)extents.z);


	// reassign the filter size if they are too big

	//float3 going_to_use_this_extent = extent_buffer[index];
	float3 going_to_use_this_extent = extent_at_tile[which_grid_calculate];

	////////////////caution!!!!!!! need to change back if not ok

	//going_to_use_this_extent.x /= 5.5;
	//going_to_use_this_extent.y /= 5.5;
	//going_to_use_this_extent.x -= 2.0;
	//going_to_use_this_extent.y -= 2.0;

	if (launch_index.x == 300u && launch_index.y == 300u) {
		//rtPrintf("which_grid_calculate = %u \n", which_grid_calculate);
		//rtPrintf("filter size x = %f, filter size y = %f, filter size z = %f \n", going_to_use_this_extent.x, going_to_use_this_extent.y, going_to_use_this_extent.z);
	}

	if (going_to_use_this_extent.x < 0)
		going_to_use_this_extent.x = 0.0f;
	if (going_to_use_this_extent.y < 0)
		going_to_use_this_extent.y = 0.0f;

	uint gaussian_filter_size_x = going_to_use_this_extent.x;
	uint gaussian_filter_size_y = going_to_use_this_extent.y;
	uint gaussian_filter_size_z = going_to_use_this_extent.z;


	//gaussian_filter_size_x = 21;
	//gaussian_filter_size_y = 21;

	// reassign the filter size to odd number

	if (gaussian_filter_size_x % 2 == 0u) {
		gaussian_filter_size_x += 1u;
	}

	if (gaussian_filter_size_y % 2 == 0u) {
		gaussian_filter_size_y += 1u;
	}

	if (gaussian_filter_size_z % 2 == 0u) {
		gaussian_filter_size_z += 1u;
	}

	uint temp_filter_size_z = gaussian_filter_size_z;

	// reassign the filter size if they are too big
	if (gaussian_filter_size_x > 33u) {
		gaussian_filter_size_x = 33u;
	}

	if (gaussian_filter_size_y > 33u) {
		gaussian_filter_size_y = 33u;
	}

	//if (gaussian_filter_size_z > 3u) {
	//	gaussian_filter_size_z = 3u;
	//}

	if (temp_filter_size_z > 21u) {
		temp_filter_size_z = 21u;
	}

	if (launch_index.x == 100u && launch_index.y == 100u) {
		//rtPrintf("In recon: x = %d, y = %d, z = %d \n", gaussian_filter_size_x, gaussian_filter_size_y, gaussian_filter_size_z);
		//rtPrintf("which_grid_calculate= %d \n", which_grid_calculate);
	}

	// Test  

	float total_weight = 0.0f;
	uchar4 total_color, final_color;
	float3 tot_color;
	tot_color.x = tot_color.y = tot_color.z = 0.0f;

	// help to find where the start pixel for Gaussian filter
	uint gaussian_start_index_x = launch_index.x - gaussian_filter_size_x / 2 + 1;
	uint gaussian_start_index_y = launch_index.y - gaussian_filter_size_y / 2 + 1;
	uint2 guassian_start_index = make_uint2(gaussian_start_index_x, gaussian_start_index_y);

	// use convolution
	for (unsigned int i = 0u; i < gaussian_filter_size_x; i++)
	{
		for (unsigned int j = 0u; j < gaussian_filter_size_y; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				float Nx, Ny, Nz, N;

				uint color_index_x = guassian_start_index.x + i;
				uint color_index_y = guassian_start_index.y + j;
				uint2 color_index = make_uint2(color_index_x, color_index_y);

				if (color_index_x < 0 || color_index_x > screen_size_buffer[0] || color_index_y < 0 || color_index_y > screen_size_buffer[1]) {
					continue;
				}


				if (gaussian_filter_size_y == 1u) {
					Ny = gaussian_1[j];
				}
				else if (gaussian_filter_size_y == 3u) {
					Ny = gaussian_3[j];
				}
				else if (gaussian_filter_size_y == 5u) {
					Ny = gaussian_5[j];
				}
				else if (gaussian_filter_size_y == 7u) {
					Ny = gaussian_7[j];
				}
				else if (gaussian_filter_size_y == 9u) {
					Ny = gaussian_9[j];
				}
				else if (gaussian_filter_size_y == 11u) {
					Ny = gaussian_11[j];
				}
				else if (gaussian_filter_size_y == 13u) {
					Ny = gaussian_13[j];
				}
				else if (gaussian_filter_size_y == 15u) {
					Ny = gaussian_15[j];
				}
				else if (gaussian_filter_size_y == 17u) {
					Ny = gaussian_17[j];
				}
				else if (gaussian_filter_size_y == 19u) {
					Ny = gaussian_19[j];
				}
				else if (gaussian_filter_size_y == 21u) {
					Ny = gaussian_21[j];
				}
				else if (gaussian_filter_size_y == 23u) {
					Ny = gaussian_23[j];
				}
				else if (gaussian_filter_size_y == 25u) {
					Ny = gaussian_25[j];
				}
				else if (gaussian_filter_size_y == 27u) {
					Ny = gaussian_27[j];
				}
				else if (gaussian_filter_size_y == 29u) {
					Ny = gaussian_29[j];
				}
				else if (gaussian_filter_size_y == 31u) {
					Ny = gaussian_31[j];
				}
				else if (gaussian_filter_size_y == 33u) {
					Ny = gaussian_33[j];
				}
				else {
					Ny = gaussian_33[j];
				}

				if (gaussian_filter_size_x == 1u) {
					Nx = gaussian_1[i];
				}
				else if (gaussian_filter_size_x == 3u) {
					Nx = gaussian_3[i];
				}
				else if (gaussian_filter_size_x == 5u) {
					Nx = gaussian_5[i];
				}
				else if (gaussian_filter_size_x == 7u) {
					Nx = gaussian_7[i];
				}
				else if (gaussian_filter_size_x == 9u) {
					Nx = gaussian_9[i];
				}
				else if (gaussian_filter_size_x == 11u) {
					Nx = gaussian_11[i];
				}
				else if (gaussian_filter_size_x == 13u) {
					Nx = gaussian_13[i];
				}
				else if (gaussian_filter_size_x == 15u) {
					Nx = gaussian_15[i];
				}
				else if (gaussian_filter_size_x == 17u) {
					Nx = gaussian_17[i];
				}
				else if (gaussian_filter_size_x == 19u) {
					Nx = gaussian_19[i];
				}
				else if (gaussian_filter_size_x == 21u) {
					Nx = gaussian_21[i];
				}
				else if (gaussian_filter_size_x == 23u) {
					Nx = gaussian_23[i];
				}
				else if (gaussian_filter_size_x == 25u) {
					Nx = gaussian_25[i];
				}
				else if (gaussian_filter_size_x == 27u) {
					Nx = gaussian_27[i];
				}
				else if (gaussian_filter_size_x == 29u) {
					Nx = gaussian_29[i];
				}
				else if (gaussian_filter_size_x == 31u) {
					Nx = gaussian_31[i];
				}
				else if (gaussian_filter_size_x == 33u) {
					Nx = gaussian_33[i];
				}
				else {
					Nx = gaussian_33[i];
				}

				float sample_age = 0.0f;

				if (k == 0) {
					sample_age = start_to_reconstruction_time - sample_time_buffer[color_index];
				}
				else if (k == 1) {
					sample_age = start_to_reconstruction_time - sample_time_temp_buffer1[color_index];
				}
				else if (k == 2) {
					sample_age = start_to_reconstruction_time - sample_time_temp_buffer2[color_index];
				}
				else if (k == 3) {
					sample_age = start_to_reconstruction_time - sample_time_temp_buffer3[color_index];
				}
				else {
					sample_age = 777.0f;
				}



				float new_temporal_extent = going_to_use_this_extent.z * 2.0;

				//sample_age -= 0.045*k;

				if (sample_age <= 0)
					sample_age = 0;

				//sample_age *= 750;

				//sample_age *= 100;

				uint sample_age_index = (uint)floorf(sample_age);
				uint sample_age_index_ceil = (uint)ceilf(sample_age);

				if (launch_index.x == 50u && launch_index.y == 50u && k == 0) {
					//rtPrintf("sample_age = %f, sample_age_cc = %u sample_age_index = %u\n", sample_age, (uint)ceilf(sample_age), sample_age_index);
					//rtPrintf("extentx = %f, extenty = %f, new temp extentz = %f\n", going_to_use_this_extent.x, going_to_use_this_extent.y, new_temporal_extent);
					//rtPrintf("which_grid_calculate = %d, extent x= %f, extent y = %f, extent z = %f\n", which_grid_calculate, going_to_use_this_extent.x, going_to_use_this_extent.y, going_to_use_this_extent.z);
				}

				// interpolation:
				// A0   : sample_age floor
				// A1   : sample_age ceil
				// A    : current extent (new_temporal_extent)
				// B0   : Gaussian weight for A0
				// B1   : Gaussian weight for A1
				// B(Nz): Expected intepolation weight
				// alpha: (A-A0)/(A1-A0) = A-A0

				float A0, A1, A, B0, B1, alpha;
				A0 = floorf(sample_age);
				A1 = ceilf(sample_age);
				alpha = sample_age - A0;

				//if (launch_index.x == 100u && launch_index.y == 100u && k == 0)
				//	rtPrintf("new_temporal_extent = %f, sample_age = %f\n", new_temporal_extent, sample_age);


				if (sample_age >= 25.0f) {
					Nz = 0.00000;
					//Nz = 0.0000001;
				}
				else if (new_temporal_extent > 49.0f) {
					B0 = gaussianT_49[sample_age_index];
					B1 = gaussianT_49[sample_age_index_ceil];
					Nz = B0 * (1 - alpha) + B1 * alpha;
				}
				else {
					if (new_temporal_extent <= 1u) {
						if (sample_age_index > 0u)
							Nz = gaussianT_1[0];
						else
							Nz = gaussianT_1[sample_age_index];
					}
					else if (new_temporal_extent <= 3u) {

						if (sample_age_index >= 1u)
							Nz = gaussianT_3[1];
						else {
							B0 = gaussianT_3[sample_age_index];
							B1 = gaussianT_3[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 5u) {
						if (sample_age_index >= 2u)
							Nz = gaussianT_5[2];
						else {
							B0 = gaussianT_5[sample_age_index];
							B1 = gaussianT_5[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 7u) {
						if (sample_age_index >= 3u)
							Nz = gaussianT_7[3];
						else {
							B0 = gaussianT_7[sample_age_index];
							B1 = gaussianT_7[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 9u) {
						if (sample_age_index >= 4u)
							Nz = gaussianT_9[4];
						else {
							B0 = gaussianT_9[sample_age_index];
							B1 = gaussianT_9[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 11u) {
						if (sample_age_index >= 5u)
							Nz = gaussianT_11[5];
						else {
							B0 = gaussianT_11[sample_age_index];
							B1 = gaussianT_11[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 13u) {
						if (sample_age_index >= 6u)
							Nz = gaussianT_13[6];
						else {
							B0 = gaussianT_13[sample_age_index];
							B1 = gaussianT_13[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 15u) {
						if (sample_age_index >= 7u)
							Nz = gaussianT_15[7];
						else {
							B0 = gaussianT_15[sample_age_index];
							B1 = gaussianT_15[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 17u) {
						if (sample_age_index >= 8u)
							Nz = gaussianT_17[8];
						else {
							B0 = gaussianT_17[sample_age_index];
							B1 = gaussianT_17[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 19u) {
						if (sample_age_index >= 9u)
							Nz = gaussianT_19[9];
						else {
							B0 = gaussianT_19[sample_age_index];
							B1 = gaussianT_19[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 21u) {
						if (sample_age_index >= 10u)
							Nz = gaussianT_21[10];
						else {
							B0 = gaussianT_21[sample_age_index];
							B1 = gaussianT_21[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 23u) {
						if (sample_age_index >= 11u)
							Nz = gaussianT_23[11];
						else {
							B0 = gaussianT_23[sample_age_index];
							B1 = gaussianT_23[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 25u) {
						if (sample_age_index >= 12u)
							Nz = gaussianT_25[12];
						else {
							B0 = gaussianT_25[sample_age_index];
							B1 = gaussianT_25[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 27u) {
						if (sample_age_index >= 13u)
							Nz = gaussianT_27[13];
						else {
							B0 = gaussianT_27[sample_age_index];
							B1 = gaussianT_27[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 29u) {
						if (sample_age_index >= 14u)
							Nz = gaussianT_29[14];
						else {
							B0 = gaussianT_29[sample_age_index];
							B1 = gaussianT_29[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 31u) {
						if (sample_age_index >= 15u)
							Nz = gaussianT_31[15];
						else {
							B0 = gaussianT_31[sample_age_index];
							B1 = gaussianT_31[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 33u) {
						if (sample_age_index >= 16u)
							Nz = gaussianT_33[16];
						else {
							B0 = gaussianT_33[sample_age_index];
							B1 = gaussianT_33[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 35u) {
						if (sample_age_index >= 17u)
							Nz = gaussianT_35[17];
						else {
							B0 = gaussianT_35[sample_age_index];
							B1 = gaussianT_35[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 37u) {
						if (sample_age_index >= 18u)
							Nz = gaussianT_37[18];
						else {
							B0 = gaussianT_37[sample_age_index];
							B1 = gaussianT_37[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 39u) {
						if (sample_age_index >= 19u)
							Nz = gaussianT_39[19];
						else {
							B0 = gaussianT_39[sample_age_index];
							B1 = gaussianT_39[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 41u) {
						if (sample_age_index >= 20u)
							Nz = gaussianT_41[20];
						else {
							B0 = gaussianT_41[sample_age_index];
							B1 = gaussianT_41[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 43u) {
						if (sample_age_index >= 21u)
							Nz = gaussianT_43[21];
						else {
							B0 = gaussianT_43[sample_age_index];
							B1 = gaussianT_43[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 45u) {
						if (sample_age_index >= 22u)
							Nz = gaussianT_45[22];
						else {
							B0 = gaussianT_45[sample_age_index];
							B1 = gaussianT_45[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 47u) {
						if (sample_age_index >= 23u)
							Nz = gaussianT_47[23];
						else {
							B0 = gaussianT_47[sample_age_index];
							B1 = gaussianT_47[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else {
						if (sample_age_index >= 24u)
							Nz = gaussianT_49[24];
						else {
							B0 = gaussianT_49[sample_age_index];
							B1 = gaussianT_49[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
				}

				if (Nz == 0.0f) {
					Nz = 0.000001;
				}

				/*
				if (gaussian_filter_size_z == 1u) {
				Nz = gaussian_1[k];
				}
				else if (gaussian_filter_size_z == 3u) {
				Nz = gaussian_3[k];
				}


				if (temp_filter_size_z == 1u) {
				Nz = gaussian_1[k];
				}
				else if (temp_filter_size_z == 3u) {
				Nz = gaussian_3[k];
				}
				else if (temp_filter_size_z == 5u) {
				Nz = gaussian_5[k];
				}
				else if (temp_filter_size_z == 7u) {
				Nz = gaussian_7[k];
				}
				else if (temp_filter_size_z == 9u) {
				Nz = gaussian_9[k];
				}
				else if (temp_filter_size_z == 11u) {
				Nz = gaussian_11[k];
				}
				else if (temp_filter_size_z == 13u) {
				Nz = gaussian_13[k];
				}
				else if (temp_filter_size_z == 15u) {
				Nz = gaussian_15[k];
				}
				else if (temp_filter_size_z == 17u) {
				Nz = gaussian_17[k];
				}
				else if (temp_filter_size_z == 19u) {
				Nz = gaussian_19[k];
				}
				else {
				Nz = gaussian_21[k];
				}
				*/


				N = Nx * Ny*Nz;


				uchar4 color_info = color_buffer[color_index];

				if (k == 0) {
					color_info = color_buffer[color_index];
				}
				else if (k == 1) {
					color_info = temp_buffer1[color_index];
				}
				else if (k == 2) {
					color_info = temp_buffer2[color_index];
				}
				else {
					color_info = temp_buffer3[color_index];
				}



				float3 color_info_float = make_float3(color_info.z, color_info.y, color_info.x)*make_float3(1.0f / 255.99f);

				color_info_float *= make_float3(N);
				tot_color += color_info_float;
				total_weight += N;
			}
		}
	}//for loop ends
	if (total_weight != 0) {
		tot_color *= make_float3(1.0f / total_weight);
		final_color = make_color(tot_color);
	}
	else {
		//tot_color.x = 255.0f;
		//tot_color.y = tot_color.z = 0.0f;
		//final_color = make_color(tot_color);
	}

	if (launch_index.x == 128u && launch_index.y == 128u) {
		//rtPrintf("In recon: x = %d, y = %d, z = %d t = %f\n", gaussian_filter_size_x, gaussian_filter_size_y, gaussian_filter_size_z, going_to_use_this_extent.z);
		//rtPrintf("which_grid_calculate= %d \n", which_grid_calculate);
	}

	// end gather process
	// draw the tiles & extent

	if (show_gradient_buffer[0] > 0)
	{
		float3 going_to_add_gradient_color = crosshair_buffer[index];

		//going_to_add_gradient_color = 

		float3 final_color_make = make_float3(final_color.z, final_color.y, final_color.x)*make_float3(1.0f / 255.99f);

		final_color_make.x = going_to_add_gradient_color.x;
		final_color_make.y = going_to_add_gradient_color.y;
		final_color_make.z = going_to_add_gradient_color.z;
		float g_x = (float)gaussian_filter_size_x - 1;
		float g_y = (float)gaussian_filter_size_y - 1;
		//if (launch_index.x == 128u && launch_index.y == 128u)
		//	rtPrintf("g_x = %f\n", g_x);
		//final_color_make.x += (g_x * (1.0f / 255.99f)*5) ;
		//final_color_make.y += (g_y * (1.0f / 255.99f)*5);
		//final_color_make.z += 0.0;

		final_color = make_color(final_color_make);
		output_buffer[index] = final_color;
	}
	else
	{
		output_buffer[index] = final_color;
	}


	if (show_tile_buffer[0] > 0) {
		//if (rand_pixel.x == centre_pixel.x - tilesize / 2u || rand_pixel.y == centre_pixel.y - tilesize / 2u) {
		if (index.x == centre_pixel.x - tilesize / 2u || index.x == centre_pixel.x + tilesize / 2u || index.y == centre_pixel.y - tilesize / 2u || index.y == centre_pixel.y + tilesize / 2u) {
			float3 whiteline = make_float3(255.0f, 255.0f, 255.0f);
			output_buffer[index] = make_color(whiteline);
		}
		else {
			output_buffer[index] = final_color;
		}

	}
	else {
		output_buffer[index] = final_color;
	}
}


// implement 3D Gaussian filter using convolution
static __device__ __inline__ void convolution_3D_to_tile_gather_copy()
{
	// record the start time from rendering

	float start_to_reconstruction_time = render_elapse_time_buffer[0];
	clock_t reconstruction_start_time = clock();

	// define the input index as thread index
	uint2 index = make_uint2(launch_index.x, launch_index.y);
	// using the input index to find the tile (tile = grid in this code)
	uint which_grid_calculate = pixel_at_tile_buffer[index];

	// assign a tile if the call is illegal, should not happen after every pixel is updated at least once
	if (which_grid_calculate > 512) {
		which_grid_calculate = launch_index.x;
	}
	// the center pixel of the tile
	uint2 centre_pixel = make_uint2(leaf_tile_indices[which_grid_calculate].x, leaf_tile_indices[which_grid_calculate].y);
	// tile size of the tile
	unsigned int tilesize = leaf_tile_sizes[which_grid_calculate];

	// find the calculated extent(filter size) in this tile
	float3 extents = extent_buffer[index];
	uint3 filter_size = make_uint3((unsigned int)extents.x, (unsigned int)extents.y, (unsigned int)extents.z);


	// reassign the filter size if they are too big

	//float3 going_to_use_this_extent = extent_buffer[index];
	float3 going_to_use_this_extent = extent_at_tile[which_grid_calculate];

	////////////////caution!!!!!!! need to change back if not ok

	//going_to_use_this_extent.x *= 1.5;
	//going_to_use_this_extent.y *= 1.5;
	//going_to_use_this_extent.x -= 2.0;
	//going_to_use_this_extent.y -= 2.0;

	if (launch_index.x == 300u && launch_index.y == 300u) {
		//rtPrintf("which_grid_calculate = %u \n", which_grid_calculate);
		//rtPrintf("filter size x = %f, filter size y = %f, filter size z = %f \n", going_to_use_this_extent.x, going_to_use_this_extent.y, going_to_use_this_extent.z);
	}

	if (going_to_use_this_extent.x < 0)
		going_to_use_this_extent.x = 0.0f;
	if (going_to_use_this_extent.y < 0)
		going_to_use_this_extent.y = 0.0f;

	uint gaussian_filter_size_x = going_to_use_this_extent.x;
	uint gaussian_filter_size_y = going_to_use_this_extent.y;
	uint gaussian_filter_size_z = going_to_use_this_extent.z;


	//gaussian_filter_size_x = 21;
	//gaussian_filter_size_y = 21;

	// reassign the filter size to odd number

	if (gaussian_filter_size_x % 2 == 0u) {
		gaussian_filter_size_x += 1u;
	}

	if (gaussian_filter_size_y % 2 == 0u) {
		gaussian_filter_size_y += 1u;
	}

	if (gaussian_filter_size_z % 2 == 0u) {
		gaussian_filter_size_z += 1u;
	}

	uint temp_filter_size_z = gaussian_filter_size_z;

	// reassign the filter size if they are too big
	if (gaussian_filter_size_x > 33u) {
		gaussian_filter_size_x = 33u;
	}

	if (gaussian_filter_size_y > 33u) {
		gaussian_filter_size_y = 33u;
	}

	//if (gaussian_filter_size_z > 3u) {
	//	gaussian_filter_size_z = 3u;
	//}

	if (temp_filter_size_z > 21u) {
		temp_filter_size_z = 21u;
	}

	if (launch_index.x == 100u && launch_index.y == 100u) {
		//rtPrintf("In recon: x = %d, y = %d, z = %d \n", gaussian_filter_size_x, gaussian_filter_size_y, gaussian_filter_size_z);
		//rtPrintf("which_grid_calculate= %d \n", which_grid_calculate);
	}

	// Test  

	float total_weight = 0.0f;
	uchar4 total_color, final_color;
	float3 tot_color;
	tot_color.x = tot_color.y = tot_color.z = 0.0f;

	if (is_moving_buffer[0] == 0) {
		gaussian_filter_size_x = 1;
		gaussian_filter_size_y = 1;
	}


	// help to find where the start pixel for Gaussian filter
	uint gaussian_start_index_x = launch_index.x - gaussian_filter_size_x / 2 + 1;
	uint gaussian_start_index_y = launch_index.y - gaussian_filter_size_y / 2 + 1;
	uint2 guassian_start_index = make_uint2(gaussian_start_index_x, gaussian_start_index_y);

	// use convolution
	for (unsigned int i = 0u; i < gaussian_filter_size_x; i++)
	{
		for (unsigned int j = 0u; j < gaussian_filter_size_y; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				float Nx, Ny, Nz, N;

				uint color_index_x = guassian_start_index.x + i;
				uint color_index_y = guassian_start_index.y + j;
				uint2 color_index = make_uint2(color_index_x, color_index_y);

				if (color_index_x < 0 || color_index_x > screen_size_buffer[0] || color_index_y < 0 || color_index_y > screen_size_buffer[1]) {
					continue;
				}


				if (gaussian_filter_size_y == 1u) {
					Ny = gaussian_1[j];
				}
				else if (gaussian_filter_size_y == 3u) {
					Ny = gaussian_3[j];
				}
				else if (gaussian_filter_size_y == 5u) {
					Ny = gaussian_5[j];
				}
				else if (gaussian_filter_size_y == 7u) {
					Ny = gaussian_7[j];
				}
				else if (gaussian_filter_size_y == 9u) {
					Ny = gaussian_9[j];
				}
				else if (gaussian_filter_size_y == 11u) {
					Ny = gaussian_11[j];
				}
				else if (gaussian_filter_size_y == 13u) {
					Ny = gaussian_13[j];
				}
				else if (gaussian_filter_size_y == 15u) {
					Ny = gaussian_15[j];
				}
				else if (gaussian_filter_size_y == 17u) {
					Ny = gaussian_17[j];
				}
				else if (gaussian_filter_size_y == 19u) {
					Ny = gaussian_19[j];
				}
				else if (gaussian_filter_size_y == 21u) {
					Ny = gaussian_21[j];
				}
				else if (gaussian_filter_size_y == 23u) {
					Ny = gaussian_23[j];
				}
				else if (gaussian_filter_size_y == 25u) {
					Ny = gaussian_25[j];
				}
				else if (gaussian_filter_size_y == 27u) {
					Ny = gaussian_27[j];
				}
				else if (gaussian_filter_size_y == 29u) {
					Ny = gaussian_29[j];
				}
				else if (gaussian_filter_size_y == 31u) {
					Ny = gaussian_31[j];
				}
				else if (gaussian_filter_size_y == 33u) {
					Ny = gaussian_33[j];
				}
				else {
					Ny = gaussian_33[j];
				}

				if (gaussian_filter_size_x == 1u) {
					Nx = gaussian_1[i];
				}
				else if (gaussian_filter_size_x == 3u) {
					Nx = gaussian_3[i];
				}
				else if (gaussian_filter_size_x == 5u) {
					Nx = gaussian_5[i];
				}
				else if (gaussian_filter_size_x == 7u) {
					Nx = gaussian_7[i];
				}
				else if (gaussian_filter_size_x == 9u) {
					Nx = gaussian_9[i];
				}
				else if (gaussian_filter_size_x == 11u) {
					Nx = gaussian_11[i];
				}
				else if (gaussian_filter_size_x == 13u) {
					Nx = gaussian_13[i];
				}
				else if (gaussian_filter_size_x == 15u) {
					Nx = gaussian_15[i];
				}
				else if (gaussian_filter_size_x == 17u) {
					Nx = gaussian_17[i];
				}
				else if (gaussian_filter_size_x == 19u) {
					Nx = gaussian_19[i];
				}
				else if (gaussian_filter_size_x == 21u) {
					Nx = gaussian_21[i];
				}
				else if (gaussian_filter_size_x == 23u) {
					Nx = gaussian_23[i];
				}
				else if (gaussian_filter_size_x == 25u) {
					Nx = gaussian_25[i];
				}
				else if (gaussian_filter_size_x == 27u) {
					Nx = gaussian_27[i];
				}
				else if (gaussian_filter_size_x == 29u) {
					Nx = gaussian_29[i];
				}
				else if (gaussian_filter_size_x == 31u) {
					Nx = gaussian_31[i];
				}
				else if (gaussian_filter_size_x == 33u) {
					Nx = gaussian_33[i];
				}
				else {
					Nx = gaussian_33[i];
				}

				float sample_age = 0.0f;

				if (k == 0) {
					sample_age = start_to_reconstruction_time - sample_time_buffer[color_index];
				}
				else if (k == 1) {
					sample_age = start_to_reconstruction_time - sample_time_temp_buffer1[color_index];
				}
				else if (k == 2) {
					sample_age = start_to_reconstruction_time - sample_time_temp_buffer2[color_index];
				}
				else if (k == 3) {
					sample_age = start_to_reconstruction_time - sample_time_temp_buffer3[color_index];
				}
				else {
					sample_age = 777.0f;
				}



				float new_temporal_extent = going_to_use_this_extent.z * 2.0;

				//sample_age -= 0.045*k;

				if (sample_age <= 0)
					sample_age = 0;

				//sample_age *= 750;

				//sample_age *= 100;

				uint sample_age_index = (uint)floorf(sample_age);
				uint sample_age_index_ceil = (uint)ceilf(sample_age);

				if (launch_index.x == 50u && launch_index.y == 50u && k == 0) {
					//rtPrintf("sample_age = %f, sample_age_cc = %u sample_age_index = %u\n", sample_age, (uint)ceilf(sample_age), sample_age_index);
					//rtPrintf("extentx = %f, extenty = %f, new temp extentz = %f\n", going_to_use_this_extent.x, going_to_use_this_extent.y, new_temporal_extent);
					//rtPrintf("which_grid_calculate = %d, extent x= %f, extent y = %f, extent z = %f\n", which_grid_calculate, going_to_use_this_extent.x, going_to_use_this_extent.y, going_to_use_this_extent.z);
				}

				// interpolation:
				// A0   : sample_age floor
				// A1   : sample_age ceil
				// A    : current extent (new_temporal_extent)
				// B0   : Gaussian weight for A0
				// B1   : Gaussian weight for A1
				// B(Nz): Expected intepolation weight
				// alpha: (A-A0)/(A1-A0) = A-A0

				float A0, A1, A, B0, B1, alpha;
				A0 = floorf(sample_age);
				A1 = ceilf(sample_age);
				alpha = sample_age - A0;

				//if (launch_index.x == 100u && launch_index.y == 100u && k == 0)
				//	rtPrintf("new_temporal_extent = %f, sample_age = %f\n", new_temporal_extent, sample_age);


				if (sample_age >= 25.0f) {
					Nz = 0.00000;
					//Nz = 0.0000001;
				}
				else if (new_temporal_extent > 49.0f) {
					B0 = gaussianT_49[sample_age_index];
					B1 = gaussianT_49[sample_age_index_ceil];
					Nz = B0 * (1 - alpha) + B1 * alpha;
				}
				else {
					if (new_temporal_extent <= 1u) {
						if (sample_age_index > 0u)
							Nz = gaussianT_1[0];
						else
							Nz = gaussianT_1[sample_age_index];
					}
					else if (new_temporal_extent <= 3u) {

						if (sample_age_index >= 1u)
							Nz = gaussianT_3[1];
						else {
							B0 = gaussianT_3[sample_age_index];
							B1 = gaussianT_3[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 5u) {
						if (sample_age_index >= 2u)
							Nz = gaussianT_5[2];
						else {
							B0 = gaussianT_5[sample_age_index];
							B1 = gaussianT_5[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 7u) {
						if (sample_age_index >= 3u)
							Nz = gaussianT_7[3];
						else {
							B0 = gaussianT_7[sample_age_index];
							B1 = gaussianT_7[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 9u) {
						if (sample_age_index >= 4u)
							Nz = gaussianT_9[4];
						else {
							B0 = gaussianT_9[sample_age_index];
							B1 = gaussianT_9[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 11u) {
						if (sample_age_index >= 5u)
							Nz = gaussianT_11[5];
						else {
							B0 = gaussianT_11[sample_age_index];
							B1 = gaussianT_11[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 13u) {
						if (sample_age_index >= 6u)
							Nz = gaussianT_13[6];
						else {
							B0 = gaussianT_13[sample_age_index];
							B1 = gaussianT_13[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 15u) {
						if (sample_age_index >= 7u)
							Nz = gaussianT_15[7];
						else {
							B0 = gaussianT_15[sample_age_index];
							B1 = gaussianT_15[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 17u) {
						if (sample_age_index >= 8u)
							Nz = gaussianT_17[8];
						else {
							B0 = gaussianT_17[sample_age_index];
							B1 = gaussianT_17[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 19u) {
						if (sample_age_index >= 9u)
							Nz = gaussianT_19[9];
						else {
							B0 = gaussianT_19[sample_age_index];
							B1 = gaussianT_19[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 21u) {
						if (sample_age_index >= 10u)
							Nz = gaussianT_21[10];
						else {
							B0 = gaussianT_21[sample_age_index];
							B1 = gaussianT_21[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 23u) {
						if (sample_age_index >= 11u)
							Nz = gaussianT_23[11];
						else {
							B0 = gaussianT_23[sample_age_index];
							B1 = gaussianT_23[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 25u) {
						if (sample_age_index >= 12u)
							Nz = gaussianT_25[12];
						else {
							B0 = gaussianT_25[sample_age_index];
							B1 = gaussianT_25[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 27u) {
						if (sample_age_index >= 13u)
							Nz = gaussianT_27[13];
						else {
							B0 = gaussianT_27[sample_age_index];
							B1 = gaussianT_27[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 29u) {
						if (sample_age_index >= 14u)
							Nz = gaussianT_29[14];
						else {
							B0 = gaussianT_29[sample_age_index];
							B1 = gaussianT_29[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 31u) {
						if (sample_age_index >= 15u)
							Nz = gaussianT_31[15];
						else {
							B0 = gaussianT_31[sample_age_index];
							B1 = gaussianT_31[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 33u) {
						if (sample_age_index >= 16u)
							Nz = gaussianT_33[16];
						else {
							B0 = gaussianT_33[sample_age_index];
							B1 = gaussianT_33[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 35u) {
						if (sample_age_index >= 17u)
							Nz = gaussianT_35[17];
						else {
							B0 = gaussianT_35[sample_age_index];
							B1 = gaussianT_35[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 37u) {
						if (sample_age_index >= 18u)
							Nz = gaussianT_37[18];
						else {
							B0 = gaussianT_37[sample_age_index];
							B1 = gaussianT_37[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 39u) {
						if (sample_age_index >= 19u)
							Nz = gaussianT_39[19];
						else {
							B0 = gaussianT_39[sample_age_index];
							B1 = gaussianT_39[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 41u) {
						if (sample_age_index >= 20u)
							Nz = gaussianT_41[20];
						else {
							B0 = gaussianT_41[sample_age_index];
							B1 = gaussianT_41[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 43u) {
						if (sample_age_index >= 21u)
							Nz = gaussianT_43[21];
						else {
							B0 = gaussianT_43[sample_age_index];
							B1 = gaussianT_43[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 45u) {
						if (sample_age_index >= 22u)
							Nz = gaussianT_45[22];
						else {
							B0 = gaussianT_45[sample_age_index];
							B1 = gaussianT_45[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else if (new_temporal_extent <= 47u) {
						if (sample_age_index >= 23u)
							Nz = gaussianT_47[23];
						else {
							B0 = gaussianT_47[sample_age_index];
							B1 = gaussianT_47[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
					else {
						if (sample_age_index >= 24u)
							Nz = gaussianT_49[24];
						else {
							B0 = gaussianT_49[sample_age_index];
							B1 = gaussianT_49[sample_age_index_ceil];
							Nz = B0 * (1 - alpha) + B1 * alpha;
						}
					}
				}

				if (Nz == 0.0f) {
					Nz = 0.000001;
				}

				/*
				if (gaussian_filter_size_z == 1u) {
				Nz = gaussian_1[k];
				}
				else if (gaussian_filter_size_z == 3u) {
				Nz = gaussian_3[k];
				}


				if (temp_filter_size_z == 1u) {
				Nz = gaussian_1[k];
				}
				else if (temp_filter_size_z == 3u) {
				Nz = gaussian_3[k];
				}
				else if (temp_filter_size_z == 5u) {
				Nz = gaussian_5[k];
				}
				else if (temp_filter_size_z == 7u) {
				Nz = gaussian_7[k];
				}
				else if (temp_filter_size_z == 9u) {
				Nz = gaussian_9[k];
				}
				else if (temp_filter_size_z == 11u) {
				Nz = gaussian_11[k];
				}
				else if (temp_filter_size_z == 13u) {
				Nz = gaussian_13[k];
				}
				else if (temp_filter_size_z == 15u) {
				Nz = gaussian_15[k];
				}
				else if (temp_filter_size_z == 17u) {
				Nz = gaussian_17[k];
				}
				else if (temp_filter_size_z == 19u) {
				Nz = gaussian_19[k];
				}
				else {
				Nz = gaussian_21[k];
				}
				*/


				N = Nx * Ny*Nz;


				uchar4 color_info = color_buffer[color_index];

				if (k == 0) {
					color_info = color_buffer[color_index];
				}
				else if (k == 1) {
					color_info = temp_buffer1[color_index];
				}
				else if (k == 2) {
					color_info = temp_buffer2[color_index];
				}
				else {
					color_info = temp_buffer3[color_index];
				}



				float3 color_info_float = make_float3(color_info.z, color_info.y, color_info.x)*make_float3(1.0f / 255.99f);

				color_info_float *= make_float3(N);
				tot_color += color_info_float;
				total_weight += N;
			}
		}
	}//for loop ends

	// averaging all the old samples
	if (is_moving_buffer[0] == 0) {
	
		tot_color.x = tot_color.y = tot_color.z = 0.0f;
		uchar4 color_info = color_buffer[launch_index];
		float3 color_info_float = make_float3(color_info.z, color_info.y, color_info.x)*make_float3(1.0f / 255.99f);

		uchar4 color_info1 = color_buffer[launch_index];
		float3 color_info_float1 = make_float3(color_info1.z, color_info1.y, color_info1.x)*make_float3(1.0f / 255.99f);

		uchar4 color_info2 = color_buffer[launch_index];
		float3 color_info_float2 = make_float3(color_info2.z, color_info2.y, color_info2.x)*make_float3(1.0f / 255.99f);

		uchar4 color_info3 = color_buffer[launch_index];
		float3 color_info_float3 = make_float3(color_info3.z, color_info3.y, color_info3.x)*make_float3(1.0f / 255.99f);

		tot_color = color_info_float + color_info_float1 + color_info_float2 + color_info_float3;
		tot_color /= 4;
		total_weight = 1.0f;
	}



	if (total_weight != 0) {
		tot_color *= make_float3(1.0f / total_weight);
		final_color = make_color(tot_color);
	}
	else {
		//tot_color.x = 255.0f;
		//tot_color.y = tot_color.z = 0.0f;
		//final_color = make_color(tot_color);
	}

	if (launch_index.x == 128u && launch_index.y == 128u) {
		//rtPrintf("In recon: x = %d, y = %d, z = %d t = %f\n", gaussian_filter_size_x, gaussian_filter_size_y, gaussian_filter_size_z, going_to_use_this_extent.z);
		//rtPrintf("which_grid_calculate= %d \n", which_grid_calculate);
	}

	// end gather process
	// draw the tiles & extent

	if (show_gradient_buffer[0] > 0)
	{
		float3 going_to_add_gradient_color = crosshair_buffer[index];

		//going_to_add_gradient_color = 

		float3 final_color_make = make_float3(final_color.z, final_color.y, final_color.x)*make_float3(1.0f / 255.99f);

		final_color_make.x = going_to_add_gradient_color.x;
		final_color_make.y = going_to_add_gradient_color.y;
		final_color_make.z = going_to_add_gradient_color.z;
		float g_x = (float)gaussian_filter_size_x - 1;
		float g_y = (float)gaussian_filter_size_y - 1;
		//if (launch_index.x == 128u && launch_index.y == 128u)
		//	rtPrintf("g_x = %f\n", g_x);
		//final_color_make.x += (g_x * (1.0f / 255.99f)*5) ;
		//final_color_make.y += (g_y * (1.0f / 255.99f)*5);
		//final_color_make.z += 0.0;

		final_color = make_color(final_color_make);
		output_buffer[index] = final_color;
	}
	else
	{
		output_buffer[index] = final_color;
	}


	if (show_tile_buffer[0] > 0) {
		//if (rand_pixel.x == centre_pixel.x - tilesize / 2u || rand_pixel.y == centre_pixel.y - tilesize / 2u) {
		if (index.x == centre_pixel.x - tilesize / 2u || index.x == centre_pixel.x + tilesize / 2u || index.y == centre_pixel.y - tilesize / 2u || index.y == centre_pixel.y + tilesize / 2u) {
			float3 whiteline = make_float3(255.0f, 255.0f, 255.0f);
			output_buffer[index] = make_color(whiteline);
		}
		else {
			output_buffer[index] = final_color;
		}

	}
	else {
		output_buffer[index] = final_color;
	}
}



RT_PROGRAM void pinhole_camera()
{
	/*
	size_t2 screen = output_buffer.size();

	float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon );

	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;

	rtTrace(top_object, ray, prd);

	output_buffer[launch_index] = make_color( prd.result );
	*/

	//frameless_rendering();
	//shoot_3rays_use_random_map_tile();
	
	shoot_3rays_new(); 
	//shoot_3rays_use_random_map_tile_test();

}

RT_PROGRAM void extent_calculation() {
	if (launch_index.y <= 0u)
	{
		new_calculate_variance(true);
	}
	else if (launch_index.y <= 1u && launch_index.x <= (unsigned int)number_of_parent_tiles)
	{
		new_calculate_variance(false);
	}
	else {
		calculate_extent();
	}
}

RT_PROGRAM void reconstruct() 
{
	//frameless_rendering();
	//no_reconstruction();
	//gaussian_filter_to_whole_image_gather();
	
	//convolution_3D_to_tile_gather();
	convolution_3D_to_tile_gather_copy();
}

//
// (NEW)
// Environment map background
//
rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
  float theta = atan2f( ray.direction.x, ray.direction.z );
  float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
  float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v     = 0.5f * ( 1.0f + sin(phi) );
  prd_radiance.result = make_float3( tex2D(envmap, u, v) );
}
  

//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);

  rtTerminateRay();
}
  

//
// Phong surface shading with shadows 
//
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(float3,   Ka, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(float,    phong_exp, , );
rtDeclareVariable(float3,   ambient_light_color, , );
rtBuffer<BasicLight>        lights;
rtDeclareVariable(rtObject, top_shadower, , );

RT_PROGRAM void closest_hit_radiance3()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 color = Ka * ambient_light_color;

  float3 hit_point = ray.origin + t_hit * ray.direction;

  for(int i = 0; i < lights.size(); ++i) {
    BasicLight light = lights[i];
    float3 L = normalize(light.pos - hit_point);
    float nDl = dot( ffnormal, L);

    if( nDl > 0.0f ){
      // cast shadow ray
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(1.0f);
      float Ldist = length(light.pos - hit_point);
      optix::Ray shadow_ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_shadower, shadow_ray, shadow_prd);
      float3 light_attenuation = shadow_prd.attenuation;

      if( fmaxf(light_attenuation) > 0.0f ){
        float3 Lc = light.color * light_attenuation;
        color += Kd * nDl * Lc;

        float3 H = normalize(L - ray.direction);
        float nDh = dot( ffnormal, H );
        if(nDh > 0)
          color += Ks * Lc * pow(nDh, phong_exp);
      }

    }
  }
  prd_radiance.result = color;
}


//
// Phong surface shading with shadows and reflections
//
rtDeclareVariable(float3, reflectivity, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(int, max_depth, , );

RT_PROGRAM void floor_closest_hit_radiance4()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 color = Ka * ambient_light_color;

  float3 hit_point = ray.origin + t_hit * ray.direction;

  for(int i = 0; i < lights.size(); ++i) {
    BasicLight light = lights[i];
    float3 L = normalize(light.pos - hit_point);
    float nDl = dot( ffnormal, L);

    if( nDl > 0.0f ){
      // cast shadow ray
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(1.0f);
      float Ldist = length(light.pos - hit_point);
      optix::Ray shadow_ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_shadower, shadow_ray, shadow_prd);
      float3 light_attenuation = shadow_prd.attenuation;

      if( fmaxf(light_attenuation) > 0.0f ){
        float3 Lc = light.color * light_attenuation;
        color += Kd * nDl * Lc;

        float3 H = normalize(L - ray.direction);
        float nDh = dot( ffnormal, H );
        if(nDh > 0)
          color += Ks * Lc * pow(nDh, phong_exp);
      }

    }
  }

  float importance = prd_radiance.importance * optix::luminance( reflectivity );

  // reflection ray
  if( importance > importance_cutoff && prd_radiance.depth < max_depth) {
    PerRayData_radiance refl_prd;
    refl_prd.importance = importance;
    refl_prd.depth = prd_radiance.depth+1;
    float3 R = reflect( ray.direction, ffnormal );
    optix::Ray refl_ray( hit_point, R, radiance_ray_type, scene_epsilon );
    rtTrace(top_object, refl_ray, refl_prd);
    color += reflectivity * refl_prd.result;
  }

  prd_radiance.result = color;
}
 
// adding some chull implementation from other tutorial to add refraction

//
// Bounding box program for programmable convex hull primitive
//
rtDeclareVariable(float3, chull_bbmin, , );
rtDeclareVariable(float3, chull_bbmax, , );

RT_PROGRAM void chull_bounds(int primIdx, float result[6])
{
	optix::Aabb* aabb = (optix::Aabb*)result;
	aabb->m_min = chull_bbmin;
	aabb->m_max = chull_bbmax;
}

//
// Intersection program for programmable convex hull primitive
//
rtBuffer<float4> planes;
RT_PROGRAM void chull_intersect(int primIdx)
{
	int n = planes.size();
	float t0 = -FLT_MAX;
	float t1 = FLT_MAX;
	float3 t0_normal = make_float3(0);
	float3 t1_normal = make_float3(0);
	for (int i = 0; i < n && t0 < t1; ++i) {
		float4 plane = planes[i];
		float3 n = make_float3(plane);
		float  d = plane.w;

		float denom = dot(n, ray.direction);
		float t = -(d + dot(n, ray.origin)) / denom;
		if (denom < 0) {
			// enter
			if (t > t0) {
				t0 = t;
				t0_normal = n;
			}
		}
		else {
			//exit
			if (t < t1) {
				t1 = t;
				t1_normal = n;
			}
		}
	}

	if (t0 > t1)
		return;

	if (rtPotentialIntersection(t0)) {
		shading_normal = geometric_normal = t0_normal;
		rtReportIntersection(0);
	}
	else if (rtPotentialIntersection(t1)) {
		shading_normal = geometric_normal = t1_normal;
		rtReportIntersection(0);
	}
}

//
// Attenuates shadow rays for shadowing transparent objects
//
rtDeclareVariable(float3, shadow_attenuation, , );

RT_PROGRAM void glass_any_hit_shadow()
{
	float3 world_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float nDi = fabs(dot(world_normal, ray.direction));

	prd_shadow.attenuation *= 1 - fresnel_schlick(nDi, 5, 1 - shadow_attenuation, make_float3(1));

	rtIgnoreIntersection();
}


//
// Dielectric surface shader
//
rtDeclareVariable(float3, cutoff_color, , );
rtDeclareVariable(float, fresnel_exponent, , );
rtDeclareVariable(float, fresnel_minimum, , );
rtDeclareVariable(float, fresnel_maximum, , );
rtDeclareVariable(float, refraction_index, , );
rtDeclareVariable(int, refraction_maxdepth, , );
rtDeclareVariable(int, reflection_maxdepth, , );
rtDeclareVariable(float3, refraction_color, , );
rtDeclareVariable(float3, reflection_color, , );
rtDeclareVariable(float3, extinction_constant, , );
RT_PROGRAM void glass_closest_hit_radiance()
{
	// intersection vectors
	const float3 h = ray.origin + t_hit * ray.direction;            // hitpoint
	const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
	const float3 i = ray.direction;                                            // incident direction

	float reflection = 1.0f;
	float3 result = make_float3(0.0f);

	float3 beer_attenuation;
	if (dot(n, ray.direction) > 0) {
		// Beer's law attenuation
		beer_attenuation = exp(extinction_constant * t_hit);
	}
	else {
		beer_attenuation = make_float3(1);
	}

	// refraction
	if (prd_radiance.depth < min(refraction_maxdepth, max_depth))
	{
		float3 t;                                                            // transmission direction
		if (refract(t, i, n, refraction_index))
		{

			// check for external or internal reflection
			float cos_theta = dot(i, n);
			if (cos_theta < 0.0f)
				cos_theta = -cos_theta;
			else
				cos_theta = dot(t, n);

			reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);

			float importance = prd_radiance.importance * (1.0f - reflection) * optix::luminance(refraction_color * beer_attenuation);
			if (importance > importance_cutoff) {
				optix::Ray ray(h, t, radiance_ray_type, scene_epsilon);
				PerRayData_radiance refr_prd;
				refr_prd.depth = prd_radiance.depth + 1;
				refr_prd.importance = importance;

				rtTrace(top_object, ray, refr_prd);
				result += (1.0f - reflection) * refraction_color * refr_prd.result;
			}
			else {
				result += (1.0f - reflection) * refraction_color * cutoff_color;
			}
		}
		// else TIR
	}

	// reflection
	if (prd_radiance.depth < min(reflection_maxdepth, max_depth))
	{
		float3 r = reflect(i, n);

		float importance = prd_radiance.importance * reflection * optix::luminance(reflection_color * beer_attenuation);
		if (importance > importance_cutoff) {
			optix::Ray ray(h, r, radiance_ray_type, scene_epsilon);
			PerRayData_radiance refl_prd;
			refl_prd.depth = prd_radiance.depth + 1;
			refl_prd.importance = importance;

			rtTrace(top_object, ray, refl_prd);
			result += reflection * reflection_color * refl_prd.result;
		}
		else {
			result += reflection * reflection_color * cutoff_color;
		}
	}

	result = result * beer_attenuation;

	prd_radiance.result = result;
}

//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
