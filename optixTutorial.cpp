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

//-----------------------------------------------------------------------------
//
//  tutorial
//
//-----------------------------------------------------------------------------

// 0 - normal shader
// 1 - lambertian
// 2 - specular
// 3 - shadows
// 4 - reflections
// 5 - miss
// 6 - schlick
// 7 - procedural texture on floor
// 8 - LGRustyMetal
// 9 - intersection
// 10 - anyhit
// 11 - camera



#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#  include <GL/wglew.h>
#  include <GL/freeglut.h>
#  else
#  include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "commonStructs.h"
#include "random.h"
#include <Arcball.h>
#include <OptiXMesh.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <stdint.h>
#include "Tile.h"
#include <queue>
#include <sys/timeb.h>

//#include "tobii.h"
//#include "tobii_advanced.h"
//#include "tobii_config.h"
//#include "tobii_licensing.h"
//#include "tobii_streams.h"
//#include "tobii_wearable.h"

using namespace optix;

const char* const SAMPLE_NAME = "optixTutorial";

static float rand_range(float min, float max)
{
    static unsigned int seed = 0u;
    return min + (max - min) * rnd(seed);
}


//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context      context;
uint32_t     width  = 512u;
uint32_t     height = 512;
bool         use_pbo = true;

std::string  texture_path;
const char*  tutorial_ptx;
int          tutorial_number = 5;

// Camera state
float3       camera_up;
float3       camera_lookat;
float3       camera_eye;
Matrix4x4    camera_rotate;
sutil::Arcball arcball;

// Mouse state
int2       mouse_prev_pos;
int        mouse_button;

// Experiments

bool singleBuffering = true;
bool add_more_chull = true;
bool irregularTile = true;
int InitLeafTileSize = 50;
int  chullNum = 1;

bool switch_variance_visual = true;
bool switch_gradient_visual = true;
bool switch_movement_visual = true;
bool switch_tile_visual = true;
bool add_lots_of_chull = true;
Geometry moving_box;
float boxmin = 330.0f;
float boxmax = 480.0f;
bool moving_box_one = true;
bool moving_box_two = false;

bool add_a_moving_box = true;
bool updateRandomMap = true;

bool to_do_sampling = true;
bool sampling_after_20s = true;
bool avg_for_5s = true;
bool camera_move_right = true, camera_move_left = false;;
bool camera_in_motion = false;

bool camera_in_motion1 = false;
bool camera_in_motion2 = false;
bool camera_in_motion3 = false;
bool mouse_clicking = false;
bool automatic_camera = false;
bool automatic_camera_moving = false;

float camera_moving_speed = 17.0f;
float keyboard_speed = 2.0f;

float avg_sample_time, avg_reconstruction_time;
float aggre_sample_time = 0.0f, aggre_reconstruction_time = 0.0f;
float avg_how_many = 0.0f;
float4 current_quaternion;
uint naive_adding_threads = 256u;
uint prev_sec_frame_generation = 0u;

// camera recording implementation

float3 automatic_moving_camera[6000];
float3 automatic_moving_lookat[6000];
float4 quaternion_moving_camera[6000];
float3 ideal_moving_camera[6000];
float3 ideal_moving_lookat[6000];
float  time_buffer_for_camera[6000];
float  max_time_for_recording = 0.0f;
void setup_automatic_camera();
void clean_automatic_camera();
void generate_ideal_camera();
void quaternion_implementation_for_camera_rotation();
float3 old_camera_eye, old_camera_lookat;
int recording_camera_position_time = 0;
int playing_camera_position_time = 0;
bool record_camera_position = false;
bool play_camera_position = false;
bool playing_record_moving = false;
std::ifstream fp_in;
std::ofstream fp_out;


int experiment_num = 11;
bool light_experiment = false;

//bool scene_sponza = true;
bool scene_breakfast = false;
bool scene_fireplace = false;
// add randomness
void genRndSeeds(unsigned int width, unsigned int height);
Buffer        m_rnd_seeds;

// add some buffers
Buffer        busy_buffer;
Buffer        gradient_buffer;
Buffer        color_buffer;
Buffer        inter_color_buffer;
Buffer        stencil_buffer;
Buffer        tiled_buffer;
Buffer        leaf_tile_indices;
Buffer        leaf_tile_sizes;
Buffer        deep_buffer;
Buffer        variance_buffer;
Buffer        parent_variance_buffer;
Buffer        raycounting_buffer, rayrejecting_buffer;
Buffer        show_variance_buffer, show_gradient_buffer, show_tile_buffer;
Buffer        temp_buffer1, temp_buffer2, temp_buffer3, float_temp_buffer1, float_temp_buffer2, float_temp_buffer3, float_color_buffer;
Buffer        crosshair_buffer, crosshair_buffer1, crosshair_buffer2, crosshair_buffer3;
Buffer        sample_time_buffer, sample_time_temp_buffer1, sample_time_temp_buffer2, sample_time_temp_buffer3;
Buffer        tile_gradient_buffer;
Buffer        extent_buffer, extent_buffer1, extent_buffer2, extent_buffer3;
Buffer        extent_at_tile;
Buffer        pixel_at_tile_buffer;
Buffer        ray_per_sec_buffer;
Buffer        screen_size_buffer;

Buffer        scatter_weight_sample_sum;
Buffer        scatter_sum_of_weight;

Buffer        mini_tile_busy_buffer;
Buffer        random_map_buffer;
Buffer        mini_tile_buffer;

Buffer        render_elapse_time_buffer;
Buffer        mid_scatter_sum_of_weight_buffer, mid_scatter_weight_sample_sum_buffer;
Buffer        scatter_time_buffer;
Buffer        is_moving_buffer;
// adding some testing buffer

Buffer        calculating_filter_used_buffer;



// add tile implementation
Tile root_tile;
std::vector<Tile*> leaf_tiles;
std::vector<Tile*> parent_tiles;
int num_of_tile = 256;
// retiling 
void retiling(int);
void splitTiles();

// Try to make box moving?
Geometry box;
float moving_speed = 0.05f;
float box_y_min = 0.0, box_y_max = 2.0, moving_range = 0.0f;
bool moveUpFlag = true;

clock_t global_time_01s = clock();
clock_t global_time_1s = clock() / CLOCKS_PER_SEC;
clock_t global_time_00001s = clock() / CLOCKS_PER_SEC;
clock_t global_time_5s = clock() / CLOCKS_PER_SEC;
clock_t current_time = clock() / CLOCKS_PER_SEC;
clock_t current_time_test = clock();
clock_t begin_rendering_time;
clock_t sample_time_start, sample_time_end, reconstruction_time_start, reconstruction_time_end;
clock_t trigger_moving_time = clock();
clock_t start_automatic_camera = clock();
clock_t start_recording_camera = clock();
clock_t start_playing_camera = clock();
clock_t prev_frame_time = clock();
unsigned int previous_raycount = 0;
unsigned int previous_rayreject = 0;
int ray_difference = 0;
int ray_reject_difference = 0;
int randomMap[64];
int randomMapSize = 64;

MinHeap merge_tile_heap(num_of_tile);
MaxHeap split_tile_heap(num_of_tile);


float4 make_plane(float3 n, float3 p)
{
	n = normalize(n);
	float d = -dot(n, p);
	return make_float4(n, d);
}

// slerp code from :https://www.lix.polytechnique.fr/~nielsen/WEBvisualcomputing/programs/slerp.cpp
class Point3D
{
public:
	float x, y, z;
};

class Quaternion {
public:
	float w;
	Point3D u;

	inline void Multiply(const Quaternion q)
	{
		Quaternion tmp;
		tmp.u.x = ((w * q.u.x) + (u.x * q.w) + (u.y * q.u.z) - (u.z * q.u.y));
		tmp.u.y = ((w * q.u.y) - (u.x * q.u.z) + (u.y * q.w) + (u.z * q.u.x));
		tmp.u.z = ((w * q.u.z) + (u.x * q.u.y) - (u.y * q.u.x) + (u.z * q.w));
		tmp.w = ((w * q.w) - (u.x * q.u.x) - (u.y * q.u.y) - (u.z * q.u.z));
		*this = tmp;
	}

	inline float Norm()
	{
		return sqrt(u.x*u.x + u.y*u.y + u.z*u.z + w * w);
	}


	inline void Normalize()
	{
		float norm = Norm();
		u.x /= norm; u.y /= norm; u.z /= norm;
	}

	inline void Conjugate()
	{
		u.x = -u.x;
		u.y = -u.y;
		u.z = -u.z;
	}

	inline void Inverse()
	{
		float norm = Norm();
		Conjugate();
		u.x /= norm;
		u.y /= norm;
		u.z /= norm;
		w /= norm;
	}

	void ExportToMatrix(float matrix[16])
	{
		float wx, wy, wz, xx, yy, yz, xy, xz, zz;
		// adapted from Shoemake
		xx = u.x * u.x;
		xy = u.x * u.y;
		xz = u.x * u.z;
		yy = u.y * u.y;
		zz = u.z * u.z;
		yz = u.y * u.z;

		wx = w * u.x;
		wy = w * u.y;
		wz = w * u.z;

		matrix[0] = 1.0f - 2.0f*(yy + zz);
		matrix[4] = 2.0f*(xy - wz);
		matrix[8] = 2.0f*(xz + wy);
		matrix[12] = 0.0;

		matrix[1] = 2.0f*(xy + wz);
		matrix[5] = 1.0f - 2.0f*(xx + zz);
		matrix[9] = 2.0f*(yz - wx);
		matrix[13] = 0.0;

		matrix[2] = 2.0f*(xz - wy);
		matrix[6] = 2.0f*(yz + wx);
		matrix[10] = 1.0f - 2.0f*(xx + yy);
		matrix[14] = 0.0;

		matrix[3] = 0;
		matrix[7] = 0;
		matrix[11] = 0;
		matrix[15] = 1;
	}

};
void Slerp(Quaternion q1, Quaternion q2, Quaternion &qr, float lambda)
{
	float dotproduct = q1.u.x * q2.u.x + q1.u.y * q2.u.y + q1.u.z * q2.u.z + q1.w * q2.w;
	float theta, st, sut, sout, coeff1, coeff2;

	// algorithm adapted from Shoemake's paper
	lambda = lambda / 2.0;

	theta = (float)acos(dotproduct);
	if (theta < 0.0) theta = -theta;

	st = (float)sin(theta);
	sut = (float)sin(lambda*theta);
	sout = (float)sin((1 - lambda)*theta);
	coeff1 = sout / st;
	coeff2 = sut / st;

	qr.u.x = coeff1 * q1.u.x + coeff2 * q2.u.x;
	qr.u.y = coeff1 * q1.u.y + coeff2 * q2.u.y;
	qr.u.z = coeff1 * q1.u.z + coeff2 * q2.u.z;
	qr.w = coeff1 * q1.w + coeff2 * q2.w;

	qr.Normalize();
}




// loading the mesh file

void loadMesh(const std::string& filename);
optix::Aabb    aabb;


void loadMesh(const std::string& filename)
{
	OptiXMesh mesh;
	mesh.context = context;
	loadMesh(filename, mesh);

	aabb.set(mesh.bbox_min, mesh.bbox_max);
	//mesh.closest_hit = 
	//mesh.closest_hit
	/*
	// Create chull
	Geometry chull = 0;
	if (add_lots_of_chull) {
		chull = context->createGeometry();
		chull->setPrimitiveCount(1u);
		chull->setBoundingBoxProgram(context->createProgramFromPTXString(tutorial_ptx, "chull_bounds"));
		chull->setIntersectionProgram(context->createProgramFromPTXString(tutorial_ptx, "chull_intersect"));
		Buffer plane_buffer = context->createBuffer(RT_BUFFER_INPUT);
		plane_buffer->setFormat(RT_FORMAT_FLOAT4);
		int nsides = 6;
		plane_buffer->setSize(nsides + 2);
		float4* chplane = (float4*)plane_buffer->map();
		float radius = 1;
		float3 xlate = make_float3(-1.4f, 0, -3.7f);

		for (int i = 0; i < nsides; i++) {
			float angle = float(i) / float(nsides) * M_PIf * 2.0f;
			float x = cos(angle);
			float y = sin(angle);
			chplane[i] = make_plane(make_float3(x, 0, y), make_float3(x*radius, 0, y*radius) + xlate);
		}
		float min = 0.02f;
		float max = 3.5f;
		chplane[nsides + 0] = make_plane(make_float3(0, -1, 0), make_float3(0, min, 0) + xlate);
		float angle = 5.f / nsides * M_PIf * 2;
		chplane[nsides + 1] = make_plane(make_float3(cos(angle), .7f, sin(angle)), make_float3(0, max, 0) + xlate);
		plane_buffer->unmap();
		chull["planes"]->setBuffer(plane_buffer);
		chull["chull_bbmin"]->setFloat(-radius + xlate.x, min + xlate.y, -radius + xlate.z);
		chull["chull_bbmax"]->setFloat(radius + xlate.x, max + xlate.y, radius + xlate.z);
	}

	// Glass material
	Material glass_matl;
	if (chull.get()) {
		Program glass_ch = context->createProgramFromPTXString(tutorial_ptx, "glass_closest_hit_radiance");
		const std::string glass_ahname =  "glass_any_hit_shadow";
		Program glass_ah = context->createProgramFromPTXString(tutorial_ptx, glass_ahname);
		glass_matl = context->createMaterial();
		glass_matl->setClosestHitProgram(0, glass_ch);
		glass_matl->setAnyHitProgram(1, glass_ah);

		glass_matl["importance_cutoff"]->setFloat(1e-2f);
		glass_matl["cutoff_color"]->setFloat(0.34f, 0.55f, 0.85f);
		glass_matl["fresnel_exponent"]->setFloat(3.0f);
		glass_matl["fresnel_minimum"]->setFloat(0.1f);
		glass_matl["fresnel_maximum"]->setFloat(1.0f);
		glass_matl["refraction_index"]->setFloat(1.4f);
		glass_matl["refraction_color"]->setFloat(1.0f, 1.0f, 1.0f);
		glass_matl["reflection_color"]->setFloat(1.0f, 1.0f, 1.0f);
		glass_matl["refraction_maxdepth"]->setInt(100);
		glass_matl["reflection_maxdepth"]->setInt(100);
		float3 extinction = make_float3(.80f, .89f, .75f);
		glass_matl["extinction_constant"]->setFloat(log(extinction.x), log(extinction.y), log(extinction.z));
		glass_matl["shadow_attenuation"]->setFloat(0.4f, 0.7f, 0.4f);
	}

	// Create a lot of chull for performance testing

	Geometry chull_new[25];
	if (add_lots_of_chull) {
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				if (true) {
					chull_new[5 * i + j] = context->createGeometry();
					chull_new[5 * i + j]->setPrimitiveCount(1u);
					chull_new[5 * i + j]->setBoundingBoxProgram(context->createProgramFromPTXString(tutorial_ptx, "chull_bounds"));
					chull_new[5 * i + j]->setIntersectionProgram(context->createProgramFromPTXString(tutorial_ptx, "chull_intersect"));
					Buffer plane_buffer = context->createBuffer(RT_BUFFER_INPUT);
					plane_buffer->setFormat(RT_FORMAT_FLOAT4);
					int nsides = 6;
					plane_buffer->setSize(nsides + 2);
					float4* chplane = (float4*)plane_buffer->map();
					float radius = 1;
					float3 xlate = make_float3(-1.4f + 9 * i, 0, -3.7f + 9 * j);

					for (int i = 0; i < nsides; i++) {
						float angle = float(i) / float(nsides) * M_PIf * 2.0f;
						float x = cos(angle);
						float y = sin(angle);
						chplane[i] = make_plane(make_float3(x, 0, y), make_float3(x*radius, 0, y*radius) + xlate);
					}
					float min = 0.02f;
					float max = 200.5f;
					chplane[nsides + 0] = make_plane(make_float3(0, -1, 0), make_float3(0, min, 0) + xlate);
					float angle = 5.f / nsides * M_PIf * 2;
					chplane[nsides + 1] = make_plane(make_float3(cos(angle), .7f, sin(angle)), make_float3(0, max, 0) + xlate);
					plane_buffer->unmap();
					chull_new[5 * i + j]["planes"]->setBuffer(plane_buffer);
					chull_new[5 * i + j]["chull_bbmin"]->setFloat(-radius + xlate.x, min + xlate.y, -radius + xlate.z);
					chull_new[5 * i + j]["chull_bbmax"]->setFloat(radius + xlate.x, max + xlate.y, radius + xlate.z);
				}
			}
		}
	}



	// Create GIs for each piece of geometry
	std::vector<GeometryInstance> gis;
	if (add_lots_of_chull) {
	if (chull.get())
	gis.push_back(context->createGeometryInstance(chull, &glass_matl, &glass_matl + 1));

	for (int i = 0; i < 25; i++) {
	if (chull_new[i].get())
	gis.push_back(context->createGeometryInstance(chull_new[i], &glass_matl, &glass_matl + 1));
	}
	}
	*/
    /*
	Material moving_box_matl = context->createMaterial();
	moving_box = context->createGeometry();
	
		const char *ptx = sutil::getPtxString(SAMPLE_NAME, "box.cu");
		Program box_bounds = context->createProgramFromPTXString(ptx, "box_bounds");
		Program box_intersect = context->createProgramFromPTXString(ptx, "box_intersect");

		// Create box
		
		moving_box->setPrimitiveCount(1u);
		moving_box->setBoundingBoxProgram(box_bounds);
		moving_box->setIntersectionProgram(box_intersect);
		moving_box["boxmin"]->setFloat(-50.0f, 330.0f, -50.0f);
		moving_box["boxmax"]->setFloat(50.0f, 480.0f, 50.0f);

		// Materials
		std::string box_chname;
		if (tutorial_number >= 8) {
			box_chname = "box_closest_hit_radiance";
		}
		else if (tutorial_number >= 3) {
			box_chname = "closest_hit_radiance3";
		}
		else if (tutorial_number >= 2) {
			box_chname = "closest_hit_radiance2";
		}
		else if (tutorial_number >= 1) {
			box_chname = "closest_hit_radiance1";
		}
		else {
			box_chname = "closest_hit_radiance0";
		}

		
		Program box_ch = context->createProgramFromPTXString(tutorial_ptx, box_chname.c_str());
		moving_box_matl->setClosestHitProgram(0, box_ch);
		if (tutorial_number >= 3) {
			Program box_ah = context->createProgramFromPTXString(tutorial_ptx, "any_hit_shadow");
			moving_box_matl->setAnyHitProgram(1, box_ah);
		}
		moving_box_matl["Ka"]->setFloat(0.3f, 0.3f, 0.3f);
		moving_box_matl["Kd"]->setFloat(0.6f, 0.7f, 0.8f);
		moving_box_matl["Ks"]->setFloat(0.8f, 0.9f, 0.8f);
		moving_box_matl["phong_exp"]->setFloat(88);
		moving_box_matl["reflectivity_n"]->setFloat(0.2f, 0.2f, 0.2f);
	
	*/
	std::vector<GeometryInstance> gis;
	//gis.push_back(context->createGeometryInstance(moving_box, &moving_box_matl, &moving_box_matl + 1));
	

	GeometryGroup geometry_group = context->createGeometryGroup();
	geometry_group->addChild(mesh.geom_instance);
	//geometry_group->addChild(gis[0]);
	
	/*
	if (add_lots_of_chull) {
		geometry_group->addChild(gis[0]);

		for (int i = 0; i < 25; i++) {
			if (chull_new[i].get())
				geometry_group->addChild(gis[i + 1]);
		}
	}
	*/

	geometry_group->setAcceleration(context->createAcceleration("Bvh"));
	context["top_object"]->set(geometry_group);
	context["top_shadower"]->set(geometry_group);
	begin_rendering_time = clock();
}


void genRndSeeds(unsigned int width, unsigned int height)
{
	unsigned int* seeds = static_cast<unsigned int*>(m_rnd_seeds->map());
	fillRandBuffer(seeds, width*height);
	m_rnd_seeds->unmap();
}

void splitTiles() {
	// testing when randomly split
	/*int do_split = rand() % 100;
	if (do_split > 5) {
	return;
	}
	else {
	int where_to_split = rand() % leaf_tiles.size();
	leaf_tiles.at(where_to_split)->splitTile();
	leaf_tiles.clear();
	leaf_tiles = root_tile.get_leaf_tiles(leaf_tiles);
	printf("After a split, leaf tiles = %u \n", leaf_tiles.size());
	}*/
	int do_split = rand() % 100;
	if (do_split > 5) {
		return;
	}
	//Perform all the mappings
	void* output_data = context["output_buffer"]->getBuffer()->map();
	optix::uchar4* op_data = (optix::uchar4*) output_data;

	void* col_data = context["color_buffer"]->getBuffer()->map();
	optix::uchar4* color_data = (optix::uchar4*)col_data;

	void* variance_data_void = context["variance_buffer"]->getBuffer()->map();
	float* variance_data = (float*)variance_data_void;

	void* parent_variance = context["parent_variance_buffer"]->getBuffer()->map();
	float* parent_variance_data = (float*)parent_variance;

	Tile* max_error_tile;
	max_error_tile = leaf_tiles.front();
	float max_error = 0.0f;

	for (int i = 0; i < leaf_tiles.size(); i++)
	{
		//leaf_tiles.at(i)->calculate_variance(color_data,op_data);
		leaf_tiles.at(i)->tile_variance = variance_data[i];
		if (leaf_tiles.at(i)->tile_variance * leaf_tiles.at(i)->tile_size >= max_error)
		{
			max_error = leaf_tiles.at(i)->tile_size*leaf_tiles.at(i)->tile_variance;
			max_error_tile = leaf_tiles.at(i);
		}
	}

	max_error_tile->splitTile(); //Split leaf tile with max error

								 //Add the newly created leaf tiles to the vector
	leaf_tiles.push_back(max_error_tile->north_east);
	leaf_tiles.push_back(max_error_tile->north_west);
	leaf_tiles.push_back(max_error_tile->south_east);
	leaf_tiles.push_back(max_error_tile->south_west);

	//Add the max_error_tile as a parent tile
	parent_tiles.push_back(max_error_tile);
	max_error_tile->number_of_descendants = 4;

	//Remove the max_error_tile from leaf tiles as it's no longer a leaf tile 
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), max_error_tile), leaf_tiles.end());

	Tile* grandparent = max_error_tile->parent;
	parent_tiles.erase(std::remove(parent_tiles.begin(), parent_tiles.end(), grandparent), parent_tiles.end());


	context["parent_variance_buffer"]->getBuffer()->unmap();
	context["variance_buffer"]->getBuffer()->unmap();
	context["output_buffer"]->getBuffer()->unmap();
	context["color_buffer"]->getBuffer()->unmap();

	//Place leaf and parent data into appropriate buffers
	//
	//

	//Fill the leaf tiles buffer
	void* leaf_data = context["leaf_tile_indices"]->getBuffer()->map();
	optix::uint2* leaf_indices = (optix::uint2*)leaf_data;

	void* leaf_tile_size_data = context["leaf_tile_sizes"]->getBuffer()->map();
	unsigned int* leaf_sizes = (unsigned int*)leaf_tile_size_data;

	for (int i = 0; i < leaf_tiles.size(); i++)
	{
		leaf_indices[i] = leaf_tiles.at(i)->centre_index;
		leaf_sizes[i] = leaf_tiles.at(i)->tile_size;

	}
	context["leaf_tile_sizes"]->getBuffer()->unmap();
	context["leaf_tile_indices"]->getBuffer()->unmap();

	//Map the parent buffers 
	void* parent_data = context["parent_tile_indices"]->getBuffer()->map();
	optix::uint2* parent_indices = (optix::uint2*) parent_data;

	void* parent_size_data = context["parent_tile_sizes"]->getBuffer()->map();
	unsigned int* parent_sizes = (unsigned int*)parent_size_data;

	for (int i = 0; i < parent_tiles.size(); i++)
	{
		parent_indices[i] = parent_tiles.at(i)->centre_index;
		parent_sizes[i] = parent_tiles.at(i)->tile_size;
	}

	context["parent_tile_sizes"]->getBuffer()->unmap();
	context["parent_tile_indices"]->getBuffer()->unmap();
}

void retiling(int num_of_retiling)
{
	//Perform all the mappings
	void* output_data = context["output_buffer"]->getBuffer()->map();
	optix::uchar4* op_data = (optix::uchar4*) output_data;

	void* col_data = context["color_buffer"]->getBuffer()->map();
	optix::uchar4* color_data = (optix::uchar4*)col_data;

	void* variance_data_void = context["variance_buffer"]->getBuffer()->map();
	float* variance_data = (float*)variance_data_void;

	void* parent_variance = context["parent_variance_buffer"]->getBuffer()->map();
	float* parent_variance_data = (float*)parent_variance;

	for (int i = 0; i < num_of_retiling; i++)
	{
		//Merge
		//
		//
		Tile* min_error_tile = &root_tile;
		float min_error = 99999.99f;

		for (int i = 0; i < parent_tiles.size(); i++)
		{
			parent_tiles.at(i)->tile_variance = parent_variance_data[i];
			//printf("variance data check = %f\n", parent_tiles.at(i)->tile_variance);
			if (//parent_tiles.at(i)->tile_size <= 2500u && Make sure tiles don't get too big 
				parent_tiles.at(i)->tile_variance* parent_tiles.at(i)->tile_size* parent_tiles.at(i)->tile_size < min_error)
			{
				min_error = parent_tiles.at(i)->tile_variance* parent_tiles.at(i)->tile_size* parent_tiles.at(i)->tile_size;
				min_error_tile = parent_tiles.at(i);
			}

		}
		/*
		// release all the value in the heap, maybe use malloc next time?
		for (int i = 0; i < merge_tile_heap.heap_size; i++) {
			merge_tile_heap.deleteKey(0);
		}

		MinHeap A_new_heap_for_merge(256);

		for (int i = 0; i<parent_tiles.size(); i++)
		{
			A_new_heap_for_merge.insertKey(i, parent_variance_data[i] * parent_tiles.at(i)->tile_size);
		}

		min_error_tile = parent_tiles.at(A_new_heap_for_merge.extractMin());
		*/
		//Remove the children of the min_error_tile from leaf tiles, and place min_error_til in leaf_tiles
		leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->north_east), leaf_tiles.end());
		leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->north_west), leaf_tiles.end());
		leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->south_east), leaf_tiles.end());
		leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->south_west), leaf_tiles.end());

		//Now, make the min_error_tile a leaf tile by calling mergeTile
		min_error_tile->mergeTile(); //Merge the parent tile with least error

									 //Now add min_error_tile to leaf_tiles, as it now has no children
		leaf_tiles.push_back(min_error_tile);

		//Calculate variance of min_error_tile as it is a newly added leaf tile
		//min_error_tile->calculate_variance(color_data, op_data);

		//Remove min_error_tile from the parent tiles too
		parent_tiles.erase(std::remove(parent_tiles.begin(), parent_tiles.end(), min_error_tile), parent_tiles.end());

		//Add grandpa tile to parent tile if it has 4 children exactly.

		//printf("children check! have children:%d\n", min_error_tile->number_of_descendants);

		//if (min_error_tile->number_of_descendants > 4) {
		//	printf("check if leaf_tile size change: %d\n", leaf_tiles.size());
		//}
		if (min_error_tile->parent != NULL) {
			Tile* grandpa = min_error_tile->parent;
			if (grandpa->north_east->is_leaf_tile &&
				grandpa->north_west->is_leaf_tile &&
				grandpa->south_east->is_leaf_tile &&
				grandpa->south_west->is_leaf_tile)
			{
				parent_tiles.push_back(grandpa);
			}
		}
		else {
			printf("not reach here (tile)!!!\n");
		}


		//Split
		//
		//
		Tile* max_error_tile;
		max_error_tile = leaf_tiles.front();
		float max_error = 0.0f;


		for (int i = 0; i < leaf_tiles.size(); i++)
		{
			//leaf_tiles.at(i)->calculate_variance(color_data,op_data);
			leaf_tiles.at(i)->tile_variance = variance_data[i];

			if (leaf_tiles.at(i)->tile_size >= 4u && //Make sure tiles don't get too small
				leaf_tiles.at(i)->tile_variance * leaf_tiles.at(i)->tile_size * leaf_tiles.at(i)->tile_size >= max_error)
			{
				max_error = leaf_tiles.at(i)->tile_variance * leaf_tiles.at(i)->tile_size * leaf_tiles.at(i)->tile_size;
				max_error_tile = leaf_tiles.at(i);
			}
		}

		/*
		MaxHeap A_new_heap_for_split(256);

		for (int i = 0; i<leaf_tiles.size(); i++)
		{
			if(leaf_tiles.at(i)->tile_size >= 16u)
				A_new_heap_for_split.insertKey(i, variance_data[i] * leaf_tiles.at(i)->tile_size);
		}

		max_error_tile = leaf_tiles.at(A_new_heap_for_split.extractMax());
		*/
		/*
		// release all the value in the heap, maybe use malloc next time?
		for (int i = 0; i < split_tile_heap.heap_size; i++) {
			split_tile_heap.deleteKey(0);
		}

		for (int i = 0; i<leaf_tiles.size(); i++)
		{
			split_tile_heap.insertKey(i, variance_data[i] * leaf_tiles.at(i)->tile_size);
		}

		max_error_tile = parent_tiles.at(split_tile_heap.extractMax());
		*/

		max_error_tile->splitTile(); //Split leaf tile with max error

									 //Add the newly created leaf tiles to the vector
		leaf_tiles.push_back(max_error_tile->north_east);
		leaf_tiles.push_back(max_error_tile->north_west);
		leaf_tiles.push_back(max_error_tile->south_east);
		leaf_tiles.push_back(max_error_tile->south_west);

		//Add the max_error_tile as a parent tile
		parent_tiles.push_back(max_error_tile);
		max_error_tile->number_of_descendants = 4;

		//Remove the max_error_tile from leaf tiles as it's no longer a leaf tile 
		leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), max_error_tile), leaf_tiles.end());

		Tile* grandparent = max_error_tile->parent;
		parent_tiles.erase(std::remove(parent_tiles.begin(), parent_tiles.end(), grandparent), parent_tiles.end());

	}//for loop ends 

	//Perform all the unmappings

	context["parent_variance_buffer"]->getBuffer()->unmap();
	context["variance_buffer"]->getBuffer()->unmap();
	context["output_buffer"]->getBuffer()->unmap();
	context["color_buffer"]->getBuffer()->unmap();
	/**/
	unsigned int* pixel_at_tile_data = (unsigned int*)context["pixel_at_tile_buffer"]->getBuffer()->map();
	// fill the position of tiles in plane

	for (int i = 0; i < leaf_tiles.size(); i++)
	{
		uint2 center_pixel = leaf_tiles.at(i)->centre_index;
		unsigned int tilesize = leaf_tiles.at(i)->tile_size;
		unsigned int min_x = center_pixel.x - (tilesize / 2);
		unsigned int min_y = center_pixel.y - (tilesize / 2);
		for (int a = min_x; a < (min_x + tilesize); ++a)
		{
			for (int b = min_y; b < (min_y + tilesize); ++b)
			{
				pixel_at_tile_data[b*width + a] = i;
			}
		}
	}
	context["pixel_at_tile_buffer"]->getBuffer()->unmap();

	//Place leaf and parent data into appropriate buffers

	//Fill the leaf tiles buffer
	void* leaf_data = context["leaf_tile_indices"]->getBuffer()->map();
	optix::uint2* leaf_indices = (optix::uint2*)leaf_data;

	void* leaf_tile_size_data = context["leaf_tile_sizes"]->getBuffer()->map();
	unsigned int* leaf_sizes = (unsigned int*)leaf_tile_size_data;

	for (int i = 0; i < leaf_tiles.size(); i++)
	{
		leaf_indices[i] = leaf_tiles.at(i)->centre_index;
		leaf_sizes[i] = leaf_tiles.at(i)->tile_size;
	}
	context["leaf_tile_sizes"]->getBuffer()->unmap();
	context["leaf_tile_indices"]->getBuffer()->unmap();

	//Map the parent buffers 
	void* parent_data = context["parent_tile_indices"]->getBuffer()->map();
	optix::uint2* parent_indices = (optix::uint2*) parent_data;

	void* parent_size_data = context["parent_tile_sizes"]->getBuffer()->map();
	unsigned int* parent_sizes = (unsigned int*)parent_size_data;

	for (int i = 0; i < parent_tiles.size(); i++)
	{
		parent_indices[i] = parent_tiles.at(i)->centre_index;
		parent_sizes[i] = parent_tiles.at(i)->tile_size;
	}

	context["parent_tile_sizes"]->getBuffer()->unmap();
	context["parent_tile_indices"]->getBuffer()->unmap();
}




//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

//std::string ptxPath(const std::string& cuda_file);
Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void createGeometry();
void setupCamera();
void setupLights();
void setupCamera_simple();
void setupLights_simple();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}


void createContext()
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 3 );
    context->setStackSize( 4640 );

	// generate a random map
	for (int i = 0; i < randomMapSize; i++) {
		randomMap[i] = i;
	}

	for (int a = 0; a < 10; a++) {
		for (int i = 0; i < randomMapSize; i++) {
			int x = rand() % 100;
			if (x < 50) {
				if (i < randomMapSize - 1) {
					int temp = randomMap[i];
					randomMap[i] = randomMap[i + 1];
					randomMap[i + 1] = temp;
				}
				else {
					int temp = randomMap[i];
					randomMap[i] = randomMap[0];
					randomMap[0] = temp;
				}
			}
		}
	}


    // Note: high max depth for reflection and refraction through glass
    context["max_depth"]->setInt( 100 );
    context["radiance_ray_type"]->setUint( 0 );
    context["shadow_ray_type"]->setUint( 1 );
    context["scene_epsilon"]->setFloat( 1.e-4f );
    context["importance_cutoff"]->setFloat( 0.01f );
    context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );

    // Output buffer
    // First allocate the memory for the GL buffer, then attach it to OptiX.
    GLuint vbo = 0;
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glBufferData( GL_ARRAY_BUFFER, 4 * width * height, 0, GL_STREAM_DRAW);
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );


    // Ray generation program
    const std::string camera_name = tutorial_number >= 11 ? "env_camera" : "pinhole_camera";
    Program ray_gen_program = context->createProgramFromPTXString( tutorial_ptx, camera_name );
    context->setRayGenerationProgram( 0, ray_gen_program );
	
	// Extent program
	Program extent_program = context->createProgramFromPTXString(tutorial_ptx, "extent_calculation");
	context->setRayGenerationProgram(1, extent_program);
	
	
	// Reconstruct program
	
	Program reconstruct_program = context->createProgramFromPTXString(tutorial_ptx, "reconstruct");
	context->setRayGenerationProgram(2, reconstruct_program);
	

    // Exception program
    Program exception_program = context->createProgramFromPTXString( tutorial_ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
	context->setExceptionProgram( 1, exception_program );
	context->setExceptionProgram( 2, exception_program);
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    const std::string miss_name = tutorial_number >= 5 ? "envmap_miss" : "miss";
    context->setMissProgram( 0, context->createProgramFromPTXString( tutorial_ptx, miss_name ) );
    const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
    const std::string texpath = texture_path + "/" + std::string( "CedarCity.hdr" );
    context["envmap"]->setTextureSampler( sutil::loadTexture( context, texpath, default_color) );
    context["bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );


	// add a new buffer : seed buffer

	Buffer seedBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1024u, 1024u);
	unsigned int* seeds = (unsigned int*)seedBuffer->map();
	for (unsigned int i = 0; i < 1024u * 1024u; ++i)
		seeds[i] = rand();
	seedBuffer->unmap();

	context["frame"]->setUint(0u);
	context["sqrt_occlusion_samples"]->setUint(2);
	context["occlusion_distance"]->setFloat(100.0f);
	context["rnd_seeds"]->set(seedBuffer);


    // 3D solid noise buffer, 1 float channel, all entries in the range [0.0, 1.0].

    const int tex_width  = 64;
    const int tex_height = 64;
    const int tex_depth  = 64;
    Buffer noiseBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, tex_width, tex_height, tex_depth);
    float *tex_data = (float *) noiseBuffer->map();

    // Random noise in range [0, 1]
    for (int i = tex_width * tex_height * tex_depth;  i > 0; i--) {
        // One channel 3D noise in [0.0, 1.0] range.
        *tex_data++ = rand_range(0.0f, 1.0f);
    }
    noiseBuffer->unmap(); 


    // Noise texture sampler
    TextureSampler noiseSampler = context->createTextureSampler();

    noiseSampler->setWrapMode(0, RT_WRAP_REPEAT);
    noiseSampler->setWrapMode(1, RT_WRAP_REPEAT);
    noiseSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    noiseSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    noiseSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    noiseSampler->setMaxAnisotropy(1.0f);
    noiseSampler->setMipLevelCount(1);
    noiseSampler->setArraySize(1);
    noiseSampler->setBuffer(0, 0, noiseBuffer);

    context["noise_texture"]->setTextureSampler(noiseSampler);

	// add some randomness? (a random seed buffer)
	m_rnd_seeds = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT, 1024u, 1024u);
	context["rnd_seeds"]->set(m_rnd_seeds);
	genRndSeeds(1024u, 1024u);

	// add a busy buffer for checking if a pixel is been writen 
	busy_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, width, height);
	int* busy_flag = (int*)busy_buffer->map();
	for (int i = 0; i < width*height; ++i)
		busy_flag[i] = 0;
	busy_buffer->unmap();
	context["busy_buffer"]->set(busy_buffer);

	//Set a color buffer

	color_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
	context["color_buffer"]->set(color_buffer);

	//Set a Stencil buffer

	stencil_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, width, height);
	context["stencil_buffer"]->set(stencil_buffer);


	// add tiles into the scene
	//Set the root tile properties to initialize it
	root_tile.tile_size = width;
	root_tile.north_east = root_tile.north_west = root_tile.south_east = root_tile.south_west = NULL;
	root_tile.parent = NULL;
	root_tile.centre_index.x = root_tile.centre_index.y = width/2;

	leaf_tiles = root_tile.get_leaf_tiles(leaf_tiles);

	printf("Initial leaf_tiles size = %u \n", leaf_tiles.size());

	//while (leaf_tiles.size() < InitLeafTileSize)
	while (leaf_tiles.size() < num_of_tile)
	{
		for (int i = 0; i < leaf_tiles.size(); i++)
		{
			leaf_tiles.at(i)->splitTile();
		}
		leaf_tiles.clear();
		leaf_tiles = root_tile.get_leaf_tiles(leaf_tiles);
		printf("Now, leaf tiles = %u \n", leaf_tiles.size());
	}

	//Add leaf tiles to the pq
	for (int i = 0; i < leaf_tiles.size(); i++)
	{
		//leaf_pq.push(leaf_tiles.at(i));
		leaf_tiles.at(i)->is_leaf_tile = true;
	}

	//leaf_tiles.clear();

	//Populate parent tiles
	parent_tiles.clear();
	parent_tiles = root_tile.calculate_descendants(parent_tiles);
	printf("Parents = %u \n", parent_tiles.size());

	for (int i = 0; i < parent_tiles.size(); i++)
	{
		parent_tiles.at(i)->is_leaf_tile = false;
	}

	float s = (float)parent_tiles.size();
	const float* six = &s;
	context["number_of_parent_tiles"]->set1fv(six);

	// create a tiled buffer
	tiled_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
	context["tiled_buffer"]->set(tiled_buffer);

	//Set the leaf_tile_indices buffer
	leaf_tile_indices = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT2, num_of_tile);
	context["leaf_tile_indices"]->set(leaf_tile_indices);

	//Set the leaf_tile_sizes buffer
	leaf_tile_sizes = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, num_of_tile);
	context["leaf_tile_sizes"]->set(leaf_tile_sizes);

	//Set the parent tile buffers
	context["parent_tile_indices"]->set(context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT2, num_of_tile));
	context["parent_tile_sizes"]->set(context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, num_of_tile));


	//Set the variance buffers
	variance_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, num_of_tile);
	context["variance_buffer"]->set(variance_buffer);

	//Set the parent variance buffers
	parent_variance_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, num_of_tile);
	context["parent_variance_buffer"]->set(parent_variance_buffer);

	//Set the leaf_tile_indices buffer
	leaf_tile_indices = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT2, num_of_tile);
	context["leaf_tile_indices"]->set(leaf_tile_indices);
	 
	//Set the leaf_tile_sizes buffer
	leaf_tile_sizes = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, num_of_tile);
	context["leaf_tile_sizes"]->set(leaf_tile_sizes);

	//Set the parent tile buffers
	context["parent_tile_indices"]->set(context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT2, num_of_tile));
	context["parent_tile_sizes"]->set(context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, num_of_tile));

	// Set pixel_at_tile buffer
	pixel_at_tile_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, width, height);
	context["pixel_at_tile_buffer"]->set(pixel_at_tile_buffer);

	// set render elapse time buffer
	render_elapse_time_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
	float* render_elapse_flag = (float*)render_elapse_time_buffer->map();
	render_elapse_flag[0] = 0.0f;
	render_elapse_time_buffer->unmap();
	context["render_elapse_time_buffer"]->set(render_elapse_time_buffer);

	// set mini tile buffer

	mini_tile_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, (width / 8), (height / 8));
	int* mini_flag = (int*)mini_tile_buffer->map();
	for (int i = 0; i < (width / 8)*(height / 8); ++i)
		mini_flag[i] = 0;
	mini_tile_buffer->unmap();
	context["mini_tile_buffer"]->set(mini_tile_buffer);

	// set random map buffer

	random_map_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 8, 8);
	int* random_map_flag = (int*)random_map_buffer->map();
	for (int i = 0; i < 64; ++i)
		random_map_flag[i] = randomMap[i];;
	random_map_buffer->unmap();
	context["random_map_buffer"]->set(random_map_buffer);


	// Set sample_time_buffer
	sample_time_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, width, height);
	context["sample_time_buffer"]->set(sample_time_buffer);

	// Set sample_time_temp_buffer
	sample_time_temp_buffer1 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, width, height);
	context["sample_time_temp_buffer1"]->set(sample_time_temp_buffer1);

	sample_time_temp_buffer2 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, width, height);
	context["sample_time_temp_buffer2"]->set(sample_time_temp_buffer2);

	sample_time_temp_buffer3 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, width, height);
	context["sample_time_temp_buffer3"]->set(sample_time_temp_buffer3);
	
	//Set the temp_buffer
	float_color_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	temp_buffer1 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
	float_temp_buffer1 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	temp_buffer2 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
	float_temp_buffer2 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	temp_buffer3 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
	float_temp_buffer3 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	context["float_color_buffer"]->set(float_color_buffer);
	context["temp_buffer1"]->set(temp_buffer1);
	context["float_temp_buffer1"]->set(float_temp_buffer1);
	context["temp_buffer2"]->set(temp_buffer2);
	context["float_temp_buffer2"]->set(float_temp_buffer2);
	context["temp_buffer3"]->set(temp_buffer3);
	context["float_temp_buffer3"]->set(float_temp_buffer3);

	// Set ray_per_sec_buffer
	ray_per_sec_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 1);
	context["ray_per_sec_buffer"]->set(ray_per_sec_buffer);

	// Set tile_gradient_buffer
	tile_gradient_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, num_of_tile, 3);
	context["tile_gradient_buffer"]->set(tile_gradient_buffer);

	// Set crosshair_buffer
	crosshair_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	context["crosshair_buffer"]->set(crosshair_buffer);

	crosshair_buffer1 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	context["crosshair_buffer1"]->set(crosshair_buffer1);

	crosshair_buffer2 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	context["crosshair_buffer2"]->set(crosshair_buffer2);

	crosshair_buffer3 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	context["crosshair_buffer3"]->set(crosshair_buffer3);

	// Set extent buffer
	extent_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	context["extent_buffer"]->set(extent_buffer);

	extent_buffer1 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	context["extent_buffer1"]->set(extent_buffer1);

	extent_buffer2 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	context["extent_buffer2"]->set(extent_buffer2);

	extent_buffer3 = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height);
	context["extent_buffer3"]->set(extent_buffer3);

	extent_at_tile = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, num_of_tile);
	context["extent_at_tile"]->set(extent_at_tile);

	//Set the ray counting buffer
	raycounting_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 1);
	context["raycounting_buffer"]->set(raycounting_buffer);

	//Set the visualization buffer
	show_tile_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 1);
	context["show_tile_buffer"]->set(show_tile_buffer);
	show_variance_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 1);
	context["show_variance_buffer"]->set(show_variance_buffer);
	show_gradient_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 1);
	context["show_gradient_buffer"]->set(show_gradient_buffer);

	// set screen size buffer

	screen_size_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 2);
	int* screen_size_flag = (int*)screen_size_buffer->map();
	screen_size_flag[0] = width;
	screen_size_flag[1] = height;
	
	screen_size_buffer->unmap();
	context["screen_size_buffer"]->set(screen_size_buffer);

	// set is_moving_buffer : if the scene is not moving, forcing the filter size to 1 
	is_moving_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 1);
	context["is_moving_buffer"]->set(is_moving_buffer);

	context->setPrintEnabled(true);
	context->setPrintBufferSize(1024);
}


void createGeometry()
{
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "box.cu" );
    Program box_bounds    = context->createProgramFromPTXString( ptx, "box_bounds" );
    Program box_intersect = context->createProgramFromPTXString( ptx, "box_intersect" );

    // Create box
    Geometry box = context->createGeometry();
    box->setPrimitiveCount( 1u );
    box->setBoundingBoxProgram( box_bounds );
    box->setIntersectionProgram( box_intersect );
    box["boxmin"]->setFloat( -2.0f, 0.0f, -2.0f );
    box["boxmax"]->setFloat(  2.0f, 7.0f,  2.0f );

    // Create chull
    Geometry chull = 0;
    if( tutorial_number >= 5){                                     // Originally it was 9, change to 5 for this project to use the refraction chull 
        chull = context->createGeometry();
        chull->setPrimitiveCount( 1u );
        chull->setBoundingBoxProgram( context->createProgramFromPTXString( tutorial_ptx, "chull_bounds" ) );
        chull->setIntersectionProgram( context->createProgramFromPTXString( tutorial_ptx, "chull_intersect" ) );
        Buffer plane_buffer = context->createBuffer(RT_BUFFER_INPUT);
        plane_buffer->setFormat(RT_FORMAT_FLOAT4);
        int nsides = 6;
        plane_buffer->setSize( nsides + 2 );
        float4* chplane = (float4*)plane_buffer->map();
        float radius = 1;
        float3 xlate = make_float3(-1.4f, 0, -3.7f);

        for(int i = 0; i < nsides; i++){
            float angle = float(i)/float(nsides) * M_PIf * 2.0f;
            float x = cos(angle);
            float y = sin(angle);
            chplane[i] = make_plane( make_float3(x, 0, y), make_float3(x*radius, 0, y*radius) + xlate);
        }
        float min = 0.02f;
        float max = 3.5f;
        chplane[nsides + 0] = make_plane( make_float3(0, -1, 0), make_float3(0, min, 0) + xlate);
        float angle = 5.f/nsides * M_PIf * 2;
        chplane[nsides + 1] = make_plane( make_float3(cos(angle),  .7f, sin(angle)), make_float3(0, max, 0) + xlate);
        plane_buffer->unmap();
        chull["planes"]->setBuffer(plane_buffer);
        chull["chull_bbmin"]->setFloat(-radius + xlate.x, min + xlate.y, -radius + xlate.z);
        chull["chull_bbmax"]->setFloat( radius + xlate.x, max + xlate.y,  radius + xlate.z);
    }

    // Floor geometry
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    parallelogram->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    parallelogram->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );
    float3 anchor = make_float3( -64.0f, 0.01f, -64.0f );
    float3 v1 = make_float3( 128.0f, 0.0f, 0.0f );
    float3 v2 = make_float3( 0.0f, 0.0f, 128.0f );
    float3 normal = cross( v2, v1 );
    normal = normalize( normal );
    float d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    float4 plane = make_float4( normal, d );
    parallelogram["plane"]->setFloat( plane );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );
    parallelogram["anchor"]->setFloat( anchor );

    // Materials
    std::string box_chname;
    if(tutorial_number >= 8){
        box_chname = "box_closest_hit_radiance";
    } else if(tutorial_number >= 3){
        box_chname = "closest_hit_radiance3";
    } else if(tutorial_number >= 2){
        box_chname = "closest_hit_radiance2";
    } else if(tutorial_number >= 1){
        box_chname = "closest_hit_radiance1";
    } else {
        box_chname = "closest_hit_radiance0";
    }

    Material box_matl = context->createMaterial();
    Program box_ch = context->createProgramFromPTXString( tutorial_ptx, box_chname.c_str() );
    box_matl->setClosestHitProgram( 0, box_ch );
    if( tutorial_number >= 3) {
        Program box_ah = context->createProgramFromPTXString( tutorial_ptx, "any_hit_shadow" );
        box_matl->setAnyHitProgram( 1, box_ah );
    }
    box_matl["Ka"]->setFloat( 0.3f, 0.3f, 0.3f );
    box_matl["Kd"]->setFloat( 0.6f, 0.7f, 0.8f );
    box_matl["Ks"]->setFloat( 0.8f, 0.9f, 0.8f );
    box_matl["phong_exp"]->setFloat( 88 );
    box_matl["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );

    std::string floor_chname;
    if(tutorial_number >= 7){
        floor_chname = "floor_closest_hit_radiance";
    } else if(tutorial_number >= 6){
        floor_chname = "floor_closest_hit_radiance5";
    } else if(tutorial_number >= 4){
        floor_chname = "floor_closest_hit_radiance4";
    } else if(tutorial_number >= 3){
        floor_chname = "closest_hit_radiance3";
    } else if(tutorial_number >= 2){
        floor_chname = "closest_hit_radiance2";
    } else if(tutorial_number >= 1){
        floor_chname = "closest_hit_radiance1";
    } else {
        floor_chname = "closest_hit_radiance0";
    }

    Material floor_matl = context->createMaterial();
    Program floor_ch = context->createProgramFromPTXString( tutorial_ptx, floor_chname.c_str() );
    floor_matl->setClosestHitProgram( 0, floor_ch );
    if(tutorial_number >= 3) {
        Program floor_ah = context->createProgramFromPTXString( tutorial_ptx, "any_hit_shadow" );
        floor_matl->setAnyHitProgram( 1, floor_ah );
    }
    floor_matl["Ka"]->setFloat( 0.3f, 0.3f, 0.1f );
    floor_matl["Kd"]->setFloat( 194/255.f*.6f, 186/255.f*.6f, 151/255.f*.6f );
    floor_matl["Ks"]->setFloat( 0.4f, 0.4f, 0.4f );
    floor_matl["reflectivity"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["reflectivity_n"]->setFloat( 0.05f, 0.05f, 0.05f );
    floor_matl["phong_exp"]->setFloat( 88 );
    floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
    floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
    floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
    floor_matl["crack_width"]->setFloat( 0.02f );

    // Glass material
    Material glass_matl;
    if( chull.get() ) {
        Program glass_ch = context->createProgramFromPTXString( tutorial_ptx, "glass_closest_hit_radiance" );
        const std::string glass_ahname = tutorial_number >= 10 ? "glass_any_hit_shadow" : "any_hit_shadow";
        Program glass_ah = context->createProgramFromPTXString( tutorial_ptx, glass_ahname.c_str() );
        glass_matl = context->createMaterial();
        glass_matl->setClosestHitProgram( 0, glass_ch );
        glass_matl->setAnyHitProgram( 1, glass_ah );

        glass_matl["importance_cutoff"]->setFloat( 1e-2f );
        glass_matl["cutoff_color"]->setFloat( 0.34f, 0.55f, 0.85f );
        glass_matl["fresnel_exponent"]->setFloat( 3.0f );
        glass_matl["fresnel_minimum"]->setFloat( 0.1f );
        glass_matl["fresnel_maximum"]->setFloat( 1.0f );
        glass_matl["refraction_index"]->setFloat( 1.4f );
        glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
        glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
        glass_matl["refraction_maxdepth"]->setInt( 100 );
        glass_matl["reflection_maxdepth"]->setInt( 100 );
        float3 extinction = make_float3(.80f, .89f, .75f);
        glass_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
        glass_matl["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );
    }

    // Create GIs for each piece of geometry 
    std::vector<GeometryInstance> gis;
    gis.push_back( context->createGeometryInstance( box, &box_matl, &box_matl+1 ) );
    gis.push_back( context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );
    if(chull.get())
        gis.push_back( context->createGeometryInstance( chull, &glass_matl, &glass_matl+1 ) );

    // Place all in group
    GeometryGroup geometrygroup = context->createGeometryGroup();
    geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
    geometrygroup->setChild( 0, gis[0] );
    geometrygroup->setChild( 1, gis[1] );
    if(chull.get()) {
        geometrygroup->setChild( 2, gis[2] );
    }
    geometrygroup->setAcceleration( context->createAcceleration("NoAccel") );

    context["top_object"]->set( geometrygroup );
    context["top_shadower"]->set( geometrygroup );

}


void setupCamera()
{
	const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components


																 //camera_eye = aabb.center() + make_float3(0.0f, 0.0f, max_dim*1.5f);
	if (scene_fireplace)
		camera_eye = make_float3(2.0f, 1.5f, -0.5f);
	else if (scene_breakfast)
		camera_eye = make_float3(2.0f, 1.5f, -0.5f);
	else
		camera_eye = make_float3(620.0f, 680.0f, -50.0f);

	camera_eye.x *= -1;
	camera_lookat = aabb.center();
	
	
	camera_up = make_float3(0.0f, 1.0f, 0.0f);

	camera_rotate = Matrix4x4::identity();
}

void setupLights()
{
	const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components


																 //BasicLight lights[] = {
																 //	
	//{ make_float3(-0.5f, 0.25f, -1.0f), make_float3(1.0f, 1.0f, 1.0f), 0, 0 },
	//{ make_float3(-0.5f,  0.0f ,  1.0f), make_float3(1.0f, 1.0f, 1.0f), 0, 0 },
	//{ make_float3(0.0f,  500.5f ,  -50.5f) , make_float3(0.5f, 0.5f, 0.5f), 1, 0 }
																 //};

	
	BasicLight lights[] = {
		    { make_float3(-0.5f,  0.25f, -1.0f), make_float3(0.6f, 0.6f, 0.6f), 0, 0 },
			{ make_float3(-0.5f,  0.0f ,  1.0f), make_float3(0.3f, 0.3f, 0.3f), 1, 0 },
			{ make_float3(0.5f,  0.5f ,  0.5f), make_float3(0.7f, 0.7f, 0.65f), 0, 0 }
	};

	//lights[0].pos *= max_dim * 10.0f;
	//lights[1].pos *= max_dim * 10.0f;
	//lights[2].pos *= max_dim * 10.0f;
	
	/*
	BasicLight lights[100];

	int lightlengthbase = 10;
	float lightdivide = float(lightlengthbase* lightlengthbase);
	for (int i = 0; i <  lightlengthbase; i++) {
		for (int j = 0; j <  lightlengthbase; j++)
			lights[lightlengthbase * i + j] = { make_float3(-0.5f + 0.01f*i,  0.3f + 0.01f*j,  0.5f), make_float3(1.0f/ lightdivide, 1.0f / lightdivide, 1.0f / lightdivide), 0, 0 };
	}

	lights[0].casts_shadow = 1;
	lights[1].casts_shadow = 1;

	for (int i = 0; i <  lightlengthbase* lightlengthbase; i++) {
		lights[i].pos *= max_dim * 10.0f;
	}
	*/

	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(BasicLight));
	light_buffer->setSize(sizeof(lights) / sizeof(lights[0]));
	memcpy(light_buffer->map(), lights, sizeof(lights));
	light_buffer->unmap();

	context["lights"]->set(light_buffer);
}


void setupCamera_simple()
{
    camera_eye    = make_float3( 7.0f, 9.2f, -6.0f );
    camera_lookat = make_float3( 0.0f, 4.0f,  0.0f );
    camera_up     = make_float3( 0.0f, 1.0f,  0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void setupLights_simple()
{

    BasicLight lights[] = { 
        { make_float3( -5.0f, 60.0f, -16.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
    };

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );
}


void updateCamera()
{
    const float vfov = 60.0f;
    const float aspect_ratio = static_cast<float>(width) /
                               static_cast<float>(height);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    const Matrix4x4 frame = Matrix4x4::fromBasis(
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans   = frame*camera_rotate*camera_rotate*frame_inv;

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );
}


void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( SAMPLE_NAME );
    glutHideWindow();
}


void glutRun()
{
    // Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1 );

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    glutShowWindow();
    glutReshapeWindow( width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
	quaternion_implementation_for_camera_rotation();

	updateCamera();

	sample_time_start = clock();
	prev_frame_time = clock();
	current_time = clock();

	//context->launch(0, num_of_tile, static_cast<unsigned int>(naive_adding_threads));
	context->launch(0, num_of_tile, static_cast<unsigned int>(512u));
	//context->launch(0, static_cast<unsigned int>(512u), static_cast<unsigned int>(512u));

	float elapse_time_from_start = (double)(current_time - begin_rendering_time) / (double)CLOCKS_PER_SEC;
	float sample_time_elapse = 0.0f, reconstruction_time_elapse = 0.0f;
	void* global_time_data = context["render_elapse_time_buffer"]->getBuffer()->map();
	float* global_time_flag = (float*)global_time_data;
	*global_time_flag = elapse_time_from_start;
	context["render_elapse_time_buffer"]->getBuffer()->unmap();

	//printf("start sampling, whole elapse time = %f \n", elapse_time_from_start);

	context->launch(1, num_of_tile, 3);
	context->launch(2, width, height);

	Buffer buffer = getOutputBuffer();
	sutil::displayBufferGL(getOutputBuffer());

	{
		static unsigned frame_count = 0;
		sutil::displayFps(frame_count++);
	}

	void* raycounting_data = context["raycounting_buffer"]->getBuffer()->map();

	unsigned int* raycount = (unsigned int*)raycounting_data;
	unsigned int rayray = *raycount;

	current_time = clock();
	if (current_time - global_time_00001s >= 1) {
		global_time_00001s = current_time;
		ray_difference = (int)rayray - (int)(previous_raycount);
		previous_raycount = rayray;

		/*
		if (moving_box_one) {
			if (boxmax <= 480.0f) {
				boxmin += 0.25f;
				boxmax += 0.25f;
			}
			else {
				moving_box_one = false;
				moving_box_two = true;
			}
		}else if (moving_box_two) {
			if (boxmax >= 330.0f) {
				boxmin -= 0.25f;
				boxmax -= 0.25f;
			}
			else {
				moving_box_one = true;
				moving_box_two = false;
			}
		}
		
		moving_box["boxmin"]->setFloat(-50.0f, boxmin, -50.0f);
		moving_box["boxmax"]->setFloat(50.0f, boxmax, 50.0f);
		*/
		//printf("bmin = %f, bmax = %f\n", boxmin, boxmax);
		
	}

	current_time = clock();
	if ((float)(current_time - global_time_1s) / 1000.0f >= 1) {
		global_time_1s = current_time;
		printf("naive_adding_threads = %u\n", naive_adding_threads);

		if (prev_sec_frame_generation < 60 && naive_adding_threads > 50u) {
			naive_adding_threads -= 10u;
		}
		else if (prev_sec_frame_generation > 70) {
			naive_adding_threads += 10u;
		}

		prev_sec_frame_generation = 0u;
		

	}
	context["raycounting_buffer"]->getBuffer()->unmap();
	//printf("ray_difference = %d\n", ray_difference);
	
	// updating ray_per_sec_buffer
	int* ray_difference_gpu = (int*)ray_per_sec_buffer->map();
	ray_difference_gpu[0] = ray_difference;
	//void* ray_difference_gpu = context["ray_per_sec_buffer"]->getBuffer()->map();
	//unsigned int* num = (unsigned int*)ray_difference_gpu;
	//*num = ray_difference;
	//printf("num = %d\n", *num);
	context["ray_per_sec_buffer"]->getBuffer()->unmap();
	
	current_time_test = clock();
	if ((float)(current_time_test - global_time_01s) / 1000.0f >= 0.1) {
		global_time_01s = current_time_test;
		if (camera_in_motion) {
			if (camera_move_left && camera_eye.z < 150.0f)
				camera_eye.z += camera_moving_speed;
			else if (camera_move_left) {
				camera_move_left = !camera_move_left;
				camera_move_right = !camera_move_right;
			}
			if (camera_move_right && camera_eye.z > -150.0f)
				camera_eye.z -= camera_moving_speed;
			else if (camera_move_right) {
				camera_move_left = !camera_move_left;
				camera_move_right = !camera_move_right;
			}
		}
		else if (camera_in_motion1) {
			if (camera_move_left && camera_eye.x < 350.0f) {
				camera_eye.x += camera_moving_speed;
				camera_lookat.x += camera_moving_speed;
			}
			else if (camera_move_left) {
				camera_move_left = !camera_move_left;
				camera_move_right = !camera_move_right;
			}
			if (camera_move_right && camera_eye.x > -350.0f) {
				camera_eye.x -= camera_moving_speed;
				camera_lookat.x -= camera_moving_speed;
			}
			else if (camera_move_right) {
				camera_move_left = !camera_move_left;
				camera_move_right = !camera_move_right;
			}
		}
		else if (camera_in_motion2) {
			if (camera_move_left && camera_eye.y < 900.0f) {
				camera_eye.y += camera_moving_speed;
				camera_lookat.y += camera_moving_speed;
			}
			else if (camera_move_left) {
				camera_move_left = !camera_move_left;
				camera_move_right = !camera_move_right;
			}
			if (camera_move_right && camera_eye.y > 150.0f) {
				camera_eye.y -= camera_moving_speed;
				camera_lookat.y -= camera_moving_speed;
			}
			else if (camera_move_right) {
				camera_move_left = !camera_move_left;
				camera_move_right = !camera_move_right;
			}
		}
		else if (camera_in_motion3) {
			if (camera_move_left && camera_eye.z < 300.0f) {
				camera_eye.z += camera_moving_speed;
				camera_lookat.z += camera_moving_speed;
			}
			else if (camera_move_left) {
				camera_move_left = !camera_move_left;
				camera_move_right = !camera_move_right;
			}
			if (camera_move_right && camera_eye.z > -350.0f) {
				camera_eye.z -= camera_moving_speed;
				camera_lookat.z -= camera_moving_speed;
			}
			else if (camera_move_right) {
				camera_move_left = !camera_move_left;
				camera_move_right = !camera_move_right;
			}
		}
		else {

		}
	}

	void* is_moving = context["is_moving_buffer"]->getBuffer()->map();
	unsigned int* is_moving_num = (unsigned int*)is_moving;

	if (camera_in_motion1 || camera_in_motion2 || camera_in_motion3 || mouse_clicking || automatic_camera_moving || playing_record_moving) {
		//*is_moving_num = (unsigned int)1;
		trigger_moving_time = clock();
	}
	//else {
	//	*is_moving_num = (unsigned int)0;
	//}

	if ((float)(current_time_test - trigger_moving_time) / 1000.0f >= 0.3) {
		*is_moving_num = (unsigned int)0;
	}
	else {
		*is_moving_num = (unsigned int)1;
	}

	if (automatic_camera) {
		current_time = clock();
		float time_difference = float(current_time - start_automatic_camera);
		time_difference /= 1000.0f;
		if (time_difference < 3.0f) {

		}
		else if (time_difference < 10.0f) {
			automatic_camera_moving = true;
			camera_eye.x -= keyboard_speed;
			camera_lookat.x -= keyboard_speed;
		}
		else if (time_difference < 15.0f) {
			camera_eye.x += keyboard_speed;
			camera_lookat.x += keyboard_speed;
		}
		else if (time_difference < 20.0f) {
			camera_eye.z -= keyboard_speed;
			camera_lookat.z -= keyboard_speed;
		}
		else if (time_difference < 25.0f) {
			camera_eye.z += keyboard_speed;
			camera_lookat.z += keyboard_speed;
		}
		else if (time_difference < 30.0f) {
			camera_eye.y -= keyboard_speed;
			camera_lookat.y -= keyboard_speed;
		}
		else if (time_difference < 35.0f) {
			camera_eye.y += keyboard_speed;
			camera_lookat.y += keyboard_speed;
		}
		else {
			automatic_camera_moving = false;
		}
	}

	/*
	if (play_camera_position && playing_camera_position_time < 6000) {
		camera_eye.x = automatic_moving_camera[playing_camera_position_time].x;
		camera_eye.y = automatic_moving_camera[playing_camera_position_time].y;
		camera_eye.z = automatic_moving_camera[playing_camera_position_time].z;
		camera_lookat.x = automatic_moving_lookat[playing_camera_position_time].x;
		camera_lookat.y = automatic_moving_lookat[playing_camera_position_time].y;
		camera_lookat.z = automatic_moving_lookat[playing_camera_position_time].z;
		


		if ((old_camera_eye.x - automatic_moving_camera[playing_camera_position_time].x <= 1) && (old_camera_eye.y - automatic_moving_camera[playing_camera_position_time].y <= 1) && ( old_camera_eye.z - automatic_moving_camera[playing_camera_position_time].z <= 1 ) && ( old_camera_lookat.x - automatic_moving_lookat[playing_camera_position_time].x <= 1 )&& (old_camera_lookat.y - automatic_moving_lookat[playing_camera_position_time].y <= 1 ) && (old_camera_lookat.z - automatic_moving_lookat[playing_camera_position_time].z <= 1 ) ) {
			playing_record_moving = false;
		}
		else {
			playing_record_moving = true;
		}
		playing_camera_position_time++;
		printf("still here\n");
	}
	*/

	// new playing strategy (testing)

	/* adding some testting functions:

	First recording the camera position from the recording file
	Later it needs to be interpolated and load into the GPU

	1. For ideal renderer, for example, use 120 frames/sec generate around 6000 views in camera, and feed into the GPU, let it produce 6000 images
	2. For framed-buffer renderer/ AFR, just use the camera position as recored before, and use the time as recorded, but when producing the image, record the rendered time. 

	*/
	if (play_camera_position) {
		current_time = clock();
		float time_difference_for_playing_naive = float((current_time - start_playing_camera) / 1000.0f);
		if (time_difference_for_playing_naive > max_time_for_recording) {
			printf("exceed %f, stop playing \n", max_time_for_recording);
			play_camera_position = false;
		}
		else {
			printf("playing record at time %f \n", time_difference_for_playing_naive);
			int find_the_correct_position_for_playing = 0;
			for (int i = 0; i < 6000; i++) {
				if (time_difference_for_playing_naive < time_buffer_for_camera[i]) {
					//printf("i = %d, case1 = %f, case2 = %f \n", i, time_difference_for_playing_naive, time_buffer_for_camera[i]);
					break;
				}
					
				find_the_correct_position_for_playing = i;
			}
			//printf("where to play = %d \n", find_the_correct_position_for_playing);
			
			// doing interpolation here

			//printf("before = %f, now = %f, after = %f \n", time_buffer_for_camera[find_the_correct_position_for_playing], time_difference_for_playing_naive, time_buffer_for_camera[find_the_correct_position_for_playing+1]);
			if (find_the_correct_position_for_playing + 1 >= 6000) { // boundary case
				camera_eye.x = automatic_moving_camera[find_the_correct_position_for_playing].x;
				camera_eye.y = automatic_moving_camera[find_the_correct_position_for_playing].y;
				camera_eye.z = automatic_moving_camera[find_the_correct_position_for_playing].z;
				camera_lookat.x = automatic_moving_lookat[find_the_correct_position_for_playing].x;
				camera_lookat.y = automatic_moving_lookat[find_the_correct_position_for_playing].y;
				camera_lookat.z = automatic_moving_lookat[find_the_correct_position_for_playing].z;
			}
			else {
				float3 previous_camera_eye = automatic_moving_camera[find_the_correct_position_for_playing];
				float3 previous_lookat     = automatic_moving_lookat[find_the_correct_position_for_playing];
				float3 future_camera_eye = automatic_moving_camera[find_the_correct_position_for_playing+1];
				float3 future_lookat = automatic_moving_lookat[find_the_correct_position_for_playing+1];
				float previous_time = time_buffer_for_camera[find_the_correct_position_for_playing];
				float future_time   = time_buffer_for_camera[find_the_correct_position_for_playing+1];
				
				float3 estimated_camera_eye, estimated_camera_lookat;

				float time_to_interpolate = (time_difference_for_playing_naive - previous_time) / (future_time - previous_time);
				
				estimated_camera_eye.x = previous_camera_eye.x * time_to_interpolate + future_camera_eye.x * (1- time_to_interpolate);
				estimated_camera_eye.y = previous_camera_eye.y * time_to_interpolate + future_camera_eye.y * (1 - time_to_interpolate);
				estimated_camera_eye.z = previous_camera_eye.z * time_to_interpolate + future_camera_eye.z * (1 - time_to_interpolate);
				estimated_camera_lookat.x = previous_lookat.x * time_to_interpolate + future_lookat.x * (1 - time_to_interpolate);
				estimated_camera_lookat.y = previous_lookat.y * time_to_interpolate + future_lookat.y * (1 - time_to_interpolate);
				estimated_camera_lookat.z = previous_lookat.z * time_to_interpolate + future_lookat.z * (1 - time_to_interpolate);

				camera_eye = estimated_camera_eye;
				camera_lookat = estimated_camera_lookat;
			}

			if ((old_camera_eye.x - camera_eye.x <= 1) && (old_camera_eye.y - camera_eye.y <= 1) && (old_camera_eye.z - camera_eye.z <= 1) && (old_camera_lookat.x - camera_lookat.x <= 1) && (old_camera_lookat.y - camera_lookat.y <= 1) && (old_camera_lookat.z - camera_lookat.z <= 1)) {
				playing_record_moving = false;
			}
			else {
				playing_record_moving = true;
			}
			const std::string outputImage = "Z "+ std::string(SAMPLE_NAME) + " " + std::to_string(time_difference_for_playing_naive) + ".ppm";
			sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer());

		}
	}



	if (record_camera_position && recording_camera_position_time<6000) {
		
		float camera_position_x = camera_eye.x;
		float camera_position_y = camera_eye.y;
		float camera_position_z = camera_eye.z;
		float lookat_position_x = camera_lookat.x;
		float lookat_position_y = camera_lookat.y;
		float lookat_position_z = camera_lookat.z;
		current_time = clock();
		float time_difference_for_recording_naive = float((current_time - start_recording_camera)/1000.0f);
		//fp_out << camera_difference_x << std::endl;
		//fp_out << camera_difference_y << std::endl;
		//fp_out << camera_difference_z << std::endl;
		//fp_out << lookat_difference_x << std::endl;
		//fp_out << lookat_difference_y << std::endl;
		//fp_out << lookat_difference_z << std::endl;
		fp_out << camera_position_x << " " << camera_position_y << " " << camera_position_z << " " << lookat_position_x << " " << lookat_position_y << " " << lookat_position_z << " " << time_difference_for_recording_naive << " " << current_quaternion.w << " " << current_quaternion.x << " " << current_quaternion.y << " " << current_quaternion.z << std::endl;

		recording_camera_position_time++;
		if (recording_camera_position_time == 6000)
			fp_out.close();
	}


	
	old_camera_eye = camera_eye;
	old_camera_lookat = camera_lookat;
	
	context["is_moving_buffer"]->getBuffer()->unmap();
	mouse_clicking = false;
	prev_sec_frame_generation++;

	for (int i = 0; i< 1; i++)
		retiling(1);
	
	
	glutSwapBuffers();
	current_time = clock();
	float time_diff = (float)(current_time) / 1000.0f - (float)(prev_frame_time) / 1000.0f;
	//printf("time flow for a frame = %f \n", time_diff);
}

void setup_automatic_camera()
{
	fp_in.open("camera_input.txt");
	max_time_for_recording = 0.0f;
	/*for (int i = 0; i < 6000; i++) {
		automatic_moving_camera[i].x = 0.0f;
		automatic_moving_camera[i].y = 0.0f;
		automatic_moving_camera[i].z = 0.0f;
		automatic_moving_lookat[i].x = 0.0f;
		automatic_moving_lookat[i].y = 0.0f;
		automatic_moving_lookat[i].z = 0.0f;
	}*/
	int i = 0;
	for (std::string line; std::getline(fp_in, line);) {
		std::istringstream in(line);
		fp_in >> automatic_moving_camera[i].x >> automatic_moving_camera[i].y >> automatic_moving_camera[i].z >> automatic_moving_lookat[i].x >> automatic_moving_lookat[i].y >> automatic_moving_lookat[i].z >> time_buffer_for_camera[i] >> quaternion_moving_camera[i].w >> quaternion_moving_camera[i].x >> quaternion_moving_camera[i].y >> quaternion_moving_camera[i].z;
		if (max_time_for_recording < time_buffer_for_camera[i]) {
			max_time_for_recording = time_buffer_for_camera[i];
		}
		//printf("time_buffer_for_camera = %f \n", time_buffer_for_camera[i]);
		i++;
	}
	printf("playing max = %f \n", max_time_for_recording);
	fp_out.open("camera_output.txt");
}

void clean_automatic_camera()
{
	for (int i = 0; i < 6000; i++) {
		automatic_moving_camera[i].x = 0.0f;
		automatic_moving_camera[i].y = 0.0f;
		automatic_moving_camera[i].z = 0.0f;
		automatic_moving_lookat[i].x = 0.0f;
		automatic_moving_lookat[i].y = 0.0f;
		automatic_moving_lookat[i].z = 0.0f;
		time_buffer_for_camera[i] = 0.0f;
	}
}

// use somecode to implement matrix to quaterion :https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

void quaternion_implementation_for_camera_rotation()
{
	float *m = camera_rotate.getData();
	//printf("current rotation output = %f, %f, %f,%f, %f, %f,%f, %f, %f,%f, %f, %f,%f, %f, %f,%f\n", m[0],m[1],m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
	float trace = m[0] + m[5] + m[10]; // I removed + 1.0f; see discussion with Ethan
	if (trace > 0) {// I changed M_EPSILON to 0
		float s = 0.5f / sqrtf(trace + 1.0f);
		current_quaternion.w = 0.25f / s;
		current_quaternion.x = (m[9] - m[6]) * s;
		current_quaternion.y = (m[2] - m[8]) * s;
		current_quaternion.z = (m[4] - m[1]) * s;
	}
	else {
		if (m[0] > m[5] && m[0] > m[10]) {
			float s = 2.0f * sqrtf(1.0f + m[0] - m[5] - m[10]);
			current_quaternion.w = (m[9] - m[6]) / s;
			current_quaternion.x = 0.25f * s;
			current_quaternion.y = (m[1] + m[4]) / s;
			current_quaternion.z = (m[2] + m[8]) / s;
		}
		else if (m[5] > m[10]) {
			float s = 2.0f * sqrtf(1.0f + m[5] - m[0] - m[10]);
			current_quaternion.w = (m[2] - m[8]) / s;
			current_quaternion.x = (m[1] + m[4]) / s;
			current_quaternion.y = 0.25f * s;
			current_quaternion.z = (m[6] + m[9]) / s;
		}
		else {
			float s = 2.0f * sqrtf(1.0f + m[10] - m[0] - m[5]);
			current_quaternion.w = (m[4] - m[1]) / s;
			current_quaternion.x = (m[2] + m[8]) / s;
			current_quaternion.y = (m[6] + m[9]) / s;
			current_quaternion.z = 0.25f * s;
		}
	}
}


void generate_ideal_camera() 
{
     // lets say the ideal camera will update in 120 hz
	float t = 1.0f / 120.0f;
	// use interpolation to get the ideal camera positions
	for (int i = 0; i < 6000; i++) {
		int find_the_correct_position_for_playing = 0;
		for (int j = 0; j < 6000; j++) {
			if (float(i)*t < time_buffer_for_camera[j]) {
				printf("j = %d, case1 = %f, case2 = %f \n", j, float(i)*t, time_buffer_for_camera[j]);
				break;
			}

			find_the_correct_position_for_playing = j;
		}
		//printf("where to play = %d \n", find_the_correct_position_for_playing);

		// doing interpolation here

		//printf("before = %f, now = %f, after = %f \n", time_buffer_for_camera[find_the_correct_position_for_playing], time_difference_for_playing_naive, time_buffer_for_camera[find_the_correct_position_for_playing+1]);
		if (find_the_correct_position_for_playing + 1 >= 6000) { // boundary case
			ideal_moving_camera[i].x = automatic_moving_camera[find_the_correct_position_for_playing].x;
			ideal_moving_camera[i].y = automatic_moving_camera[find_the_correct_position_for_playing].y;
			ideal_moving_camera[i].z = automatic_moving_camera[find_the_correct_position_for_playing].z;
			ideal_moving_lookat[i].x = automatic_moving_lookat[find_the_correct_position_for_playing].x;
			ideal_moving_lookat[i].y = automatic_moving_lookat[find_the_correct_position_for_playing].y;
			ideal_moving_lookat[i].z = automatic_moving_lookat[find_the_correct_position_for_playing].z;
		}
		else {
			float3 previous_camera_eye = automatic_moving_camera[find_the_correct_position_for_playing];
			float3 previous_lookat = automatic_moving_lookat[find_the_correct_position_for_playing];
			float3 future_camera_eye = automatic_moving_camera[find_the_correct_position_for_playing + 1];
			float3 future_lookat = automatic_moving_lookat[find_the_correct_position_for_playing + 1];
			float previous_time = time_buffer_for_camera[find_the_correct_position_for_playing];
			float future_time = time_buffer_for_camera[find_the_correct_position_for_playing + 1];

			float3 estimated_camera_eye, estimated_camera_lookat;
			
			float time_to_interpolate = (float(i)*t - previous_time) / (future_time - previous_time);
			
			estimated_camera_eye.x = previous_camera_eye.x * time_to_interpolate + future_camera_eye.x * (1 - time_to_interpolate);
			estimated_camera_eye.y = previous_camera_eye.y * time_to_interpolate + future_camera_eye.y * (1 - time_to_interpolate);
			estimated_camera_eye.z = previous_camera_eye.z * time_to_interpolate + future_camera_eye.z * (1 - time_to_interpolate);
			estimated_camera_lookat.x = previous_lookat.x * time_to_interpolate + future_lookat.x * (1 - time_to_interpolate);
			estimated_camera_lookat.y = previous_lookat.y * time_to_interpolate + future_lookat.y * (1 - time_to_interpolate);
			estimated_camera_lookat.z = previous_lookat.z * time_to_interpolate + future_lookat.z * (1 - time_to_interpolate);
			
			ideal_moving_camera[i] = estimated_camera_eye;
			ideal_moving_lookat[i] = estimated_camera_lookat;
			//printf("x = %f, y = %f, z = %f \n", ideal_moving_camera[i].x, ideal_moving_camera[i].y, ideal_moving_camera[i].z);
		}
	}

}
void glutKeyboardPress( unsigned char k, int x, int y )
{

    switch( k )
    {
        case( 'q' ):
        case( 27 ): // ESC
        {
            destroyContext();
            exit(0);
        }
        case( 's' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
		case('w'):
		{
			switch_tile_visual = !switch_tile_visual;
			void* show_tile_num = context["show_tile_buffer"]->getBuffer()->map();
			unsigned int* num = (unsigned int*)show_tile_num;
			if (switch_tile_visual) {
				*num = (unsigned int)0;
				std::cerr << "Switch to show normal mode\n";
			}
			else {
				*num = (unsigned int)1;
				std::cerr << "Switch to tile mode\n";
			}
			context["show_tile_buffer"]->getBuffer()->unmap();
			break;
		}
		case('x'):
		{
			switch_gradient_visual = !switch_gradient_visual;
			void* show_gradient_num = context["show_gradient_buffer"]->getBuffer()->map();
			unsigned int* num = (unsigned int*)show_gradient_num;
			if (switch_gradient_visual) {
				*num = (unsigned int)0;
				std::cerr << "Switch to show gradient mode\n";
			}
			else {
				*num = (unsigned int)1;
				std::cerr << "Switch to gradient mode\n";
			}
			context["show_gradient_buffer"]->getBuffer()->unmap();
			break;
		}
		case('m'):
		{
			switch_movement_visual = !switch_movement_visual;
			camera_in_motion = (!camera_in_motion);
			if (camera_in_motion) {
				std::cerr << "Switch to moving camera\n";
			}
			else {
				std::cerr << "Camera stop moving\n";
			}
			break;
		}
		case('['):
		{
			camera_in_motion1 = (!camera_in_motion1);
			camera_in_motion2 = false;
			camera_in_motion3 = false;
			break;
		}
		case(']'):
		{
			camera_in_motion2 = (!camera_in_motion2);
			camera_in_motion1 = false;
			camera_in_motion3 = false;
			break;
		}
		case('p'):
		{
			camera_in_motion3 = (!camera_in_motion3);
			camera_in_motion1 = false;
			camera_in_motion2 = false;
			break;
		}
		case('1'):
		{
			if (camera_moving_speed < 100.0f) {
				camera_moving_speed += 10.0f;
			}
			else {
				std::cerr << "Camera moving too fast!";
			}
			break;
		}
		case('2'):
		{
			if (camera_moving_speed > 2.0f) {
				camera_moving_speed -= 2.0f;
			}
			else {
				std::cerr << "Camera moving too slow!";
			}
			break;
		}
		case('r'):
		{
			camera_eye = make_float3(-620.0f, 680.0f, -50.0f);
			camera_moving_speed = 17.0f;
			camera_lookat = aabb.center();
			camera_up = make_float3(0.0f, 1.0f, 0.0f);
			camera_rotate = Matrix4x4::identity();
			std::cerr << "Reset the camera_eye\n";
			break;

		}
		case('u'):
		{
			trigger_moving_time = clock();
			camera_eye.x -= keyboard_speed;
			camera_lookat.x -= keyboard_speed;
			break;
		}
		case('o'):
		{
			trigger_moving_time = clock();
			camera_eye.x += keyboard_speed;
			camera_lookat.x += keyboard_speed;
			break;
		}
		case('k'):
		{
			trigger_moving_time = clock();
			camera_eye.y -= keyboard_speed;
			camera_lookat.y -= keyboard_speed;
			break;
		}
		case('i'):
		{
			trigger_moving_time = clock();
			camera_eye.y += keyboard_speed;
			camera_lookat.y += keyboard_speed;
			break;
		}
		case('j'):
		{
			trigger_moving_time = clock();
			camera_eye.z += keyboard_speed;
			camera_lookat.z += keyboard_speed;
			break;
		}
		case('l'):
		{
			trigger_moving_time = clock();
			camera_eye.z -= keyboard_speed;
			camera_lookat.z -= keyboard_speed;
			break;
		}
		case('3'):
		{
			keyboard_speed += 1.0f;
			printf("adding moving speed to %f\n", keyboard_speed);
			break;
		}
		case('4'):
		{
			keyboard_speed -= 1.0f;
			printf("reducing moving speed to %f\n", keyboard_speed);
			break;
		}
		case('y'): {
			const float2 from = { static_cast<float>(1),
				static_cast<float>(3) };
			const float2 to = { static_cast<float>(1),
				static_cast<float>(1) };

			const float2 a = { from.x / width, from.y / height };
			const float2 b = { to.x / width, to.y / height };
			camera_rotate = arcball.rotate(b, a);
			trigger_moving_time = clock();
			break;
		}
		case('h'): {
			const float2 from = { static_cast<float>(1),
				static_cast<float>(3) };
			const float2 to = { static_cast<float>(1),
				static_cast<float>(1) };

			const float2 a = { from.x / width, from.y / height };
			const float2 b = { to.x / width, to.y / height };
			camera_rotate = arcball.rotate(a, b);
			trigger_moving_time = clock();
			break;
		}
		case('t'): {
			camera_eye.x += keyboard_speed;
			trigger_moving_time = clock();
			break;
		}
		case('g'): {
			camera_eye.x -= keyboard_speed;
			trigger_moving_time = clock();
			break;
		}
		case('n'): {
			camera_eye.y -= keyboard_speed;
			trigger_moving_time = clock();
			break;
		}
		case('b'): {
			camera_eye.y += keyboard_speed;
			trigger_moving_time = clock();
			break;
		}
		case('c'): {
			camera_eye.z -= keyboard_speed;
			trigger_moving_time = clock();
			break;
		}
		case('v'): {
			camera_eye.z += keyboard_speed;
			trigger_moving_time = clock();
			break;
		}
		case('a'): {
			automatic_camera = (!automatic_camera);
			start_automatic_camera = clock();
			std::cerr << "start moving!\n";
			break;
		}
		case('9'): {
			record_camera_position = !record_camera_position;
			if (record_camera_position) {
				recording_camera_position_time = 0;
				clean_automatic_camera();
				start_recording_camera = clock();
				std::cerr << "start recording\n";
			}
			else
				std::cerr << "end recording\n";
			break;
		}
		case('0'): {
			play_camera_position = !play_camera_position;
			if (play_camera_position) {
				playing_camera_position_time = 0;
				start_playing_camera = clock();
				std::cerr << "start playing camera record!\n";
			}
			else
				std::cerr << "end playing record!\n";
			break;
		}
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
		mouse_clicking = true;
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = fminf( dmax, 0.9f );
        camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
		mouse_clicking = true;
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_rotate = arcball.rotate( b, a );
    }
	
    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer( getOutputBuffer(), width, height );

    glViewport(0, 0, width, height);

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help         Print this usage message and exit.\n"
        "  -f | --file         Save single frame to file and exit.\n"
        "  -n | --nopbo        Disable GL interop for display buffer.\n"
        "  -T | --tutorial-number <num>              Specify tutorial number\n"
        "  -t | --texture-path <path>                Specify path to texture directory\n"
        "App Keystrokes:\n"
        "  q  Quit\n"
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        << std::endl;

    exit(1);
}

int main( int argc, char** argv )
{
    std::string out_file;

	std::string mesh_file;
	//std::string mesh_file = std::string(sutil::samplesDir()) + "/data/fireplace_room_ppm/fireplace_room.obj";

	if (scene_breakfast) {
		mesh_file = std::string(sutil::samplesDir()) + "/data/San_Miguel_ppm/san-miguel.obj";
	}
	else if (scene_fireplace) {
		mesh_file = std::string(sutil::samplesDir()) + "/data/fireplace_room_ppm/fireplace_room.obj";
	}
	else {
		mesh_file = std::string(sutil::samplesDir()) + "/data/sponza_ppm/sponza.obj";
	}

    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if ( arg == "-f" || arg == "--file" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        } 
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            use_pbo = false;
        }
        else if ( arg == "-t" || arg == "--texture-path" )
        {
            if ( i == argc-1 ) {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            texture_path = argv[++i];
        }
        else if ( arg == "-T" || arg == "--tutorial-number" )
        {
            if ( i == argc-1 ) {
                printUsageAndExit( argv[0] );
            }
            tutorial_number = atoi(argv[++i]);
            if ( tutorial_number < 0 || tutorial_number > 11 ) {
                std::cerr << "Tutorial number (" << tutorial_number << ") is out of range [0..11]\n";
                printUsageAndExit( argv[0] );
            }
        }
		else if (arg == "-m" || arg == "--mesh")
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			mesh_file = argv[++i];
		}
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if( texture_path.empty() ) {
        texture_path = std::string( sutil::samplesDir() ) + "/data";
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        // load the ptx source associated with tutorial number
        std::stringstream ss;
        ss << "tutorial" << tutorial_number << ".cu";
        std::string tutorial_ptx_path = ss.str();
        tutorial_ptx = sutil::getPtxString( SAMPLE_NAME, tutorial_ptx_path.c_str() );

        createContext();
        
		//createGeometry();
		//setupCamera_simple();
		//setupLights_simple();
		
		loadMesh(mesh_file);
        setupCamera();
        setupLights();
		
		setup_automatic_camera();
		generate_ideal_camera();
        context->validate();

        if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

