#ifndef NEURON_H_
#define NEURON_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct _Axon
{
	float f_weight;
};

typedef struct _Axon Axon;

struct _Neuron
{
	unsigned n_index;
	Axon* p_list_axons;
	float f_output;
	float f_output_sum;
};

typedef struct _Neuron Neuron;

struct _Layer
{
	Neuron** p_neuron_list;
	size_t n_size;
	size_t n_out_count;
	float* p_bias;
};

typedef struct _Layer Layer;

struct _Ann
{
	Layer** pp_layers;
	size_t n_layers_count;
	size_t n_max_layers;
};

void neuron_create(Neuron** p_neuron, unsigned n_index, unsigned n_axons, const float * p_weight, unsigned num_input);
void neuron_forward(Neuron* p_neuron, const Layer* p_previous_layer);
float neuron_sigmoid(float f_x);

#ifdef __cplusplus
}
#endif

#endif  // NEURON_H_
