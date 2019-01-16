#include "neuron.h"
#include <math.h>
#include <stdlib.h>

void neuron_create(Neuron** p_neuron, unsigned n_index, unsigned n_axons, const float * p_weight, unsigned n_num_input)
{
	*p_neuron = (Neuron*)calloc(1, sizeof(Neuron));
	
	if(*p_neuron != NULL)
	{
		(*p_neuron)->n_index = n_index;
		(*p_neuron)->f_output = 0.0;
	
		Axon* p_list_axons = (Axon*)calloc(n_axons, sizeof(Axon));
		if (p_list_axons != NULL)
		{
			for (unsigned i = 0; i < n_axons; i++)
			{
				p_list_axons[i].f_weight = p_weight[i*n_num_input + n_index];
			}
			(*p_neuron)->p_list_axons = p_list_axons;
		}
		else
		{
			free(*p_neuron);
			*p_neuron = NULL;
		}
	}
}

void neuron_forward(Neuron* p_neuron, const Layer* p_previous_layer)
{
	p_neuron->f_output_sum = 0.0;

	/* Ñalculate the sum of inputs * weights */
	for (unsigned i = 0; i < p_previous_layer->n_size; i++)
	{
		p_neuron->f_output_sum += p_previous_layer->p_neuron_list[i]->f_output * p_previous_layer->p_neuron_list[i]->p_list_axons[p_neuron->n_index].f_weight;
	}

	/* Calculate sigmoid function */
	p_neuron->f_output = neuron_sigmoid(p_neuron->f_output_sum + p_previous_layer->p_bias[p_neuron->n_index]);
}

float neuron_sigmoid(float f_x)
{
	return (1/(1 + expf(-f_x)));
}
