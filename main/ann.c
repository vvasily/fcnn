#include "ann.h"
#include "neuron.h"
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>

AnnStatus ann_create(size_t max_layers, Ann ** pp_ann)
{
	AnnStatus status = ANN_STATUS_OK;
	
	if ((max_layers > 0) && (max_layers < SIZE_MAX))
	{
		if (pp_ann != NULL)
		{
			*pp_ann = (Ann*)calloc(1, sizeof(Ann));
			if (*pp_ann != NULL)
			{
				(*pp_ann)->n_max_layers = max_layers;
				(*pp_ann)->n_layers_count = 0;
				(*pp_ann)->pp_layers = NULL;
			}
			else
			{
				status = ANN_STATUS_OUT_OF_MEMORY;
			}
		}
		else
		{
			status = ANN_STATUS_NULL_POINTER;
		}
	}
	else
	{
		status = ANN_STATUS_OUT_OF_RANGE;
	}

	return status;
}

AnnStatus ann_add(Ann * p_ann, size_t num_input, size_t num_output, const float * p_weight, const float * p_bias)
{
	AnnStatus status = ANN_STATUS_OK;
		
	if ((p_ann != NULL) && 
		(p_weight != NULL) && 
	(	p_bias != NULL))
	{
		if (p_ann->n_layers_count < p_ann->n_max_layers)
		{
			if ((num_input > 0) && 
				(num_output > 0) &&
				(num_input < SIZE_MAX) &&
				(num_output < SIZE_MAX))
			{
				
				if ((p_ann->n_layers_count > 0) && 
					(p_ann->pp_layers[p_ann->n_layers_count - 1]->n_out_count != num_input))
				{
					status = ANN_STATUS_INCOMPATIBLE;
				}
				else
				{
					Layer* p_layer = (Layer*)calloc(1, sizeof(Layer));
					p_layer->p_neuron_list = (Neuron**)calloc(num_input, sizeof(Neuron*));
					p_layer->p_bias = (float*)calloc(num_output, sizeof(float));
					p_layer->n_out_count = num_output;
					
					if ((p_layer == NULL) || 
						(p_layer->p_neuron_list == NULL) || 
						(p_layer->p_bias == NULL))
					{
						status = ANN_STATUS_OUT_OF_MEMORY;
					}
					else
					{
						p_layer->n_size = 0;
	
						unsigned n_axons = num_output;
	
						for (unsigned i = 0; i < num_input; i++)
						{
							Neuron* p_neuron = NULL;
							neuron_create(&p_neuron, i, n_axons, p_weight, num_input);
							if(p_neuron != NULL)
							{
								p_layer->p_neuron_list[i] = p_neuron;
								p_layer->n_size++;
							}
							else
							{
								status = ANN_STATUS_OUT_OF_MEMORY;
								break;
							}
						}
						
						if (status == ANN_STATUS_OK)
						{
							for (unsigned i = 0; i < num_output; i++)
							{
								p_layer->p_bias[i] = p_bias[i];
							}
							p_ann->n_layers_count++;
							p_ann->pp_layers = (Layer**)realloc(p_ann->pp_layers, p_ann->n_layers_count * sizeof(Layer*));
							if (p_ann->pp_layers != NULL)
							{
								p_ann->pp_layers[p_ann->n_layers_count -1] = p_layer;
							}
							else
							{
								status = ANN_STATUS_OUT_OF_MEMORY;
							}
						}
					}
				}
			}
			else
			{
				status = ANN_STATUS_OUT_OF_RANGE;
			}	
		}
		else
		{
			status = ANN_STATUS_MAX_LAYER_EXCEEDED;
		}
	}
	else
	{
		status = ANN_STATUS_NULL_POINTER;
	}
	return status;
}
	
AnnStatus ann_forward(Ann * p_ann, size_t num_input, size_t num_output, const float * p_input, float * p_output)
{
	AnnStatus status = ANN_STATUS_OK;
	
	if ((p_ann != NULL) && (p_input != NULL))
	{
		if ((num_input > 0) && 
			(num_output > 0) &&
			(num_input < SIZE_MAX) &&
			(num_output < SIZE_MAX))
		{
			unsigned n_layers_size = p_ann->n_layers_count;
			
			if (n_layers_size > 0)
			{
				if ((p_ann->pp_layers[0]->n_size == num_input) &&
					(p_ann->pp_layers[n_layers_size - 1]->n_out_count == num_output))
				{
					// Set outputs for input layers neurons
					for (unsigned i = 0; i < num_input; i++)
					{
						p_ann->pp_layers[0]->p_neuron_list[i]->f_output = p_input[i];
					}
	
					// Forward
					for (unsigned i = 1; i < n_layers_size; i++)
					{
						Layer* p_layer = p_ann->pp_layers[i];
						for (unsigned j = 0; j < p_layer->n_size; j++)
						{
							neuron_forward(p_layer->p_neuron_list[j], p_ann->pp_layers[i - 1]);
						}
					}
	
					// Output calculate
					if (p_output != NULL)
					{
						Layer* p_last_layer = p_ann->pp_layers[n_layers_size - 1];
	
						for (unsigned i = 0; i < num_output; i++)
						{
							Neuron neuron;
							neuron.n_index = i;
							neuron_forward(&neuron, p_last_layer);
							p_output[i] = neuron.f_output;
						}
					}
					else
					{
						status = ANN_STATUS_OUT_OF_MEMORY;
					}
				}
				else
				{
					status = ANN_STATUS_INCOMPATIBLE;
				}
			}
			else
			{
				status = ANN_STATUS_NO_LAYERS;
			}
		}
		else
		{
			status = ANN_STATUS_OUT_OF_RANGE;
		}
	}
	else
	{
		status = ANN_STATUS_NULL_POINTER;
	}
	return status;
}

void ann_release(Ann ** pp_ann)
{
	for (unsigned i = 0; i < (*pp_ann)->n_layers_count; i++)
	{
		Layer* p_layer = (*pp_ann)->pp_layers[i];
		for (unsigned j = 0; j < p_layer->n_size; j++)
		{
			Axon* list_axons = p_layer->p_neuron_list[j]->p_list_axons;
			if (list_axons != NULL)
			{
				free(list_axons);
				list_axons = NULL;
			}
			if (p_layer->p_neuron_list[j] != NULL)
			{
				free(p_layer->p_neuron_list[j]);
				p_layer->p_neuron_list[j] = NULL;
			}
		}
		if (p_layer->p_neuron_list != NULL)
		{
			free(p_layer->p_neuron_list);
			p_layer->p_neuron_list = NULL;
		}
		if (p_layer->p_bias != NULL)
		{
			free(p_layer->p_bias);
			p_layer->p_bias = NULL;
		}
		if (p_layer)
		{
			free(p_layer);
			p_layer = NULL;
		}
	}
	
	if ((*pp_ann)->pp_layers != NULL)
	{
		free((*pp_ann)->pp_layers);
		(*pp_ann)->pp_layers = NULL;
	}
	if ((*pp_ann) != NULL)
	{
		free(*pp_ann);
		(*pp_ann) = NULL;
	}
}
