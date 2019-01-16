#include "ann.h"
#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	AnnStatus status = ANN_STATUS_OK;
	Ann** pp_ann = (Ann**)calloc(1, sizeof(Ann*));
	status = ann_create(5, pp_ann);
	printf("Create status = %d\n", status);

	const float p_weight[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f };
	const float p_bias[] = { 0.1f, 0.2f, 0.3f, 0.4f };
	status = ann_add(*pp_ann, 3, 4, p_weight, p_bias);
	printf("Add layer status = %d\n", status);

	const float p_weight_1[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f };
	const float p_bias_1[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
	status = ann_add(*pp_ann, 4, 5, p_weight_1, p_bias_1);
	printf("Add layer status = %d\n", status);

	const float p_input[] = { 1, 2, 3 };
	unsigned num_output = 5;
	float* p_output = (float*)calloc(num_output, sizeof(float));
	status = ann_forward(*pp_ann, 3, num_output, p_input, p_output);
	printf("Forward status = %d\n", status);

	for (unsigned i = 0; i < num_output; i++)
	{
		printf("Output[%d] = %.4f\n", i, p_output[i]);
	}

	ann_release(pp_ann);
	return 0;
}
