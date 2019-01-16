#ifndef ARTIFICIAL_NN_H_
#define ARTIFICIAL_NN_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
	ANN_STATUS_OK = 0,
	ANN_STATUS_NULL_POINTER,
	ANN_STATUS_OUT_OF_RANGE,
	ANN_STATUS_OUT_OF_MEMORY,
	ANN_STATUS_MAX_LAYER_EXCEEDED,
	ANN_STATUS_NO_LAYERS,
	ANN_STATUS_INCOMPATIBLE
} AnnStatus;

typedef struct _Ann Ann;

AnnStatus ann_create(
	size_t max_layers, 
	Ann ** pp_ann);
	/* Brief: Create artificial neural network.
Input:
	max_layers - Maximum number of layers.
	pp_ann - Pointer to a variable to fill with new ANN pointer.
Output:
	*pp_ann - Should contain a pointer to a newly created ANN.
Return:
	ANN_STATUS_OK - Operation completed successfully.
	ANN_STATUS_NULL_POINTER - Null pointer passed.
	ANN_STATUS_OUT_OF_RANGE - Parameter 'max_layers' is out of range.
	ANN_STATUS_OUT_OF_MEMORY - Not enough memory to allocate the ANN.
*/

void ann_release(
	Ann ** pp_ann);
	/* Brief: Destroy ANN and free all buffers.
Input:
	pp_ann - A pointer to ANN pointer.
Output:
	*pp_ann should be freed if non-null.
	*pp_ann should be set to null.
Return:
	<none>
*/

AnnStatus ann_add(
	Ann * p_ann,
	size_t num_input,
	size_t num_output,
	const float * p_weight,
	const float * p_bias);
	/* Brief: Add layer and copy layer data into internal structures.
Input:
	p_ann - ANN pointer.
	num_input - Number of elements in input vector
	num_output - Number of elements in output vector
	p_weight - Weight matrix of size 'num_output' x 'num_input'.
	So it has 'num_output' rows and 'num_input' columns.
	Data is stored continuosly:
	W11, W12, ... W1n,
	W21, W22, ... W2n,
	...
	Wm1, Wm2, ... Wmn
	where m = num_output and n = num_input
	p_bias - Bias vector. It has 'num_output' elements.
Output:
	<none>
Return:
	ANN_STATUS_OK - Operation completed successfully.
	ANN_STATUS_NULL_POINTER - Null pointer passed.
	ANN_STATUS_OUT_OF_RANGE - Number of input or output elements
	is out of range.
	ANN_STATUS_OUT_OF_MEMORY - Not enough memory to allocate buffer and copy
	weight matrix or bias vector.
	ANN_STATUS_MAX_LAYER_EXCEEDED - No more layers can be added.
	ANN_STATUS_INCOMPATIBLE - Layer input is incompatible with previous
	layer output (size mismatch).
	*/

AnnStatus ann_forward(
	Ann * p_ann,
	size_t num_input,
	size_t num_output,
	const float * p_input,
	float * p_output);
	/* Brief: Perform 'forward' operation.
Input:
	p_ann - ANN pointer.
	num_input - Number of elements in input vector
	num_output - Number of elements in output vector
	p_input - Input vector
	p_output - Output vector
Output:
	*p_output - should be filled with result of forward operation
Return:
	ANN_STATUS_OK - Operation completed successfully.
	ANN_STATUS_NULL_POINTER - Null pointer passed.
	ANN_STATUS_OUT_OF_RANGE - Number of input or output elements
	is out of range.
	ANN_STATUS_OUT_OF_MEMORY - Not enough memory to allocate temporal buffer.
	ANN_STATUS_NO_LAYERS - No layers in the network. Call ann_add() first
	ANN_STATUS_INCOMPATIBLE - Input or output vector has incompatible size.
*/

#ifdef __cplusplus
}
#endif

#endif  // ARTIFICIAL_NN_H_