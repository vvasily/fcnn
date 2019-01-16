# Fully Connected Neural Network

Fully Connected Neural Network implementation on C.

Please see `main.c` to set the settings of network.

## Getting Started

```
Build (all):
  make
  
Build (only network):
  make main
  
Build (only tests):
  make unittest
```

To build this code on VS just create new project and put them to it.

For VS 2015 it's needed to rename `*.c to *.cpp`

## Source code description

```
  AnnStatus ann_create(size_t max_layers, Ann ** pp_ann);
  Brief: Create artificial neural network.
  Input:
    max_layers - Maximum number of layers.
    pp_ann - Pointer to a variable to fill with new ANN pointer.
  Output:
    *pp_ann - Should contain a pointer to a newly created ANN.
  
  AnnStatus ann_add(Ann * p_ann, size_t num_input, size_t num_output, const float * p_weight, const float * p_bias);
  Brief: Add layer and copy layer data into internal structures.
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

  AnnStatus ann_forward(Ann * p_ann, size_t num_input, size_t num_output, const float * p_input, float * p_output);
  Brief: Perform 'forward' operation.
  Input:
    p_ann - ANN pointer.
    num_input - Number of elements in input vector
    num_output - Number of elements in output vector
    p_input - Input vector
    p_output - Output vector
  Output:
    *p_output - should be filled with result of forward operation

  void ann_release(Ann ** pp_ann);
   /* Brief: Destroy ANN and free all buffers.
  Input:
    pp_ann - A pointer to ANN pointer.
  Output:
    *pp_ann should be freed if non-null.
    *pp_ann should be set to null.
```
