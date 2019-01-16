//#include "add you function"
#include "gtest/gtest.h"
#include "ann.h"
#include "neuron.h"

class RunnerTest : public testing::Test
{
protected:
	void SetUp() override {}
	void TearDown() override {}
};

TEST_F(RunnerTest, ann_create)
{
	size_t max_layers = 5;
	Ann** pp_ann = NULL;
	EXPECT_EQ(ANN_STATUS_NULL_POINTER, ann_create(max_layers, pp_ann));
  
	max_layers = -1;
	EXPECT_EQ(ANN_STATUS_OUT_OF_RANGE, ann_create(max_layers, pp_ann));
	
	max_layers = SIZE_MAX;
	EXPECT_EQ(ANN_STATUS_OUT_OF_RANGE, ann_create(max_layers, pp_ann));
}

TEST_F(RunnerTest, ann_add)
{
	Ann** pp_ann = (Ann**)calloc(1, sizeof(Ann*));
	(void)ann_create(2, pp_ann);

	const float p_weight[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f };
	const float p_bias[] = { 0.1f, 0.2f, 0.3f, 0.4f };
	EXPECT_EQ(ANN_STATUS_NULL_POINTER, ann_add(NULL, 3, 4, p_weight, p_bias));
	
	EXPECT_EQ(ANN_STATUS_OK, ann_add(*pp_ann, 3, 4, p_weight, p_bias));
	
	EXPECT_EQ(ANN_STATUS_INCOMPATIBLE, ann_add(*pp_ann, 3, 4, p_weight, p_bias));

	EXPECT_EQ(ANN_STATUS_OUT_OF_RANGE, ann_add(*pp_ann, 0, 4, p_weight, p_bias));

	const float p_weight_2[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f };
	const float p_bias_2[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
	EXPECT_EQ(ANN_STATUS_OK, ann_add(*pp_ann, 4, 5, p_weight_2, p_bias_2));

	EXPECT_EQ(ANN_STATUS_MAX_LAYER_EXCEEDED, ann_add(*pp_ann, 4, 5, p_weight_2, p_weight_2));

	ann_release(pp_ann);
}

TEST_F(RunnerTest, ann_forward)
{
	Ann** pp_ann = (Ann**)calloc(1, sizeof(Ann*));
	(void)ann_create(5, pp_ann);
	
	size_t num_output = 5;
	const float p_input[] = { 1, 2, 3 };
	float* p_output = (float*)calloc(num_output, sizeof(float));

	EXPECT_EQ(ANN_STATUS_NO_LAYERS, ann_forward(*pp_ann, 3, num_output, p_input, p_output));
	
	const float p_weight[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f };
	const float p_bias[] = { 0.1f, 0.2f, 0.3f, 0.4f };
	EXPECT_EQ(ANN_STATUS_OK, ann_add(*pp_ann, 3, 4, p_weight, p_bias));

	const float p_weight_1[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f };
	const float p_bias_1[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
	EXPECT_EQ(ANN_STATUS_OK, ann_add(*pp_ann, 4, 5, p_weight_1, p_bias_1));
	
	EXPECT_EQ(ANN_STATUS_OK, ann_forward(*pp_ann, 3, num_output, p_input, p_output));
		
	EXPECT_EQ(ANN_STATUS_OUT_OF_RANGE, ann_forward(*pp_ann, 0, num_output, p_input, p_output));
	
	EXPECT_EQ(ANN_STATUS_INCOMPATIBLE, ann_forward(*pp_ann, 3, num_output + 1, p_input, p_output));

	ann_release(pp_ann);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


