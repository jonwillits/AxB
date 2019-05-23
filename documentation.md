
# Calculating Perplexity 

## Theoretical Minimum Sequence Perplexity

When,

- NUM_AB_TYPES = 2
- NUM_X_TRAIN_TYPES = 4
- NUM_X_TEST_TYPES = 8
- MAX_DISTANCE = 1
- MIN_DISTANCE = 1

Then, the theoretical minimum should be 13/8=1.625, where 13=2+4+1+1 (sum of pps at each position)


## Category Perplexity 

The initial value should be:

-np.log(1/master_vocab-size)*num_windows_where_target / num_windows

An example, given the hyper parameters above:
-np.log(1/13)*8 / 32 = 0.641
(8 is the number of times in the sequences where the answer is correct)

## Type Perplexity

The initial value should be:

-np.log(1/num_types_in_category)*num_windows_where_target / num_windows


An example, given the hyper parameters above, for types in A:
-np.log(1/2)*8/32 = 0.1732

