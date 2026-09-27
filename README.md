# numpyTm
Tsetlin machine implementation with focus on numpy and boolean algebra - proof of concept


## Structure
This repo currently contains two implementations of the original binary classifier tsetlin machine with focus on using as few for loops as possiable, while utelizing numpy. numpyTm is closer to the original paper, while numpyTmReduced is numpyTm, but with some optimizations and some changes to the logic, while still prodicing the same result.

## Main ideas

### Using boolean algebra

Since the Tsetlin machine is so closely related to boolean algebra, have I tried to use it as much as possiable. An if else is therefore for example represeted as a multiplexer with the condition choosing the signal used. 

### Random variables

The random variables in this repo is for the most part generated all at once. I.e. random()<1/s can be represented as a precalculated matrix, one for each state in each clause. The same is done with (T-v)/(2T) and so on. By doing this this repo treats the random variables as any other boolean variable.

### Reducing feedback table to boolean algebra

When random variables are expressed as boolean variables, feedback I and II can also be described as boolean variables.
Feedbakck type II is for example represented as something like:

```python
(y ^ clause_sign) & ( y & sum_tar_neg | ~y & sum_tar_pos) & clause_evaluated & ~literal
```

Where ```(y ^ clause_sign)``` checks if the polarity of a clause any y is not the same so that type II feedback should be used,

The section ```( y & sum_tar_neg | ~y & sum_tar_pos)``` is a multiplexer, where y is used to choose the signal. If y = 1, then sum_tar_neg is chosen.

```sum_tar_neg``` is a precalculated array of random variables from ```random()<(T-v)/(2T)```, where v is the evaluated som of positive and negatve clauses on the literal set.

Anding with ```clause_evaluated``` is just of the clause evaluates to 1, and the same with inverted ```literal```, as it is shown in the feedback type II table.

From this punish only comes from type I feedback, while reward both comes from type I and II feedback and are combined with a simple or

Note that this boolean algebra is then again reduced to decrease the number of operations.

### Checking if a clause evaluates to 1

This is done with literal + ~clause, and then checking if the result contains only ones for each clause. There are probably faster ways to do this, but this works for now

### Reward and punish

two matrices are produced in the end which represent what states should be increaced and what states should be decreased and are added and subtracted from the ta state matrix. Note that the punish and reward matrices doesn't overlap values. If one of them have a true value for a state, the other is false.

### Reducing even further in numpyTmReduced

It turns out that with some small changes one can probably calculate the boolean algebra a bit easier. First off if the tsetlin automatas states have negative values as exlude literal and positive as include, one can just check one bit (probably msb as this is probably in 2's complimnet). Numpy got a function for that, and it's pretty fast!

It also turns out that since both the feedback sum clip function and the 1/s functions are symmetrical with their counterparts, one only needs to generate one of each, and then invert them, when the values are needed.

To reduce complecity the polarity of the clauses are now not dependent if they are in an odd or even number index, but rather below half or above half of the clause size.

