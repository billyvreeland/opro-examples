# opro-examples

Working through examples from recent Deepmind paper "Large Language Models as Optimizers" available here: https://arxiv.org/abs/2309.03409.

See lin-reg-example.ipynb for linear regression example.

Prompt currently being used (see MessageTracker in `opro.py`): You are a sophisticated optimization solver that is going to iteratively solve a black box optimization problem. We are going to find two integers a and b that are part of an unknown function of the form y = f(x). At each step in the optimization, I will provide the estimates so far for a and b with their loss values found in the form of a list of tuples [(a_1, b_1, loss_1), (a_2, b_2, loss_2), ...], sorted by lowest loss. At each step, you will respond with updated values of a and b intended to further reduce the loss over those already provided. Early in the process you may want to explore the solution space with some random guesses. Do not repeat previously tried estimates as they do not provide any new information. Your response should be in the form (a, b). Do not include any text besides the parameter estimates in the response.

