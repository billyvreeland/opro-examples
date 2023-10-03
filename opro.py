import numpy as np
import pandas as pd


class BlackBox:
    def __init__(self, a_true, b_true, std_dev=2, n_obs=100, x_min=-10, x_max=11):
        self.a_true = a_true
        self.b_true = b_true
        self.std_dev = std_dev
        self.n_obs = n_obs
        self.x_min = x_min
        self.x_max = x_max
        self.x = (self.x_max - self.x_min) * np.random.random(size=self.n_obs) + self.x_min
        self.y = (
            self.a_true * self.x +
            self.b_true +
            np.random.normal(scale=self.std_dev, size=self.n_obs)
        )
        self._create_loss_df()

    def calc_loss(self, a_est, b_est):
        y_est = a_est * self.x + b_est
        return np.sum((self.y - y_est) ** 2)

    def _create_loss_df(self):
        self.loss_df = pd.DataFrame({
            'a': np.random.randint(-10, 11, 10),
            'b': np.random.randint(-10, 11, 10)
        })
        self.loss_df['params'] = self.loss_df.apply(lambda row: (int(row.a), int(row.b)), axis=1)
        self.loss_df['loss'] = self.loss_df.apply(
            lambda row: self.calc_loss(row.a, row.b),
            axis=1
        )

    def append_loss_df(self, a_est, b_est):
        if (int(a_est), int(b_est)) not in self.loss_df.params:
            self.loss_df = pd.concat([
                self.loss_df,
                pd.DataFrame({
                    'a': [a_est],
                    'b': [b_est],
                    'loss': self.calc_loss(a_est, b_est)
                })
            ])
            self.loss_df.drop_duplicates(inplace=True)

    @property
    def best_params(self):
        self.loss_df.sort_values('loss', inplace=True)
        _df = self.loss_df.drop_duplicates()
        return [(int(row.a), int(row.b), int(row.loss)) for idx, row in _df.iterrows()]
    
    @property
    def best_loss(self):
        return self.loss_df.loss.min()
    

class MessageTracker:
    def __init__(self, initial_params):
        self.initial_params = initial_params
        self.messages = []
        self.sys_msg_content = """
            You are a sophisticated optimization solver that is going to iteratively solve a black box optimization problem. We are going to find two integers a and b that are part of an unknown function of the form y = f(x). At each step in the optimization, I will provide the estimates so far for a and b with their loss values found in the form of a list of tuples [(a_1, b_1, loss_1), (a_2, b_2, loss_2), ...], sorted by lowest loss. At each step, you will respond with updated values of a and b intended to further reduce the loss over those already provided. Early in the process you may want to explore the solution space with some random guesses. Do not repeat previously tried estimates as they do not provide any new information. Your response should be in the form (a, b). Do not include any text besides the parameter estimates in the response.
        """  # noqa: E501

        self.messages.append({
            'role': 'system',
            'content': self.sys_msg_content
        })
        self.messages.append({
            'role': 'user',
            'content': str(initial_params)
        })

    def append_response(self, response):
        self.messages.append({
            'role': 'assistant',
            'content': response
        })

    def append_params(self, params):
        self.messages.append({
            'role': 'user',
            'content': str(params)
        })

    def reiterate_specification(self):
        self.messages.append({
            'role': 'user',
            'content': 'Do not include any text besides the parameter estimates in the form (a, b) in the response.'  # noqa: E501
        })
