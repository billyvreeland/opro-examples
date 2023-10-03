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