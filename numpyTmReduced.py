# Based on https://arxiv.org/pdf/1804.01508

import numpy as np


class NumpyTsetlinMachineR:
    def __init__(
        self,
        number_of_clauses: int,
        number_of_features: int,
        number_of_states: int,
        s: float,
        threshold: int,
    ):

        self.number_of_clauses = number_of_clauses
        self.number_of_features = number_of_features
        self.number_of_states = number_of_states
        self.s = s
        self.threshold = threshold

        self.state_shape = [number_of_clauses, number_of_features * 2]
        self.state_size = number_of_clauses * number_of_features * 2

        # negative is exluded literal, positive is included literal to the clause (invert this for less calculations?)
        self.states = np.random.choice([-1, 0], size=self.state_size).reshape(
            self.state_shape
        )

        # sign of clause
        self.clause_sign_treshold = self.number_of_clauses // 2

        # hardcoded for now
        self.feedback_type_y = (
            np.arange(self.number_of_clauses) < self.clause_sign_treshold
        )

    # perhaps we can remove these?
    def C_to_L_reshape(self, arr):
        one_test = np.ones(self.state_shape).T
        combined = (one_test * arr).T > 0
        return combined

    def L_to_C_reshape(self, arr):
        one_test = np.ones(self.state_shape)
        combined = (one_test * arr) > 0
        return combined

    # probably slow
    def calculate_clauses_output(self, literals, clauses):
        # tries to find matches where literal and clauses matches. If they do, all values are true for that clause
        resloved = literals | ~clauses
        return np.all(
            resloved,
            axis=1,
        )

    def sum_tar_select_pos(self, group_sum):
        prob_pos = (group_sum + self.threshold) / (2 * self.threshold)
        return prob_pos > np.random.rand(self.number_of_clauses)

    def rand_s_generator(self):
        s_inv = 1 / self.s
        self.s_inv_l = (s_inv > np.random.rand(self.state_size)).reshape(
            self.state_shape
        )

    def fit(self, X_all, y_all, epochs):
        rng = np.random.default_rng()
        bool_X_all = X_all > 0
        bool_y_all = y_all > 0
        literals_and_y = np.concat(
            (bool_X_all.T, ~bool_X_all.T, np.atleast_2d(bool_y_all))
        ).T

        for i in range(epochs):
            rng.shuffle(literals_and_y)
            # potentially slow, but we can only do one and one example at a time anyways
            for literal_and_y in literals_and_y:
                y = literal_and_y[-1]
                literals = literal_and_y[:-1]
                self.update(literals, y)
            print("epoch", i, "of", epochs)

    def class_sum(self, clauses_evaluated):
        return np.count_nonzero(
            clauses_evaluated[: self.clause_sign_treshold]
        ) - np.count_nonzero(clauses_evaluated[self.clause_sign_treshold :])

    def update(self, literals, y):
        clauses = ~np.signbit(
            self.states
        )  # negative number -> exlude, positive -> include
        clauses_evaluated = self.calculate_clauses_output(literals, clauses)
        class_sum = self.class_sum(clauses_evaluated)
        sum_tar = self.sum_tar_select_pos(class_sum) ^ ~y  # her er det en feil lol
        feedback_type = (
            y ^ self.feedback_type_y
        )  # [0,0,0 ... 1, 1, 1] if y = 1 [1,1,1 ... 0, 0, 0] if y = 0
        self.rand_s_generator()

        # a bunch of reshaping - See if we can reduce this stuff to a minimum with broadcast
        sum_tar_r = self.C_to_L_reshape(sum_tar)
        type_I_fb_sum_tar = self.C_to_L_reshape(feedback_type & sum_tar)
        clauses_evaluated_r = self.C_to_L_reshape(clauses_evaluated)
        feedback_type_r = self.C_to_L_reshape(feedback_type)

        reward_short = (
            clauses_evaluated_r
            & sum_tar_r
            & (
                (~feedback_type_r & ~literals)
                | (feedback_type_r & literals & ~self.s_inv_l)
            )
        )

        type_I_p = (type_I_fb_sum_tar & self.s_inv_l) & (
            (clauses_evaluated_r & ~literals) | ~clauses_evaluated_r
        )

        reward = reward_short.astype(dtype=np.int32)
        punish = type_I_p.astype(dtype=np.int32)

        self.states += reward - punish
        self.states = np.clip(
            self.states, a_min=-self.number_of_states, a_max=self.number_of_states
        )

    def evaluate(self, X_all: list[int, int], y_all: list[int]):

        bool_X_all = X_all > 0
        literals_all = np.concat((bool_X_all.T, ~bool_X_all.T)).T
        bool_y_all = y_all > 0

        clauses = ~np.signbit(
            self.states
        )  # negative number -> exlude, positive -> include

        errors = 0
        for l in range(bool_y_all.shape[0]):
            literals = literals_all[l]
            clauses_evaluated = self.calculate_clauses_output(literals, clauses)
            output_sum = self.class_sum(clauses_evaluated)

            if output_sum >= 0 and bool_y_all[l] == 0:
                errors += 1

            elif output_sum < 0 and bool_y_all[l] == 1:
                errors += 1

        return (
            1.0 * errors / bool_y_all.shape[0]
        )  # somehow I got the 1 - accuracy out from my model now
