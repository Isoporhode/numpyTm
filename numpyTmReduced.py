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

        # Negative is exluded literal, positive is included literal to the clause. Use msb to check what literal should be included and not.
        # Turns out that this will give less calculation than the inverted version, check out calculate_clauses_output
        self.states = np.random.choice([-1, 0], size=self.state_size).reshape(
            self.state_shape
        )

        # sign of clause
        self.clause_sign_treshold = self.number_of_clauses // 2

        # hardcoded for now
        self.feedback_type_y = (
            np.arange(self.number_of_clauses) < self.clause_sign_treshold
        )

    # probably slow, need to find a way to short circuit it (hopefully in numpy).
    def calculate_clauses_output(self, literals):

        # tries to find matches where literal and clauses matches. If they do, all values are true for that clause
        return np.all(
            literals
            | np.signbit(self.states),  # negative number -> exlude, positive -> include
            axis=1,
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
        # Since there's a lot of reshaping, the variables, will have a C in them if its pr clause, or F if its the full state size

        # part 1, Limited to class_sum
        clauses_evaluated_C = self.calculate_clauses_output(literals)
        class_sum = self.class_sum(clauses_evaluated_C)

        # part 2: calculates reward and punish of literal inclusion
        # fixes both (T+v)/(2T) and (T-v)/(2T), as both are dependent on y, and symmetrical

        prob_pos = (class_sum + self.threshold) / (2 * self.threshold)
        sum_tar_C = (prob_pos > np.random.rand(self.number_of_clauses)) ^ ~y

        # [0,0,0 ... 1, 1, 1] if y = 1 [1,1,1 ... 0, 0, 0] if y = 0 (can just do a lookup for this one)
        # I think my invert comes from here. Check this later
        feedback_type_C = y ^ self.feedback_type_y

        # generate s values
        s_inv_F = (
            (1 / self.s > np.random.rand(self.state_size)).reshape(self.state_shape).T
        )

        literals_F = np.tile(literals, (self.number_of_clauses, 1)).T

        reward_short = (
            clauses_evaluated_C
            & sum_tar_C
            & ~(feedback_type_C ^ literals_F)
            & ~(literals_F & s_inv_F)
        )

        type_I_p = (
            feedback_type_C
            & sum_tar_C
            & s_inv_F
            & ((clauses_evaluated_C & ~literals_F) | ~clauses_evaluated_C)
        )

        # somehow a bit slower? Needs more testing
        # type_I_p = (
        #     feedback_type_C
        #     & sum_tar_C
        #     & s_inv_F
        #     & ~(clauses_evaluated_C & literals_F)
        # )

        reward = reward_short.astype(dtype=np.int32)
        punish = type_I_p.astype(dtype=np.int32)

        self.states += (reward - punish).T
        self.states = np.clip(
            self.states, a_min=-self.number_of_states, a_max=self.number_of_states
        )

    def evaluate(self, X_all: list[int, int], y_all: list[int]):

        bool_X_all = X_all > 0
        literals_all = np.concat((bool_X_all.T, ~bool_X_all.T)).T
        bool_y_all = y_all > 0

        errors = 0
        for l in range(bool_y_all.shape[0]):
            literals = literals_all[l]
            clauses_evaluated = self.calculate_clauses_output(literals)
            output_sum = self.class_sum(clauses_evaluated)

            if output_sum >= 0 and bool_y_all[l] == 0:
                errors += 1

            elif output_sum < 0 and bool_y_all[l] == 1:
                errors += 1

        return 1.0 * errors / bool_y_all.shape[0]
