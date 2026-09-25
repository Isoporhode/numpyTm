# Based on https://github.com/cair/TsetlinMachine/blob/master/TsetlinMachine.pyx and https://arxiv.org/pdf/1804.01508

import numpy as np


class NumpyTsetlinMachine:
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

        self.ta_state = np.random.choice(
            [self.number_of_states, self.number_of_states + 1],
            size=(self.number_of_clauses, self.number_of_features, 2),
        ).astype(dtype=np.int32)

        self.clause_sign = np.zeros(self.number_of_clauses, dtype=np.int32)
        for j in range(self.number_of_clauses):
            if j % 2 == 0:
                self.clause_sign[j] = 1
            else:
                self.clause_sign[j] = 0

        self.clause_shape = [self.number_of_clauses, self.number_of_features, 2]
        self.clause_shape_size = self.number_of_clauses * self.number_of_features * 2

    def C_to_L_reshape(self, arr):
        one_test = np.ones(self.clause_shape).T
        combined = (one_test * arr).T > 0
        return combined

    def L_to_C_reshape(self, arr):
        one_test = np.ones(self.clause_shape)
        combined = (one_test * arr) > 0
        return combined

    def calculate_clauses_output(self, literals, clauses):
        # will only give true, if the "clause & literals" output is true
        # tries to find matches where literal and clauses matches. If they do, all values are true for that clause
        # print(clauses.shape, literals.shape)
        resloved = literals | ~clauses
        return np.all(
            np.reshape(
                resloved, shape=[self.number_of_clauses, self.number_of_features * 2]
            ),
            axis=1,
        )

    def predict(self, X: list[int]) -> int:
        clauses = self.ta_state > self.number_of_states
        my_clause_sign = self.clause_sign > 0

        X_bool = np.array(X, dtype="bool")
        literals = np.vstack((X_bool, ~X_bool)).T
        clauses_evaluated = self.calculate_clauses_output(literals, clauses)
        output_sum = self.my_sum_up_clause_votes(clauses_evaluated, my_clause_sign)

        if output_sum >= 0:
            return 1
        else:
            return 0

    def evaluate(self, X: list[int, int], y: list[int]):
        number_of_examples = y.shape[0]
        clauses = self.ta_state > self.number_of_states
        my_clause_sign = self.clause_sign > 0
        errors = 0
        for l in range(number_of_examples):

            X_bool = np.array(X[l], dtype="bool")
            literals = np.vstack((X_bool, ~X_bool)).T
            clauses_evaluated = self.calculate_clauses_output(literals, clauses)
            output_sum = self.my_sum_up_clause_votes(clauses_evaluated, my_clause_sign)

            if output_sum >= 0 and y[l] == 0:
                errors += 1

            elif output_sum < 0 and y[l] == 1:
                errors += 1

        return 1.0 - 1.0 * errors / number_of_examples

    def fit(self, X: list[int, int], y: list[int], epochs=100) -> None:

        number_of_examples = y.shape[0]
        Xi = np.zeros((self.number_of_features,), dtype=np.int32)

        random_index = np.arange(number_of_examples)

        for epoch in range(epochs):
            np.random.shuffle(random_index)

            for l in range(number_of_examples):
                example_id = random_index[l]
                target_class = y[example_id]

                for j in range(self.number_of_features):
                    Xi[j] = X[example_id, j]
                self.update(Xi, target_class)
        return

    def my_sum_up_clause_votes(self, clauses_evaluated, my_clause_sign) -> int:
        output_pos = my_clause_sign & clauses_evaluated
        output_neg = ~my_clause_sign & clauses_evaluated
        return np.sum(output_pos) - np.sum(
            output_neg
        )  # det finnes sikkert en bedre måte å gjøre denne på

    def my_sum_up_clause_votes_save(self, clauses_evaluated, my_clause_sign) -> int:
        output_pos = np.sum(my_clause_sign & clauses_evaluated)
        output_neg = np.sum(~my_clause_sign & clauses_evaluated)
        return output_pos, output_neg

    def sum_tar_select(self, group_sum):
        T = self.threshold
        if group_sum < -T:
            group_sum = -T
        if group_sum > T:
            group_sum = T

        prob_pos = (group_sum + self.threshold) / (2 * self.threshold)
        self.sum_tar_pos_l = prob_pos > np.random.rand(self.number_of_clauses)
        self.sum_tar_neg_l = np.random.rand(self.number_of_clauses) > prob_pos

    # def sum_tar_select_no_T(self, output_pos, output_neg):
    #       # Does not work well on other datasets. Only gets up to ~93.3 acc (test) and 95.6 (train) on binary iris for multiclass with
    #      # s = 80, number_of_clauses = 300 states = 100
    #
    #     prob_pos = 0.5
    #     if not output_pos == 0 and output_pos == 0:
    #         prob_pos = output_pos / (output_pos + output_neg)
    #     self.sum_tar_pos_l = prob_pos > np.random.rand(self.number_of_clauses)
    #     self.sum_tar_neg_l = np.random.rand(self.number_of_clauses) > prob_pos

    def rand_s_generator(self):
        # can probably just use one of these arrays, and invert it when i have to
        s_inv = 1 / self.s
        self.s_inv_l = (s_inv > np.random.rand(self.clause_shape_size)).reshape(
            self.clause_shape
        )
        self.s_inv_neg_l = (np.random.rand(self.clause_shape_size) > s_inv).reshape(
            self.clause_shape
        )

    def update(self, X: list[int], y: int) -> None:

        self.rand_s_generator()

        X_bool = np.array(X, dtype="bool")
        literals = np.vstack((X_bool, ~X_bool)).T  # sjekk transposen på denne
        clauses = self.ta_state > self.number_of_states

        my_clauses_evaluated = self.calculate_clauses_output(literals, clauses)

        my_clause_sign = self.clause_sign > 0

        my_output_sum = self.my_sum_up_clause_votes(
            my_clauses_evaluated, my_clause_sign
        )
        self.sum_tar_select(group_sum=my_output_sum)

        # setup for creating types of feedback and sum_tars
        bool_y = bool(y)
        feedback_type = ~(bool_y ^ my_clause_sign)  # 0 is type II 1 is type I
        sum_tar = (bool_y & self.sum_tar_neg_l) | (
            np.invert(bool_y) & self.sum_tar_pos_l
        )
        sum_tar_r = self.C_to_L_reshape(sum_tar)
        type_I_fb_sum_tar = self.C_to_L_reshape(feedback_type & sum_tar)
        clauses_evaluated_r = self.C_to_L_reshape(my_clauses_evaluated)
        feedback_type_r = self.C_to_L_reshape(feedback_type)

        # # Type I reward
        # type_I_r = type_I_fb_sum_tar & clauses_evaluated_r & self.s_inv_neg_l & literals

        # # Type 2 reward
        # type_II_r = (
        #     type_II_fb_sum_tar & ~literals & clauses_evaluated_r
        # )

        # C - Clause evaluates to
        # Z - sum_tar
        # G - Feedback type
        # L - Literal
        # S - 1-1/s
        # C⋅~Z⋅((~G⋅~L)+(G⋅L⋅S)) is equivalent to Type I | Type II reward feedback:
        # Type I | Type II => (G & ~Z & C & S & L)|(~G & ~Z& ~L & C) => C & ~Z & ((~G & ~L)|(G & L & S))

        reward_short = (
            clauses_evaluated_r
            & sum_tar_r
            & (
                (~feedback_type_r & ~literals)
                | (feedback_type_r & literals & self.s_inv_neg_l)
            )
        )

        # (G∧C∧S)∨(~G∧C∧~l)

        # (G∧~Z∧C∧S∧L)∨(~G∧~Z∧~L∧C)

        # # Type I reward
        # type_I_r = type_I_fb_sum_tar & clauses_evaluated_r & self.s_inv_neg_l & literals

        # # Type 2 reward
        # type_II_r = (
        #     type_II_fb_sum_tar & ~literals & clauses_evaluated_r
        # )

        # Type 1 punish
        type_I_p = (type_I_fb_sum_tar & self.s_inv_l) & (
            (clauses_evaluated_r & ~literals) | ~clauses_evaluated_r
        )

        reward = (reward_short).astype(dtype=np.int32)
        punish = type_I_p.astype(dtype=np.int32)

        self.ta_state += reward - punish
        self.ta_state = np.clip(self.ta_state, a_min=1, a_max=self.number_of_states * 2)
