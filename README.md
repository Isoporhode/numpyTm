# numpyTm
Tsetlin machine implementation with focus on numpy and boolean algebra - In progress



Combined reward types:

        # # Type I reward
        # type_I_r = type_I_fb_skip & clauses_evaluated_r & self.s_inv_neg_l & literals

        # # Type 2 reward
        # type_II_r = (
        #     type_II_fb_skip & ~literals & clauses_evaluated_r
        # )  # denne er grei! (tror jeg)

        # C⋅~Z⋅((~G⋅~L)+(G⋅L⋅S))
        # equivalent to Type I | Type II reward feedback
        # C - Clause evaluates to
        # Z - skip
        # G - Feedback type
        # L - Literal
        # S - 1-1/s

        # Type I | Type II => (G∧~Z∧C∧S∧L)∨(~G∧~Z∧~L∧C) => C⋅~Z⋅((~G⋅~L)+(G⋅L⋅S))


Eliminating T:

pos_probability = positive/(positive+negative)

if positive+negative == 0:
    pos_probability = 0.5
