class KLAnnealer:
    """
    Class which implements KL Anealing.
    This updates beta depending on the epoch, which weights how much impact the kl divergence should have on the
    final loss function
    """

    def __init__(self, initial_beta, incriment_beta, increment_steps, beginning_step, max_beta):
        self.initial_beta = initial_beta
        self.incriment_beta = incriment_beta
        self.beginning_step = beginning_step
        self.max_beta = max_beta
        self.increment_steps = increment_steps

    def calculate_beta(self, steps):
        """
        DESCRIPTION:
            calculate beta calculates the beta given the current epoch.
        INPUT:
            steps: INT: current step of training.
        OUTPUT:
            beta: The calculated beta.
        """
        n_increments = steps // self.increment_steps

        if n_increments >= self.beginning_step:
            beta = self.initial_beta + self.incriment_beta * ((n_increments + 1) - self.beginning_step)
        else:
            beta = self.initial_beta
        return min(beta, self.max_beta)
