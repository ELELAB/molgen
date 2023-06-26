class KLAnnealer:
    """
    Class which implements KL Anealing.
    This updates beta depending on the epoch, which weights how much impact the kl divergence should have on the
    final loss function
    """

    def __init__(self, initial_beta, incriment_beta, beginning_epoch, max_beta):
        self.initial_beta = initial_beta
        self.incriment_beta = incriment_beta
        self.beginning_epoch = beginning_epoch
        self.max_beta = max_beta

    def calculate_beta(self, epoch):
        """
        DESCRIPTION:
            calculate beta calculates the beta given the current epoch.
        INPUT:
            epoch: INT: current epoch of training.
        OUTPUT:
            beta: The calculated beta.
        """

        if epoch >= self.beginning_epoch:
            beta = self.initial_beta + self.incriment_beta * ((epoch + 1) - self.beginning_epoch)
        else:
            beta = self.initial_beta
        return min(beta, self.max_beta)
