import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        z = np.asarray(z)
        return np.round(np.tanh(z @ self.G_W), 4).tolist()
    
    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        x = np.asarray(x)
        return np.round(self.sigmoid(x @ self.D_W), 4).tolist()
    
    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        # Your implementation here
        real_data = np.asarray(real_data)
        z = np.asarray(z)
        fake_data = self.generate(z)
        fake_data = np.asarray(fake_data)
        real_probs = np.asarray(self.discriminate(real_data))
        fake_probs = np.asarray(self.discriminate(fake_data))
        real_probs = np.clip(real_probs, a_min=1e-8, a_max=None)
        d_loss = np.round(-np.mean(np.log(real_probs) + np.log((1 - fake_probs).clip(min=1e-8, max=None))), 4)
        g_loss = np.round(-np.mean(np.log(fake_probs.clip(min=1e-8, max=None))), 4)

        return {"d_loss": d_loss, "g_loss": g_loss}
        
    def sigmoid(self, x):
        return np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(np.exp(x)+1))