import torch
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from deepLFM.likelihoods import Gaussian
from deepLFM.utils import DKL_Gaussian

class deepLFM(torch.nn.Module):
    def __init__(self, d_in, d_out, n_hidden_layers=1, n_lfm=2, n_rff=20, n_lf=2, mc=100,
                 q_Omega_fixed_epochs=0, q_theta_fixed_epochs=0, local_reparam=True, feed_forward=True):
        """ PyTorch implementation of the deep latent force model. The underlying architecture is a
            deep Gaussian process with random feature expansions, and in this case we derive these
            random Fourier features from an ODE1 LFM kernel.

        :param d_in: Input dimensionality
        :param d_out: Output dimensionality
        :param n_hidden_layers: Number of hidden layers
        :param n_lfm: Dimensionality of hidden layers
        :param n_rff: No. random features per latent force
        :param n_lf: No. latent forces per node
        :param mc: Number of Monte Carlo samples
        :param q_Omega_fixed_epochs: Number of epochs to fix Omega for
        :param q_theta_fixed_epochs: Number of epochs to fix theta for
        :param local_reparam: Option for reparameterisation trick
        :param feed_forward: Option for feeding forward of inputs at each layer
        """
        super(deepLFM, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        n_layers = self.n_hidden_layers + 1
        self.n_layers = n_layers
        self.mc = mc
        self.local_reparam = local_reparam
        self.feed_forward = feed_forward
        self.q_Omega_fixed_epochs = q_Omega_fixed_epochs
        self.q_theta_fixed_epochs = q_theta_fixed_epochs
        self.q_Omega_fixed = q_Omega_fixed_epochs > 0
        self.q_theta_fixed = q_theta_fixed_epochs > 0        
        self.n_rff = n_rff * np.ones(n_layers, dtype=np.int32)
        self.n_lfm = n_lfm * np.ones(n_layers, dtype=np.int32)
        self.n_lf = n_lf * np.ones(n_layers, dtype=np.int32)
        self.D = d_out
        self.likelihood = Gaussian()
        
        # Define Omega matrix dimensionalities for each layer
        if self.feed_forward:
            self.d_in = np.concatenate([[d_in], self.n_lfm[:(n_layers - 1)] + d_in])
        else:
            self.d_in = np.concatenate([[d_in], self.n_lfm[:(n_layers - 1)]])
        self.d_out = self.n_rff * self.n_lf

        # Define W matrix dimensionalites for each layer
        self.dhat_in = 2 * self.n_rff * self.n_lf
        self.dhat_out = np.concatenate([self.n_lfm[:-1], [d_out]])
        
        # Define the fixed standard normals used for the VAR-FIXED treatment of Omega       
        self.z_for_Omega_fixed = []
        for i in range(self.n_layers):
            temp = torch.randn(1, self.d_in[i], self.d_out[i], requires_grad=False)
            self.z_for_Omega_fixed.append(temp[0, :, :])        
        
        # Define parameters which govern prior over Omega & the kernel parameters; all theta parameters are
        # constrained to be positive by means of optimising the log of the parameter
        
        # Define layer-wise sensitivity parameters for inner layers to weight contribution of each input/LF
        self.rho = [torch.randn(self.d_in[i], self.n_lf[i], requires_grad=True) for i in range(self.n_hidden_layers)]

        # Define layer-wise prior lengthscales across whole model (ARD behaviour)
        self.theta_log_lengthscale_prior = [np.log(0.01) \
                                            for i in range(self.n_hidden_layers)]
        
        # Using the prior, define layer-wise lengthscale parameters for inner layers (ARD behaviour)
        self.theta_log_lengthscale = [(torch.ones(self.d_in[i]) *
                                       self.theta_log_lengthscale_prior[i]).requires_grad_(True) \
                                      for i in range(self.n_hidden_layers)]

        # Initialise ODE1 decay (gamma) parameters which are involved in the computation of the RFRFs within the
        # inner layers; these can differ according to the input dimension r
        self.theta_log_decay = [torch.log(torch.ones(self.d_in[i], 1) * 0.01).requires_grad_(True) \
                                for i in range(self.n_hidden_layers)]

        # Output layer parameters; these can differ according to both input dimension r and output dimension d
        self.rho_out = torch.randn(self.d_in[-1], self.D, self.n_lf[-1], requires_grad=True)
        self.theta_log_lengthscale_prior_out = 0.01
        self.theta_log_lengthscale_out = (torch.ones(self.d_in[-1],
                                                     self.D) * self.theta_log_lengthscale_prior_out).requires_grad_(True)

        # Output layer ODE1 decay parameters; these can differ according to both input dimension r 
        # and output dimension d
        self.theta_log_decay_out = torch.log(torch.ones(self.d_in[-1], self.D) * 0.01).requires_grad_(True)

        # Log variance parameter of output Gaussian likelihood
        self.output_log_var = torch.tensor([-2.0], dtype=torch.float, requires_grad=True)
        
        # Fetch priors over Omega using the lengthscales, and W
        self.Omega_mean_prior, self.Omega_log_var_prior = self.get_Omega_prior((self.theta_log_lengthscale +
                                                                                   [self.theta_log_lengthscale_out]))
        self.W_mean_prior, self.W_log_var_prior = self.get_W_prior()
        
        # Initialise posteriors over Omega and W
        self.Omega_mean, self.Omega_log_var = self.init_posterior_Omega()
        self.W_mean, self.W_log_var = self.init_posterior_W()

    def get_Omega_prior(self, log_lengthscale):
        """ 
        Define prior over Omega, given the lengthscales; prior over variance is \frac{2}{lengthscale^2}
        """
        Omega_mean_prior = [torch.zeros(self.d_in[i], 1, requires_grad=False) for i in range(self.n_layers)]
        Omega_log_var_prior = [np.log(2) + (-2.0 * log_lengthscale[i]) for i in range(self.n_layers)]
        return Omega_mean_prior, Omega_log_var_prior

    def get_W_prior(self):
        """
        Define prior over W using standard normals
        """
        W_mean_prior = torch.zeros(self.n_layers, requires_grad=False)
        W_log_var_prior = torch.zeros(self.n_layers, requires_grad=False)
        return W_mean_prior, W_log_var_prior        

    def init_posterior_Omega(self):
        """ 
        Initialise posterior over Omega
        """
        mu, log_var = self.get_Omega_prior(self.theta_log_lengthscale_prior + [self.theta_log_lengthscale_prior_out])
        Omega_mean = [(mu[i] * torch.ones(self.d_in[i], self.d_out[i])).requires_grad_(True) \
                       for i in range(self.n_layers)]
        Omega_log_var = [(log_var[i] * torch.ones(self.d_in[i], self.d_out[i])).requires_grad_(True) \
                          for i in range(self.n_layers)]
        return Omega_mean, Omega_log_var

    def init_posterior_W(self):
        """ 
        Initialise posterior over W
        """
        W_mean = [torch.zeros(self.dhat_in[i], self.dhat_out[i], requires_grad=True) \
                  for i in range(self.n_layers)]
        W_log_var = [torch.zeros(self.dhat_in[i], self.dhat_out[i], requires_grad=True) \
                     for i in range(self.n_layers)]
        return W_mean, W_log_var

    def get_kl(self):
        """ 
        Compute KL divergence between priors and approximate posteriors over Omega & W
        """
        kl = 0

        # Sum terms arising from hidden layers first
        for i in range(self.n_hidden_layers):

            Omega_log_var_prior_reshaped = (self.Omega_log_var_prior[i].clone().detach().reshape(-1, 1) *
                                             torch.ones_like(self.Omega_log_var[i]))
            Omega_mean_prior_reshaped = self.Omega_mean_prior[i] * torch.ones_like(self.Omega_mean[i])

            kl += torch.sum(Omega_log_var_prior_reshaped)

            kl += DKL_Gaussian(self.Omega_mean[i], self.Omega_log_var[i],
                               Omega_mean_prior_reshaped, Omega_log_var_prior_reshaped)
            kl += DKL_Gaussian(self.W_mean[i], self.W_log_var[i],
                               self.W_mean_prior[i], self.W_log_var_prior[i].reshape(-1, 1))

        # Deal with output layer separately due to separate treatment of different output dimensions
        for d in range(self.D):

            Omega_log_var_prior_reshaped = (self.Omega_log_var_prior[-1][:, d].clone().detach().reshape(-1, 1) *
                                             torch.ones_like(self.Omega_log_var[-1]))
            Omega_mean_prior_reshaped = self.Omega_mean_prior[-1] * torch.ones_like(self.Omega_mean[-1])

            kl += DKL_Gaussian(self.Omega_mean[-1], self.Omega_log_var[-1],
                               Omega_mean_prior_reshaped, Omega_log_var_prior_reshaped)
            kl += DKL_Gaussian(self.W_mean[-1], self.W_log_var[-1],
                               self.W_mean_prior[-1], self.W_log_var_prior[-1].reshape(-1, 1))

        return kl

    def sample_from_Omega(self):
        """ 
        Sample from Omega when computed using fixed random variables and two optimised parameters, 
        the mean and variance of the approximating distribution (VAR-FIXED case)
        """
        Omega_from_q = []
        for i in range(self.n_layers):
            z = self.z_for_Omega_fixed[i] * torch.ones(self.mc, self.d_in[i], self.d_out[i])
            Omega_from_q.append((z * torch.exp(self.Omega_log_var[i] / 2)) + self.Omega_mean[i])
        return Omega_from_q

    def sample_from_W(self):
        """ 
        Sample from approximate posterior over W
        """
        W_from_q = []
        for i in range(self.n_layers):
            z = torch.randn(self.mc, self.dhat_in[i], self.dhat_out[i])
            self.z = z
            W_from_q.append((z * torch.exp(self.W_log_var[i] / 2)) + self.W_mean[i])
        return W_from_q

    def get_ell(self, X, y=None, ignore_nan=False):
        """ 
        Compute expected log-likelihood term in variational lower bound
        """
        try:
            test_N = self.N
        except:
            raise Exception(('If using the model.get_ell() function outside of training, manually supply a N value' \
                             ' using model.N = some_value'))

        d_input = self.d_in[0]
        batch_size = X.size()[0]

        # 3D tensor for each layer, where each slice [i,:,:] is a MC realisation of the hidden layer
        # values; the initial layer is just the input replicated self.mc times
        self.layer = []
        self.layer.append(torch.ones(self.mc, batch_size, d_input) * X)

        # Sample Omega values
        Omega_from_q = self.sample_from_Omega()

        # Propagate input data through the hidden layers
        for i in range(self.n_layers):

            # If any other layer besides the final/output layer
            if i < (self.n_layers - 1):

                # Transform input to be of shape (d_input x self.mc x batch_size x 1), expand Omega 
                # samples to same shape, and multiply together
                layer = torch.transpose(torch.unsqueeze(self.layer[i], 0), 0, -1) 
                Omega = torch.unsqueeze(Omega_from_q[i], -1).permute(1, 0, 3, 2).repeat(1, 1, batch_size, 1)
                layer_Omega = Omega * layer

                # Convert some quantities needed for kernel to complex tensor format
                layer_Omega_c = layer_Omega.type(torch.cfloat)
                Omega_c = Omega.type(torch.cfloat)
                layer_c = layer.type(torch.cfloat)

                # Define 0 + 1*i as a variable for convenience whilst computing kernels
                imag_1 = torch.zeros(1, dtype=torch.cfloat)  # i.e. 0 + 1i
                imag_1.imag = torch.tensor([1.0])
                   
                # Fetch decay hyperparameters and convert to complex tensor form
                gamma = torch.exp(self.theta_log_decay[i]).unsqueeze(-1).unsqueeze(-1).type(torch.cfloat)

                # Compute random features for ODE1 kernel, for all input dimensions
                B = 1.0 / (gamma + (imag_1 * Omega_c))
                A = - B

                features = ((B * torch.exp(imag_1 * layer_Omega_c)) +
                            (A * torch.exp(- gamma * layer_c)))

                # Construct matrix of coefficients to multiply RFs by (small epsilon to avoid NaNs in backprop)
                v_coeff = torch.sqrt((torch.square(self.rho[i]) + 1e-7) / (self.n_lf[i] * self.n_rff[i]))
                v_coeff = v_coeff.unsqueeze(1).unsqueeze(1).repeat_interleave(self.n_rff[i], dim=-1)
                features *= v_coeff

                # Construct composite kernel by combining RFRFs from each input dimension via addition
                v = features.sum(dim=0)

                # Construct Phi matrix from the real and imaginary portions of the RFRFs computed for this layer
                Phi = torch.cat([v.real, v.imag], 2)

                # Compute F using reparameterisation trick (with small epsilon added to avoid NaNs in backprop)
                if self.local_reparam:
                    z_for_F_sample = torch.randn(self.mc, Phi.size()[1], self.dhat_out[i])
                    F_mean = torch.tensordot(Phi, self.W_mean[i], [[2], [0]])
                    F_var = torch.tensordot(torch.pow(Phi, 2), torch.exp(self.W_log_var[i]), [[2], [0]])
                    F = (z_for_F_sample * torch.sqrt(F_var + 1e-7)) + F_mean
                # Otherwise compute samples conventionally
                else:
                    W_from_q = self.sample_from_W()
                    F = torch.matmul(Phi, W_from_q[i])

                # If feed-forward specified, concatenate layer with original input
                if self.feed_forward:
                    F = torch.cat([F, self.layer[0]], 2)

                self.layer.append(F)

            # If final layer, proceed with handling each output separately
            else:

                # Initialise variable to hold outputs
                outputs = None

                # Loop through each of the D outputs
                for d in range(self.D):

                    # Transform input to be of shape (d_input x self.mc x batch_size x 1), expand Omega 
                    # samples to same shape, and multiply together
                    layer = torch.transpose(torch.unsqueeze(self.layer[i], 0), 0, -1) 
                    Omega = torch.unsqueeze(Omega_from_q[i], -1).permute(1, 0, 3, 2).repeat(1, 1, batch_size, 1)
                    layer_Omega = Omega * layer

                    # Convert some quantities needed for kernel to complex tensor format
                    layer_Omega_c = layer_Omega.type(torch.cfloat)
                    Omega_c = Omega.type(torch.cfloat)
                    layer_c = layer.type(torch.cfloat)

                    # Define 0 + 1*i as a variable for convenience whilst computing kernels
                    imag_1 = torch.zeros(1, dtype=torch.cfloat)  # i.e. 0 + 1i
                    imag_1.imag = torch.tensor([1.0])

                    # Fetch decay hyperparameters and convert to complex tensor form
                    gamma = torch.exp(self.theta_log_decay_out[:, d]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).type(torch.cfloat)

                    # Compute features for ODE1 kernel, for all input dimensions
                    B = 1.0 / (gamma + (imag_1 * Omega_c))
                    A = - B

                    features = ((B * torch.exp(imag_1 * layer_Omega_c)) +
                            (A * torch.exp(- gamma * layer_c)))

                    # Construct matrix of coefficients to multiply RFs by (small epsilon to avoid NaNs in backprop)
                    v_coeff = torch.sqrt((torch.square(self.rho_out[:, d, :]) + 1e-7) / (self.n_lf[i] * self.n_rff[i]))
                    v_coeff = v_coeff.unsqueeze(1).unsqueeze(1).repeat_interleave(self.n_rff[i], dim=-1)
                    features *= v_coeff

                    # Construct composite kernel by combining RFRFs from each input dimension via addition
                    v = features.sum(dim=0)

                    # Construct Phi matrix from the real and imaginary portions of the RFRFs computed for this layer
                    Phi = torch.cat([v.real, v.imag], 2)

                    # Compute y_d using reparameterisation trick (with small epsilon added to avoid NaNs in backprop)
                    if self.local_reparam:
                        z_for_y_sample = torch.randn(self.mc, Phi.size()[1], 1)
                        y_d_mean = torch.tensordot(Phi, torch.unsqueeze(self.W_mean[i][:, d], 1), [[2], [0]])
                        y_d_var = torch.tensordot(torch.pow(Phi, 2),
                                                  torch.exp(torch.unsqueeze(self.W_log_var[i][:, d], 1)), [[2], [0]])
                        y_d = (z_for_y_sample * torch.sqrt(y_d_var + 1e-7)) + y_d_mean
                    # Otherwise compute samples conventionally for the d-th output
                    else:
                        W_from_q = self.sample_from_W()
                        y_d = torch.matmul(Phi, torch.unsqueeze(W_from_q[i][:, :, d], 2))

                    # Store these output values and proceed onto the next
                    if outputs is None:
                        outputs = y_d
                    else:
                        outputs = torch.cat([outputs, y_d], dim=2)

                self.layer.append(outputs)

        output_layer = self.layer[self.n_layers]

        # If targets specified, use output to calculate conditional likelihood across N samples
        if y is not None:

            # Mini-batch estimate of the expected log-likelihood, computed output by output, such that observations
            # with missing data-points can be present (as NaNs) in the training targets/y values, without stopping
            # the model from training by introducing NaNs in backprop
            if ignore_nan == True:
                ell = 0
                for d in range(self.D):
                    # Remove missing (NaN) y values
                    y_d = y[:, d]
                    output_d = output_layer[:, :, d]
                    is_nan = torch.isnan(y_d)
                    y_d = y_d[~is_nan]
                    output_d = output_d[:, ~is_nan]
                
                    # Compute ELL of non-NaN data-points in batch
                    ll = self.likelihood.log_cond_prob(y_d, output_d, self.output_log_var)
                    ell += torch.sum(torch.logsumexp(ll, 0)) * self.N / len(y_d)
                    # ell += torch.sum(torch.mean(ll, 0)) * self.N / len(y_d)

            # Otherwise, compute regular mini-batch estimate of the expected log-likelihood
            else:
                ll = self.likelihood.log_cond_prob(y, output_layer, self.output_log_var)
                ell = torch.sum(torch.logsumexp(ll, 0)) * self.N / batch_size

            return ell, output_layer

        else:
            return output_layer

    def get_nelbo(self, X, y, ignore_nan=False):
        """ 
        Get negative ELBO; minimising NELBO == maximising ELBO
        """
        kl = self.get_kl()
        ell, output_layer = self.get_ell(X, y, ignore_nan=ignore_nan)
        nelbo = kl - ell
        return nelbo, kl, ell, output_layer

    def predict(self, X, y=None, get_layers=False):
        """ 
        Generate predictions for given data
        """
        with torch.no_grad():
            output_layer = self.get_ell(X)
            output = self.likelihood.predict(output_layer)
        y_pred_mean = torch.mean(output, 0) #+ (torch.exp(self.output_log_var / 2.0) *  torch.randn(output.size()[1], 1))
        y_pred_std = torch.std(output, 0)

        # Compute MNLL, NMSE & NLPD if target values specified
        if y is not None:
            metrics = {}

            # Compute mean negative log likelihood, normalised MSE and RMSE
            mnll = - torch.mean(-np.log(self.mc) + torch.logsumexp(self.likelihood.log_cond_prob(y, output_layer, self.output_log_var), 0))
            metrics['mnll'] = mnll.item()
            metrics['nmse'] = (torch.mean((y_pred_mean - y)**2) / torch.mean((torch.mean(y) - y)**2)).item()
            metrics['rmse'] = torch.sqrt(torch.mean((y_pred_mean - y)**2)).item()

        # Fetch values at each layer as well as overall predictions
        if get_layers:
            pred_layers = self.layer

        # Return correct outputs dependent on arguments specified
        if get_layers and (y is not None):
            return y_pred_mean, y_pred_std, pred_layers, metrics
        elif get_layers:
            return y_pred_mean, y_pred_std, pred_layers
        elif y is not None:
            return y_pred_mean, y_pred_std, metrics
        else:
            return y_pred_mean, y_pred_std

    def set_opt_vars(self):
        """ 
        Set variables to be optimised. Note that whilst we can treat Omega and W variationally 
        within this model, the other covariance parameters are simply optimised for now.
        """
        # Initial case, Omega and theta fixed
        opt_vars = self.W_mean + self.W_log_var
        opt_vars.append(self.output_log_var)

        # Omega fixed
        if self.q_Omega_fixed and (not self.q_theta_fixed):
            opt_vars += (self.theta_log_lengthscale + self.rho)
            opt_vars.append(self.rho_out)
            opt_vars.append(self.theta_log_lengthscale_out)

            opt_vars += self.theta_log_decay
            opt_vars.append(self.theta_log_decay_out)

        # Theta fixed
        elif self.q_theta_fixed and (not self.q_Omega_fixed):
            opt_vars += (self.Omega_mean + self.Omega_log_var)

        # Nothing fixed:
        elif (not self.q_theta_fixed) and (not self.q_Omega_fixed):
            opt_vars += (self.Omega_mean + self.Omega_log_var + self.theta_log_lengthscale + self.rho)
            opt_vars.append(self.rho_out)
            opt_vars.append(self.theta_log_lengthscale_out)

            opt_vars += self.theta_log_decay
            opt_vars.append(self.theta_log_decay_out)

        return opt_vars

    def train(self, X, y, X_valid=None, y_valid=None, lr=0.01, batch_size=64, epochs=10, verbosity=1,
              single_mc_epochs=0, train_time_limit=None, ignore_nan=False, csv_output=False):
        """ Optimisation of the deep LFM. Using a value > 0 for single_mc_epochs (as well
            as fixing theta and Omega for a number of epochs) can be beneficial if training
            happens to be unstable.

        :param X: Training inputs
        :param y: Training targets
        :param X_valid: Optional validation inputs
        :param y_valid: Optional validation targets
        :param lr: Learning rate
        :param batch_size: Batch size
        :param epochs: Training epochs
        :param verbosity: Verbosity of metric evaluations
        :param single_mc_epochs: Number of epochs to use MC = 1
        :param train_time_limit: Training time limit (overrides no. epochs)
        :param ignore_nan: Must be set to True if observations in training data have NaN datapoints
        :param csv_output: Option for easy-to-parse output of validation metrics
        """
        train_time = 0

        # If no validation set specified, just evaluate on the training data
        if (X_valid is None) or (y_valid is None):
            X_valid, y_valid = X, y

        # Set optimiser and parameters to be optimised
        optimizer = torch.optim.AdamW(self.set_opt_vars(), lr=lr)

        # Initialise dataloader for minibatch training
        self.N = X.size()[0] # Total number of training examples
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # If desired, fix number of Monte Carlo samples to one for first 'single_mc_epochs' of training
        if single_mc_epochs != 0:
            mc_placeholder = self.mc
            self.mc = 1

        if csv_output:
            print('Epoch, Time, NMSE, RMSE, MNLL')
        
        # Perform epochs of minibatch training
        for i in range(epochs):

            # Stop training after time limit elapsed
            if train_time_limit is not None:
                if (train_time > 1000 * 60 * train_time_limit):
                    break

            # If given number of epochs have passed, transition to using more than one MC sample
            if single_mc_epochs != 0:
                if i > single_mc_epochs:
                    self.mc = mc_placeholder

            # Use each batch to train model
            batch_start = int(round(time.time() * 1000))

            for X_minibatch, y_minibatch in train_dataloader:
                optimizer.zero_grad()
                loss, _, _, _ = self.get_nelbo(X_minibatch, y_minibatch, ignore_nan=ignore_nan)
                loss.backward()
                optimizer.step()

            train_time += (int(round(time.time() * 1000)) - batch_start)

            # Unfix Omega/theta if sufficient number of iterations passed
            if self.q_Omega_fixed:
                if i >= self.q_Omega_fixed_epochs:
                    self.q_Omega_fixed = False
                    optimizer = torch.optim.AdamW(self.set_opt_vars(), lr=lr)

            if self.q_theta_fixed:
                if i >= self.q_theta_fixed_epochs:
                    self.q_theta_fixed = False
                    optimizer = torch.optim.AdamW(self.set_opt_vars(), lr=lr)

            # Display validation metrics at specified intervals
            if verbosity == 0:
                print('Epoch %d complete.' % i)
            elif i % verbosity == 0:
                with torch.no_grad():
                    _, _, metrics = self.predict(X_valid, y_valid)
                if csv_output: # If .csv output style required for analysis of training metrics
                    print('%d,%.4f,%.4f,%.4f,%.4f' % (i, train_time / 1000, metrics['nmse'], metrics['rmse'], metrics['mnll']))
                else: # Otherwise, print metrics in an easily readable fashion
                    print('Epoch %d' % (i))
                    print('Validation N-MSE = %.4f, Validation RMSE = %.4f, Validation MNLL = %.4f\n' % (metrics['nmse'], metrics['rmse'], metrics['mnll']))
