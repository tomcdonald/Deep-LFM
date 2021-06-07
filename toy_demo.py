import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from deepLFM.deepLFM import deepLFM

# CUDA initialisations (Note, if you aren't using a GPU, you'll likely 
# need to train for longer than specified in this demo)
SEED = 99
cuda_ = "cuda:0"
device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")
print('Device:', device)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

# Generate toy data from a hierarchical ODE system
t = np.linspace(0, 15, 750)
f1_init, f2_init = [0], [0]
decay1, decay2 = [0.01], [0.02]
tau = np.zeros(750)
lambd = 1

def u_func(t, array=None):
    return np.cos(t/2) + 6*np.sin(3*t)

def f1(tau):
    i = np.array([0+1j])
    G1 = (np.exp(i*lambd*(t-tau)) - np.exp(-decay1[0]*(t-tau))) / (decay1[0] + i*lambd)
    return G1 * u_func(tau)

f1 = integrate.quad_vec(f1, 0, max(t))[0]

def f2(tau):
    i = np.array([0+1j])
    G2 = (np.exp(i*lambd*(f1-tau)) - np.exp(-decay2[0]*(f1-tau))) / (decay2[0] + i*lambd)
    return G2 * u_func(tau)

f2 = integrate.quad_vec(f2, 0, max(f1))[0].reshape(-1, 1)
f2 = f2.real.reshape(1, -1)
f2 += 0.15 * np.random.randn(*f2.shape) # Add Gaussian noise to outputs

# Scale inputs & outputs, and split into train and test sets
ss_x = MinMaxScaler()
X = t.reshape(-1, 1)
X_tr = ss_x.fit_transform(np.concatenate([t[:125], t[225:600]]).reshape(-1, 1))
X_test_interp = ss_x.transform(t[125:225].reshape(-1, 1))
X_test_extrap = ss_x.transform(t[600:].reshape(-1, 1))

ss_y = StandardScaler()
y_tr = ss_y.fit_transform(np.concatenate([f2[0, :125], f2[0, 225:600]]).reshape(-1, 1))
y_test_interp = ss_y.transform(f2[0, 125:225].reshape(-1, 1))
y_test_extrap = ss_y.transform(f2[0, 600:].reshape(-1, 1))

# Initialise and train model
dlfm = deepLFM(d_in=1, d_out=1, n_hidden_layers=1, n_lfm=3, n_rff=100, n_lf=1, mc=100, q_Omega_fixed_epochs=200, 
               q_theta_fixed_epochs=400, feed_forward=True, local_reparam=True).to(device)

dlfm.train(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32), 
           X_valid=torch.tensor(X_test_extrap, dtype=torch.float32), y_valid=torch.tensor(y_test_extrap, dtype=torch.float32),
           lr=0.01, epochs=100000, batch_size=len(X_tr), verbosity=100, single_mc_epochs=50, train_time_limit=2)

def plot_model_results(model):
    # Compute predictions across all data points
    X_all = ss_x.transform(t.reshape(-1, 1))
    mean_all, std_all = model.predict(torch.tensor(X_all, dtype=torch.float32))
    mean_all, std_all = mean_all.detach().cpu().numpy().flatten(), std_all.detach().cpu().numpy().flatten()

    # Plot predictions for model
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.scatter(X_tr.flatten(), y_tr.flatten(), label='Training Data', color='grey', s=1, alpha=0.5)
    ax.scatter(X_test_interp.flatten(), y_test_interp.flatten(), label='Test Data (Interpolation)', s=1, color='orange', alpha=0.5)
    ax.scatter(X_test_extrap.flatten(), y_test_extrap.flatten(), label='Test Data (Extrapolation)', s=1, color='red', alpha=0.5)
    ax.plot(X_all.flatten(), mean_all.flatten() , label='Predictive Mean (Train)', color='purple')
    ax.fill_between(X_all.flatten(), mean_all.flatten() + 2.0 * std_all.flatten(), mean_all.flatten() - 2.0 * std_all.flatten(), alpha=0.2,
                    color='black', linewidth=0.1)
    ax.set_xlabel('$t$', fontsize=16)
    ax.set_ylabel('$y_2$', fontsize=16)
    ax.set_ylim(bottom=-6.0, top=3.75)
    plt.tight_layout()
    plt.savefig('toy.pdf', dpi=300, bbox_inches='tight')

plot_model_results(dlfm)
