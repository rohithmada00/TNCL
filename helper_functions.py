import numpy as np
from scipy.stats import multivariate_normal
import random
from sklearn.metrics import f1_score
from time import time
import matplotlib.pyplot as plt
from numpy.ma.extras import average
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
import os

def B_mat_symmetric(a, p):
    '''To generate a symmetric toeplitz matrix with a given vector 'a' '''
    # Initialize a zeros matrix of size nxn
    result = np.zeros((p, p))

    # Fill in the main diagonal with center element of a
    np.fill_diagonal(result, a[0])

    # Fill in sub/super-diagonals (below the main diagonal)
    for i in range(1, p):
        np.fill_diagonal(result[i:], -1*a[i]) # Sub-diagonals
        np.fill_diagonal(result[:,i:], -1*a[i]) # Super-diagonals
    return result

def cov_x(p):
    """ Generate the covariance matrix for x"""
    sigma_x = np.eye(p)
    return sigma_x

def cov_y(sigma_x, B):
    """ Generate covariance matrix for Y i.e. Sigma_Y = (B^(-1).Sigma_X.B^(-T)) since Y=B^(-1).X"""
    B_inv = np.linalg.solve(B, np.eye(B.shape[0]))
    sigma_y = B_inv@sigma_x@B_inv.T
    return sigma_y

def generate_y(mean_y, sigma_y, n_samples, random_state=None):
    """ Generate samples of Y"""
    y_samples = multivariate_normal.rvs(mean=mean_y, cov=sigma_y, size=n_samples, random_state=random_state)
    print('Y_samples: ', y_samples[0])
    return y_samples

def samp_cov(y_samples):
    """ Generating sample covariance matrix """
    n_samples = len(y_samples)
    samp_cov_y = np.sum(np.array([np.outer(y_samples[i],y_samples[i]) for i in range(n_samples)]), axis=0)/n_samples 
    return samp_cov_y

def soft(x, y):
    return np.sign(x) * np.maximum(np.abs(x) - y, 0)

def threshold(vec, threshold_value=0.002):
    vec1 = vec.copy()
    vec1[np.abs(vec) < threshold_value] = 0
    return vec1

def binarize_matrix(matrix):
    '''Converts non-zero elements to 1 and keeps zeros as zeros.'''
    binary_matrix = np.where(matrix != 0, 1, 0)
    return binary_matrix

def create_sparse_vec_pos_def_2(dim, nonzeros, diag=20, random_state= None):
    ''' Function to create sparse vector(p x 1) which can generate symmetric +ve definite toeplitz matrix'''
    vec = np.zeros((dim,))
    vec[0] = diag
    nonzeros -= 1
    rng = random.Random(random_state)
    # selection of indicies with the non-zero entry (random selection)
    # non_zero_indices = [1] # tri-diagonal case
    non_zero_indices = rng.sample(range(1,dim),nonzeros)
    for idx in non_zero_indices:
        # Randomly assign values to non-zero entries 
        vec[idx] = rng.randint(1,5)
    print(f'a: {vec}')
    return vec

def f1(a, S, sigma_x, rho, a2, mu1, regularized = False):
    """ Function evaluation for Newton method with regularization term based on ADMM Tr[S.Theta_y] - logdet(Theta_y) + rho/2*||a1 - a2 + mu1||_1"""
    p = S.shape[0]
    a.reshape((-1,))
    B = B_mat_symmetric(a, p)
    theta_x = np.linalg.solve(sigma_x, np.eye(p))
    if regularized:
        # Added [1:] since we don't want to regularize a0
        reg_term = (rho/2)*(np.linalg.norm(a[1:]-a2+mu1)**2)
    else:
        reg_term = 0
    return np.trace(S@B.T@theta_x@B)-np.log(np.linalg.det(B.T@theta_x@B))+reg_term 

def make_T_matrices(p, symmetric=True):
    """
    Returns a list [T0, T1, …, T(p-1)]
    for every j!=0 T[j] is a pxp matrix with jth super and subdiagonals with entries 1
    for j=0 its an Identity matrix
    """
    T_list = []
    I = np.eye(p)
    T_list.append(I)
    for j in range(1, p):
        M = np.zeros((p, p))
        # sub‑diagonal
        M[np.arange(j, p), np.arange(0, p-j)] = -1
        if symmetric:
            # super‑diagonal
            M[np.arange(0, p-j), np.arange(j, p)] = -1
        T_list.append(M)
    return T_list


def save_to_pickle(data_object, destination_path):
    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    try:
        with open(destination_path, 'wb') as f: 
            pickle.dump(data_object, f)
        print(f"Data successfully pickled and saved to: {destination_path}")
    except Exception as e:
        print(f"Error saving data to {destination_path}: {e}")

def load_from_pickle(source_path):
    if not os.path.exists(source_path):
        print(f"Error: File not found at {source_path}")
        return None

    try:
        with open(source_path, 'rb') as f: 
            loaded_object = pickle.load(f)
        print(f"Data successfully unpickled from: {source_path}")
        return loaded_object
    except Exception as e:
        print(f"Error loading data from {source_path}: {e}")
        return None

def create_comparison_dataframe(data):   
    num_rep = len(data)
    comparison_series = {}

    for i in range(num_rep):
        ground_truth_a = data[i]['a']
        estimate_a1 = data[i]['a1']

        ground_truth_a = np.asarray(ground_truth_a).flatten()
        estimate_a1 = np.asarray(estimate_a1).flatten()

        comparison_series[(f'Run {i+1}', 'ground_truth')] = pd.Series(ground_truth_a)
        comparison_series[(f'Run {i+1}', 'estimate')] = pd.Series(estimate_a1)

    df_comparison = pd.DataFrame(comparison_series)
    df_comparison.index.name = 'index'

    return df_comparison

def compute_recovery_rate_numpy(a_binary, true_a_binary):
    joint_ones = np.sum((a_binary == 1) & (true_a_binary == 1)) # True positives + False positives
    ones_true_a = np.sum(true_a_binary == 1) # True positives
    if ones_true_a == 0:
        return 0
    return joint_ones / ones_true_a

def kll(a, a_hat, sigma_x):
    ''' Computing the KL loss for the covariance matrices of Y'''
    p = len(a)
    B = B_mat_symmetric(a, p)
    B_hat = B_mat_symmetric(a_hat, p)
    sigma_y = cov_y(sigma_x, B)
    sigma_y_hat = cov_y(sigma_x, B_hat)
    theta_y_hat = np.linalg.solve(sigma_y_hat, np.eye(p))
    # theta_y = np.linalg.solve(sigma_y, np.eye(p))
    return np.trace(sigma_y@theta_y_hat) - np.log(np.linalg.det(sigma_y@theta_y_hat)) - p

def rkll(a, a_hat, sigma_x):
    ''' Computing the reverse KL loss for the covariance matrices of Y'''
    return kll(a_hat, a, sigma_x)

def rte(a, a_hat, sigma_x):
    ''' Computing the relative trace error loss for the precision matrix'''
    p = len(a)
    B = B_mat_symmetric(a, p)
    B_hat = B_mat_symmetric(a_hat, p)
    # sigma_y = cov_y(sigma_x, B)
    # sigma_y_hat = cov_y(sigma_x, B_hat)
    theta_y_hat = B_hat.T@B_hat
    theta_y = B.T@B
    return abs(1 - (np.trace(theta_y_hat)/np.trace(theta_y)))

def norm_loss(a, a_hat):
    ''' Computing the norm loss for the precision matrix'''
    p = len(a)
    B = B_mat_symmetric(a, p)
    B_hat = B_mat_symmetric(a_hat, p)
    # sigma_y = cov_y(sigma_x, B)
    # sigma_y_hat = cov_y(sigma_x, B_hat)
    # theta_y_hat = np.linalg.solve(sigma_y_hat, np.eye(p))
    # theta_y = np.linalg.solve(sigma_y, np.eye(p))
    theta_y_hat = B_hat.T@B_hat
    theta_y = B.T@B
    fro = np.linalg.norm(theta_y-theta_y_hat,ord='fro')
    spe = np.linalg.norm(theta_y-theta_y_hat,ord=2)
    l1 = np.linalg.norm(theta_y-theta_y_hat,ord=1)
    return fro, spe, l1

def scores(a, a_hat, threshold_value):
    ''' Computing the support recovery based scores for the precision matrix'''
    a_thr = threshold(a_hat, threshold_value=threshold_value) # Apply the threshold
    a_binary = binarize_matrix(a_thr)  # Change the output to a binary matrix to check for support recovery
    true_a_binary = binarize_matrix(a)  # True Support Vector
    cm = confusion_matrix(true_a_binary, a_binary)
    tn,fp,fn,tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    accuracy = (tn + tp)/np.sum(cm)
    specificity = (tn)/(tn + fp) # TN/(TN+FP)
    sensitivity = (tp)/(tp + fn) # TP/(TP+FN)
    MCC = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return accuracy, specificity, sensitivity, MCC, cm

def compute_gH(x, S, rho, a2, mu1, T_list, B, B_inv, theta_x, regularized = False):
    p=S.shape[0]
    # compute gradient
    if regularized:
        k = np.zeros_like(x)
        k[1:] = x[1:]-a2+mu1
    else:
        k = np.zeros_like(x)
    gradient = np.zeros((p,))
    for i in range(p):
            Ti = T_list[i] 
            gradient[i] = 2*np.trace((S@B@theta_x@Ti) - (B_inv@Ti)) + (rho*k[i])

    # compute hessian
    hessian = np.zeros((p,p))
    for i in range(p):
        Ti = T_list[i]
        for j in range(p):
            Tj = T_list[j]
            hessian[i,j] = 2*(np.trace(S@Ti@theta_x@Tj) + np.trace(B_inv@Ti@B_inv@Tj))
    if regularized:
        reg = rho*np.eye(p)
        reg[0,0] = 0
        hessian+=reg
    print(f"Condition number of Hessian: {np.linalg.cond(hessian)}")
    return gradient, hessian

def compute_newton_step(gradient, hessian):
    """
    Computes the Newton step and decrement.

    Parameters:
    gradient (np.array): Gradient vector.
    hessian (np.array): Hessian matrix.

    Returns:
    delta_x_nt (np.array): Newton step.
    lambd (float): Newton decrement.
    """
    delta_x_nt = -np.linalg.solve(hessian, gradient)  # Newton step
    # lambd_sqr = np.dot(gradient.T, hessian@gradient)  # Newton decrement
    # lambd_sqr = np.dot(gradient.T, np.linalg.solve(hessian, gradient))
    lambd_sqr = -np.dot(gradient.T, delta_x_nt)
    return delta_x_nt, lambd_sqr

def backtracking_line_search(f, gradient, x, delta_x_nt, S, sigma_x, rho, a2, mu1,
                             regularized=False, alpha=0.01, beta=0.5):
    """
    Backtracking line search to ensure sufficient decrease.

    Parameters:
    f (function): Function to minimize.
    gradient (np.array): Gradient vector.
    x (np.array): Current point.
    delta_x_nt (np.array): Newton step.
    alpha (float): Alpha parameter for backtracking.
    beta (float): Beta parameter for backtracking.

    Returns:
    t (float): Step size.
    """
    t = 1  # Start with full step size
    p = S.shape[0]
    x_ = x + t * delta_x_nt
    e_val, _ = np.linalg.eig(B_mat_symmetric(x_, p))
    i = 0
    while not np.all(e_val>0): 
        t *= beta # Reduce step size so that x + del_x is PSD
        x_ = x + t * delta_x_nt
        e_val, _ = np.linalg.eig(B_mat_symmetric(x_, p))
        i+=1
        print(f"   PSD-loop  iter={i:2d}  t={t:.2e}  λ_min={e_val[0]:.2e}")
        # if(i%5==0):
        #     print(f"   PSD-loop  iter={i:2d}  t={t:.2e}  λ_min={e_val[0]:.2e}")
        if i>20:
            t = 0
            print('Backtracking line search (I) went beyond 20 iterations')
            break
    x_ = x + t * delta_x_nt
    i = 0
    while (f(x_,S,sigma_x,rho,a2,mu1,regularized=regularized)\
           >f(x,S,sigma_x,rho,a2,mu1,regularized=regularized)\
           + alpha*t*np.dot(gradient.T,delta_x_nt)):
        t *= beta  # Reduce step size
        x_ = x + t * delta_x_nt
        if (i>20) or (t==0):
            t = 0
            print('Backtracking line search (II) went beyond 20 iterations')
            break
    return t


def projection_Rp(x):
    y = np.maximum(x, np.zeros_like(x))
    return y


def newton_method(f, x0, S, sigma_x, rho, a2, mu1, T_list, epsilon=1e-6, alpha=0.01, beta=0.5,
                  regularized=False, backtracking=False, projection=False):
    """
    Implements Newton's method for optimization.

    Parameters:
    f (function): Function to minimize.
    grad_f (function): Gradient function.
    hessian_f (function): Hessian function.
    x0 (np.array): Initial point.
    epsilon (float): Stopping criterion for Newton decrement.
    alpha (float): Alpha parameter for backtracking.
    beta (float): Beta parameter for backtracking.

    Returns:
    x (np.array): Optimized point.
    """
    x = x0
    i = 0
    p=S.shape[0]
    theta_x = np.linalg.solve(sigma_x, np.eye(p))

    while True:
        B = sum(x[j]*T_list[j] for j in range(p))
        B_inv = np.linalg.solve(B, np.eye(p))
        
        gradient,hessian = compute_gH(x, S, rho, a2, mu1, T_list, B, B_inv,theta_x, regularized=regularized)
        
        # Compute Newton step and decrement
        delta_x_nt, lambd_sqr = compute_newton_step(gradient, hessian)
        
        # Stopping criterion
        if lambd_sqr / 2 <= epsilon:
            print('Stopping Criteria Met at iteration ', i)
            print(lambd_sqr, delta_x_nt)
            # print(i)
            break

        # Backtracking line search
        if backtracking:
            # print("line search")
            t = backtracking_line_search(f, gradient, x, delta_x_nt, S, sigma_x, rho, a2, mu1,
                                         regularized=regularized, alpha=alpha, beta=beta)
            print(f"line search step size t = {t:.2e}")
        else:
            t = 1
        # print(t)
            
        # Update x
        x_old = x.copy()
        x = x + t*delta_x_nt

        # Project x on R+
        if projection:
            x = projection_Rp(x)
        
        print(f"Newton iteration {i}. lambda_sqr: {lambd_sqr}, t: {t}, delta_x_nt: {delta_x_nt}")
        # if i%10==0:
        #     # print(lambd_sqr, delta_x_nt)
        #     # print(i)
        #     print('Newton algorithm finished 10 iterations, lambda_sqr: ', lambd_sqr)
            
        if (i>=50):
            print('Newton algorithm finished 50 iterations')
            break
        
        if (i>1) and (np.linalg.norm(x_old-x)<=epsilon):
            print('not much change in x observed')
            print(i)
            break
        i+=1
    return x

def ADMM_newton(f, iterations, rho, lambda_param, S, sigma_x, a, T_list, epsilon=1e-6,
                alpha=0.5, beta=0.7, regularized=True, backtracking=True,
                projection=True, perturbed=False, tol=1e-5):
    p = S.shape[0]
    e0 = np.zeros((p,))
    e0[0] = 1
    a1 = e0.copy()
    a2 = np.zeros((p-1,))
    mu1 = np.zeros((p-1,))
    if perturbed:
        a1 = a + 0.1*np.random.randn(p)
        a2 = a[1:] + 0.1*np.random.randn(p-1)
    if regularized:
        print('Regularized - ADMM + Newton')
        for i in range(iterations):       
            print('='*50, "iteration ", i, '='*50)
            a2_old = a2.copy()
            a1_ = newton_method(f, a1, S, sigma_x, rho, a2, mu1, T_list, epsilon=epsilon,
                                alpha=alpha, beta=beta, regularized=regularized,
                                backtracking=backtracking, projection=projection)
            print("Completed Newton Iteration")
            a2_ = soft(a1_[1:]+mu1,lambda_param/rho)
            mu1_ = mu1 + (a1_[1:]-a2_)
            a1, a2, mu1 = a1_, a2_, mu1_

            primal_res = np.linalg.norm(a1[1:] - a2)
            dual_res = np.linalg.norm(rho * (a2 - a2_old))
            # if(i%10==0):
            #     print(f"Primal: {primal_res}, Dual: {dual_res}, tol: {tol}")
            print(f"Iteration: {i}. Primal: {primal_res}, Dual: {dual_res}, tol: {tol}")
            if primal_res <= tol and dual_res <= tol:
                break

    else:
        print('Unregularized - Newton')
        for i in range(iterations):       
            print('='*50, "iteration ", i, '='*50)
            a1_ = newton_method(f, a1, S, sigma_x, 0, a2, mu1, T_list, epsilon=epsilon,
                                alpha=alpha, beta=beta, regularized=regularized,
                                backtracking=backtracking, projection=projection)
            a1 = a1_
    return a1, a2, mu1

class SolverArgs:
    def __init__(self, p=50, d=15, const=5, rho=1, lambda_param=0.01, iterations=50,
                 regularize=True, backtrack=True, project=True, perturb=False, num_rep=50, n_samples=None):
        self.p = p
        self.d = d
        self.const = const
        self.rho = rho
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.regularize = regularize
        self.backtrack = backtrack
        self.project = project
        self.perturb = perturb
        self.num_rep = num_rep
        self.n_samples = n_samples

class Solver:
    def __init__(self, args: SolverArgs):
        self.args = args

    def solve(self,epsilon=1e-4,alpha=0.1,beta=0.7,):
        p = self.args.p
        d = self.args.d
        const = self.args.const
        rho = self.args.rho
        lambda_param = self.args.lambda_param
        iterations = self.args.iterations
        regularize = self.args.regularize
        backtrack = self.args.backtrack
        project = self.args.project
        perturb = self.args.perturb
        num_rep = self.args.num_rep
        n_samples = self.args.n_samples

        if n_samples is None:
            n_samples = int(const * (d * d * np.log(p))) + 2 # minimum number
        print("Number of samples N:", n_samples)
        data = {}
        T_list=make_T_matrices(p, symmetric=True)

        print('Dimensions of Matrix = ', p, 'Sparsity = ', d)
        for rep in range(num_rep):
            a = create_sparse_vec_pos_def_2(p, d, diag=d * 10, random_state=rep)
            a=a/a[0]
            B0 = B_mat_symmetric(a, p)
            sigma_x = cov_x(p)
            sigma_y = cov_y(sigma_x, B0)
            y_samp = generate_y(np.zeros((p,)), sigma_y, n_samples, random_state= rep)
            S = samp_cov(y_samp)

            total_t1 = time()
            print('_' * 50, 'Run = ', rep, '_' * 50)
            a1, _, _ = ADMM_newton(
                f1,
                iterations,
                rho,
                lambda_param,
                S,
                sigma_x,
                a,
                T_list,
                epsilon=epsilon,
                alpha=alpha,
                beta=beta,
                regularized=regularize,
                backtracking=backtrack,
                projection=project,
                perturbed=perturb
            )
            total_t2 = time()
            print('Total time taken for a run = ', total_t2 - total_t1)

            run_data = {
                'p': p,
                'd': d,
                'const': const,
                'num_samples': n_samples,
                'lambda': lambda_param,
                'rho': rho,
                'iterations': iterations,
                'a': a,
                'samp_y': y_samp,
                'f_reg': regularize,
                'f_back': backtrack,
                'f_proj': project,
                'f_perturb': perturb,
                'a1': a1,
            }
            data[rep] = run_data

        return data

def evaluate(data, threshold_range=[0.01]):
        num_rep = len(data)
        # keys = list(data[0].keys())
        # threshold_range = 0.01 * np.arange(101)

        best_thr = []
        best_rate = []
        best_acc = []
        best_f1 = []
        best_cm = []

        for rep in range(num_rep):
            a = data[rep]['a']
            a1 = data[rep]['a1']
            true_a_binary = binarize_matrix(a)
            thr_rate = []
            thr_cm = []
            thr_acc = []
            thr_f1 = []

            for thr in threshold_range:
                a_thr = threshold(a1, threshold_value=thr)
                a_binary = binarize_matrix(a_thr)
                rate = compute_recovery_rate_numpy(a_binary, true_a_binary)
                cm = confusion_matrix(true_a_binary, a_binary)
                accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)

                thr_rate.append(rate)
                thr_cm.append(cm)
                thr_acc.append(accuracy)
                thr_f1.append(f1_score(true_a_binary, a_binary))

            idx = np.argmax(np.array(thr_f1))
            best_thr.append(threshold_range[idx])
            best_rate.append(thr_rate[idx])
            best_acc.append(thr_acc[idx])
            best_f1.append(thr_f1[idx])
            best_cm.append(thr_cm[idx])

        
        print(f"Best threshold: {best_thr}")
        

        metrics = {
            'avg_acc': average(best_acc),
            'avg_f1': average(best_f1),
            'best_thr': best_thr,
        }

        return metrics