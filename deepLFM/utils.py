import torch

def DKL_Gaussian(m_q, lv_q, m_p, lv_p):
    """ Returns the Kullback Leibler divergence for MV Gaussian distributions
        q and p with diagonal covariance matrices.

    :param m_q: Means for q
    :param lv_q: Log-variances for q
    :param m_p: Means for p
    :param lv_p: Log-variances for p
    :return: KL(q||p)
    """
    # Flatten tensors to 1D
    m_q, m_p = m_q.view(-1), m_p.view(-1)
    lv_q, lv_p = lv_q.view(-1), lv_p.view(-1)

    # Compute constituent terms of DKL
    term_a = lv_p - lv_q
    term_b = torch.pow(m_q - m_p, 2) / torch.exp(lv_p)
    term_c = torch.exp(lv_q - lv_p) - 1

    return 0.5 * torch.sum(term_a + term_b + term_c)