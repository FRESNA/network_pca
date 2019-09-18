# -*- coding: utf-8 -*-


#from sklearn.decomposition import PCA as skpca
from .heuristics import renewable_mismatch, flow, renewables_i
from collections import namedtuple
import pandas as pd
import numpy as np
import pypsa
from pypsa.descriptors import Dict


class pcs:
    def __init__(self, n, data_in=None):
        if data_in is None:
            data_in = {'generation':n.generators_t.p
                                    .groupby(n.generators.bus, axis=1).sum(),
                       'flow': flow(n),
                       'mismatch': renewable_mismatch(n),
                       'topology': pypsa.allocation.PTDF(n)}
        for k, v in data_in.items():
            setattr(self, k, variance_pcs(v, abbrev=k[0]))
        n.pca = self


#def calculate_pc(ojectiv, return_extended=True, abbrev=None):
#    ''''Returns val, vec, beta, A_squared for default mode.
#    Returns val, vec, beta, A_squared, val_orig, cov, beta_normed for
#    extended mode. '''
#    data = skpca().fit(ojectiv)
#    pc = namedtuple('PrincipalComponents', ['val', 'val_total', 'vec',
#                                            'beta', 'norm', 'cov', 'abbr'])
#    pc.val = pd.Series(data.explained_variance_ratio_)
#    pc.vec = pd.DataFrame(data.components_, columns=ojectiv.columns).T
#    pc.beta = pd.DataFrame(data.transform(ojectiv), index=ojectiv.index)
#    pc.norm = 1/data.get_covariance().trace()
#    pc.val_total = pd.Series(data.explained_variance_)
#    pc.cov = pd.DataFrame(data.get_covariance())
#    pc.abbr = abbrev
#    return pc


def covariance(df):
    return df.T @ df

def eigvec(df):
    return pd.DataFrame(np.linalg.eig(df)[1], df.index, range(1, 1+len(df)))

def eigval(df):
    return pd.Series(np.linalg.eigvals(df), range(1, 1+len(df)))

def decomposition_pcs(df, abbrev=None):
    '''
    PCA without substracting mean without scaling the covariance matrix. The
    eigenvalues of those components reflect the real variance covered by the
    components.
    '''
    mean = df.mean()
    C = covariance(df)
    val = eigval(C)
    vec = eigvec(C)
    beta = df @ vec
    return Dict(vec=vec, val=val, beta=beta, mean=mean, C=C, abbr=abbrev)


def adjust_sign(pcs):
    sign = np.sign(pcs.beta.mean())
    pcs.vec *= sign
    pcs.beta *= sign
    return pcs

def variance_pcs(df, abbrev=None):
    '''
    Ordinary PCA with substracting mean from original data set and scaling of the
    Covariance matrix.
    '''
    mean = df.mean()
    C = covariance(df - mean).pipe(lambda C: C/np.trace(C))
    val = eigval(C)
    vec = eigvec(C)
    beta = df @ vec
    return Dict(vec=vec, val=val, beta=beta, mean=mean, C=C, abbr=abbrev)


def approximate(pcs, M=0):
    '''
    Approximation for decomposition components, with updated mean value. The
    latter changes as it consists of the mean values of none included components.
    '''
    vec = pcs.vec.loc[:, :M]
    val = pcs.val[:M]
    beta = pcs.beta.loc[:, :M]
    mean = pcs.mean
    mean = mean - (mean @ vec * vec).sum(1)
    return Dict(vec=vec, val=val, beta=beta, mean=mean)



def wrap_fft(df, framefunction_hanning=True, logx=True):
    if framefunction_hanning:
        framefunc = np.hanning(len(df))
    else:
        framefunc = 1
    df_f = pd.DataFrame({c:np.fft.rfft(df[c]*framefunc) for c in df}).apply(
                                     lambda df:(2.0/len(df)*abs(df))**2)
    if logx:
        df_f.index=np.log(1./np.fft.rfftfreq(len(df), 1.0)/24.)
    else:
        df_f.index=1./np.fft.rfftfreq(len(df), 1.0)/24.
    return df_f



def majorization_theorem(vt, vp, vf, normed=True):
    # drop last zero eigenvalue
    vt = vt[:-1]
    vf = vf[:len(vt)]
    vp = vp[:-1]

    vt_r = pd.Series(vt.sort_values(ascending=True).values)

    left = (vp * vt_r).sort_values().reset_index(drop=True)
    middle = vf
    right = vp * vt

    if normed:
        left, middle, right = map(
            lambda x: x/middle.sum(), [left, middle, right])

    majorization = pd.DataFrame({
        r'$\lambda^p_\downarrow \circ \mathbf{s}_\uparrow$': left,
        r'$\lambda^f$': middle,
        r'$\lambda^p_\downarrow \circ \mathbf{s}_\downarrow$': right})
    majorization.index += 1

    return majorization.apply(pd.Series.cumsum)
