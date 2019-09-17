#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:56:56 2016

@author: fabian
"""
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

from .analysis import wrap_fft
from .heuristics import renewable_mismatch, renewables_i
from cartopy import crs as ccrs
import numpy as np
import pandas as pd
import seaborn as sns


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#sns.set_style('whitegrid')
colors=sns.color_palette()


xlim=(-11,32)
ylim=(35,72)
xlim_red=(-11,25)
ylim_red=(36,65)


def components(n, region_data=None, line_data=None, index=None,
               plot_colorbar=False, figsize=(20,6),
               cmap='RdBu_r', subplots=(1,3),
               bounds=[-10. , 45, 36, 72], line_widths={'Line':1, 'Link':1},
               line_colors={'Line':'darkgreen', 'Link':'darkgreen'},
               plot_regions=None,
               regionsline_width=0.005, title_prefix='',
               ev_str='\lambda',
               starting_component=1, busscale=0.1,
               colorbar_kw={}, flow_quantile=0.):
    """
    This plots the principal components of the network.
    """

    if isinstance(region_data, str):
        region_data = getattr(n.pca, region_data)
    if isinstance(line_data, str):
        line_data = getattr(n.pca, line_data)

    if plot_regions is None:
        if 'regions' in n.__dir__():
            regions = n.regions
            plot_regions = True
        else:
            plot_regions = False

    if index is None:
        index = region_data.abbr if region_data is not None else line_data.abbr

    crs = ccrs.EqualEarth()
    fig, axes = plt.subplots(*subplots, figsize=figsize, squeeze=0,
                             subplot_kw={"projection":crs})
    for i in range(axes.size):
        ax = axes.flatten()[i]
        region_comp = get_comp(region_data, i+starting_component)
        region_comp = pd.Series(region_comp, index=n.buses.index).fillna(0)
        if line_data is None:
            line_comp = None
        else:
            line_comp = get_comp(line_data, i+starting_component)
            line_comp = (line_comp.div(line_comp.abs().max())
                         .where(lambda ds: ds.abs() >
                                ds.abs().quantile(flow_quantile),0))
        if plot_regions:
            region_comp /= region_comp.abs().max()
            regions.loc[:,'weight'] = region_comp
            regions.plot(cmap=cmap, column='weight', ax=ax, vmin=-1., vmax=1.,
                         linewidth=regionsline_width*figsize[0], edgecolor='k',
                         transform=ccrs.PlateCarree())
            region_comp[:] = 0

        n.plot(ax=ax,
               bus_sizes=region_comp.abs()*busscale,
               flow=line_comp,
               line_widths=line_widths,
               line_colors = line_colors,
               bus_colors=np.sign(region_comp),
               bus_cmap=cmap,
               boundaries=bounds,
               geomap=True)

        val = region_data.val if region_data is not None else line_data.val
        ax.set_title(fr'{title_prefix}${ev_str}_{i+starting_component}'
                        fr' = {round(val.loc[i+starting_component], 2)}$')
        ax.set_facecolor('white')
    fig.canvas.draw()
    fig.tight_layout(w_pad=7.)
    fig.colorbar(plt.cm.ScalarMappable(Normalize(-1,1), cmap=cmap), ax=axes,
                 **colorbar_kw)
    return fig, axes


def get_comp(pca, index):
    if pca is None:
        return None
    else:
        return pca.vec[index]


def fourrier_and_daytime(pcs, figsize=(7,3), subplots=(2,3)):
    beta, val = pcs.beta, pcs.val
    beta_fourier = wrap_fft(beta.loc[:,:6])
    with sns.axes_style('darkgrid'):

        fig, axes = plt.subplots(*subplots, sharey=False, sharex=False,
                               squeeze=0, figsize=figsize)
        xt = [0.2, 0.5, 1, 2, 3.5, 7, 14, 28, 56,102, 182, 365]
        colors=sns.color_palette()
        ax_iter = axes.flat
        for i in range(axes.size//2):
            ax = next(ax_iter)
            ax.set_xlim(np.log(0.2), np.log(600))
            ax.plot(beta_fourier.index, beta_fourier.loc[:,i], c=colors[i])
        #    ax.set_ylim(top=1.2 * beta_fourier.loc[:,i].max())
            ax.set_xticks(np.log(xt))
            ax.set_xticklabels(xt, rotation=75)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        #    ax.set_yscale('log')
    #        ax.locator_params(axis='y',nbins=4)  #specify number of ticks
            if i==0:
                ax.set_ylabel(r"PSD $[MW^2]$")
            ax.set_xlabel(r"period [days]")
            ax.set_title(r'$\lambda^{}_{} = {:.2}$'.format(pcs.abbr,
                         i+1, val.loc[i]))

        for p in range(axes.size//2):
            ax = next(ax_iter)
            hmean = pd.Series([beta.loc[beta.index.hour==i, p].mean()
                               for i in range(24)])
            hmean.plot(ax=ax, c=colors[p])
            ax.set_xticks(range(24)[::3])
            ax.set_xticklabels((hmean.index.astype(str) + ':00')[::3], rotation=75)
            ax.locator_params(axis='y',nbins=5, tight=False)
            ax.set_xlabel('daytime [h]')
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            ax.set_ylim(- max(abs(ax.get_yticks())), max(abs(ax.get_yticks())))
            if (p==0 or p==3):
                ax.set_ylabel(r'$\left<\, \beta_k(t=h)\; \right>_t$')
        fig.tight_layout(pad=0.4)
    #    plt.subplots_adjust(hspace=0.5)
        return fig, ax



def fourrier(pcs, subplots=(3,2), figsize=(7,3), ev_str=r'$\lambda'):
    beta, val = pcs.beta, pcs.val
    beta_fourier = wrap_fft(beta.loc[:,:6])
    fig, ax = plt.subplots(*subplots, sharey=False, sharex=True, figsize=figsize)
    xt = [0.2, 0.5, 1, 2, 3.5, 7, 14, 28, 56,102, 182, 365]
    colors=sns.color_palette()
    if sum(subplots)>2:
        ax_iter = ax.flat
    else:
        ax_iter = np.array(ax).flat
    for i in beta.columns[:len(ax_iter)]:
        ax = next(ax_iter)
        ax.set_xlim(np.log(0.2), np.log(600))
        ax.plot(beta_fourier.index.values, beta_fourier[i], color=colors[i])
        ax.hlines(0, *ax.get_xlim(), linestyles='dashed', alpha=0.4,
                  color='grey')
        ax.set_xticks(np.log(xt))
        ax.set_xticklabels(xt, rotation=75)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax.locator_params(axis='y',nbins=4)  #specify number of ticks
        if i%(subplots[1])==0:
            ax.set_ylabel(r"PSD $[MW^2]$")
        if i > subplots[0] * (subplots[1] - 1):
            ax.set_xlabel(r"period $[days]$")
        ax.set_title(r'${}_{} = {:.2}$'.format(ev_str, i+1, val.loc[i]))
    fig.tight_layout(pad=0.5)
    return fig, ax


def yearly_profile(pcs, std_free=False, weeks=4, ev_str=r'\lambda',
                   figsize=(7,4.), subplots=(2, 2), rolling=False):
    beta, val = pcs.beta, pcs.val


    fig, ax = plt.subplots(*subplots, figsize=figsize)
    if rolling:
        mean = beta.loc[:,:5].rolling(7*24*weeks, center=True).mean()
        #std just for a quarter of mean window
        std = beta.loc[:,:5].rolling(7*6*weeks, center=True).std()
        title = r'$beta$ monthly window mean'

    else:
        mean = beta.loc[:,:5].resample(f'{weeks}w').mean()
        #std just for a quarter of mean window
        std = beta.loc[:,:5].resample(f'{weeks}w').std()
        title = rf'$\beta$ averaged over {weeks} weeks'

    ax_iter = ax.flat

    for p in beta.columns[:len(ax_iter)]:
        ax = next(ax_iter)
#        ax.set_ylim(-np.abs(mean[p]).max(), np.abs(mean[p]).max())
        line = mean[p].plot(ax=ax, sharex=True, color=colors[p])
        ax.fill_between(mean.index, (mean-std)[p], (mean+std)[p], alpha=0.4,
                        color=line.lines[0].get_color())
        ax.hlines(0, *ax.get_xlim(), linestyles='dashed', alpha=0.5,
                  color='grey')
        ax.locator_params(axis='y',nbins=5, tight=False)
        ax.set_xlabel('')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax.set_title(r'$ %s_{%s} = %.2f$'%(ev_str, p, val.loc[p]))
        if p%(subplots[1])==0:
            ax.set_ylabel(title)
    fig.tight_layout(pad=0.5)
    return fig, ax



def daytime_profile(pcs, std_free=False, subplots=(2,3),
                    ev_str=r'\lambda', figsize=(7,4)):
    beta, val = pcs.beta, pcs.val
    fig, ax = plt.subplots(*subplots, figsize=figsize)
    ax_iter = ax.flat
    beta_string = r'\tilde{\beta}' if std_free else r'\beta'

    for p in beta.columns[:len(ax_iter)]:
        ax = next(ax_iter)
        data = beta[p]
        mean = data.groupby(beta.index.hour).mean()
        std = data.groupby(beta.index.hour).std()

        line = mean.plot(ax=ax, c=colors[p], sharex=True)
        ax.fill_between(mean.index, mean-std, mean+std, alpha=0.5,
                        color=line.lines[0].get_color())
        ax.hlines(0, *ax.get_xlim(), linestyles='dashed', alpha=0.5,
                  color='grey')
        ax.set_xticks(range(24)[::3])
        ax.set_xticklabels((mean.index.astype(str) + ':00')[::3], rotation=75)
        ax.locator_params(axis='y',nbins=5, tight=False)
        ax.set_xlabel('daytime h')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax.set_ylim(- max(abs(ax.get_yticks())), max(abs(ax.get_yticks())))
        ax.set_title(r'$ %s_{%s} = %.2f$'%(ev_str, p, val.loc[p]))
        if p%(subplots[1])==1:
            ax.set_ylabel(
            r'$\left<\, %s_k(t=h)\; \right>_t$'%beta_string)
    fig.tight_layout(pad=0.5)
    return fig, ax


def weekly_mean(pcs, figsize=(7,4), subplots=(2,2), ev_str=r'\lambda'):
    beta, val = pcs.beta, pcs.val
    fig, ax = plt.subplots(*subplots,  figsize=figsize)
    data = (beta.groupby((beta.index.dayofweek) * 24 +
                         (beta.index.hour)).mean()\
                .rename_axis('Hour of week'))
    ax_iter = ax.flat

    for p in data.columns[:len(ax_iter)]:
        ax = next(ax_iter)
#        ax.set_ylim(-np.abs(data[p]).max(), np.abs(data[p]).max())
        data[p].plot(ax=ax, sharex=True, color=colors[p])
        ax.hlines(0, *ax.get_xlim(), linestyles='dashed', alpha=0.5,
                  color='grey')
        ax.locator_params(axis='y',nbins=5, tight=False)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax.set_title(r'$ %s_{%s} = %.2f$'%(ev_str, p, val.loc[p]))
        if p%(subplots[1])==1:
            ax.set_ylabel(r'$\left<\beta\right>$ hour of week')
    fig.tight_layout(pad=0.5)
    return fig, ax


def farmer_and_cumsum(pcs, head=None, tail=None):
    val = pcs.val
    K = (val.cumsum() <= .9).sum().sum()+1
    if head is None:
        if int(int(np.log10(len(val))) * K)<len(val)/2.5:
            head = int(int(np.log10(len(val))) * K)
        else:
            head = int(1.1 * K)
    if tail is None:
        tail = int(0.8*head)
    fig, ax = plt.subplots(1,2)
    ax1, ax2 = ax.tolist()
    fig.set_size_inches(7,2.5)
    val_shifted = val.copy()
    val_shifted.index = val_shifted.index+1

    np.log(val_shifted).plot(ax=ax1, legend=None, linestyle='-')
    linfunct = np.polyfit(val_shifted.index[head:-tail:int((len(val)/30.))],
                    np.log(val_shifted)[head:-tail:int((len(val)/30.))],1)
    pd.DataFrame(linfunct[1]+ val_shifted.index.to_series()*linfunct[0]
                    ).plot(ax=ax1, legend=None)
    ax1.set_xlim(0,len(val))
    ax1.set_ylabel(r'$log(\lambda_k)$')
    ax1.set_xlabel(r'$k$')
    ax1.set_xticklabels(ax1.get_xticks().astype(int), rotation=75)

    upperlimit=len(val)
    cumsum = np.cumsum(val[:upperlimit])
    cumsum.index = cumsum.index + 1
    ax2.plot(pd.Series(data=[0]).append(cumsum))
    ax2.plot(pd.Series(data=0.9, index=range(upperlimit+1)), '--' )
    ax2.set_xlim(0,upperlimit)
    ax2.set_ylim(0.,1.)
    ax2.set_ylabel(r'$\sum\lambda_k$')
    ax2.set_xlabel(r'$k$')
    plt.tight_layout(pad=0.5)
    return fig, ax



def duration_curve(n, figsize=(7,4)):
    mismatch = renewable_mismatch(n)
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1,1, figsize=figsize)
        Gmismatch =  ((-mismatch).sum(axis=1).sort_values(ascending=False)
                    .reset_index(drop=True))
        duration_index = (-mismatch).sum(axis=1).sort_values(ascending=False).index

        conventionals_i = n.generators.index.difference(renewables_i(n))
#        merit_rank = ['Run-of-river', 'Lignite', 'Nuclear', 'Coal', 'CCGT', 'OCGT', 'Hydro']
        (n.generators_t.p.reindex(columns=conventionals_i)
                     .groupby(n.generators.carrier, axis=1).sum()
#                     .reindex(columns=merit_rank)
                     .reindex(duration_index)
                     .reset_index(drop=True)
                     .loc[Gmismatch>0.]
                     .plot.area(stacked=True, ax=ax,
                                color=sns.color_palette(n_colors=7),
                                linewidth=0))
#        Gmismatch.name = 'Mismatch'
#        Gmismatch.plot(ax=ax, zorder=3)
        curtailment = Gmismatch.loc[lambda ds : ds<0.]
        curtailment.name = 'Curtailment'
        curtailment.plot.area(color='darkorange')
        ax.set_ylim((1*(-mismatch).sum(axis=1).min(), 1.1*(-mismatch).sum(axis=1).max()))
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:-1][::-1]+[handles[-1]]
        labels = labels[:-1][::-1]+[labels[-1]]
        ax.legend(handles, labels, title=None, frameon=True)
        ax.set_xlabel('h')
        ax.set_ylabel('Power [MW]')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        fig.tight_layout(pad=0.5)
    return fig, ax
