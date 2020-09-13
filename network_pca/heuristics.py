#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 22:16:54 2017

@author: fabian
"""

import numpy as np
import pandas as pd
import pypsa


def renewables_i(n):
    return n.generators_t.p_max_pu.columns


def renewable_mismatch(n):
    return (n.generators_t.p[renewables_i(n)]
            .groupby(n.generators.bus, axis=1).sum()
            - n.loads_t.p_set).dropna(axis=1)


def renewable_generation(n):
    return (n.generators_t.p[renewables_i(n)]
            .groupby(n.generators.bus, axis=1).sum())


def flow(n):
    return pd.concat([n.lines_t.p0, n.links_t.p0], keys=['Line', 'Link'],
                     axis=1, sort=False)



def set_country_alpha(n, alpha=None, networksize=None, minimized_bus_variance=True):
    from os.path import dirname
    renewables_b = n.generators.carrier.isin(['solar', 'windon', 'windoff'])
    solar_b = n.generators.carrier.isin(['solar'])
    wind_b = n.generators.carrier.isin(['windon', 'windoff'])

    country_map = n.generators.bus.map(n.buses.country)
    loads_country = n.loads_t.p_set.sum().groupby(n.buses.country).sum()

    n.generators_t.p_set.loc[:, renewables_b] = \
        n.generators_t.p_max_pu.loc[:, renewables_b].copy()
    n.generators_t.p_set.loc[:, wind_b] *= \
        country_map.map(alpha * loads_country /
                        n.generators_t.p_set.loc[:, wind_b]
                        .groupby(country_map, axis=1).sum().sum())[wind_b]
    n.generators_t.p_set.loc[:, solar_b] *= \
        country_map.map((1-alpha) * loads_country /
                        n.generators_t.p_set.loc[:, solar_b]
                        .groupby(country_map, axis=1).sum().sum())[solar_b]



def harmonized_nodal_balancing(network, proportional_to='mixed'):
    """
    Define the balancing scheme according to harmonized balancing. By default
    (proportional_to='mixed') mode, this sets the curtailment proportional
    to the current nodal surplus and the backup proportional to the nodal
    average load. Setting proportional_to='mismatch', both curtailment and
    backup are set proportional to the current nodal mismatch.
    """

    generation = renewable_generation(network)
    mismatch = renewable_mismatch(network)
    carrier_t = network.generators_t.p_set.loc[:, renewables_i(network)]
    loads = network.loads_t.p_set
    total_curtailment = mismatch.sum(axis=1).clip_lower(0.)
    if proportional_to is None:
#        curtailment = (pd.DataFrame(0, mismatch.index, mismatch.columns)
#                       mismatch.mean(1).clip_lower(0.))
        c_mean = mismatch.mean(1).clip_lower(0.)
        curtailment = (pd.DataFrame({c:c_mean for c in mismatch.columns}))
#        assert curtailment.sum(1).equals(total_curtailment)
    else:
        curtailment = (mismatch.clip_lower(0.)
                       .pipe(lambda df: df.div(df.sum(axis=1), axis=0))
                       .mul(total_curtailment, axis=0)).fillna(0)
    total_backup = mismatch.sum(axis=1).clip_upper(0).abs()
    if proportional_to == 'mixed':
        backup = (pd.DataFrame(
                  np.outer(total_backup,
                           loads.mean().pipe(lambda df: df/df.sum())),
                  index=loads.index,
                  columns=loads.columns))
    elif proportional_to == 'mismatch':
        backup = (mismatch.clip_upper(0.).abs()
                  .pipe(lambda df: df.div(df.sum(axis=1), axis=0))
                  .mul(total_backup, axis=0)).fillna(0)
    elif proportional_to is None:
        b_mean = mismatch.mean(1).clip_upper(0.).abs()
        backup = (pd.DataFrame({c:b_mean for c in mismatch.columns}))

    network.generators_t.p_set.loc[:, renewables_i(network)] = (
        reduce_generation_by_curtailement(carrier_t, generation,
                                          curtailment, network))
#    balancing = curtailment - backup
#    injection = mismatch - balancing
    generator_backup = backup.rename(
            columns=network.generators[network.generators.carrier == "OCGT"]
            .reset_index(drop=False).set_index('bus')['index'])
    network.import_series_from_dataframe(
        generator_backup, 'Generator', 'p_set')
    network.generators.loc[network.generators.carrier ==
                           "OCGT", 'p_nom'] = generator_backup.max()
    return curtailment



def reduce_generation_by_curtailement(carrier_t, generation, curtailement, network):
    ratio_curtailed = (
        generation-curtailement).div(generation).where(generation > 0, 0.)
    ratio_curtailed = ratio_curtailed.reindex(
        columns=network.generators.bus.values)
    ratio_curtailed.columns = network.generators.bus.index
    ratio_curtailed = ratio_curtailed.loc[:, carrier_t.columns]
    curtailed_gen = carrier_t.mul(ratio_curtailed)
    return curtailed_gen



def convert_dc_to_ac(network):
    from collections import namedtuple
    LineParam = namedtuple("LineParam", ["v_nom", "wires", "r", "x", "c", "i"])
    p = LineParam(v_nom=300., wires=3.0, r=0.04, x=0.265, c=13.2, i=1.935)
    newlines = network.links.rename(
        columns={'p_nom': 's_nom', 's_nom_max': 'p_nom_max'})
    circuits = newlines.s_nom * \
        newlines.circuits.fillna(1.) / (np.sqrt(3.) * p.v_nom * p.i)
    length = newlines.loc[:, "length"].fillna(50.)
    newlines.loc[:, "r"] = length * p.r / circuits
    newlines.loc[:, "x"] = length * p.x / circuits
    newlines.loc[:, "b"] = length * (2*np.pi*50*1e-9) * p.c * circuits
    newlines.loc[:, 'g'] = 0
    newlines.loc[:, 'g_pu'] = 0
    newlines.loc[:, 'num_parallel'] = 1
    newlines.loc[:, 's_nom_max'] = np.inf
    newlines.loc[:, 's_nom_min'] = 0
    newlines.loc[:, 's_nom_opt'] = 0
    newlines.loc[:, 'v_ang_max'] = np.inf
    newlines.loc[:, 'v_ang_min'] = -np.inf
    newlines = newlines.reindex(columns=network.lines.columns)
    newlines.index = pd.RangeIndex(len(network.lines.index)+1,
                                   len(network.lines.index) + len(newlines)+1)
    network.links = network.links.drop(network.links.index)
    network.import_components_from_dataframe(newlines, 'Line')


