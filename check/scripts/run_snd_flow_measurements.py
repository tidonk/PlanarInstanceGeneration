#!/usr/bin/env python3
"""Measure fixed-capacity SND routing flows on prepared composites."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import socket
import time

import gurobipy as gp
from gurobipy import GRB


def stats(values):
    values = sorted(values)
    def quantile(q):
        i = (len(values) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        return values[lo] + (i-lo) * (values[hi]-values[lo])
    return dict(count=len(values), minimum=values[0], mean=math.fsum(values)/len(values),
                median=quantile(.5), p95=quantile(.95), maximum=values[-1])


def solve(data, destination, baseline=False, *, time_limit=3600):
    started = time.monotonic()
    nodes = data['nodes']
    edges = data['edges']
    demands = [d for d in data['demands'] if d['value'] > 0]
    position = {n: i for i, n in enumerate(nodes)}
    incident = [[] for _ in nodes]
    for j, e in enumerate(edges):
        incident[position[e['source']]].append((j, 1))
        incident[position[e['target']]].append((j, -1))
    capacities = [0.0 if baseline and e['coupling'] else e['capacity'] for e in edges]
    mode = 'baseline' if baseline else 'enabled'
    with gp.Env(params={'OutputFlag': 1, 'LogToConsole': 0, 'LogFile': str(destination / (mode+'.log'))}) as env:
        with gp.Model('snd_fixed_capacity_routing', env=env) as model:
            model.Params.Threads = 1
            model.Params.Seed = 0
            model.Params.FeasibilityTol = 1e-8
            model.Params.OptimalityTol = 1e-8
            model.Params.SoftMemLimit = 6
            model.Params.TimeLimit = time_limit
            f = model.addVars(len(demands), len(edges), 2, lb=0, name='f')
            model.addConstrs((gp.quicksum(f[k,j,0]+f[k,j,1] for k in range(len(demands))) <= capacities[j]
                             for j in range(len(edges))), name='capacity')
            for k, d in enumerate(demands):
                model.addConstrs((gp.quicksum(sign*(f[k,j,0]-f[k,j,1]) for j,sign in incident[i]) ==
                                  (d['value'] if n == d['source'] else -d['value'] if n == d['target'] else 0)
                                  for i,n in enumerate(nodes)), name=f'balance_{k}')
            cost = gp.quicksum(e['cost']*(f[k,j,0]+f[k,j,1]) for k in range(len(demands)) for j,e in enumerate(edges))
            model.setObjective(cost, GRB.MINIMIZE)
            model.update()
            build_seconds = time.monotonic() - started
            print(f'{mode}: built {model.NumVars} variables in {build_seconds:.2f}s', flush=True)
            model.optimize()
            primary_status = int(model.Status)
            primary_seconds = float(model.Runtime)
            print(f'{mode}: primary status {primary_status}, {primary_seconds:.2f}s', flush=True)
            if model.Status != GRB.OPTIMAL:
                raise RuntimeError(f'{mode}: routing cost not optimal, status={model.Status}')
            optimum = float(model.ObjVal)
            summary = dict(mode=mode, primary_objective=optimum, primary_status=primary_status,
                           build_seconds=build_seconds, primary_seconds=primary_seconds,
                           variables=int(model.NumVars), constraints=int(model.NumConstrs))
            if baseline:
                expected = data['baseline_objective']
                error = abs(optimum-expected)/max(1, abs(expected))
                summary.update(expected_objective=expected, relative_error=error)
                if error > 1e-7:
                    raise ValueError(f'baseline differs from archived result: {optimum} versus {expected}')
                return summary
            tolerance = max(1e-7, abs(optimum)*1e-9)
            model.addConstr(cost <= optimum+tolerance, name='minimum_routing_cost')
            model.setObjective(gp.quicksum(f.values()), GRB.MINIMIZE)
            model.optimize()
            print(f'{mode}: secondary status {model.Status}, {model.Runtime:.2f}s', flush=True)
            if model.Status != GRB.OPTIMAL:
                raise RuntimeError(f'minimum total flow not optimal, status={model.Status}')
            summary.update(secondary_seconds=float(model.Runtime), secondary_objective=float(model.ObjVal),
                           secondary_status=int(model.Status), routing_cost_tolerance=tolerance)
            values = model.getAttr(GRB.Attr.X, f)
            loads = [0.0]*len(edges)
            largest_balance_residual = 0.0
            largest_balance_scaled_residual = 0.0
            largest_negative_flow = 0.0
            with gzip.open(destination/'commodity_flows.csv.gz', 'wt', newline='') as out:
                writer = csv.writer(out)
                writer.writerow(['demand','edge','forward','reverse'])
                for k,d in enumerate(demands):
                    balances = [0.0]*len(nodes)
                    for j,e in enumerate(edges):
                        forward, reverse = values[k,j,0], values[k,j,1]
                        largest_negative_flow = max(largest_negative_flow, -forward, -reverse)
                        loads[j] += forward+reverse
                        net = forward-reverse
                        balances[position[e['source']]] += net
                        balances[position[e['target']]] -= net
                        if abs(forward)>1e-10 or abs(reverse)>1e-10:
                            writer.writerow([d['id'],e['id'],format(forward,'.17g'),format(reverse,'.17g')])
                    for i,n in enumerate(nodes):
                        expected = d['value'] if n==d['source'] else -d['value'] if n==d['target'] else 0
                        residual = abs(balances[i]-expected)
                        largest_balance_residual = max(largest_balance_residual,residual)
                        largest_balance_scaled_residual = max(largest_balance_scaled_residual,residual/max(1,d['value']))
            overflow = max(max(0, load-cap)/max(1,cap) for load,cap in zip(loads,capacities))
            reconstructed_cost = math.fsum(e['cost']*load for e,load in zip(edges,loads))
            assert largest_negative_flow <= 1e-6
            assert largest_balance_scaled_residual <= 1e-6
            assert overflow <= 1e-6
            assert abs(reconstructed_cost-optimum) <= tolerance*2+1e-5
            assert optimum <= data['baseline_objective']+max(1,abs(data['baseline_objective']))*1e-7
            with (destination/'edge_flows.csv').open('w',newline='') as out:
                writer = csv.writer(out)
                writer.writerow(['edge','source','target','coupling','capacity','routing_cost','aggregate_flow','utilization'])
                for e,load,cap in zip(edges,loads,capacities):
                    writer.writerow([e['id'],e['source'],e['target'],e['coupling'],cap,e['cost'],load,load/cap if cap else ''])
            coupling = [max(0,load) for e,load in zip(edges,loads) if e['coupling']]
            inherited = [max(0,load) for e,load in zip(edges,loads) if not e['coupling']]
            cs, ins = stats(coupling),stats(inherited)
            summary.update(coupling=cs,inherited=ins,
                           ratios={k:cs[k]/ins[k] if ins[k] else None for k in ['minimum','mean','median','p95','maximum']},
                           positive_coupling_edges=sum(x>1e-8*max(1,ins['mean']) for x in coupling),
                           max_balance_residual=largest_balance_residual,
                           max_scaled_balance_residual=largest_balance_scaled_residual,
                           max_scaled_capacity_violation=overflow,
                           reconstructed_routing_cost=reconstructed_cost,
                           baseline_objective=data['baseline_objective'],
                           relative_cost_change=(optimum/data['baseline_objective']-1) if data['baseline_objective'] else None,
                           wall_seconds=time.monotonic()-started)
            return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--validate-baseline',action='store_true')
    parser.add_argument('--time-limit', type=float, default=3600, help='Seconds per optimization step.')
    parser.add_argument('--memory-limit-gib', type=float, default=8, help='Process address-space limit; zero disables it.')
    args=parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error('--time-limit must be finite and positive')
    if not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib < 0:
        parser.error('--memory-limit-gib must be finite and nonnegative')
    if args.memory_limit_gib:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = int(args.memory_limit_gib * 1024**3)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    with gzip.open(args.input,'rt') as f:
        data=json.load(f)
    result=dict(stem=data['stem'],strategy=data['strategy'],input_index=data['input_index'],
                host=socket.gethostname(),solver_version=gp.gurobi.version(),threads=1,
                solver_limit_seconds=args.time_limit,
                input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                model='continuous fixed-capacity minimum-cost multicommodity flow, no failures')
    try:
        if args.validate_baseline:
            result['baseline_validation']=solve(data,args.output,baseline=True,time_limit=args.time_limit)
        result.update(solve(data,args.output,time_limit=args.time_limit))
        result['success']=True
    except Exception as error:
        result.update(success=False,error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['total_seconds']=time.monotonic()-started
        (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result.get(k) for k in ['stem','success','total_seconds','ratios','error']},allow_nan=False))

if __name__=='__main__':
    main()
