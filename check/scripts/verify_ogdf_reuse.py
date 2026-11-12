#!/usr/bin/env python3
"""Check OGDF reuse against small instances with known exact skewness."""
from itertools import combinations
from pathlib import Path
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
from ogdf_validation import certificate_input,graph_input,invoke,CertifiedPlanarPiece,write_json

def optimum_solution(g,value):
    # The values supplied below also have independent lower bounds:
    # bipartite Euler bounds, and additivity over two nonplanar blocks.
    edges=list(g.edges)
    for omitted in combinations(edges,value):
        h=g.copy();h.remove_edges_from(omitted)
        if nx.check_planarity(h)[0] and nx.is_connected(h):return h
    raise ValueError('no solution at claimed optimum')

def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
    driver=Path(sys.argv[2]).resolve()
    joined=nx.disjoint_union(nx.complete_bipartite_graph(3,3),nx.complete_bipartite_graph(3,3));joined.add_edge(0,6)
    cases=[('K34',nx.complete_bipartite_graph(3,4),2),('K44',nx.complete_bipartite_graph(4,4),4),('two_K33',joined,2)]
    shuffled=nx.Graph();nodes=list(range(8));random.Random(41).shuffle(nodes);shuffled.add_nodes_from(nodes)
    edges=list(nx.complete_bipartite_graph(4,4).edges);random.Random(17).shuffle(edges);shuffled.add_edges_from(edges)
    cases.append(('K44_reordered',shuffled,4));jobs=[]
    for name,g,value in cases:
        h=optimum_solution(g,value);graph=out/f'{name}.graph';cert=out/f'{name}.cert'
        mapped=graph_input(g,graph)
        certificate_input(g,(CertifiedPlanarPiece(g,h,value-1),),(),7,cert)
        item={'n':len(g),'m':g.number_of_edges(),'edges':mapped,'components':1,'historical_bounds':[value,value]}
        for mode in range(4):jobs.append((driver,graph,cert,out/f'{name}.{mode}.json',10,7,mode,40+mode,item,value))
    records={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(invoke,*job[:-1]):job for job in jobs}
        for f in as_completed(fs):
            job=fs[f];r=f.result()
            assert r['deletion_lower_bound']==r['deletion_upper_bound']==job[-1],job[3]
            records[job[3].stem]=r
    write_json(out/'verification.json',{'passed':len(records),'records':records})
    print('All 16 exact comparisons passed.')
if __name__=='__main__':main()
