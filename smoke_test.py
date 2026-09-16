#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parent
def load(name,rel):
 p=ROOT/rel;sys.path.insert(0,str(p.parent))
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 import numpy as np
 import pandas as pd
 import torch
 torch.set_num_threads(1)
 checks=[]
 boundary=load('boundary','tb/boundary/scripts/boundary_distance.py')
 ring=np.zeros((32,32),bool);ring[4:28,4:28]=True;ring[10:22,10:22]=False
 assert not boundary.fill_small_holes(ring,143)[16,16]
 assert boundary.fill_small_holes(ring,144)[16,16]
 cells=pd.DataFrame({'x_centroid':[0.,9.9,10.1],'y_centroid':[0.,0.,0.]})
 counts=boundary.rasterize(cells,2,2,0.,0.)
 assert counts.sum()==3 and counts[0,0]==2 and counts[0,1]==1
 checks.append('TB: inclusive hole-area threshold and counted 10-micron raster bins')
 nerve=load('nerve','nerve/define/scripts/schwann_clusters.py')
 n,lab=nerve._linkage_labels(np.array([[0.,0.],[1.,0.],[8.,0.]]),2.)
 assert n==2 and lab[0]==lab[1] and lab[1]!=lab[2]
 checks.append('Nerve: graph connectivity preserves separated components')
 tls=load('tls','tls/define/scripts/tls_maturity.py')
 assert [tls.classify_tls_state(b,d) for b,d in [(1,1),(1,-1),(-1,1),(-1,-1)]]==['Mature TLS','FDC-deficient TLS','Premature TLS','Early TLS']
 checks.append('TLS: four molecular states and release-root import')
 load('tls_regions','tls/define/scripts/build_regions.py')
 checks.append('TLS reporting: local geometry dependency imports')
 model=load('tls_prediction','tls/prediction/scripts/train.py')
 torch.manual_seed(42);m=model.GatedAttentionHead(8,2).eval()
 center=torch.randn(3,8);nbr=torch.randn(3,24,8);mask=torch.zeros(3,24);oh=torch.zeros(3,2)
 with torch.no_grad():
  a=m(center,nbr,mask,oh);b=m(center,nbr+100,mask,oh)
 assert torch.isfinite(a).all() and torch.equal(a,b)
 checks.append('TLS prediction: absent neighbors contribute zero regardless of filler values')
 with tempfile.TemporaryDirectory(prefix='heid-smoke-') as td:
  t=Path(td);(t/'white').mkdir();(t/'blur').mkdir()
  pd.DataFrame({'cell_id':['synthetic_a','synthetic_b'],'block':[1,1],'x_um':[10.,20.],'y_um':[30.,40.],'dx_um':[1.,1.],'dy_um':[-2.,-2.],'has_model':[True,True],'white_frac':[0.,1.],'drop_white':[False,True]}).to_parquet(t/'white/synthetic_cells.parquet')
  pd.DataFrame({'cell_id':['synthetic_b','synthetic_a'],'contrast':[20.,20.],'tenengrad':[.1,.1],'mask_delta':[10.,10.],'rim_grad':[3.,3.],'drop_blur':[False,False],'drop_nomask':[False,False],'drop_flat':[False,False]}).to_parquet(t/'blur/synthetic_cells.parquet')
  p=subprocess.run([sys.executable,str(ROOT/'align/cell_qc/scripts/12_cell_keep_table.py'),'--field','synthetic','--cancer','CRC','--white-dir',str(t/'white'),'--blur-dir',str(t/'blur'),'--out',str(t/'out')],capture_output=True,text=True)
  assert p.returncode==0,p.stderr
  result=pd.read_parquet(t/'out/synthetic_cells_for_training.parquet')
  assert result.cell_id.tolist()==['synthetic_a']
  assert np.isclose(result.he_px.iloc[0],11/.2125) and np.isclose(result.he_py.iloc[0],28/.2125)
 checks.append('Align: join by cell identity, blank exclusion, positive displacement sign and micron-to-pixel conversion')
 print(json.dumps({'status':'PASS','checks':checks,'scope':'synthetic CPU contracts, not scientific re-evaluation'},indent=2))
if __name__=='__main__':main()
