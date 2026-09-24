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
 tbcfg=load('experiment_config','tb/he_prediction/scripts/experiment_config.py')
 try:
  tbcfg.merge({'cohort':{'folds':5}},{'cohort':{'folds':4}});raise AssertionError('overlap accepted')
 except ValueError:pass
 joined=tbcfg.load_experiment(ROOT/'tb/he_prediction/configs/experiment.json')
 if tbcfg.BINDING.is_file():assert tbcfg.canonical_sha256(joined)=='362381d29bcafe80a9f8865a8f134d9f1eae3087ac892e2ae4f76f70cb5fa95b'
 else:assert 'n_samples' not in joined['cohort']
 checks.append('TB: configuration joined with its cohort binding keeps the checkpoint-bound hash; overlapping fields are refused')
 tlsbind=load('cohort_binding','tls/prediction/scripts/cohort_binding.py')
 try:
  tlsbind.load_contract(ROOT/'tls/prediction/configs/data_contract.json',ROOT/'absent_binding.json');raise AssertionError('missing binding accepted')
 except FileNotFoundError:pass
 assert tlsbind.merge({'a':{'path':'x'}},{'a':{'sha256':'y'}})=={'a':{'path':'x','sha256':'y'}}
 checks.append('TLS: the data contract requires its cohort binding and joins bound hashes without overwriting fields')
 nerve=load('nerve','nerve/define/scripts/schwann_clusters.py')
 n,lab=nerve._linkage_labels(np.array([[0.,0.],[1.,0.],[8.,0.]]),2.)
 assert n==2 and lab[0]==lab[1] and lab[1]!=lab[2]
 checks.append('Nerve: graph connectivity preserves separated components')
 nin=load('evaluation_inputs','nerve/prediction/scripts/evaluation_inputs.py')
 with tempfile.TemporaryDirectory(prefix='heid-smoke-') as td:
  f=Path(td)/'a.npz';np.savez(f,experiment_id=np.array('other_head'))
  try:
   nin.check_probability_source(np.load(f),'v8_all5k_cls_sigma3_v1',f);raise AssertionError('foreign head accepted')
  except ValueError:pass
 checks.append('Nerve: probabilities of a head other than the frame source are refused')
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
  pub=t/'aligned_he';reg=t/'registered';reg.mkdir();(reg/'he_aligned.ome.tif').write_bytes(b'')
  (t/'offsets').mkdir();(t/'per_block').mkdir()
  for field,prom in [('synthetic',.05),('weakfield',.01)]:
   pd.DataFrame({'status':['ok']*40,'he_prominence':[prom]*40}).to_csv(t/f'offsets/{field}_tiles.csv',index=False)
   (t/f'per_block/{field}.json').write_text('{}')
   if field!='synthetic':
    for suffix in ('_cells.parquet','_cells_for_training.parquet','.json'):
     (t/'out'/f'{field}{suffix}').write_bytes((t/'out'/f'synthetic{suffix}').read_bytes())
   p=subprocess.run([sys.executable,str(ROOT/'align/cell_qc/scripts/publish_aligned_he.py'),'--publish-root',str(pub),'field','--field',field,'--cancer','CRC','--registered-dir',str(reg),'--tiles-dir',str(t/'offsets'),'--per-block-dir',str(t/'per_block'),'--keep-dir',str(t/'out')],capture_output=True,text=True)
   assert p.returncode==0,p.stderr
  for command in (['mark-blur'],['report']):
   p=subprocess.run([sys.executable,str(ROOT/'align/cell_qc/scripts/publish_aligned_he.py'),'--publish-root',str(pub)]+command,capture_output=True,text=True)
   assert p.returncode==0,p.stderr
  assert (pub/'5k/CRC/synthetic-blur/micro/synthetic_cells_for_training.parquet').is_file()
  assert (pub/'5k/CRC/weakfield-weak/micro/cell_keep/weakfield.json').is_file()
  summary=pd.read_csv(pub/'_report/cohort_summary.csv')
  assert summary.field.tolist()==['synthetic','weakfield'] and summary.stage_reached.eq('cell_keep').all()
  assert pd.read_csv(pub/'_report/blur_rename_map.tsv',sep='\t').field.tolist()==['synthetic']
 checks.append('Align: join by cell identity, blank exclusion, positive displacement sign, micron-to-pixel conversion and the published layout with section marks')
 chain=load('placement_chain','3d/scripts/placement_chain.py')
 rot=lambda d,tx,ty:np.array([[np.cos(np.radians(d)),-np.sin(np.radians(d)),tx],[np.sin(np.radians(d)),np.cos(np.radians(d)),ty]])
 edges={'e1':{'fixed':'s1','moving':'s2','dz':5.,'M':rot(2,10,-4)},'e2':{'fixed':'s3','moving':'s2','dz':5.,'M':rot(-1,-6,3)}}
 os.environ.pop('HTAN3D_SECTION_CORRECTIONS',None)
 ab=chain.chain_on(chain.prereg_tree(edges,['s1','s2','s3']),edges,'M','s1')
 assert np.allclose(ab['s2'],edges['e1']['M']) and np.allclose(chain.compose(ab['s2'],chain.invert(edges['e2']['M'])),ab['s3'])
 checks.append('3D: the placement chain composes solved edges forward and by their inverse from the anchor')
 with tempfile.TemporaryDirectory(prefix='heid-smoke-') as td:
  t=Path(td);n=7;sids=[f'S-U{i}' for i in range(1,n+1)];W=H=4000.
  truth={s:rot(0.8*np.sin(i),25*np.cos(i),-15*np.sin(2*i)) for i,s in enumerate(sids)}
  (t/'vol').mkdir();(t/'vol/metadata.json').write_text(json.dumps({'canvas_origin_um':[0.,0.],'canvas_px':{'width':500,'height':500},'in_plane_um_per_px':8.,'encodings':{'G_withdrawn':{'planes':[{'section_id':s,'z_um':5.*(i+1)} for i,s in enumerate(sids)]}}}))
  for i in range(n):
   for j in (i+1,i+2):
    if j<n:
     A=chain.compose(chain.invert(truth[sids[j]]),truth[sids[i]])
     d=t/'pairs'/f'{i}_{j}';d.mkdir(parents=True)
     d.joinpath('pair_fits.json').write_text(json.dumps([{'a':sids[i],'b':sids[j],'affine_a_to_b_um':A.tolist(),'affine_rms_um':1.,'anchors':[{'pa':[0,0],'pb':[0,0]}]*6}]))
  p=subprocess.run([sys.executable,str(ROOT/'3d/scripts/align3d_consensus.py'),'--volume',str(t/'vol'),'--pairs-dir',str(t/'pairs'),'--out',str(t/'out')],capture_output=True,text=True,env={**os.environ,'MPLBACKEND':'Agg'})
  assert p.returncode==0,p.stderr
  got={s:np.array(m) for s,m in json.loads((t/'out/section_corrections.json').read_text())['sections'].items()}
  G=[chain.compose(got[s],chain.invert(truth[s])) for s in sids]
  pts=np.array([[0,0],[W,0],[0,H],[W,H]],float)
  assert max(np.abs((g[:,:2]@pts.T).T+g[:,2]-((G[0][:,:2]@pts.T).T+G[0][:,2])).max() for g in G)<0.5
 checks.append('3D: the alignment consensus recovers known per-section corrections from adjacent and one-apart pair fits up to one common transform')
 link=load('tls_linker','3d/scripts/tls_linker.py')
 a=pd.DataFrame({'cx_um_canvas':[0.,1000.],'cy_um_canvas':[0.,0.],'core_area_mm2':[0.001,0.001]})
 c=pd.DataFrame({'cx_um_canvas':[170.,1000.],'cy_um_canvas':[0.,40.],'core_area_mm2':[0.001,0.001]})
 assert [r[:2] for r in link.mutual_links(a,c,150.,dz=5.)[0]]==[(1,1)]
 assert [r[:2] for r in link.mutual_links(a,c,150.,dz=15.)[0]]==[(0,0),(1,1)]
 big=pd.DataFrame({'cx_um_canvas':[0.],'cy_um_canvas':[0.],'core_area_mm2':[0.2]})
 assert len(link.mutual_links(big,pd.DataFrame({'cx_um_canvas':[400.],'cy_um_canvas':[0.],'core_area_mm2':[0.2]}),150.)[0])==1
 checks.append('3D: TLS links are mutual nearest neighbours within max(150 um + 2 dz, 0.8 of the summed core radii)')
 duct=load('duct_anchor_align','3d/scripts/duct_anchor_align.py')
 M=np.array([[1.03,0.02,40.],[-0.01,0.97,-25.]]);pb=np.random.default_rng(0).uniform(0,3000,(8,2))
 fit,rms,_=duct.fit_affine([{'pa':list(M[:,:2]@q+M[:,2]),'pb':list(q)} for q in pb])
 assert np.allclose(fit,M,atol=1e-6) and rms<1e-6
 checks.append('3D: the anchor affine fit recovers a known anisotropic affine')
 print(json.dumps({'status':'PASS','checks':checks,'scope':'synthetic CPU contracts, not scientific re-evaluation'},indent=2))
if __name__=='__main__':main()
