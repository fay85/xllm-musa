import subprocess,json,socket,datetime
h={'worker32007':'sj9','worker32009':'sj10'}[socket.gethostname()];name='xiaopang1117_v41_sg438_'+('prefill' if h=='sj9' else 'decode')+'_'+h
c=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
expected={'sj9':'52abeddab774ff06522e2ffe452b0714065c579f95137660f51f54ed3d9ec94a','sj10':'7eefcf3aedcb80825833e6b47295689965905518e51823cab8bedecfad6374f3'}[h];assert c['Id']==expected
script="import importlib.metadata as m,json,sys;names=['torch','torch_musa','mate','mate-mubin','deep_ep','tvm-ffi'];d={x.metadata['Name'].lower().replace('_','-'):x.version for x in m.distributions()};print(json.dumps({'python':sys.executable,'versions':{n:d.get(n.replace('_','-')) for n in names}}))"
p=subprocess.run(['docker','exec','-e','MUSA_VISIBLE_DEVICES=-1',name,'python3','-c',script],text=True,capture_output=True,timeout=20);assert p.returncode==0,p.stderr
v={'at':datetime.datetime.now().astimezone().isoformat(),'host':h,'container':name,'id':c['Id'],'image_id':c['Image'],'image_tag':c['Config']['Image'],'python_package_metadata':json.loads(p.stdout),'scope':'Package metadata only; actual loaded C++ SO paths and hashes are recorded separately in each final role runtime_identity.'};print(json.dumps(v))
