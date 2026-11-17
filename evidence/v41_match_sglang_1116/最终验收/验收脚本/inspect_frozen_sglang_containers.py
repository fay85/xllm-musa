from pathlib import Path
import json,subprocess,socket,datetime
host={'worker32007':'sj9','worker32009':'sj10'}[socket.gethostname()]
role='prefill' if host=='sj9' else 'decode'
name=f'xiaopang1116_sglang_{role}_{host}'
p=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
out={'at':datetime.datetime.now().astimezone().isoformat(),'host':host,'container':name,'inspect_exit':p.returncode}
if p.returncode==0:
 c=json.loads(p.stdout)[0];out.update(cid=c['Id'],image=c['Image'],state=c['State'],labels=c['Config'].get('Labels'),mounts=[{'source':x['Source'],'destination':x['Destination']} for x in c['Mounts']])
im=subprocess.run(['docker','image','inspect','sha256:1612b9c34896d7efb631975bf49542693c4c6d129554c63f32a31b13edc37337','--format','{{.Id}}'],capture_output=True,text=True)
out['cached_image']=im.stdout.strip()
print(json.dumps(out))

