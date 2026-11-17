import urllib.request,json,datetime
v={'at':datetime.datetime.now().astimezone().isoformat(),'roles':{}}
for role,url in [('prefill','http://10.121.32.7:28586'),('decode','http://10.121.32.9:28588')]:
 try:
  with urllib.request.urlopen(url+'/v1/models',timeout=5) as z:body=json.load(z);status=z.status
  ok=status==200 and any(x.get('id')=='deepseek-v4.1-flash-1017-w4afp8-mt' for x in body.get('data',[]))
  v['roles'][role]={'ready':ok,'status':status,'body':body}
 except Exception as e:v['roles'][role]={'ready':False,'error':repr(e)}
v['ready']=all(x['ready'] for x in v['roles'].values());print(json.dumps(v))
