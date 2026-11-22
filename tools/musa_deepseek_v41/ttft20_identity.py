"""Content digests for the 20k native screen -> F/R identity contract. No launch, files or network."""
import hashlib
import json
import re


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def config_fingerprint(config, uid):
    if not re.fullmatch(r'L20[A-Za-z0-9]+', uid):
        raise ValueError('invalid 20k uid')
    namespace = 'xiaopang1119-long-' + uid
    case_dir = '/goal20k200k_20261119/cases/' + uid + '/'
    if config['run_id'] != 'long_' + uid or config['namespace'] != namespace:
        raise ValueError('config not prepared for declared round')
    # These fields identify the invocation/treatment, not rank behavior. Normalize *only*
    # the case directory segment and exact etcd namespace, never a bare UID in flags or assets.
    c = dict(config)
    c.pop('run_id')
    c.pop('treatment')
    c['namespace'] = '<round-namespace>'
    def normalize(x):
        if isinstance(x, dict):
            return {k: normalize(v) for k,v in x.items()}
        if isinstance(x, list):
            return [normalize(v) for v in x]
        if isinstance(x, str):
            if x == '--etcd_namespace=' + namespace:
                return '--etcd_namespace=<round-namespace>'
            return x.replace(case_dir, '/goal20k200k_20261119/cases/<round>/')
        return x
    return fingerprint(normalize(c))


def asset_set(host_assets, router):
    out = {}
    for h in ('sj9','sj10'):
        a = host_assets[h]
        if a['host'] != h or not all(a.get(k) for k in (
                'container','image','binary_sha256','libraries','aot_assets_sha256',
                'actual_data_mounts','model_metadata_sha256','weight_file_metadata')):
            raise ValueError('incomplete loaded role identity')
        out[h] = {k:v for k,v in a.items() if k not in ('at','rank0')}
    out['router'] = {k:router[k] for k in ('host','container','exe','exe_sha256')}
    if out['router']['host'] != 'sj10' or out['router']['container'] != out['sj10']['container']:
        raise ValueError('unexpected native router location')
    if 'tokenizer_initializer' in router:
        initializer = router['tokenizer_initializer']
        fields = ('module', 'sha256', 'model_path', 'tokenizer_sha256',
                  'original_so', 'original_so_sha256', 'router_sha256', 'handles')
        if (set(initializer) != set(fields) or
                initializer['router_sha256'] != router['exe_sha256'] or
                type(initializer['handles']) is not int or
                not 1 <= initializer['handles'] <= 32):
            raise ValueError('incomplete loaded Router tokenizer initializer')
        out['router']['tokenizer_initializer'] = initializer
    return out
