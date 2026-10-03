"""Strict, opt-in per-request/prefix SAVE reference bank.

Rows may be assembled across different *global* decode states. This measures
numeric equality for this causal Qwen3 configuration; it does not establish
arbitrary MoE packing/batch independence. All GPU tensors are copied after the
one natural backend replay, outside the event timing region. No launch here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PROTOCOL = 'foundry-per-request-prefix-bank-v1'
SCOPE = ('Causal Qwen3/Qwen3MoE, dummy BF16 weights, fixed seed/config/capture shape; '
         'per-request prefix reference assembly, not identical global batch state. '
         'No inference for capacity dropping, batch-coupled models, routing simulation or EPLB.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def signature(args, config):
    if config.get('architectures') not in (['Qwen3ForCausalLM'], ['Qwen3MoeForCausalLM']):
        raise ValueError('State bank supports only the tested causal Qwen3 architectures')
    return {'protocol': PROTOCOL, 'model_config': config, 'seed': args.seed,
            'dtype': 'bfloat16', 'target_backend': args.target_backend,
            'capture_batches': sorted(map(int,args.capture_batches.split(','))),
            'prompt_tokens': args.prompt_tokens, 'generate_tokens': args.generate_tokens,
            'context_length':2048, 'attention_backend':'fa3', 'load_format':'dummy',
            'enable_eplb':False,'ep_num_redundant_experts':0,'init_expert_location':'trivial',
            'scope_assumption': SCOPE}


def semantic_rows(inputs, phase, generation, rank, capture_batch):
    """Validate consumed prefix, with position P+j consuming generated[j]."""
    live, graph = inputs['forward_batch'], inputs['captured_input_buffers']
    rids = live.get('rids')
    submitted = phase['request_ids']
    if not isinstance(rids, list) or len(set(submitted)) != len(submitted):
        raise ValueError('Missing or ambiguous request identity')
    index = {rid:i for i,rid in enumerate(submitted)}
    tokens = graph['input_ids']['values']
    positions = graph['positions']['values']
    lengths = graph['seq_lens']['values']
    for name in ('input_ids','positions','seq_lens'):
        if not isinstance(live.get(name),dict) or live[name].get('values')!=graph[name].get('values'):
            raise ValueError(f'Live request metadata disagrees with captured {name}')
    if not len(rids) == len(tokens) == len(positions) == len(lengths) == capture_batch:
        raise ValueError('State bank requires exact complete capture rows')
    if len(set(rids)) != len(rids):
        raise ValueError('Repeated live request identity')
    if len(generation['input_ids']) != phase['global_batch'] or len(generation['tokens']) != phase['global_batch']:
        raise ValueError('Generation evidence has wrong global batch')
    result = []
    for row,(rid,token,pos,length) in enumerate(zip(rids,tokens,positions,lengths)):
        if rid not in index:
            raise ValueError('Live request not in armed submission')
        i = index[rid]
        prompt, generated = generation['input_ids'][i], generation['tokens'][i]
        step = pos-len(prompt)
        if type(pos) is not int or type(length) is not int or step<0 or step>=len(generated) or length!=pos+1:
            raise ValueError('Invalid decode position/sequence length')
        if token != generated[step]:
            raise ValueError('Live input does not match independently generated consumed prefix')
        prefix = prompt+generated[:step+1]
        if len(prefix)!=length or prefix[-1]!=token:
            raise ValueError('Consumed prefix length/input disagreement')
        result.append({'rank':rank,'capture_batch':capture_batch,
                       'global_batch':phase['global_batch'],'request_index':i,
                       'prompt':prompt,'consumed_prefix':prefix,'position':pos,
                       'seq_len':length,'input_id':token,
                       'expected_next_token':generated[step+1] if step+1<len(generated) else None})
    return result


def row_key(row):
    # Next token is an output check, never part of the consumed input identity.
    return digest({k:v for k,v in row.items() if k!='expected_next_token'})


def validate_global_rows(rows_by_rank):
    if not rows_by_rank or any(not rows for rows in rows_by_rank):
        raise ValueError('Missing global request rows')
    first=rows_by_rank[0][0]; count=first['global_batch']; capture=first['capture_batch']
    for rank,rows in enumerate(rows_by_rank):
        if any(row['rank']!=rank or row['global_batch']!=count or row['capture_batch']!=capture for row in rows):
            raise ValueError('Global row rank/capture metadata disagree')
    indices=[[row['request_index'] for row in rows] for rows in rows_by_rank]
    if count==capture:
        if any(sorted(values)!=list(range(count)) for values in indices):
            raise ValueError('TP replica omits or duplicates submitted requests')
        return 'replicated_tp'
    if count==capture*len(rows_by_rank):
        if sorted(i for values in indices for i in values)!=list(range(count)):
            raise ValueError('DP global batch omits or duplicates submitted requests')
        return 'partitioned_dp'
    raise ValueError('Unsupported global request partition')


def global_state_id(rows_by_rank):
    return digest([[{k:v for k,v in row.items() if k!='expected_next_token'}
                    for row in rows] for rows in rows_by_rank])


def common_lookup(verdicts):
    if not verdicts or any(v.get('error') for v in verdicts):
        return 'reject'
    return 'probe' if all(v.get('available') is True for v in verdicts) else 'defer'


def add_entry(index, key, entry):
    previous = index.get(key)
    if previous is None:
        index[key] = entry
    elif previous['row_sha256'] != entry['row_sha256'] or previous['semantic'] != entry['semantic']:
        raise ValueError('Ambiguous bank key: repeated consumed state has different complete logits')
    else:
        previous['observations'].extend(entry['observations'])


def tensor_sha(tensor, torch):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Bank:
    def __init__(self, cfg, rank, torch):
        self.root=Path(cfg['state_reference_bank']); self.rank=rank; self.torch=torch
        self.cfg=cfg; self.serial=0; self.cache={}
        if cfg['mode']=='load':
            self.manifest=json.loads((self.root/'sealed.json').read_text())
            if self.manifest['protocol']!=PROTOCOL or self.manifest['signature']!=cfg['bank_signature']:
                raise ValueError('State bank protocol/config mismatch')
            path=self.root/f'rank_{rank}'/'index.json'
            if file_sha(path)!=self.manifest['indices'][str(rank)]:
                raise ValueError('State bank index checksum mismatch')
            self.index=json.loads(path.read_text())
            self.generations=json.loads((Path(cfg['save_reference'])/'generation_checks.json').read_text())
            if digest(self.generations)!=self.manifest['generation_sha256']:
                raise ValueError('Independent SAVE generation evidence does not belong to bank')

    def save_frame(self, logits, inputs, global_inputs, phase, replay_index):
        torch=self.torch
        cpu=logits.detach().cpu().clone()
        if not bool(torch.isfinite(cpu).all()):
            raise ValueError('Nonfinite natural SAVE output')
        folder=self.root/f'rank_{self.rank}'; folder.mkdir(exist_ok=True)
        sha=tensor_sha(cpu,torch); path=folder/(sha+'.pt')
        if not path.exists(): torch.save(cpu,path)
        frame={'ordinal':self.serial,'natural_replay_index':replay_index,'phase':phase,
               'inputs':inputs,'global_inputs':global_inputs,'tensor':path.name,
               'tensor_sha256':sha,'file_sha256':file_sha(path),
               'row_sha256':[tensor_sha(row,torch) for row in cpu],
               'argmax':cpu.argmax(-1).tolist(),'shape':list(cpu.shape),'dtype':str(cpu.dtype)}
        with (folder/'frames.jsonl').open('a') as f: f.write(json.dumps(frame)+'\n')
        self.serial+=1

    def lookup(self, inputs, global_inputs, phase, size):
        generations=[g for g in self.generations if g['phase']['batch']==phase['batch']]
        if not generations: raise ValueError('No SAVE generation for capture batch')
        generation=generations[0]
        rows=semantic_rows(inputs,phase,generation,self.rank,size)
        keys=[row_key(row) for row in rows]
        missing=[key for key in keys if key not in self.index]
        # Prefixes here are provisional until the real LOAD generation is checked.
        global_rows=[semantic_rows(item,phase,generation,r,size) for r,item in enumerate(global_inputs)]
        coverage=validate_global_rows(global_rows)
        global_id=global_state_id(global_rows)
        evidence={'protocol':PROTOCOL,'assembly_mode':'per_request_prefix',
                  'scope_assumption':SCOPE,'rows':rows,'keys':keys,'missing_keys':missing,
                  'global_request_coverage':coverage,
                  'global_logical_state_id':global_id,
                  'global_logical_state_identity':not missing and all(
                      any(o['global_logical_state_id']==global_id for o in self.index[k]['observations']) for k in keys),
                  'actual_complete_prefix_verified':False,
                  'prefix_verification':'pending independent final LOAD generation check',
                  'live_inputs':inputs,'global_inputs':global_inputs}
        if missing: return None,evidence
        tensors=[]
        for key in keys:
            entry=self.index[key]; path=self.root/f'rank_{self.rank}'/entry['tensor']
            if path.name not in self.cache:
                if file_sha(path)!=entry['file_sha256']: raise ValueError('Bank tensor checksum mismatch')
                self.cache[path.name]=self.torch.load(path,map_location='cpu',weights_only=True)
            row=self.cache[path.name][entry['row']]
            if tensor_sha(row,self.torch)!=entry['row_sha256']: raise ValueError('Bank complete row checksum mismatch')
            tensors.append(row)
        return self.torch.stack(tensors),evidence


def seal(root, generations, signature_value, manifests, world, torch):
    """Called after actual/repeat generation equality; no GPU work."""
    root=Path(root); byphase={g['phase']['id']:g for g in generations}
    if len(byphase)!=len(generations) or any(not g['exact_match'] for g in generations):
        raise ValueError('Bank generation evidence missing or inconsistent')
    indices={}; summary={}
    for rank in range(world):
        folder=root/f'rank_{rank}'; index={}; frames=0; usable=0; globals_seen=set()
        for line in (folder/'frames.jsonl').read_text().splitlines():
            frame=json.loads(line); phase=frame['phase']; generation=byphase[phase['id']]
            rows=semantic_rows(frame['inputs'],phase,generation,rank,phase['batch'])
            global_rows=[semantic_rows(item,phase,generation,r,phase['batch']) for r,item in enumerate(frame['global_inputs'])]
            validate_global_rows(global_rows)
            global_id=global_state_id(global_rows); globals_seen.add(global_id)
            path=folder/frame['tensor']
            if file_sha(path)!=frame['file_sha256']: raise ValueError('SAVE frame tensor changed')
            for i,row in enumerate(rows):
                if row['expected_next_token'] is None: continue
                if frame['argmax'][i]!=row['expected_next_token']:
                    raise ValueError('Natural SAVE row argmax differs from actual next generated token')
                entry={'semantic':row,'row_sha256':frame['row_sha256'][i],
                       'tensor':frame['tensor'],'file_sha256':frame['file_sha256'],'row':i,
                       'observations':[{'frame_ordinal':frame['ordinal'],'natural_replay_index':frame['natural_replay_index'],
                                        'global_logical_state_id':global_id,'phase_id':phase['id'],'role':phase.get('role','probe')}]}
                add_entry(index,row_key(row),entry); usable+=1
            frames+=1
        if not index: raise ValueError('Empty reference bank rank')
        batches={e['semantic']['capture_batch'] for e in index.values()}
        if batches!=set(signature_value['capture_batches']): raise ValueError('Bank missing capture shapes')
        path=folder/'index.json'; path.write_text(json.dumps(index,sort_keys=True))
        indices[str(rank)]=file_sha(path)
        summary[str(rank)]={'natural_frames':frames,'usable_rows':usable,'unique_keys':len(index),
                            'distinct_global_logical_states':len(globals_seen),
                            'repeated_keys':sum(len(e['observations'])>1 for e in index.values())}
    manifest={'protocol':PROTOCOL,'signature':signature_value,'indices':indices,
              'generation_sha256':digest(generations),'archive_manifest_sha256':[m['sha256'] for m in manifests],
              'summary':summary,'scope_assumption':SCOPE,'sealed_after_generation_validation':True}
    (root/'sealed.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    return manifest


def verify_load_prefixes(reports, checks):
    for report in reports:
        evidence=report['state_reference_bank']; phase=report['phase']
        check=checks[phase['id']]
        if not check['exact_match'] or not check.get('independent_save_exact_match'):
            raise ValueError('LOAD generation did not verify the reference prefix')
        rows=semantic_rows(evidence['live_inputs'],phase,check,report['rank'],phase['batch'])
        if rows!=evidence['rows']:
            raise ValueError('Actual complete LOAD prefix differs from selected SAVE prefix')
        if 'global_inputs' in evidence:
            global_rows=[semantic_rows(item,phase,check,r,phase['batch']) for r,item in enumerate(evidence['global_inputs'])]
            validate_global_rows(global_rows)
            if global_state_id(global_rows)!=evidence['global_logical_state_id']:
                raise ValueError('Actual complete global LOAD logical state differs from lookup provenance')
        evidence['actual_complete_prefix_verified']=True
        evidence['prefix_verification']='exact complete LOAD/SAVE generation and live-input/position match'
