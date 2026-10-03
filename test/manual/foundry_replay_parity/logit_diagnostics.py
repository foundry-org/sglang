"""CPU tensor diagnostics. No comparison threshold or pass criterion lives here."""
import hashlib


def _sha(tensor, torch):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_logits(actual, saved, torch):
    if actual.device.type != 'cpu' or saved.device.type != 'cpu':
        raise ValueError('Diagnostics require retained CPU tensors')
    if actual.shape != saved.shape or actual.ndim != 2:
        raise ValueError('Diagnostics require equal 2D logit shapes')
    a,b=actual.float(),saved.float()
    delta=(a-b).abs()
    rows=[]
    saved_hashes=[_sha(row,torch) for row in saved]
    actual_hashes=[_sha(row,torch) for row in actual]
    exact_matches=[[j for j,h in enumerate(saved_hashes) if h==key] for key in actual_hashes]
    for i in range(actual.shape[0]):
        aa,bb=torch.topk(a[i],2),torch.topk(b[i],2)
        rows.append({'row':i,'actual_sha256':actual_hashes[i],'saved_sha256':saved_hashes[i],
                     'changed_elements':int(torch.count_nonzero(actual[i]!=saved[i]).item()),
                     'max_abs':float(delta[i].max().item()),'mean_abs':float(delta[i].mean().item()),
                     'actual_argmax':int(aa.indices[0]),'saved_argmax':int(bb.indices[0]),
                     'actual_top2_ids':aa.indices.tolist(),'saved_top2_ids':bb.indices.tolist(),
                     'actual_top2_logits':aa.values.tolist(),'saved_top2_logits':bb.values.tolist(),
                     'actual_top1_margin':float((aa.values[0]-aa.values[1]).item()),
                     'saved_top1_margin':float((bb.values[0]-bb.values[1]).item()),
                     'exact_saved_row_candidates':exact_matches[i]})
    result={'shape':list(actual.shape),'dtype':str(actual.dtype),
            'actual_sha256':_sha(actual,torch),'saved_sha256':_sha(saved,torch),
            'actual_finite':bool(torch.isfinite(actual).all()),'saved_finite':bool(torch.isfinite(saved).all()),
            'bitwise':bool(torch.equal(actual,saved)),'argmax_equal':bool(torch.equal(a.argmax(-1),b.argmax(-1))),
            'changed_elements':int(torch.count_nonzero(actual!=saved).item()),
            'max_abs':float(delta.max().item()),'mean_abs':float(delta.mean().item()),
            'actual_max_abs':float(a.abs().max().item()),'saved_max_abs':float(b.abs().max().item()),
            'exact_row_multiset_equal':sorted(actual_hashes)==sorted(saved_hashes),'rows':rows}
    if actual.shape[0] <= 32:
        result['row_pairwise_max_abs']=[[float((a[i]-b[j]).abs().max().item()) for j in range(len(b))] for i in range(len(a))]
    return result


def live_inputs(batch, runner, size, torch):
    def fields(owner, *, truncate=False):
        result={}
        for name in ('input_ids','positions','seq_lens','seq_lens_cpu','orig_seq_lens',
                     'req_pool_indices','req_pool_indices_cpu','out_cache_loc','rids'):
            value=getattr(owner,name,None)
            if isinstance(value,torch.Tensor):
                if truncate and value.ndim:
                    value=value[:size]
                result[name]={'shape':list(value.shape),'dtype':str(value.dtype),'values':value.detach().cpu().tolist()}
            elif value is None or isinstance(value,(str,int,bool,list,tuple)):
                result[name]=value
        return result
    return {'forward_batch':fields(batch),'captured_input_buffers':fields(runner.buffers,truncate=True),
            'note':'Recorded before selected actual replay; CUDA-to-CPU copies excluded from timing.'}
