#!/usr/bin/env python3
"""Inspect a RETFound checkpoint without unpacking the whole thing."""
import sys, torch, os

CKPT = "/root/autodl-tmp/RETFound/RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth"
print(f"[ckpt] {CKPT}")
print(f"[ckpt] size = {os.path.getsize(CKPT) / 1024**3:.2f} GB")

# Don't materialize to GPU; map_location cpu is enough
obj = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"[ckpt] top-level type = {type(obj).__name__}")

if isinstance(obj, dict):
    print(f"[ckpt] top-level keys ({len(obj)}):")
    for k, v in obj.items():
        if torch.is_tensor(v):
            print(f"   {k!r:30s} tensor  shape={tuple(v.shape)}  dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"   {k!r:30s} dict   {len(v)} entries")
            # show first 3 keys
            for kk in list(v.keys())[:3]:
                vv = v[kk]
                if torch.is_tensor(vv):
                    print(f"        .{kk!r:30s} tensor  shape={tuple(vv.shape)}  dtype={vv.dtype}")
                else:
                    print(f"        .{kk!r:30s} {type(vv).__name__} = {str(vv)[:80]}")
            if len(v) > 3:
                print(f"        ... +{len(v)-3} more")
        else:
            print(f"   {k!r:30s} {type(v).__name__} = {str(v)[:80]}")

# If it's a state_dict, also try to find a classifier head
if isinstance(obj, dict):
    flat = obj
    if "model" in obj and isinstance(obj["model"], dict):
        flat = obj["model"]
    head_keys = [k for k in flat.keys() if "head" in k.lower() or "fc" in k.lower() or "classifier" in k.lower()]
    print(f"[ckpt] head-like keys ({len(head_keys)}):")
    for k in head_keys[:20]:
        v = flat[k]
        if torch.is_tensor(v):
            print(f"   {k:50s} shape={tuple(v.shape)}  dtype={v.dtype}")
