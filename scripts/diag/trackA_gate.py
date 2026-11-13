"""Thin-Bolmo gate from ~/blt/results/trackA/prefill.json (rule: STATUS 2026-11-11 19:20, written before the run).
ceiling = OLMo e2e median / [trunk + 0.154 * (Bolmo e2e median - trunk)], forced P=6, batch 1, 8 and 16 KiB; both >= 1.20.
Also prints every cell (stock, 4.4, 6, 8; batch 1 and 4), gpu_resident and end_to_end, and the optimistic ceiling."""
import json, os, sys

path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/blt/results/trackA/prefill.json")
data = json.load(open(path))
LOCAL = 0.154
gate = {}
for row in data["results"]:
    key = (row["prompt_kib"], row["batch"], row["forced_p"])
    if "error" in row:
        print(key, "ERROR", row["error"][:200]); continue
    out = [f"{row['prompt_kib']:>2} KiB b{row['batch']} P={row['forced_p'] or 'stock':<5}"]
    for variant in ("gpu_resident", "end_to_end"):
        v = row["variants"][variant]
        bol, olm = v["models"][0], v["models"][1] if len(v["models"]) > 1 else None
        if olm is None:
            out.append(f"{variant}: olmo missing ({v.get('olmo_error')})"); continue
        pb = v["paired_bootstrap"]
        bpp = sum(bol["realised_bytes_per_unit"]) / len(bol["realised_bytes_per_unit"])
        bpt = sum(olm["realised_bytes_per_unit"]) / len(olm["realised_bytes_per_unit"])
        out.append(f"{variant}: bolmo {bol['median_seconds']*1e3:7.1f} ms, olmo {olm['median_seconds']*1e3:7.1f} ms, "
                   f"bolmo/olmo speed {pb.get('speed_a_over_b', float('nan')):.3f} [{pb.get('ci95_low', float('nan')):.3f}, "
                   f"{pb.get('ci95_high', float('nan')):.3f}], bytes/patch {bpp:.2f}, bytes/token {bpt:.2f}")
    prof = row["bolmo_event_profile"]["prefill"]
    e2e = row["variants"]["end_to_end"]["models"]
    if len(e2e) > 1:
        trunk = prof["global_trunk"] / 1e3
        bol, olm = e2e[0]["median_seconds"], e2e[1]["median_seconds"]
        ceiling = olm / (trunk + LOCAL * max(0.0, bol - trunk))
        optimistic = olm / trunk
        out.append(f"events ms {json.dumps({k: round(x, 1) for k, x in prof.items()})}; trunk share of e2e {trunk / bol:.2f}; "
                   f"ceiling {ceiling:.3f}; optimistic {optimistic:.3f}")
        gate[key] = (ceiling, optimistic, trunk, bol, olm)
    print("\n    ".join(out))
print()
cells = [gate.get((kib, 1, 6.0)) or gate.get((kib, 1, 6)) for kib in (8, 16)]
for kib, cell in zip((8, 16), cells):
    print(f"GATE P=6 batch 1 {kib} KiB:", "missing" if cell is None else f"ceiling {cell[0]:.3f} optimistic {cell[1]:.3f}")
ok = all(cell is not None and cell[0] >= 1.20 for cell in cells)
print("THIN_BOLMO_GATE", "PASS" if ok else "FAIL")
