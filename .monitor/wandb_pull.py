import wandb
api = wandb.Api()
proj = 'ashish397-university-of-exeter/longlive-phase-3'
targets = {'fwd': '9q400ati', 'lowt': 'pibqwijy', 'baseline': 'us6ybfww', 'frozenT': '2q8vvi44'}
pat = ['pred_real', 'pred_fake', 'real_teacher', 'aux_teacher', 'real_score', '_rms', 'real_lora', 'gen_loss', 'critic_loss', 'dmd_grad']

out = open('/scratch/u6ex/as1748.u6ex/ARRWM/.monitor/wandb_dump.txt', 'w')
def w(s): out.write(s + '\n'); out.flush()

# discover keys from fwd
r0 = api.run(f'{proj}/9q400ati')
allk = sorted(r0.summary.keys())
w('ALL KEYS:')
for k in allk:
    w('  ' + k)

# the metrics we care most about for the overtrain-vs-diverge + GAN-imbalance question
focus = ['gen/pred_real_rms', 'gen/pred_fake_rms', 'gen/student_pred_rms',
         'gen/aux_teacher_loss', 'gen/aux_teacher_pred_mae', 'gen/aux_teacher_t_mean',
         'gen/real_score_mae_vs_gt', 'critic/real_teacher_grad_norm']
# add any GAN / gate / dmd-weight keys we find
gan_pat = ['d_real', 'd_fake', 'disc', 'gan', 'mae_gate', 'dmd_loss_weight', 'dmd_grad',
           'ladd', 'gate', 'dmd_mae']
focus += [k for k in allk if any(p in k.lower() for p in gan_pat)]
focus = sorted(set(focus))
w('\nFOCUS KEYS: ' + str(focus))

for tag, rid in targets.items():
    w('\n' + '=' * 60)
    w(f'{tag} ({rid})')
    r = api.run(f'{proj}/{rid}')
    cols = ['_step'] + focus
    try:
        hist = r.history(keys=focus, samples=2000, pandas=True)
    except Exception as e:
        w(f'  history error: {e}')
        continue
    if hist is None or len(hist) == 0:
        w('  (no history)')
        continue
    # print every row from step 110 onward (the collapse window), plus a few early
    have = [c for c in cols if c in hist.columns]
    w('  cols: ' + str(have))
    hh = hist[have].dropna(how='all')
    for _, row in hh.iterrows():
        st = row.get('_step')
        if st is None:
            continue
        st = int(st) if st == st else -1
        if st < 100 and st % 30 != 0:
            continue
        bits = [f'step={st:>3}']
        for c in have:
            if c == '_step':
                continue
            v = row.get(c)
            if v == v and v is not None:
                bits.append(f'{c.split("/")[-1]}={v:.4f}')
        w('  ' + ' '.join(bits))
out.close()
print('DONE -> .monitor/wandb_dump.txt')
