"""Regenerate report and figures from the distributed summary tables."""
from pathlib import Path
import base64
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
NAMES = {'full':'Full','original030':'HiDDeN 0.30','default045':'HiDDeN default 0.45','psnr_matched':'HiDDeN closer PSNR'}

def main():
    summary=pd.read_csv(OUT/'results/core_macro_summary.csv')
    attack=pd.read_csv(OUT/'results/all_attack_summary.csv')
    ci=pd.read_csv(OUT/'results/paired_differences.csv')
    order=['full','psnr_matched','default045','original030']
    (OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for variant in order:
        sub=attack[attack.variant==variant].set_index('attack_id')
        axes[0].plot([50,75],[sub.loc[f'jpeg{q}','exact'] for q in [50,75]],'o-',label=NAMES[variant])
        axes[1].plot([.5,.75],[sub.loc[[f'crop{p}_{i}' for i in range(3)],'exact'].mean() for p in [50,75]],'o-')
        angles=[-10,-5,-2,-1,1,2,5,10]
        axes[2].plot(angles,[sub.loc[f'rotate{x:+d}','exact'] for x in angles],'o-')
    for ax,label in zip(axes,['JPEG quality','Retained crop area','Rotation (degrees)']):
        ax.set_xlabel(label); ax.set_ylim(-.03,1.04); ax.grid(alpha=.2)
    axes[0].set_ylabel('Exact 32-bit recovery')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.savefig(OUT/'figures/attack_curves.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    hidden=summary.set_index('variant').loc[['original030','psnr_matched','default045']]
    axes[0].plot(hidden.psnr,hidden.exact*100,'o-',label='HiDDeN native-resolution')
    for v,r in hidden.iterrows(): axes[0].annotate({'original030':'0.30','default045':'0.45','psnr_matched':'0.3615'}[v],(r.psnr,100*r.exact),xytext=(4,6),textcoords='offset points')
    for variant in ['full']:
        sub=summary[summary.variant==variant]
        if not sub.empty: axes[0].scatter(sub.psnr,sub.exact*100,marker='*',s=160,label=NAMES[variant])
    axes[0].set_xlabel('Embedding PSNR (dB)'); axes[0].set_ylabel('Core exact recovery (%)'); axes[0].legend(); axes[0].grid(alpha=.2)
    macroci=ci[ci.condition=='core_macro'].set_index('variant').loc[order[1:]]
    axes[1].errorbar(macroci.difference*100,range(len(macroci)),
       xerr=np.vstack([macroci.difference-macroci.lower,macroci.upper-macroci.difference])*100,fmt='o',capsize=4)
    axes[1].set_yticks(range(len(macroci)),[NAMES[v] for v in macroci.index]); axes[1].axvline(0,color='gray'); axes[1].set_xlabel('Full minus control (pp), paired 95% CI')
    fig.savefig(OUT/'figures/quality_recovery.png',dpi=180); plt.close(fig)
    def table(path):
        return pd.read_csv(path).to_html(index=False, border=0, float_format=lambda x:f'{x:.4f}', na_rep='—')
    def picture(name, caption):
        encoded=base64.b64encode((OUT/'figures'/name).read_bytes()).decode()
        return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    paragraphs=(OUT/'TEAM_REPORT_INSERT_EN.txt').read_text(encoding='utf-8').strip().split('\n\n')
    body=['<h1>Component ablation and HiDDeN comparison</h1>']
    body += ['<p>'+html.escape(p)+'</p>' for p in paragraphs[1:]]
    body += ['<h2>Component ablation</h2>',table(OUT.parent/'ablation/results/main_ablation.csv'),
        '<p>Rates are percentages and differences are percentage points. A3 averages 16 fixed layouts, not 16 independent datasets. A1 bypasses both correction and BCH validity rejection. E1 changes the embedded carrier distribution. These controls support components within this architecture, not general superiority over the underlying pretrained model.</p>',
        '<h2>System comparison</h2>',table(OUT/'results/main_comparison.csv'),
        '<p>Exact, wrong and erasure are percentages; BER is a fraction. Core scores equally weight six attack families; the eight extended conditions are separate. PSNR and SSIM describe embedding distortion. Missing Full raw payload BER is not zero.</p>',
        picture('attack_curves.png','Complete-message recovery under JPEG, cropping and rotation.'),
        picture('quality_recovery.png','Measured HiDDeN quality–recovery points and paired image-bootstrap intervals. Connecting lines are not a fitted frontier.'),
        '<h2>Bit-level diagnostics</h2>',table(OUT/'results/bit_metrics.csv'),
        '<p>Full/E1 BER pools bit errors over localized 32-bit carriers; missing regions are reflected in coverage. HiDDeN BER uses every raw 32-bit payload before marker rejection with six-family macro weighting. The denominators differ and must not be ranked directly. Bipolar NC = 1 − 2 BER.</p>',
        '<h2>Resolution sensitivity — calibration only</h2>',table(OUT/'hidden/results/resolution_smoke_summary.csv'),
        '<p>The direct-resolution implementation follows the OmnisealBench reference. The resized variant is inspired by VideoSeal, which uses different checkpoint names and scaling. At strength 0.45 the prespecified expansion rule was clean BER below 1%, or JPEG75 BER improvement of at least 2 percentage points. Neither occurred. This 20-image, two-condition check does not establish equivalence of all protocols or conclusively explain residual clean errors.</p>',
        '<h2>Negative controls and paired outcomes</h2>',
        '<p>Each of Full and the three HiDDeN working points accepted 0/100 unmarked test images. The same images are reused across points; they are not additional independent negatives. Zero observed events do not imply zero population risk (one-sided 95% upper bound approximately 2.95%). Full returned no clusters, so these negatives cannot isolate BCH or CRC effects.</p>',
        table(OUT/'results/paired_outcomes.csv'),
        '<p>Counts cover all 8,400 test cases; other_only means HiDDeN recovered correctly while Full did not. Higher averages do not imply success on every case.</p>',
        '<h2>All attack conditions</h2>',table(OUT/'results/all_attack_summary.csv'),
        '<h2>Scope and references</h2><p>Released weights were used without retraining. Conclusions are limited to these DIV2K images, 32-bit payloads, attack strengths and operating points. They do not establish adversarial security, print–scan robustness or superiority to every HiDDeN release. Additional teammate results require the same image, payload, quantization, attack and aggregation protocol.</p>',
        '<p>References: <a href="https://github.com/facebookresearch/watermark-anything">WAM</a>; <a href="https://github.com/facebookresearch/omnisealbench">Meta OmnisealBench</a>; <a href="https://github.com/facebookresearch/videoseal">VideoSeal</a>; <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a>. Upstream terms apply; weights and images are not included.</p>']
    css='body{font:16px/1.55 system-ui;color:#17303b;margin:30px auto;max-width:1100px;padding:20px}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:6px;border-bottom:1px solid #ccd7df;text-align:right}img{max-width:100%}h1,h2{color:#135b74}figure{margin:24px 0}'
    (OUT/'REPORT_EN.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Component ablation and HiDDeN comparison</title><style>'+css+'</style></head><body>'+''.join(body)+'</body></html>',encoding='utf-8')


if __name__=='__main__':
    main()
