"""Reprocess an existing participant ZIP: one signed taped-reference FPA per landing."""
from __future__ import annotations
import argparse,csv,json,tempfile
from pathlib import Path
from zipfile import ZipFile
from fpa_unified_pipeline import process_csv_pair,export_results


def replay(source,output,aligned_start):
    source=Path(source).expanduser().resolve();output=Path(output).expanduser().resolve()
    if not source.exists():raise FileNotFoundError(source)
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp)
        if source.is_file() and source.suffix.lower()=='.zip':
            with ZipFile(source) as z:
                for item in z.infolist():
                    entry=Path(item.filename)
                    if entry.is_absolute() or '..' in entry.parts or item.filename.startswith('\\'):
                        raise ValueError('Unsafe participant archive member')
                z.extractall(tmp)
            roots=list(tmp.rglob('left_raw.csv'))
        elif source.is_dir():roots=list(source.rglob('left_raw.csv'))
        else:raise ValueError('Expected participant ZIP or extracted session folder')
        if len(roots)!=1:raise ValueError('Expected exactly one left_raw.csv')
        folder=roots[0].parent
        if not (folder/'right_raw.csv').is_file():raise ValueError('Missing right_raw.csv')
        cal=folder/'calibration'
        if not cal.is_dir():cal=Path(__file__).resolve().parent/'calibration_frozen_v1'
        results,stage=process_csv_pair(folder/'left_raw.csv',folder/'right_raw.csv',cal,
            reference_line_mode=aligned_start)
        output.mkdir(parents=True,exist_ok=True)
        manifest=export_results(results,stage,output)
        rows=[]
        for side in ('left','right'):
            for r in stage[side]['rows']:
                rows.append(dict(foot=side,contact_time_s=r['contact_time_s'],
                    swing_id=r['swing_id'],fpa_deg=r['fpa_deg'],
                    detector_confidence=r.get('detector_confidence'),
                    detector_evidence=r.get('detector_evidence'),
                    fpa_status=r['fpa_status']))
        rows.sort(key=lambda x:(x['contact_time_s'],x['foot']))
        with (output/'reference_line_fpa.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['foot','contact_time_s','swing_id','fpa_deg',
                'detector_confidence','detector_evidence','fpa_status'])
            writer.writeheader();writer.writerows(rows)
        summary={'source':source.name,'aligned_start_explicitly_asserted':aligned_start,
                 'reference_is_independently_measured':False,
                 'requirement':'start shoe axes aligned and actual progression parallel to line',
                 'calibration_source':'session_zip_calibration' if (folder/'calibration').is_dir() else 'frozen_bundled_fallback',
                 'first_accepted_landing':{side:next((r for r in rows if r['foot']==side),None) for side in ('left','right')},
                 'stage4_schema':manifest['schema'],
                 'bilateral_window':manifest.get('bilateral_window')}
        (output/'reference_line_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        return rows,summary


def main():
    ap=argparse.ArgumentParser(description='Offline replay: signed initial-reference FPA only')
    ap.add_argument('source',help='Raw session ZIP or extracted folder')
    ap.add_argument('--aligned-start',action='store_true',help='Explicitly declare initial taped alignment and straight walkway')
    ap.add_argument('--output',default='reference_line_replay')
    args=ap.parse_args();rows,_=replay(args.source,args.output,args.aligned_start)
    print('Exported',len(rows),'contact evidence rows (A/B/C) to',args.output)
    for side in ('left','right'):
        sample=next((r for r in rows if r['foot']==side),None)
        print(side.upper(),'first FPA:',sample['fpa_deg'] if sample else None)

if __name__=='__main__':main()
