"""Generate a small, time-preserving video proxy for interactive FPA annotation.
Install: py -m pip install imageio-ffmpeg
Usage: py make_video_proxy.py "CrossreferenceTest.mp4" --height 720
The original video is untouched. Frame rate and duration are preserved.
"""
import argparse
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('--height',type=int,default=720)
    p.add_argument('--output')
    args=p.parse_args()
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit('Install video proxy dependency: py -m pip install imageio-ffmpeg') from exc
    src=Path(args.video).resolve()
    if not src.exists():raise SystemExit(f'Missing source: {src}')
    dst=Path(args.output).resolve() if args.output else src.with_name(src.stem+'_proxy720p.mp4')
    if dst==src:raise SystemExit('Output must differ from source video')
    h=max(240,args.height)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-y','-i',str(src),
         '-vf',f'scale=-2:{h}','-c:v','libx264','-preset','veryfast','-crf','26',
         '-fps_mode','passthrough','-an','-movflags','+faststart',str(dst)]
    print('Creating preview video:',dst,flush=True)
    subprocess.run(cmd,check=True)
    print('Saved:',dst)

if __name__=='__main__':main()
