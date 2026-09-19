"""Launch existing Vortex/SGLang integration from a recorded CLI config."""
import json
import os
import sys

if os.environ.get('PAPER_DECODE_WINDOW_OUTPUT'):
    from decode_observer import install
    install()

def main():
    import vortex_torch
    import sglang
    import torch
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    print(json.dumps({'vortex': vortex_torch.__file__, 'sglang': sglang.__file__, 'torch': torch.__version__}), flush=True)
    args = prepare_server_args(json.load(open(sys.argv[1]))['server_args'])
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

if __name__ == '__main__':
    main()
