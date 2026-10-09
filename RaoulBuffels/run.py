"""CLI: voorbereiding, onderzoek, modellen, eindtest en notebooks."""
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','discover','download','research','train','evaluate','notebooks','verify'])
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if args.command in ('prepare','discover','download'):
        from citibike import prepare
        function=getattr(prepare,args.command)
        function(**({'source':args.source} if args.source else {}))
    elif args.command=='research':
        from citibike.analysis import research
        research()
    elif args.command=='train':
        from citibike.modeling import train
        train()
    elif args.command=='evaluate':
        from citibike.modeling import evaluate
        evaluate()
    elif args.command=='notebooks':
        from citibike.notebooks import execute
        execute()
    elif args.command=='verify':
        from citibike.verify import verify
        verify()
