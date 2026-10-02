#!/usr/bin/env python3
import argparse,json,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8080');sub=p.add_subparsers(dest='cmd',required=True)
for cmd in ['status','sessions']:sub.add_parser(cmd)
c=sub.add_parser('create');c.add_argument('name');c.add_argument('--notes',default='')
a=sub.add_parser('action');a.add_argument('action');a.add_argument('--session');a.add_argument('--preset',default='jetson_cpu');a.add_argument('--run')
args=p.parse_args();data=None;path='/api/'+args.cmd
if args.cmd=='create':path='/api/sessions';data={'name':args.name,'notes':args.notes}
if args.cmd=='action':data={k:getattr(args,k) for k in ('action','session','preset','run')}
req=urllib.request.Request(args.url+path,data=json.dumps(data).encode() if data else None,headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=420) as r: print(json.dumps(json.load(r),indent=2))
