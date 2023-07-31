import json
import argparse
import os.path as osp
from collections import defaultdict


def fix_lex_list(lex):
    new_lex = defaultdict(dict)
    for v in lex:
        new_lex[int(v['xml_id'].replace('Id', ''))][v['lang']] = v['lex']
    return new_lex


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help="The path to the JSON data file")
    parser.add_argument('--lang', type=str, required=True, 
                        choices=['en', 'ru', 'br', 'cy', 'ga', 'mt'])
    args = parser.parse_args()

    if osp.isfile(args.file) and args.file.endswith('.json'):
        save_name = osp.join(osp.dirname(args.file),
                            args.lang + '_ref_' + osp.basename(args.file)[:-4] + 'txt')
    else: 
        parser.error('File does not exist or not a json')
    
    with open(args.file) as fp:
        data = json.load(fp)
    data = {k: fix_lex_list(v["lexicalisations"]) for x in data['entries'] for k, v in x.items()}

    num_refs = max(len(v) for v in data.values())

    sorted_data = sorted(data.items(), key=lambda x: int(x[0]))

    fp = [open(save_name + str(i), 'a') for i in range(num_refs)]
    for eid, lex in sorted_data:
        for i in range(num_refs):
            cur_lex = lex.get(i+1, None)
            if cur_lex is not None:
                fp[i].write(cur_lex[args.lang] + '\n')
            else:
                fp[i].write('\n')

    for i in fp:
        i.close()
