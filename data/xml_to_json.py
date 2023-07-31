import os
import argparse
from benchmark_reader import Benchmark, select_files


def xml_to_json(file):
    target_dir = os.path.dirname(file)
    target_basename = os.path.splitext(os.path.basename(file))[0]
    b = Benchmark()
    files = select_files(file)
    b.fill_benchmark(files)
    b.b2json(target_dir, target_basename + '.json')
    json_file_path = os.path.join(target_dir, target_basename + '.json')
    return json_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="The path to the XML data file")
    args = parser.parse_args()

    if not (os.path.exists(args.file) and args.file.endswith('.xml')):
        parser.error('Invalid file path') 

    json_file = xml_to_json(args.file)
    print(f'Data saved to {json_file}')
