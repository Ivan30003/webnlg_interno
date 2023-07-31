import os
import json
import re
import argparse


def prepare_dataset(file, target_dir, category_translation_file, predicate_translation_file, translate, 
                    add_links, add_metadata, lang="ru", instruct_lang="ru"):
    with open(file, "r") as f:
        data = json.load(f)
    entries = data["entries"]

    def camel_remove(text):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text)
        if len(text) > 0:
            if text[0].isupper() or text[0] == '"':
                result = ' '.join([m.group(0) for m in matches])
                result = result[0].capitalize() + result[1:]
            else:
                result = ' '.join([m.group(0).lower() for m in matches])
        else:
            result = text
        return result
    
    def text_clean(text):
        text = text.replace('_', ' ')
        text = text.replace('\\"', '"')
        text = camel_remove(text)
        return text
    
    PREDICATE_MAPPING = {"sameAs": "=", "includes": ">="}

    dataset = []
    inference_dataset = []
    count = 0

    for idx, line in enumerate(entries):
        completion_data = ''
        id_key = str(idx+1)
        entry = line[id_key]

        def parse_property(prop):
            return text_clean(prop)
        
        def parse_subject(subj):
            return text_clean(subj)
        
        def parse_triples_simple(tripleset):
            result = list()
            for j, triple in enumerate(tripleset):
                obj, prop, subj = triple["object"], triple["property"], triple["subject"]
                if instruct_lang == "ru":
                    parsed_prop = parse_property(prop)
                    if translate:
                        with open(predicate_translation_file, 'r') as src_f:
                            predicate_translations = json.load(src_f)
                        if parsed_prop in predicate_translations:
                            parsed_prop = predicate_translations[parsed_prop]
                    cur_str = f"{{ {parse_subject(subj)} | {parsed_prop} | {parse_subject(obj)} }}"
                else:
                    cur_str = f"{j+1}: Subject {{{parse_subject(subj)}; Property {parse_property(prop)}; Object {parse_subject(obj)}}}."
                result.append(cur_str)
            num_triples = len(result)
            return "; ".join(result), num_triples
        
        triples_string, num_triples = parse_triples_simple(entry['modifiedtripleset'])

        def parse_knowledge(tripleset):
            result = list()
            for j, triple in enumerate(tripleset):
                obj, prop, subj = triple["object"], triple["property"], triple["subject"]
                if instruct_lang == "ru":
                    cur_str = f"{{ {parse_subject(subj)} {PREDICATE_MAPPING[prop]} {parse_subject(obj)} }}"
                else:
                    cur_str = f"Knowledge entry {j+1}: Subject {parse_subject(subj)}; Property {parse_property(prop)}; Object {parse_subject(obj)}."
                result.append(cur_str)
            if len(result) == 0:
                return "{}"
            return "; ".join(result)
        
        def parse_category(category):
            result = camel_remove(category)
            if translate:
                with open(category_translation_file, 'r') as src_f:
                    category_translations = json.load(src_f)
                if result in category_translations:
                    result = category_translations[result]
            return result
        
        knowledge_string = parse_knowledge(entry["dbpedialinks"])

        def parse_links(tripleset):
            result = list()
            for j, triple in enumerate(tripleset):
                obj, prop, subj = triple["object"], triple["property"], triple["subject"]
                if instruct_lang == "ru":
                    cur_str = f"{{ {parse_subject(subj)} {PREDICATE_MAPPING[prop]} {parse_subject(obj)} }}"
                else:
                    cur_str = f"Link {j+1}: Subject {parse_subject(subj)}; Property {parse_property(prop)}; Object {parse_subject(obj)}."
                result.append(cur_str)
            if len(result) == 0:
                return "{ }"
            return "; ".join(result)
        
        links_string = parse_links(entry["links"])

        if instruct_lang == "ru":
            prompt = f'{triples_string}'
            if add_links or add_metadata:
                prompt += '.'
            if add_links:
                prompt = f'''Соотношения: {prompt}
                Дополнительные соотношения: {links_string}.
                Ссылки: {knowledge_string}.'''
            if add_metadata:
                prompt = f'''Категория: {parse_category(entry["category"])}.
                Число соотношений: {num_triples}. {prompt}
                Короткое высказывание:'''
            if translate:
                prompt = prompt.replace('comics Character', 'Персонаж комиксов')

        else:
            prompt = f"""Write a short statement about category {parse_category(entry["category"])} based on {num_triples} following lines:\n
            {triples_string}
            Objects and subjects from knowledge base entries could be used:\n
            {knowledge_string}
            Following relations between objects and subjects could be used:\n
            {links_string}
            Short statement:
            """
        prompt = re.sub("^\s+", "", prompt, flags=re.MULTILINE)

        for res in entry['lexicalisations']:
            data = {}
            if res['lang'] == lang:
                data['original_id'] = id_key
                data['id'] = str(count)
                data['prompt'] = prompt
                completion_data = res['lex']
                data['completion'] = completion_data
                dataset.append(data)
                count += 1
        
        ### ONE EXAMPLE PER ENTRY FOR INFERENCE
        for res in entry['lexicalisations']:
            cur_data = {}
            if res['lang'] == lang:
                cur_data['original_id'] = id_key
                cur_data['id'] = id_key
                cur_data['prompt'] = prompt
                completion_data = res['lex']
                cur_data['completion'] = completion_data
                inference_dataset.append(cur_data)
                break

    target_basename = os.path.splitext(os.path.basename(file))[0]
    if translate:
        target_basename = target_basename + '_translated'
    if add_links:
        target_basename = target_basename + '_with_links'
    if add_metadata:
        target_basename = target_basename + '_with_metadata'
    if not translate and not add_links and not add_metadata:
        target_basename = target_basename + '_simple'

    os.makedirs(target_dir, exist_ok=True)

    with open(os.path.join(target_dir, target_basename + '.jsonl'), 'w') as fw:
        for entry in dataset:
            json.dump(entry, fw, ensure_ascii=False)
            fw.write('\n')

    with open(os.path.join(target_dir, target_basename + '_inference.jsonl'), 'w') as fw:
        for entry in inference_dataset:
            json.dump(entry, fw, ensure_ascii=False)
            fw.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='original_datasets/ru_train.json', 
                        help="The path to the JSON data file")
    parser.add_argument("--target_dir", type=str, default='created_datasets', 
                        help="The path to the target directory")
    parser.add_argument("--lang", type=str, default='ru', choices=['en', 'ru'],
                        help="Language of the dataset - ru or en")
    parser.add_argument("--add_links", action="store_true", 
                        help="To create a dataset with links")
    parser.add_argument("--add_metadata", action="store_true", 
                        help="To create a dataset with metadata, such as category, number of triples in tripleset")
    parser.add_argument("--translate", action="store_true", help="To create a dataset with translated predicates and categories")
    parser.add_argument("--category_translation_file", type=str, default='translation/category_translations.json', 
                        help="The path to the category translation file")
    parser.add_argument("--predicate_translation_file", type=str, default='translation/predicate_translations.json', 
                        help="The path to the predicate translation file")

    args = parser.parse_args()
    json_file = args.file
    predicate_translation_file = args.predicate_translation_file
    category_translation_file = args.category_translation_file
    target_dir = args.target_dir
    lang = args.lang
    translate = args.translate
    add_links = args.add_links
    add_metadata = args.add_metadata

    prepare_dataset(json_file, target_dir, category_translation_file, predicate_translation_file, translate=translate, add_links=add_links, add_metadata=add_metadata)


if __name__ == "__main__":
    main()    

