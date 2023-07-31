import logging
from collections import defaultdict
import copy
from razdel import tokenize
import nltk

from metrics.webnlg_2023.evaluation.automatic.scripts.eval import (sacrebleu_score,
                                                                   meteor_score,
                                                                   chrF_score,
                                                                   bert_score_,
                                                                   ter_score)

class HuggingFaceEvaluator:
    """This is a modification of WebNLG metrics evaluation for training"""
    def __init__(self, meteor_path, metrics="bleu,meteor,chrf++", language="ru", num_refs=9) -> None:
        self.metrics = metrics.lower().split(',')
        self.lng = language
        self.num_refs = num_refs
        self.meteor_path = meteor_path

    def parse(self, refs, hyps, sample_ids):
        logging.info('STARTING TO PARSE INPUTS...')
        print('STARTING TO PARSE INPUTS...')
        # references       
        references, hypothesis = [], []
        sample_id_2_references = defaultdict(list)
        sample_id_2_predictions = dict()

        for prediction, target, sample_id in zip(hyps, refs, sample_ids):
            if sample_id not in sample_id_2_references:
                sample_id_2_predictions[sample_id] = prediction
            sample_id_2_references[sample_id].append(target)
        
        for sample_id in sample_id_2_references:
            references.append(sample_id_2_references[sample_id][:self.num_refs])
            hypothesis.append(sample_id_2_predictions[sample_id])

        # references tokenized
        references_tok = copy.copy(references)
        for i, refs in enumerate(references_tok):
            if self.lng == 'ru':
                references_tok[i] = [' '.join([_.text for _ in tokenize(ref)]) for ref in refs]
            else:
                references_tok[i] = [' '.join(nltk.word_tokenize(ref)) for ref in refs]

        # hypothesis tokenized
        hypothesis_tok = copy.copy(hypothesis)
        if self.lng == 'ru':
            hypothesis_tok = [' '.join([_.text for _ in tokenize(hyp)]) for hyp in hypothesis_tok]
        else:
            hypothesis_tok = [' '.join(nltk.word_tokenize(hyp)) for hyp in hypothesis_tok]

        logging.info('FINISHING TO PARSE INPUTS...')
        print('FINISHING TO PARSE INPUTS...')
        return references, references_tok, hypothesis, hypothesis_tok
    
    def compute(self, refs, hyps, sample_ids, ncorder=6, nworder=2, beta=2):
        references, references_tok, hypothesis, hypothesis_tok = self.parse(refs, hyps, sample_ids)
        
        result = {}
        
        logging.info('STARTING EVALUATION...')
        if 'bleu' in self.metrics:
            bleu = sacrebleu_score(references, hypothesis, self.num_refs)
            result["bleu"] = bleu
        if 'meteor' in self.metrics:
            meteor = meteor_score(references_tok, hypothesis_tok, self.num_refs, meteor_path=self.meteor_path, lng=self.lng)
            result['meteor'] = meteor
        if 'chrf++' in self.metrics:
            chrf, _, _, _ = chrF_score(references, hypothesis, self.num_refs, nworder, ncorder, beta)
            result['chrf++'] = chrf
        if 'ter' in self.metrics:
            ter = ter_score(references_tok, hypothesis_tok, self.num_refs)
            result['ter'] = ter
        if 'bert' in self.metrics:
            P, R, F1 = bert_score_(references, hypothesis, lng=self.lng)
            result['bert_precision'] = P
            result['bert_recall'] = R
            result['bert_f1'] = F1

        logging.info('FINISHING EVALUATION...')
        
        return result
