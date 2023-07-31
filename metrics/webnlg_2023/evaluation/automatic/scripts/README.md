# Evaluation Scripts

Before running evaluation, you must have a number of dependencies installed on your system.

```bash
./install_dependencies.sh
```

The `eval.py` script can be used to perform automatic evaluation of system outputs. The following example illustrates a potential use-case:

```bash
python eval.py -hyp outputs/baselines/forge2017_en2br_test.br -ref breton_test.txt -nr 1 -m bleu,meteor,chrf++,ter,bert -lng br
```

```
usage: eval.py [-h] -R REFERENCE -H HYPOTHESIS [-lng LANGUAGE] [-nr NUM_REFS]
               [-m METRICS] [-nc NCORDER] [-nw NWORDER] [-b BETA]

optional arguments:
  -h, --help            show this help message and exit
  -ref REFERENCE, --reference REFERENCE
                        reference translation
  -hyp HYPOTHESIS, --hypothesis HYPOTHESIS
                        hypothesis translation
  -lng LANGUAGE, --language LANGUAGE
                        evaluated language
  -nr NUM_REFS, --num_refs NUM_REFS
                        number of references
  -m METRICS, --metrics METRICS
                        evaluation metrics to be computed
  -nc NCORDER, --ncorder NCORDER
                        chrF metric: character n-gram order (default=6)
  -nw NWORDER, --nworder NWORDER
                        chrF metric: word n-gram order (default=2)
  -b BETA, --beta BETA  chrF metric: beta parameter (default=2)
```

---

This evaluation script is a modified version of the original from [this repo](https://github.com/WebNLG/GenerationEval).
