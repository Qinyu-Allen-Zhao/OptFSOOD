echo -----MSP-----; python eval_ood.py -b 128 -e imagenet -m rn50 -d msp

echo -----ODIN-----; python eval_ood.py -b 32 -e imagenet -m rn50 -d odin

echo -----EBO-----; python eval_ood.py -b 128 -e imagenet -m rn50 -d ebo

echo -----GradNorm-----; python eval_ood.py -b 32 -e imagenet -m rn50 -d gradnorm

echo -----DICE-----; python eval_ood.py -b 128 -e imagenet -m rn50 -d dice

echo -----VIM-----; python eval_ood.py -b 128 -e imagenet -m rn50 -d vim

echo -----KNN-----; python eval_ood.py -b 128 -e imagenet -m rn50 -d knn

echo -----ReAct-----; python eval_ood.py -b 128 -e imagenet -m rn50 -d react

echo -----Ash-P-----; python eval_ood.py -b 128 -e imagenet -m mb -d ash_p

echo -----Ash-B-----; python eval_ood.py -b 128 -e imagenet -m rn50 -d ash_b

echo -----Ash-S-----; python eval_ood.py -b 128 -e imagenet -m rn50 -d ash_s

echo -----Vra-P-----; python eval_ood.py -b 128 -e imagenet -m rn50 -d vra

echo -----Ours-----; python eval_ood.py -b 128 -e imagenet -m rn50 -d optact