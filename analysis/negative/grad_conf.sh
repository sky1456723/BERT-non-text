#python gradient_confusion.py --model bert-large-uncased --task MNLI\ SST-2\ STS-B --op l2 --save_dir ./Analysis/grad_l2 -a 2
#python gradient_confusion.py --model bert-large-uncased --task MNLI\ SST-2\ STS-B --seed 101 --op l2 --save_dir ./Analysis/grad_l2 -a 2
#python gradient_confusion.py --model bert-large-uncased --task MNLI\ SST-2\ STS-B --seed 102 --op l2 --save_dir ./Analysis/grad_l2 -a 2

#python gradient_confusion.py --model bert-large-uncased --task MNLI\ SST-2\ STS-B --op cosine --save_dir ./Analysis/cosine_sim -a 2
#python gradient_confusion.py --model bert-large-uncased --task MNLI\ SST-2\ STS-B --seed 101 --op cosine --save_dir ./Analysis/cosine_sim -a 2
#python gradient_confusion.py --model bert-large-uncased --task MNLI\ SST-2\ STS-B --seed 102 --op cosine --save_dir ./Analysis/cosine_sim -a 2

#python gradient_confusion.py --model bert-large-uncased --task MNLI\ SST-2\ STS-B --op dot --save_dir ./Analysis/grad_dot -a 2
#python gradient_confusion.py --model bert-large-uncased --task MNLI\ SST-2\ STS-B --seed 101 --op dot --save_dir ./Analysis/grad_dot -a 2
#python gradient_confusion.py --model bert-large-uncased --task MNLI\ SST-2\ STS-B --seed 102 --op dot --save_dir ./Analysis/grad_dot -a 2

#python gradient_confusion.py --model bert-base-uncased --task MNLI\ SST-2\ STS-B --op cosine --save_dir ./Analysis/grad_conf_new/cosine_shft1000 -a 2 --shift 1000
python gradient_confusion.py --model bert-base-uncased --task MNLI --seed 101 --op cosine --save_dir ./Analysis/grad_conf_new/cosine_shift1000 -a 2 --shift 1000
#python gradient_confusion.py --model bert-base-uncased --task MNLI\ SST-2\ STS-B --seed 102 --op cosine --save_dir ./Analysis/grad_conf_new/cosine_shft1000 -a 2 --shift 1000

#python gradient_confusion.py --model albert-base-v1 --task MNLI\ SST-2\ STS-B --op cosine --save_dir ./Analysis/grad_conf_new/cosine_shft1000 -a 2 --shift 1000
#python gradient_confusion.py --model albert-base-v1 --task MNLI\ SST-2\ STS-B --seed 101 --op cosine --save_dir ./Analysis/grad_conf_new/cosine_shft1000 -a 2 --shift 1000
#python gradient_confusion.py --model albert-base-v1 --task MNLI\ SST-2\ STS-B --seed 102 --op cosine --save_dir ./Analysis/grad_conf_new/cosine_shft1000 -a 2 --shift 1000