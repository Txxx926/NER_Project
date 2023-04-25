CUDA_VISIBLE_DEVICES=1 python3 transformers_ner/evaluate.py \
  "model.model.language_model=bert-base-uncased" \
  evaluate.checkpoint_path="/usr/local/NER/NLP_Course_Project/experiments/bert-base-uncased/2023-04-11/07-12-38/ner/eufuoao0/checkpoints/checkpoint-val_f1_0.6600-epoch_03.ckpt"