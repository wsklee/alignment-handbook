Submission for CZ4042 Project

Repository was forked from [Alignment Handbook](https://github.com/huggingface/alignment-handbook)

Code was added to support SFT and CFT training on sentiment analysis tasks on DistilBERT models.

Jupyter notebooks were ran in Google Colab.
For result analysis, please refer to the report and the jupyter notebooks.

Files added

- recipes/accelerate_configs/single_gpu.yaml
- recipes/distilbert/config_imdb_CFT.yaml
- recipes/distilbert/config_imdb_SFT.yaml
- recipes/distilbert/config_imdb_SFTCFT.yaml

- scripts/run_sentiment_sft.py
- scripts/run_sentiment_cft.py

- src/alignment/cft/imdb_preprocess.py
- src/alignment/cft/contrastive_trainer.py
- src/alignment/cft/loss.py
- src/alignment/models/distilbert_cl.py

To run the code (preferably on a single GPU),

1. Create and activate environment
   conda env create -f environment.yml
   conda activate alignment-handbook

2. Install the alignment package
   pip install -e .

3. Login securely using interactive prompts
   wandb login

huggingface-cli login

4. Create logs directory
   mkdir -p logs

5. Submit training job
   '''

# SFT imdb

sbatch --job-name=handbook_sentiment recipes/train.sh \
 distilbert \
 sentiment_sft \
 imdb_SFT \
 single_gpu

# for CFT imdb

sbatch --job-name=handbook_sentiment recipes/train.sh \
 distilbert \
 sentiment_cft \
 imdb_CFT \
 single_gpu

# for SFTCFT imdb

sbatch --job-name=handbook_sentiment recipes/train.sh \
 distilbert \
 sentiment_cft \
 imdb_SFTCFT \
 single_gpu
'''
