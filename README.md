## Installation


   ```bash
   pip install -r requirements.txt
   ```


## How To Use

To train a model, run the following command:

```bash
python train.py
```

To download model:

```bash
python model_download.py
```


To run inference (evaluate the model or save predictions):

```bash
python inference.py
```

## Микро-отчет

Там явно что-то не то, лосс из отрицательного быстро выходит на плато ~1.6 и зависает там,
без градиент клиппинга градиент норм за несколько эпох уходит в бесконечность.
[Ссылка на wandb](https://api.wandb.ai/links/titan_foundation/lqdk0w28)
