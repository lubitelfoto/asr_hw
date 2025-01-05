## Installation


   ```bash
   pip install -r requirements.txt
   ```


## How To Use

To train a model, run the following command:

```bash
python train.py
```

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py
```

##Микро-отчет

Там явно что-то не то, лосс из отрицательного быстро выходит на плато ~2.35 и зависает там,
без градиент клиппинга градиент норм за несколько эпох уходит в бесконечность.
[Ссылка на wandb](https://wandb.ai/titan_foundation/pytorch_template_asr_example/runs/00ddzkfc?nw=nwuserhydrophonyx)
