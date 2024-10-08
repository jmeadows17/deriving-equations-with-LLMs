Paper: TBA

LLaMa-3.1-8B QLoRA: https://huggingface.co/jmeadows17/DerivationGeneration8B trained using ```llama_3.1_qlora_sft.py```.

A customisable "from scratch" LLaMa-2/3 architecture training script: ```custom_llama_3_training.py```. 

Best T5-base model: https://huggingface.co/jmeadows17/MathT5-base

Best T5-large model: https://huggingface.co/jmeadows17/MathT5-large

Other fine-tuned models: https://drive.google.com/drive/folders/1cLy_UhJf9kEN4tKrr8M9DcBQVQotgf58?usp=sharing

Other data including training set: https://drive.google.com/drive/folders/1an_7TYO_V8U9lFHuesN-ZIBplAYcdJmo?usp=sharing

For 100 evaluation examples including prompts, ground truth derivations, model generations, and perturbations, download the ```*_results.json``` datasets.

To generate your own synthetic data run ```derivation_generation.py``` with ```sympy==1.5.1```. 
