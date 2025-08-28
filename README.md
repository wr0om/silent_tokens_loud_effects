# 📢 Silent Tokens, Loud Effects  

> **Understanding the Hidden Impact of Padding in LLMs**  

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  

Padding tokens are commonly used in large language models (LLMs) to equalize sequence lengths during batched inference.  
When mishandled, these tokens can **leak into computation**, but the extent of their influence has remained unclear.  

In this work, we systematically investigate padding effects across three open-source model families:  
- 🦙 **Llama**  
- 💎 **Gemma**  
- 🐦 **Qwen**  

We evaluate controlled amounts of pad tokens along **four axes**:  

| Axis               | Impact Observed |
|--------------------|-----------------|
| 🔬 Internal Activations | Drift in hidden representations |
| ✍️ Generation Quality | Reduction in output fluency & coherence |
| ⚖️ Bias            | Altered demographic biases |
| 🛡️ Safety          | Weakened safeguards & harmful outputs |

➡️ **Key takeaway**: Even small amounts of padding can harm LLM behavior. Padding requires **careful handling** in deployment pipelines.  

---

## 🚀 Getting Started  

### 1. Clone the Repository  
```bash
git clone https://github.com/wr0om/silent_tokens_loud_effects.git
cd silent_tokens_loud_effects
```

### 2. Set Up the Environment  
```bash
pip install -r requirements.txt
```

### 3. Run the Experiments  

#### Activation, Generation & Safety  
```bash
python full_activation_analysis.py
python full_generation_analysis.py
python full_safety_analysis.py
```

#### Bias (per demographic category)  
```bash
python full_bias_analysis.py --model_path meta-llama/Llama-3.1-8B-Instruct --category_num 0
python full_bias_analysis.py --model_path meta-llama/Llama-3.1-8B-Instruct --category_num 1
...
python full_bias_analysis.py --model_path meta-llama/Llama-3.1-8B-Instruct --category_num 10
```

> 💡 Replace `meta-llama/Llama-3.1-8B-Instruct` with any other model used in the paper.  

---

## 📊 Results & Visualization  

- All results are saved under the `results/` directory.  
- Bias experiment outputs are saved separately in `BBQ/results/`.  

Use the included Jupyter notebooks for visualization:  
- `full_activation_analysis.ipynb`  
- `full_generation_analysis.ipynb`  
- `full_safety_analysis.ipynb`  
- `BBQ/analysis_scripts/bias_analysis.ipynb`  

---

## 🙏 Acknowledgements  
We adapt the following GitHub repositories in our experiments:  
- https://github.com/thu-coai/AISafetyLab  
- https://github.com/neulab/BARTScore  
- https://github.com/nyu-mll/BBQ  
- https://github.com/andyrdt/refusal_direction

## 📜 Citation  
If you use this work, please cite our paper (coming soon).  
