# DS Final Project: Watermarking Framework

This project provides a basic framework for experimenting with watermark insertion and detection in deep learning models.  
Please follow the instructions below to set up your environment and begin working.

---

## Installation Guide

You can set up the environment using Anaconda.  
If you are new to Conda, please refer to the official guide:  
[Getting Started with Anaconda](https://www.anaconda.com/docs/getting-started/getting-started)

---

### Step 1: Create the Environment

Use the following command to create the environment from the provided file:

```bash
conda env create -f env.yaml
```
Use the following command to activate the environment:
```bash
conda activate DS_final_project
```

### Step 2: Complete your watermark implementation
You only have modify the file wm_student.py to complete this project.
Implement your version of "insert_watermark", "extract_features" and "train_detector"
Please do not modify the file wm_base.py and main.py.
You may trace the code to understand how the framework work.
```python
    def insert_watermark(self, model: nn.Module) -> nn.Module:
        #[TODO]
        pass

    def extract_features(self, model: nn.Module) -> torch.Tensor:
        #[TODO]

    def train_detector(self, clean_models: list, wm_models: list, epochs: int = 100):
        #[TODO]
        pass
```

### Step 3: Evaluate your watermark insertion
After you finish the "insert_watermark", "extract_features" and "train_detector",
you can simply evaluate your watermark insertion by using the command.
```bash
python main.py
```