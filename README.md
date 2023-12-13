# Vega RealTime ImageGen

Realtime image generation using segmind vega RT latent consistency adapter and streamlit + streamlit-keyup

[demo_video.webm](https://github.com/tdolan21/vega-rt-streamlit/assets/40906019/79994ceb-b859-4e8f-8dfc-26392466fcfe)

This demo assumes you have torch and CUDA properly installed for your machine.

Demo uses just under 16GB of VRAM

## Prerequisites

```
pip install streamlit streamlit-keyup transformers diffusers accelerate
```

## Installation

```bash
git clone https://github.com/tdolan21/vega-rt-streamlit
cd vega-rt-streamlit
pip install -r requirements.txt
```

## Usage

```
streamlit run app.py
```
## More information 

[Model-Card](https://huggingface.co/segmind/Segmind-VegaRT)
