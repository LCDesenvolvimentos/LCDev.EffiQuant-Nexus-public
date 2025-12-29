# LC Desenvolvimentos - EffiQuant-Nexus

[![License: PolyForm Strict](https://img.shields.io/badge/License-PolyForm--Strict-blue.svg)](https://polyformproject.org/licenses/strict/1.0/)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.1234567[TBD]-blue.svg)](https://doi.org/10.5281/zenodo.1234567)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-lcdev-yellowgreen.svg)](https://huggingface.co/lcdev)
[![Research](https://img.shields.io/badge/Research-AI%20%7C%20ML-orange.svg)](#linhas-de-pesquisa)

> Este repositório implementa e documenta a pesquisa **EffiQuant Nexus**.
> 
> [EN] This repository implements and documents the EffiQuant Nexus research.

## Table of Contents

- [Overview](#overview)
- [Motivation and Rationale for the Research](#motivation-and-rationale-for-the-research)
- [Contribution of EffiQuant Nexus](#contribution-of-effiquant-nexus)
- [Current State (What Is Already Implemented)](#current-state-what-is-already-implemented)
- [How to Run (Reproducibility)](#how-to-run-reproducibility)
- [Methodology](#methodology)
- [Contributing](#contributing)
- [Statistics](#statistics)
- [Acknowledgment](#acknowledgment)
- [Support](#support)
- [Additional Resources](#additional-resources)
- [About LC Desenvolvimentos](#about-lc-desenvolvimentos)
- [License](#license)

## Overview

A pesquisa **EffiQuant Nexus** é um framework de compressão adaptativa que integra quantização, poda estrutural e destilação de conhecimento em um ciclo fechado de feedback. O objetivo é reduzir custo de inferência (latência, memória e tamanho do modelo) sem violar restrições operacionais explícitas (por exemplo, acurácia mínima).

[EN] The EffiQuant Nexus research is an adaptive compression framework that integrates quantization, structural pruning, and knowledge distillation within a closed-loop feedback cycle. The goal is to reduce inference cost (latency, memory, and model size) without violating explicit operational constraints (e.g., minimum accuracy).

**Framework for Adaptive Model Compression with Performance Guarantee**
- Combination of dynamic quantization, structured pruning, and knowledge distillation
- Real-time feedback to maintain operational accuracy
- Drastic reduction in computational footprint

## Motivation and Rationale for the Research

Em sistemas com LLMs e outros modelos grandes, técnicas de compressão são eficazes, mas:
- Os trade-offs variam por modelo, por hardware e por runtime (CPU/GPU, kernels, compiladores).
- Configurações fixas podem degradar qualidade de forma imprevisível quando há mudança de distribuição, prompts ou carga.
- A combinação de técnicas (quantização + poda + destilação) gera efeitos não lineares e difíceis de calibrar manualmente.

A hipótese central desta pesquisa é que **compressão deve ser tratada como um processo controlado e auditável**, com decisões iterativas, medições em tempo (quase) real e capacidade de rollback quando uma restrição é violada.

[EN]
In systems using LLMs and other large models, compression techniques are effective, but:
- Trade-offs vary by model, hardware, and runtime (CPU/GPU, kernels, compilers).
- Fixed configurations can degrade quality unpredictably when there are shifts in distribution, prompts, or workload.
- Combining techniques (quantization + pruning + distillation) produces non-linear effects that are difficult to calibrate manually.

The central hypothesis of this research is that **compression should be treated as a controlled and auditable process**, with iterative decision-making, (near) real-time measurements, and the ability to roll back when a constraint is violated.

## Contribution of EffiQuant Nexus
- **Controle por restrições (garantia operacional)**: cada intervenção é aceita apenas se as métricas observadas permanecerem dentro de limites definidos.
- **Rollback por snapshot**: o estado do modelo pode ser restaurado quando uma ação candidata degrada o desempenho além do permitido.
- **Ações plugáveis**: quantização, poda e destilação são implementadas como ações independentes, permitindo extensões para diferentes stacks de LLM (PyTorch, TensorRT-LLM, vLLM, etc.).
- **Logging estruturado**: o processo gera rastros auditáveis do “antes/depois”, decisão (aceito/rejeitado) e motivo.

[EN]
- **Constraint-based control (operational guarantees)**: each intervention is accepted only if the observed metrics remain within predefined limits.
- **Snapshot-based rollback**: the model state can be restored when a candidate action degrades performance beyond acceptable thresholds.
- **Pluggable actions**: quantization, pruning, and distillation are implemented as independent actions, enabling extensions to different LLM stacks (PyTorch, TensorRT-LLM, vLLM, etc.).
- **Structured logging**: the process generates auditable traces of the “before/after” state, the decision (accepted/rejected), and the rationale.

## Current State (What Is Already Implemented)
- Núcleo do controlador e pipeline:
  - `effiquant_nexus/controller.py`
  - `effiquant_nexus/pipeline.py`
  - `effiquant_nexus/types.py`
- Ações de compressão (integrações PyTorch, quando aplicável):
  - `effiquant_nexus/compression/quantization.py`
  - `effiquant_nexus/compression/pruning.py`
  - `effiquant_nexus/compression/distillation.py`
- Experimento toy (demonstra o mecanismo de feedback + rollback e gera artefatos):
  - `scripts/run_toy_pipeline.py`
  - saída em `outputs/toy_result.json`
- Experimento real com LLMs:
  - `scripts/run_llm_experiment.py` 
- Texto acadêmico:
  - versão completa em `research/paper.tex` (PDF: `research/paper.pdf`)
  - versão curta em `research/paper_short.tex` (PDF: `research/paper_short.pdf`)

[EN]
- Core controller and pipeline:
  - `effiquant_nexus/controller.py`
  - `effiquant_nexus/pipeline.py`
  - `effiquant_nexus/types.py`
- Compression actions (PyTorch integrations where applicable):
  - `effiquant_nexus/compression/quantization.py`
  - `effiquant_nexus/compression/pruning.py`
  - `effiquant_nexus/compression/distillation.py`
- Toy experiment (demonstrates the feedback + rollback mechanism and generates artifacts):
  - `scripts/run_toy_pipeline.py`
  - output in `outputs/toy_result.json`
- Real LLM experiment:
  - `scripts/run_llm_experiment.py`
- Academic text:
  - full version in `research/paper.tex` (PDF: `research/paper.pdf`)
  - short version in `research/paper_short.tex` (PDF: `research/paper_short.pdf`)

## How to Run (Reproducibility)

### Prerequisites

- **Git** installed
- **GitHub account**
- **Code editor** (VS Code recommended)
- **Basic knowledge** of Git and GitHub

### Requirements
- Python 3.10+

### Install development dependencies
```bash
python -m pip install -r requirements-dev.txt
```

### Install dependencies for LLM experiments (optional)
O protótipo de LLM usa `transformers` e `torch`. Para quantização 8-bit/4-bit (bitsandbytes) e alguns caminhos de carregamento em GPU, instale também `accelerate`:
```bash
python -m pip install -r requirements-llm.txt
```

[EN]
The LLM prototype uses transformers and torch. For 8-bit/4-bit quantization (bitsandbytes) and some GPU loading paths, also install accelerate:
```bash
python -m pip install -r requirements-llm.txt
```

### Run quality checks
```bash
python -m ruff check .
python -m mypy .
python -m pytest -q
```

### Run the toy experiment
```bash
python -m scripts.run_toy_pipeline
```
O arquivo `outputs/toy_result.json` registra baseline, modelo final e cada passo (aceito/rejeitado) com detalhes e motivo.

[EN]
```bash
python -m scripts.run_toy_pipeline
```
The file outputs/toy_result.json records the baseline, final model, and each step (accepted/rejected) with details and rationale.

### Run LLM experiments (runtime-default, grid, sequential, adaptive)
O runner principal é `scripts/run_llm_experiment.py`. Ele gera um JSON em `outputs/` com baseline/final/steps.
- Runbook para RTX 3060: `research/rtx3060_runbook.txt`
- Exemplo rápido (baseline de runtime, sem compressão):
```bash
python -m scripts.run_llm_experiment --model-id sshleifer/tiny-gpt2 --mode runtime-default --out outputs/llm_runtime_default.json
```

[EN]
The main runner is scripts/run_llm_experiment.py. It generates a JSON in outputs/ with baseline/final/steps.
- Runbook for RTX 3060: research/rtx3060_runbook.txt
- Quick example (runtime baseline, no compression):
```bash
python -m scripts.run_llm_experiment --model-id sshleifer/tiny-gpt2 --mode runtime-default --out outputs/llm_runtime_default.json
```

## Methodology

### Our 4-Step Approach

#### 1. Rigorous Theoretical Formulation
- **Solid mathematical foundation** with theorems and propositions
- **Theoretical limits** clearly established
- **Conceptual frameworks** well defined
- **Research hypotheses** clearly formulated

#### 2. Controlled Experimentation
- **Methodically designed** experimental protocol
- **Established reference** benchmarks
- **Quantifiable evaluation** metrics
- **Rigorous experimental** controls

#### 3. Academic Reproducibility
- **Open source** with complete documentation
- **Training data** clearly defined
- **Experimental procedures** documented
- **Reproducible environment** specified

#### 4. Practical Application
- **Real use cases** identified and validated
- **Industrial partners** involved in the process
- **Validation scenarios** in real environment
- **Practical impact** demonstrated and measured

## Contributing

We welcome contributions from the entire community! See our [Contributing Guide](CONTRIBUTING.md) for details on:

- How to fork and set up the environment
- Code and documentation standards
- Pull request process
- Types of contributions accepted
- Research methodology

### Types of Contributions

- **Bug Fixes**: Bugs, performance issues
- **Features**: New features, improvements
- **Research**: Algorithms, methodologies
- **Documentation**: Guides, tutorials, references
- **Tests**: Test cases, coverage
- **Infrastructure**: CI/CD, tools

### Communication Channels

- **GitHub Issues**: For bugs and technical discussions
- **GitHub Discussions**: For general questions
- **Email**: [lcdev@lcdesenvolvimentos.com.br](mailto:lcdev@lcdesenvolvimentos.com.br)
- **Hugging Face**: [https://huggingface.co/lcdev](https://huggingface.co/lcdev)

## Support

### Main Contact
- **Email**: [lcdev@lcdesenvolvimentos.com.br](mailto:lcdev@lcdesenvolvimentos.com.br)
- **Website**: [https://lcdesenvolvimentos.github.io/](https://lcdesenvolvimentos.github.io/)
- **Hugging Face**: [https://huggingface.co/lcdev](https://huggingface.co/lcdev)

### Support Types

- **Research Support**: Methodology, experimentation
- **Industrial Partnerships**: Collaboration, R&D
- **Technical Support**: Implementation, troubleshooting
- **Security**: Vulnerabilities, security issues

### Response Times

| Type | Time |
|------|------|
| **Security** | 4-8 hours |
| **Emergency** | 4-8 hours |
| **Critical Bug** | 24 hours |
| **General Question** | 24-48 hours |
| **Research** | 48-72 hours |
| **Partnerships** | 72+ hours |

## Statistics

![GitHub stars](https://img.shields.io/github/stars/LCDesenvolvimentos/LCDev.GitRepoTemplate?style=social)
![GitHub forks](https://img.shields.io/github/forks/LCDesenvolvimentos/LCDev.GitRepoTemplate?style=social)
![GitHub issues](https://img.shields.io/github/issues/LCDesenvolvimentos/LCDev.GitRepoTemplate)
![GitHub pull requests](https://img.shields.io/github/issues-pr/LCDesenvolvimentos/LCDev.GitRepoTemplate)
![GitHub license](https://img.shields.io/github/license/LCDesenvolvimentos/LCDev.GitRepoTemplate)

## Acknowledgment

### Contributors
We thank all contributors who help improve this template:

<!-- Contributors will be automatically updated via GitHub -->

### Papers and Publications
When using this template for research, please cite:

```bibtex
@software{LCDesenvolvimentos2025,
  title={LCDev EffiQuant Nexus Research},
  author={{LC Desenvolvimentos Team}},
  version={1.0.0},
  date={2025-12-10},
  url={https://github.com/LCDesenvolvimentos/LCDev.EffiQuant-Nexus-public},
  license={Polyform-Strict}
}
```

## Additional Resources

### Documentation
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](.github/SECURITY.md)
- [Support Guide](.github/SUPPORT.md)
- [Code of Conduct](.github/CODE_OF_CONDUCT.md)

### Learning
- [AI Guide for Researchers](https://lcdesenvolvimentos.github.io/ai-guide/)
- [MLOps Best Practices](https://lcdesenvolvimentos.github.io/mlops-guide/)
- [Research Methodology](https://lcdesenvolvimentos.github.io/research-methodology/)

## About LC Desenvolvimentos

**LC Desenvolvimentos** is a company specialized in Artificial Intelligence research, specialized hardware, and software development. Our mission is to explore the frontiers of AI research, focusing on computational efficiency, specialized models, and brain-inspired architectures.

### Our Mission
- **Advanced Research**: Advance the state of the art in AI
- **Collaboration**: Connect academia and industry
- **Innovation**: Develop transformative solutions
- **Education**: Share knowledge with the community

### Where to Find Us
- **Rio de Janeiro, RJ, Brazil**
- **Email**: [lcdev@lcdesenvolvimentos.com.br](mailto:lcdev@lcdesenvolvimentos.com.br)
- **LinkedIn**: [LC Desenvolvimentos](https://linkedin.com/company/lcdesenvolvimentos)
- **Twitter**: [@LCDesenvolvimentos](https://twitter.com/lcdesenvolvimentos)

## License

This project is licensed under the Polyform-Strict License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Building the Future of Artificial Intelligence Together**

Made with heart by the LC Desenvolvimentos Team

[Star this repo](https://github.com/LCDesenvolvimentos/LCDev.GitRepoTemplate) • 
[Report a bug](https://github.com/LCDesenvolvimentos/LCDev.GitRepoTemplate/issues) • 
[Request a feature](https://github.com/LCDesenvolvimentos/LCDev.GitRepoTemplate/issues)

</div>
