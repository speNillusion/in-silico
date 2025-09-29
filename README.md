# Computational Prototype for Plastic Degradation Analysis

## AVISO DE SEGURANÇA
Este projeto é apenas computacional. Não contém instruções de laboratório nem protocolos para manipulação de organismos. Não tente realizar procedimentos de cultura, manipulação ou liberação de microrganismos com base neste código. Consulte sempre especialistas e normas de biossegurança.

## Descrição
Este é um protótipo computacional que utiliza a API Gemini para análise de imagens e um pipeline impulsionado por IA para priorização de candidatos em degradação de plásticos. Ele encapsula chamadas à API, gerencia fluxos de dados e fornece uma interface CLI simples.
---
* O protótipo avalia a degradação do plástico e gera recomendações computacionais usando IA generativa.

### 2. Pipeline de Prioridade de Candidatos

```bash
python prototype.py run-sim
```

* Gera dataset sintético de 10 candidatos usando IA.
* Executa ingestion → features → priorização → relatório, tudo impulsionado por Gemini AI.
* Saídas:

  * `output/top_candidates.csv`: ranking computacional.
  * `output/report.md`: relatório detalhado.

### 3. Testes do Protótipo

```bash
pytest -q
```

* Testa funções críticas em modo offline/mocks.
* Verifica geração de features, priorização e análise de imagens.

---

## ⚙️ Configuração

1. Instale dependências:

```bash
pip install -r requirements.txt
```

2. Crie um arquivo `.env`:

```env
GOOGLE_API_KEY=your_key_here
USE_MOCKS=true
```

* `GOOGLE_API_KEY`: chave da Gemini (necessária para chamadas reais; use uma chave válida para evitar fallbacks).
* `USE_MOCKS=true`: ativa modo offline/simulado. Defina como false para usar chamadas reais à API.

---

## 🛠 Tecnologias e Bibliotecas

* **Linguagem:** Python 3.10+
* **IA:** Gemini (google-generativeai)
* **Manipulação de dados:** pandas
* **Visão computacional:** Pillow
* **Testes:** pytest

---

## 🧪 Funcionalidades Avançadas

### GeminiClient

* Abstrai chamadas para IA generativa e visão computacional.
* Aceita imagens PIL ou caminhos de arquivo.
* Retorna texto estruturado com recomendações e análise de degradação, incluindo tipo de plástico.

### Pipeline

* Feature Engineering: Usando Gemini AI para extrair e gerar features.
* Priorização de Candidatos: Usando IA para selecionar top candidatos.
* Geração de Relatório: Relatórios gerados por IA.

### AgentOrchestrator

* Mantém loop completo com logging.
* Usa Gemini AI para todas as etapas do pipeline.

---

## 📂 Estrutura de Arquivos

```
project-root/
│
├─ prototype.py           # Código principal
├─ requirements.txt
├─ README.md
├─ .env                   # Chave Gemini + config de mocks
├─ output/
│  ├─ top_candidates.csv
│  └─ report.md
└─ tests/
   └─ test_pipeline.py    # Testes unitários
```

---

## 🔧 Exemplos de Uso

### Executar análise de imagens

```bash
python prototype.py analyze-images samples/img1.jpg samples/img2.jpg
```

### Rodar pipeline completo com dataset sintético

```bash
python prototype.py run-sim
```

### Executar testes

```bash
pytest -q
```

---

## 📈 Observabilidade e Logs

* Logging em nível INFO/DEBUG.
* Métricas disponíveis:

  * Número de candidatos processados.
  * Scores e justificativas gerados por IA.

---

## 📝 Notas Importantes

* Este protótipo é **100% computacional** e **não substitui ensaios laboratoriais**.
* Todas as análises são **simulações baseadas em IA**.
* Para chamadas reais à API, certifique-se de que GOOGLE_API_KEY é válida; caso contrário, o sistema fará fallback para dados sintéticos.
* Sempre siga normas de biossegurança quando trabalhar com experimentos reais.

---

## 📚 Referências e Recursos

* [NASA Open Data](https://data.nasa.gov/)
* [UniProt](https://www.uniprot.org/)
* [Protein Data Bank (PDB)](https://www.rcsb.org/)
* Artigos sobre biodegradação de plásticos por fungos (literatura pública)

---

## 🏆 Próximos Passos

* Integrar dados reais de literatura e APIs.
* Expansão do agente orquestrador para pipelines escaláveis.

---

**Projeto:** Plastic Busters - Protótipo IA de Biodegradação de Plásticos
**Licença:** MIT
---
`requirements.txt`

Instale as dependências com:
```
pip install -r requirements.txt
```

## Configuração
Crie um arquivo `.env` no diretório raiz com sua chave API:
```
GOOGLE_API_KEY=your_key_here
USE_MOCKS=true  # Para modo de simulação sem chamadas reais à API
```

## Uso
O protótipo fornece comandos CLI para executar o pipeline e analisar imagens.

### Executar o Pipeline de Simulação
Gera um dataset sintético e executa o pipeline completo, salvando relatórios em `output/`.
```
python src/prototype.py run-sim --dataset sample.csv  # Opcional: forneça um caminho para CSV
```

### Analisar Imagens para Degradação de Plásticos
Analisa imagens fornecidas, incluindo identificação e tipificação do tipo de plástico (ex: PET, HDPE, etc.), e retorna/salva a análise.
```
python src/prototype.py analyze-images samples/img1.jpg samples/img2.jpg samples/img3.jpg
```

### Executar Testes
Execute os testes com pytest:
```
python -m pytest tests/test_pipeline.py
```

## Integrações
- **Gemini API**: Use a chave em `.env`. Modo mock ativado se `USE_MOCKS=true`.
- **APIs Externas**: Placeholders no código para integrações com UniProt, NASA, PDB, AlphaFold. Insira chamadas reais em `ingest_sources()`.

## Observações
- O código usa mocks para testes e execução offline.
- Relatórios são gerados em Markdown e CSV.
- Para dados reais, substitua placeholders e forneça datasets rotulados para treinamento.

Para mais detalhes, consulte o código em `src/prototype.py`.