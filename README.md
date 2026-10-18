# Predição de preços de imóveis: escolha e preparação do conjunto de dados

**Objetivo do projeto:** encontrar ou criar um conjunto de dados que permita prever o preço de venda de imóveis. A base escolhida é lida direto da fonte, inspecionada, diagnosticada e limpa com regras explícitas. Uma prova de conceito com modelos de regressão confirma que ela serve para a predição. Tudo está em um notebook reprodutível no Google Colab.

**Autor:** Vinicius Machado

## Estrutura dos arquivos

| Arquivo | Descrição |
|---|---|
| `Predicao_Precos_Imoveis_Vinicius_Machado.ipynb` | Notebook principal, com as seções 0 a 10. É entregue já executado, com as saídas e os gráficos |
| `quintoandar_rio_modelagem_Vinicius_Machado.csv` | Base final para modelagem, com 41.300 linhas e 16 colunas, gravada com `encoding="utf-8-sig"` |
| `README.md` | Este arquivo |
| `_ferramentas/` (opcional) | Scripts auxiliares do autor. `build_notebook.py` gera o `.ipynb` a partir das células escritas nele, e `executar_notebook.py` executa o notebook do zero e grava as saídas. **Não são necessários** para reproduzir o trabalho, porque o notebook é autossuficiente |

## Conteúdo do notebook

1. **Definição do problema:** regressão supervisionada e os critérios de escolha da base (C1–C6)
2. **Fontes de dados candidatas:** comparação entre Kaggle SP 2019, Kaggle SP 2023 (aluguel), a biblioteca `dadosimob` e QuintoAndar Rio 2026, com a justificativa da escolha
3. **Extração** da base direto da URL
4. **Inspeção inicial:** dicionário de dados, tipos, ausentes e o vazamento de dados em `preco_m2`
5. **Diagnóstico de qualidade:** 8 problemas, cada um com a sua evidência
6. **Limpeza e preparação:** uma regra por problema, com o impacto de cada uma (99,64% das linhas preservadas)
7. **Análise exploratória** voltada à predição
8. **Prova de conceito:** referência ingênua, regressão linear e Gradient Boosting, mais validação cruzada e importância dos atributos
9. **Base final** para modelagem
10. **Conclusão:** limitações e próximos passos

### Resultado da prova de conceito (conjunto de teste)

| Modelo | MAE (R$) | MAPE | Erro ≤ 15% | R² (log) |
|---|---|---|---|---|
| Referência: m² do bairro × área | 470.789 | 33,4% | 34,0% | 0,821 |
| Regressão linear | 373.676 | 24,6% | 42,0% | 0,888 |
| Gradient Boosting | 323.766 | 21,2% | 46,5% | 0,916 |

## Como executar no Google Colab, do zero

1. Acesse [colab.research.google.com](https://colab.research.google.com) e escolha **Arquivo → Fazer upload de notebook**.
2. Selecione `Predicao_Precos_Imoveis_Vinicius_Machado.ipynb`.
3. Escolha **Ambiente de execução → Executar tudo** (ou `Ctrl + F9`).
4. Ao final, o arquivo `quintoandar_rio_modelagem_Vinicius_Machado.csv` aparece no painel **Arquivos** (ícone de pasta, à esquerda).

Não é preciso fazer *upload* de dados. O próprio código lê a base **direto da URL**:

```
https://raw.githubusercontent.com/alvaroriz/datascience_datasets/refs/heads/main/quintoandar_rio_20261005.csv
```

A leitura exige acesso à internet. A execução completa leva poucos minutos.

## Dependências

Todas já vêm instaladas no Google Colab:

- Python 3.10 ou superior
- `pandas` (testado na versão 3.0.5)
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn` (testado na versão 1.9.1)

Para executar localmente, fora do Colab:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Predicao_Precos_Imoveis_Vinicius_Machado.ipynb
```

## Observações sobre reprodutibilidade

- **Fonte única:** não há CSV corrigido à mão. Toda transformação está no código, e o notebook gera sempre a mesma base final a partir da mesma URL.
- **Base original intocada:** a limpeza é feita em uma cópia de `df_raw`, o que permite comparar o antes e o depois.
- **Resultados determinísticos:** a semente (`SEMENTE = 42`) fixa a divisão treino/teste, a validação cruzada e os modelos.
- **Sem vazamento de dados:** `preco_m2` (derivada do preço) não entra no modelo. A imputação de ausentes e o agrupamento de bairros raros são aprendidos só no conjunto de treino.
- **Verificações automáticas:** alguns `assert` interrompem a execução se uma regra de limpeza não tiver surtido efeito, se alguma coluna ficar sem tratamento no *pipeline* ou se o arquivo final relido for diferente do gravado.
- **Se a fonte mudar:** o arquivo está em um repositório público de terceiros. Se ele for alterado, os números citados no texto podem deixar de coincidir com as saídas. As regras, porém, continuam válidas, porque são genéricas.
