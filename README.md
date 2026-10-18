# Predição de preços de imóveis: criação do conjunto de dados por *scraping*

**Objetivo do projeto:** criar um conjunto de dados que permita prever o preço de venda de imóveis. Os dados são **coletados por *scraping*** do site do QuintoAndar (imóveis à venda no Rio de Janeiro até R$ 1 milhão), validados contra a base de referência do professor, inspecionados, diagnosticados e limpos com regras explícitas. Uma prova de conceito com modelos de regressão confirma que a base serve para a predição, e um segundo notebook audita a qualidade da coleta. Tudo é reprodutível no Google Colab.

**Autor:** Vinicius Machado

## Estrutura dos arquivos

| Arquivo | Descrição |
|---|---|
| `Predicao_Precos_Imoveis_Vinicius_Machado.ipynb` | **Etapa 1:** *scraping*, preparação da base e prova de conceito (seções 0 a 10). É entregue já executado, com as saídas e os gráficos |
| `EDA_Qualidade_Dados_Vinicius_Machado.ipynb` | **Etapa 2:** análise exploratória da qualidade dos dados do *scraping* (itens 1 a 10 do enunciado). Também é entregue executado |
| `quintoandar_rio_scraping_20261018.csv` | **Coleta oficial do nosso *scraping*** (18.081 anúncios, 18/10/2026), sem nenhuma correção. É a base usada na modelagem da etapa 1 e analisada na etapa 2 |
| `quintoandar_rio_modelagem_Vinicius_Machado.csv` | Base final para modelagem (15.588 linhas e 16 colunas), gravada com `encoding="utf-8-sig"` |
| `README.md` | Este arquivo |
| `_ferramentas/` (opcional) | Scripts auxiliares do autor. `build_notebook.py` e `build_notebook_eda.py` geram os `.ipynb` a partir das células escritas neles, e `executar_notebook.py [notebook.ipynb]` executa um notebook do zero e grava as saídas. **Não são necessários** para reproduzir o trabalho, porque os notebooks são autossuficientes |

## Conteúdo do notebook da etapa 1

1. **Definição do problema:** regressão supervisionada e os critérios de escolha da base (C1–C6)
2. **Fontes de dados candidatas:** comparação entre Kaggle SP 2019, Kaggle SP 2023 (aluguel), a biblioteca `dadosimob`, a base do professor (QuintoAndar Rio, 05/10/2026) e a nossa coleta, com a justificativa da escolha
3. **Extração por *scraping*:** investigação do site, `robots.txt`, *scraper* com pausas e novas tentativas, demonstração da paginação instável da API, divisão da busca em faixas de preço de até 500 anúncios, gravação do CSV, conferência da coleta e validação contra a base do professor
4. **Inspeção inicial:** dicionário de dados, tipos, ausentes (incluindo o código −1) e estatísticas
5. **Diagnóstico de qualidade:** 9 problemas, cada um com a sua evidência
6. **Limpeza e preparação:** uma regra por problema, com o impacto de cada uma (86% das linhas preservadas; a maior perda é a remoção dos preços no piso de R$ 200 mil)
7. **Análise exploratória** voltada à predição
8. **Prova de conceito:** referência ingênua, regressão linear e Gradient Boosting, mais validação cruzada e importância dos atributos
9. **Base final** para modelagem
10. **Conclusão:** limitações e próximos passos

### Resultado da prova de conceito (conjunto de teste)

| Modelo | MAE (R$) | MAPE | Erro ≤ 15% | R² (log) |
|---|---|---|---|---|
| Referência: m² do bairro × área | 165.148 | 31,2% | 35,8% | 0,379 |
| Regressão linear | 85.888 | 17,6% | 52,4% | 0,773 |
| Gradient Boosting | 78.773 | 16,3% | 56,4% | 0,793 |

O modelo vale para o segmento coletado (anúncios próprios do QuintoAndar até R$ 1 milhão). O R² é moderado porque a faixa de preços é estreita. A métrica mais relevante é o erro percentual, que cai quase pela metade em relação à referência.

## Etapa 2: análise da qualidade dos dados

O notebook `EDA_Qualidade_Dados_Vinicius_Machado.ipynb` audita o CSV gerado pelo **nosso *scraping*** (`quintoandar_rio_scraping_20261018.csv`: 18.081 anúncios próprios do QuintoAndar à venda no Rio até R$ 1 milhão), sem nenhuma correção prévia. Ele segue os 10 itens do enunciado: inspeção inicial, ausentes, duplicados, tipos, valores inconsistentes, *outliers*, variáveis categóricas (bairro e rua), indicadores de qualidade, visualizações e diagnóstico final.

Principais achados:

- **Código −1 como ausente disfarçado:** no condomínio, ele significa "sem condomínio" nas casas de rua e "não informado" nos demais tipos (812 anúncios). No IPTU, aparece em 1.046 anúncios. Nenhuma célula fica vazia, então o `isna()` não detecta nada.
- **Preço concentrado em exatamente R$ 200 mil** (2.165 anúncios, 12%): provável piso de preço da plataforma.
- **Bairros:** 215 grafias, com espaços extras, maiúsculas/minúsculas, acentos e abreviações ("Barra Olímpica" aparece de 5 formas), que caem para 144 após a padronização.
- **Ruas:** formatos misturados (só o nome, endereço completo com CEP, abreviações, nome do condomínio, bloco e apartamento).
- **Valores impossíveis:** condomínio e IPTU de centenas de milhares de reais por mês e áreas de 0 a 2 m².
- **Duplicatas prováveis:** 77 cópias de anúncios idênticos em todas as características, com códigos diferentes.

| Indicador | Base original | Base tratada |
|---|---|---|
| Completude: todas as 22 colunas (`isna`) | 100,00% | 98,98% |
| Completude: 6 variáveis do enunciado (`isna`) | 100,00% | 98,09% |
| Completude: 6 variáveis do enunciado, contando os ausentes disfarçados | 98,28% | 98,09% |
| Unicidade: código do anúncio | 100,00% | 100,00% |
| Unicidade: todas as características | 99,57% | 99,57% |
| Validade: regras do enunciado (preço > 0, ≤ R$ 1 mi, metragem > 0) | 99,99% | 100,00% |
| Validade: regras ampliadas (inclui condomínio e IPTU) | 88,98% | 99,99% |

**Diagnóstico:** a base é confiável, com ressalvas e depois dos tratamentos, para analisar preço e área por bairro no segmento de imóveis de até R$ 1 milhão anunciados diretamente pelo QuintoAndar. Ela não representa o mercado inteiro: os classificados e os imóveis acima de R$ 1 milhão ficaram de fora, e o segmento mais barato está distorcido pelo piso de R$ 200 mil.

## Como executar no Google Colab, do zero

1. Acesse [colab.research.google.com](https://colab.research.google.com) e escolha **Arquivo → Fazer upload de notebook**.
2. Selecione `Predicao_Precos_Imoveis_Vinicius_Machado.ipynb` (etapa 1) ou `EDA_Qualidade_Dados_Vinicius_Machado.ipynb` (etapa 2) e faça o *upload* de `quintoandar_rio_scraping_20261018.csv` (a coleta oficial) no painel **Arquivos** (ícone de pasta, à esquerda). A etapa 2 exige esse arquivo. Na etapa 1, sem ele, a modelagem usa a coleta feita na própria execução, e os números vão diferir dos citados no texto.
3. Escolha **Ambiente de execução → Executar tudo** (ou `Ctrl + F9`).
4. Ao final, o arquivo `quintoandar_rio_modelagem_Vinicius_Machado.csv` aparece no painel **Arquivos** (ícone de pasta, à esquerda).

Na etapa 1, o *scraping* acessa o site do QuintoAndar e leva cerca de 5 minutos. Ele grava um arquivo novo e nunca sobrescreve a coleta oficial. Para não acessar o site, use `COLETAR = False` na seção 3.5. A base de referência do professor, usada para validar o *scraper*, é lida **direto da URL**:

```
https://raw.githubusercontent.com/alvaroriz/datascience_datasets/refs/heads/main/quintoandar_rio_20261005.csv
```

A leitura exige acesso à internet. A etapa 1 leva cerca de 10 minutos (5 deles de coleta), e a etapa 2, menos de 1 minuto.

## Dependências

Todas já vêm instaladas no Google Colab:

- Python 3.10 ou superior
- `pandas` (testado na versão 3.0.5)
- `numpy`
- `matplotlib`
- `seaborn`
- `requests` (para o *scraping*)
- `scikit-learn` (testado na versão 1.9.1)

Para executar localmente, fora do Colab:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests jupyter
jupyter notebook Predicao_Precos_Imoveis_Vinicius_Machado.ipynb
```

## Observações sobre reprodutibilidade

- **Fonte única:** não há CSV corrigido à mão. Toda transformação está no código, e o notebook gera sempre a mesma base final a partir da coleta oficial.
- **Base original intocada:** a limpeza é feita em uma cópia de `df_raw`, o que permite comparar o antes e o depois.
- **Resultados determinísticos:** a semente (`SEMENTE = 42`) fixa a divisão treino/teste, a validação cruzada e os modelos.
- **Sem vazamento de dados:** o preço por m² (derivado do preço) não entra no modelo, e os anúncios repetidos são removidos antes da divisão treino/teste. A imputação de ausentes e o agrupamento de bairros raros são aprendidos só no conjunto de treino.
- **Verificações automáticas:** alguns `assert` interrompem a execução se uma regra de limpeza não tiver surtido efeito, se alguma coluna ficar sem tratamento no *pipeline* ou se o arquivo final relido for diferente do gravado.
- ***Scraping* muda a cada execução:** uma nova coleta traz os anúncios do dia e grava um CSV com a data. A modelagem da etapa 1 e a etapa 2 usam a coleta oficial de 18/10/2026, que fica salva no projeto, para que os números citados no texto continuem valendo.
- **Se a fonte mudar:** a base de referência está em um repositório público de terceiros, e o *scraper* depende de uma API interna do QuintoAndar, não documentada. Se ele for alterado, os números citados no texto podem deixar de coincidir com as saídas. As regras, porém, continuam válidas, porque são genéricas.
