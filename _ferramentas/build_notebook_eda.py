"""Gera o notebook EDA_Qualidade_Dados_Vinicius_Machado.ipynb a partir das células definidas aqui."""
import os

import nbformat as nbf

PASTA_FERRAMENTAS = os.path.dirname(os.path.abspath(__file__))
PASTA_PROJETO = os.path.dirname(PASTA_FERRAMENTAS)
ARQUIVO = os.path.join(PASTA_PROJETO, "EDA_Qualidade_Dados_Vinicius_Machado.ipynb")

cells = []


def md(texto):
    cells.append(nbf.v4.new_markdown_cell(texto.strip("\n")))


def code(texto):
    cells.append(nbf.v4.new_code_cell(texto.strip("\n")))


# =====================================================================
# Cabeçalho
# =====================================================================
md("""
# Análise exploratória da qualidade dos dados: imóveis à venda no Rio de Janeiro (*scraping* do QuintoAndar)

**Etapa 2 do trabalho:** avaliar se os dados coletados por *scraping* são consistentes e confiáveis o bastante para uma análise do mercado imobiliário.

**Autor:** Vinicius Machado

O objetivo aqui **não** é descrever o mercado, e sim **auditar a coleta**: o que falta, o que está repetido, o que está no formato errado, o que é impossível e o que é apenas incomum. Cada seção corresponde a um item do enunciado e termina com a interpretação dos resultados.

**Sumário**

0. Ambiente e leitura da base
1. Inspeção inicial
2. Dados ausentes
3. Registros duplicados
4. Consistência dos tipos de dados
5. Valores inconsistentes
6. *Outliers*
7. Consistência das variáveis categóricas (bairro e rua)
8. Indicadores de qualidade
9. Visualizações
10. Diagnóstico final
""")

md("""
## A base analisada

A base é o CSV gerado pelo **nosso *scraper*** na etapa 1 (notebook `Predicao_Precos_Imoveis_Vinicius_Machado.ipynb`, seção 3): os anúncios próprios do QuintoAndar à venda no município do Rio de Janeiro, com **preço de até R$ 1.000.000** (o limite definido na coleta), coletados em 18/10/2026. Os valores foram gravados **exatamente como o site os entrega**, sem nenhuma correção. É essa base "crua" que auditamos aqui.

As seis variáveis do enunciado e as colunas correspondentes:

| Variável do enunciado | Coluna | Origem no site |
|---|---|---|
| Preço | `preco` | `salePrice` |
| Valor Condomínio | `condominio` | `condominium` |
| Valor IPTU | `iptu` | `iptu` |
| Metragem quadrada | `area_m2` | `area` |
| Bairro | `bairro` | `neighbourhood` |
| Rua | `rua` | `address` |

A base também tem `condominio_iptu` (a soma que o próprio site calcula), o tipo do imóvel, cômodos, vagas, indicadores de lazer, a cidade e a `regiao_qa`, um agrupamento de bairros usado pelo QuintoAndar.
""")

# =====================================================================
# 0. Ambiente
# =====================================================================
md("""
## 0. Ambiente e leitura da base

O CSV precisa estar na mesma pasta do notebook. No Google Colab, basta fazer o *upload* do arquivo pelo painel **Arquivos** (ícone de pasta, à esquerda) antes de executar.

A base é lida **duas vezes**:

- `df_raw`: leitura normal, em que o `pandas` deduz o tipo de cada coluna;
- `df_texto`: leitura com **todas as colunas como texto** e sem converter nada em ausente (`dtype=str`, `keep_default_na=False`). Ela mostra os valores exatamente como estão no arquivo, inclusive células vazias e espaços extras.

Nenhuma das duas é modificada. Os tratamentos são feitos em uma cópia (`df_tratado`, seção 8), para que sempre dê para comparar o antes e o depois.
""")

code("""
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 170)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
sns.set_theme(style="whitegrid")

ARQUIVO = "quintoandar_rio_scraping_20261018.csv"
assert os.path.exists(ARQUIVO), f"Faça o upload de {ARQUIVO} (gerado pela etapa 1) para a pasta do notebook."

df_raw = pd.read_csv(ARQUIVO, encoding="utf-8")
df_texto = pd.read_csv(ARQUIVO, encoding="utf-8", dtype=str, keep_default_na=False)

PRECO, CONDOMINIO, IPTU, AREA, BAIRRO, RUA = "preco", "condominio", "iptu", "area_m2", "bairro", "rua"
NUMERICAS = [PRECO, CONDOMINIO, IPTU, AREA]
LIMITE_PRECO_COLETA = 1_000_000

print(f"pandas {pd.__version__} | numpy {np.__version__}")
""")

# =====================================================================
# 1. Inspeção inicial
# =====================================================================
md("""
# 1. Inspeção inicial da base

## 1.1 Dimensões, variáveis e tipos
""")

code("""
print(f"Linhas : {df_raw.shape[0]:,}")
print(f"Colunas: {df_raw.shape[1]}\\n")

pd.DataFrame({
    "tipo identificado pelo Python": df_raw.dtypes.astype(str),
    "exemplo (1ª linha)": df_raw.iloc[0],
})
""")

md("""
Preço, condomínio, IPTU e metragem foram lidos como inteiros (`int64`), e os textos como `str`. A exceção é `suites`, lida como `float64`: isso acontece quando uma coluna numérica tem algum valor ausente, porque uma coluna inteira comum do `pandas` não aceita ausentes. É o primeiro indício de uma célula vazia, confirmado na seção 2.

## 1.2 Cinco primeiras e cinco últimas observações
""")

code("""
df_raw.head()
""")

code("""
df_raw.tail()
""")

md("""
As linhas estão completas, mas já aparecem sinais de problema:

- a base está **aproximadamente ordenada por preço**, do mais barato ao mais caro, porque o *scraper* percorreu a busca faixa de preço por faixa de preço. Por isso `head()` e `tail()` não são amostras representativas, e qualquer amostragem deve ser aleatória;
- há imóveis com **`condominio` igual a −1**, um valor impossível para um custo;
- a coluna `condominio_iptu` não é sempre a soma de `condominio` com `iptu`: quando um deles vale −1, o site o trata como zero na soma.

## 1.3 Estatísticas descritivas das variáveis numéricas
""")

code("""
df_raw.describe().T
""")

md("""
**Problemas já perceptíveis nessa primeira inspeção:**

1. **`condominio` e `iptu` têm mínimo igual a −1.** Um custo negativo é impossível: −1 é um **código**, não um valor (seção 2).
2. **Os máximos de condomínio e de IPTU são absurdos:** centenas de milhares de reais **por mês** em imóveis de no máximo R$ 1 milhão.
3. **`area_m2` tem mínimo igual a 0**, o que é impossível, e máximo de alguns milhares de m², o que precisa ser checado.
4. **`preco` vai de R$ 125 mil a exatamente R$ 1.000.000**, respeitando o limite da coleta. A média é maior que a mediana: a distribuição é assimétrica, com mais imóveis baratos que caros.
5. A **mediana do IPTU é baixa** e o primeiro quartil é zero. Há muitos IPTUs zerados, o que pode ser isenção ou falta de informação.
""")

# =====================================================================
# 2. Ausentes
# =====================================================================
md("""
# 2. Dados ausentes

## 2.1 Ausentes declarados (células vazias)
""")

code("""
ausentes = pd.DataFrame({
    "ausentes": df_raw.isna().sum(),
    "% ausentes": (df_raw.isna().mean() * 100).round(3),
}).sort_values("ausentes", ascending=False)
ausentes
""")

code("""
colunas_ver = ["tipo", RUA, BAIRRO, "regiao_qa", "cidade", PRECO, AREA]
df_raw.loc[df_raw[[BAIRRO, "cidade", "suites"]].isna().any(axis=1), colunas_ver]
""")

md("""
Pelas células vazias, a base seria quase completa: das 22 colunas, só `bairro`, `cidade`, `regiao_qa` e `suites` têm ausentes, em pouquíssimas linhas. As demais, inclusive preço, condomínio, IPTU, metragem e rua, não têm nenhuma célula vazia. Nos anúncios sem cidade, o bairro e a rua mostram que o imóvel fica no Rio: a cidade ficou vazia, mas a localização não se perdeu.

## 2.2 Ausentes disfarçados

Um ausente nem sempre aparece como célula vazia. É comum o site ou o *scraper* preencherem a falta de informação com um código. Aqui, o código é **−1**:
""")

code("""
codigos = pd.DataFrame({
    "condomínio = -1": df_raw[CONDOMINIO].eq(-1).groupby(df_raw["tipo"]).sum(),
    "IPTU = -1": df_raw[IPTU].eq(-1).groupby(df_raw["tipo"]).sum(),
    "IPTU = 0": df_raw[IPTU].eq(0).groupby(df_raw["tipo"]).sum(),
    "anúncios": df_raw["tipo"].value_counts(),
})
codigos.loc["Total"] = codigos.sum()
codigos
""")

md("""
O cruzamento com o tipo de imóvel muda a interpretação do −1 no **condomínio**:

- **em casas de rua (`Casa`), o −1 é a regra**, e não a exceção. Casas de rua normalmente **não têm condomínio**: o −1 ali quer dizer "não se aplica". Isso é confirmado pelo fato de o **condomínio igual a zero quase não existir na base** (só 2 anúncios): o site usa −1, e não 0, para "sem condomínio";
- **em apartamentos, studios e casas em condomínio, o −1 é um ausente de verdade**: esses imóveis sempre pagam condomínio, e o valor simplesmente não foi informado.

No **IPTU**, o −1 aparece em todos os tipos e indica que o valor não foi informado. O **zero** é ambíguo: o Rio de Janeiro isenta de IPTU os imóveis residenciais de menor valor, então muitos zeros são reais, mas parte deles pode ser apenas um campo deixado em branco. Não há como distinguir os dois casos, e por isso o zero é mantido.

## 2.3 Completude das variáveis do enunciado
""")

code("""
nao_casa = df_raw["tipo"] != "Casa"
completude_enunciado = pd.DataFrame({
    "ausentes declarados": [df_raw[c].isna().sum() for c in [PRECO, CONDOMINIO, IPTU, AREA, BAIRRO, RUA]],
    "ausentes disfarçados": [
        0,
        (nao_casa & df_raw[CONDOMINIO].eq(-1)).sum(),  # -1 em casa de rua = sem condomínio, não é ausente
        df_raw[IPTU].eq(-1).sum(),
        df_raw[AREA].eq(0).sum(),
        0,
        0,
    ],
}, index=["Preço", "Condomínio", "IPTU", "Metragem", "Bairro", "Rua"])
completude_enunciado["% ausente total"] = (completude_enunciado.sum(axis=1) / len(df_raw) * 100).round(2)
completude_enunciado
""")

md("""
**Variáveis com maior problema de completude:**

1. **IPTU**: o −1 aparece em mais de 5% dos anúncios, sem contar os zeros ambíguos.
2. **Condomínio**: cerca de 5% dos apartamentos e casas em condomínio não informam o valor.
3. **Metragem, bairro e rua**: problemas pontuais.
4. **Preço**: 100% preenchido. É o campo principal do anúncio, e não há anúncio sem ele.

**Possíveis razões para a ausência no site:**

- **Condomínio e IPTU** são informados pelo proprietário no cadastro e muitas vezes não são obrigatórios. O proprietário pode não saber o valor do IPTU de cabeça, ou o imóvel pode estar com o carnê atrasado. O site grava a falta como −1 e mostra o anúncio sem esses custos.
- **IPTU zero:** isenção legal para imóveis de menor valor venal ou, simplesmente, um campo deixado em branco.
- **Cidade e região vazias:** ambas são derivadas da localização. Quando o endereço foi cadastrado de forma incompleta, a plataforma não consegue preenchê-las.
- **Suítes:** campo opcional, não preenchido em um anúncio.
""")

# =====================================================================
# 3. Duplicados
# =====================================================================
md("""
# 3. Registros duplicados

## 3.1 Linhas completamente duplicadas
""")

code("""
COLUNAS_IMOVEL = [c for c in df_raw.columns if c not in ["id", "url"]]
print(f"Linhas completamente duplicadas (todas as colunas)      : {df_raw.duplicated().sum()}")
print(f"Códigos de anúncio (id) repetidos                       : {df_raw['id'].duplicated().sum()}")
print(f"Linhas idênticas em todas as colunas, exceto id e url   : {df_raw.duplicated(subset=COLUNAS_IMOVEL).sum()}")
""")

md("""
Não há linhas repetidas nem códigos de anúncio repetidos: o *scraper* não coletou o mesmo anúncio duas vezes, como já tinha sido verificado na etapa 1.

Mas **há anúncios diferentes (com códigos diferentes) idênticos em todas as 20 características**: mesma rua, bairro, preço, área, condomínio, IPTU, cômodos e itens de lazer.

## 3.2 Imóveis potencialmente duplicados

Seguindo a sugestão do enunciado, combinamos rua, bairro, preço e metragem e depois acrescentamos outras características para tornar o critério mais rigoroso:
""")

code("""
COMBINACOES = {
    "rua + bairro + preço + metragem": [RUA, BAIRRO, PRECO, AREA],
    "+ quartos + condomínio + IPTU": [RUA, BAIRRO, PRECO, AREA, "quartos", CONDOMINIO, IPTU],
    "todas as colunas, exceto id e url": COLUNAS_IMOVEL,
}
pd.DataFrame([{
    "critério": nome,
    "registros em grupos repetidos": df_raw.duplicated(subset=colunas, keep=False).sum(),
    "cópias excedentes": df_raw.duplicated(subset=colunas).sum(),
} for nome, colunas in COMBINACOES.items()])
""")

code("""
candidatos = df_raw[df_raw.duplicated(subset=COLUNAS_IMOVEL, keep=False)].sort_values([RUA, PRECO, "id"])
candidatos[["id", "tipo", RUA, BAIRRO, PRECO, AREA, CONDOMINIO, IPTU, "quartos"]].head(10)
""")

md("""
Os grupos idênticos em tudo costumam ter **códigos de anúncio muito próximos**, o que indica anúncios publicados em sequência pelo mesmo anunciante. Há duas explicações possíveis:

1. **o mesmo imóvel anunciado duas vezes** (por engano ou para ganhar visibilidade), uma duplicata de verdade;
2. **unidades diferentes do mesmo prédio**, com a mesma planta e o mesmo preço, comuns em empreendimentos novos. Como a rua vem, na grande maioria dos anúncios, **sem o número** e sem o apartamento, não há como diferenciá-las.

**Devem ser removidos?**

- **Rua + bairro + preço + metragem é um critério fraco**: no mesmo prédio é normal haver vários apartamentos com a mesma planta e o mesmo preço. Remover esses registros apagaria oferta real.
- **Os registros idênticos em todas as características são duplicatas prováveis**, mas não certas, pela razão 2. O impacto de mantê-los é pequeno (menos de 1% da base).

**Decisão:** não remover automaticamente e **sinalizar** os registros (coluna `possivel_duplicata` na base tratada, seção 8). Uma análise que conte **imóveis**, e não anúncios, deve manter só uma cópia de cada grupo.
""")

# =====================================================================
# 4. Tipos
# =====================================================================
md("""
# 4. Consistência dos tipos de dados

## 4.1 As variáveis numéricas estão armazenadas como números?
""")

code("""
pd.DataFrame({
    "variável": ["Preço", "Condomínio", "IPTU", "Metragem"],
    "coluna": NUMERICAS,
    "tipo": [str(df_raw[c].dtype) for c in NUMERICAS],
    "é numérica?": [pd.api.types.is_numeric_dtype(df_raw[c]) for c in NUMERICAS],
})
""")

md("""
As quatro colunas foram lidas como numéricas. Se uma única célula tivesse "R$ 500.000", o `pandas` leria a coluna inteira como texto, e isso não aconteceu. Mesmo assim, conferimos o texto bruto (`df_texto`) para ter certeza de que não há símbolos, separadores ou unidades:
""")

code("""
PADRAO_SUJEIRA = r"[^0-9\\-]"  # qualquer caractere que não seja dígito ou sinal de menos

for coluna in NUMERICAS:
    sujos = df_texto[coluna].str.contains(PADRAO_SUJEIRA)
    print(f"{coluna:12s}: {sujos.sum()} valores com R$, pontos, vírgulas, unidades ou outros símbolos")

print("\\nAmostra do texto bruto, como está no arquivo:")
df_texto[NUMERICAS].sample(5, random_state=42)
""")

md("""
**Nenhum valor tem símbolos.** O motivo está na forma da coleta: o *scraper* lê os dados da API do site em formato JSON, em que os números já vêm como números, e não do texto exibido na tela ("R$ 410.000", "75 m²"). Essa é uma vantagem de coletar pela API em vez de raspar o HTML.

## 4.2 Função de limpeza para os formatos típicos de *scraping*

Se a coleta fosse feita a partir do texto da página, os valores viriam como aparecem para o usuário. Por isso escrevemos uma função de limpeza, mostramos o antes e o depois em exemplos com os formatos típicos e, em seguida, a aplicamos à base real para provar que ela não altera valores que já estão corretos.

As regras são estas:

1. remover `R$`, `m²`/`m2`, espaços e textos como `/mês`;
2. no padrão brasileiro, **o ponto é separador de milhar** e a **vírgula é a casa decimal** ("1.250,50" = 1250.5);
3. valores anuais (`/ano`, comum no IPTU) são **divididos por 12**, para ficarem comparáveis ao condomínio mensal;
4. textos sem número ("Consulte", "Não informado") viram ausentes, em vez de quebrar a conversão.
""")

code("""
def limpar_numero(valor):
    \"\"\"Converte textos como 'R$ 1.250.000', 'R$ 850/mês', 'R$ 3.600/ano' ou '75 m²' em número (anual -> mensal).\"\"\"
    if pd.isna(valor):
        return np.nan
    texto = str(valor).lower().strip()
    anual = "/ano" in texto
    texto = re.sub(r"r\\$|m²|m2|/m[eê]s|/ano|\\s", "", texto)
    texto = texto.replace(".", "").replace(",", ".")  # padrão brasileiro: ponto = milhar, vírgula = decimal
    texto = re.sub(r"[^0-9.\\-]", "", texto)
    if texto in ("", "-", "."):
        return np.nan
    numero = float(texto)
    return numero / 12 if anual else numero


exemplos = pd.Series(["R$ 1.250.000", "R$ 850/mês", "R$ 1.234,56", "R$ 3.600/ano", "75 m²",
                      " 120m2 ", "Consulte", None, "-1"])
pd.DataFrame({"antes (texto do site)": exemplos, "depois (número)": exemplos.map(limpar_numero)})
""")

code("""
# Aplicação na base real: o resultado tem de ser idêntico ao que o pandas leu
for coluna in NUMERICAS:
    convertido = df_texto[coluna].map(limpar_numero)
    assert (convertido == df_raw[coluna]).all(), coluna
    print(f"{coluna:12s}: antes {df_texto[coluna].dtype!s:4s} {df_texto[coluna].iloc[0]!r:>10} -> "
          f"depois {convertido.dtype} {convertido.iloc[0]:>10,.0f}  (os {len(convertido):,} valores conferem)")
""")

md("""
**Conclusão sobre os tipos:** as quatro variáveis numéricas estão armazenadas corretamente, e a função de limpeza está pronta (e validada) para uma coleta baseada em texto. O problema desta base não é de **formato**, e sim de **conteúdo**: números bem formatados, mas impossíveis ou codificados (−1, área zero, IPTU de R$ 500 mil por mês), analisados a seguir.

O tipo da coluna `rua` também esconde um problema de formato: ela deveria conter **só o nome do logradouro**, mas em parte dos registros traz número, bairro, CEP e até "Brasil" (seção 7.2).
""")

# =====================================================================
# 5. Inconsistentes
# =====================================================================
md("""
# 5. Identificação de valores inconsistentes

Nesta seção os casos suspeitos são **apenas identificados e apresentados**. Nada é removido. Cada regra vira uma coluna booleana (`True` = suspeito), o que permite contar, cruzar e decidir o tratamento caso a caso.

## 5.1 Regras de verificação

| Regra | Por quê |
|---|---|
| Preço ≤ 0 | Preço precisa ser positivo |
| Metragem ≤ 0 | Área precisa ser positiva |
| Metragem entre 1 e 9 m² | Nem uma quitinete tem menos de 10 m² |
| Condomínio negativo (−1) fora de casas de rua | Código de "não informado" |
| Condomínio entre R$ 1 e R$ 9 por mês | Valor simbólico |
| Condomínio mensal > 1% do preço | Significaria pagar mais de 12% do valor do imóvel por ano só de condomínio |
| IPTU negativo (−1) | Código de "não informado" |
| IPTU anual > 2% do preço | A alíquota residencial do IPTU no Rio fica em torno de 1% do valor venal, que costuma ser **menor** que o preço de mercado. Mais de 2% ao ano é incompatível |
| Preço > R$ 1.000.000 | Fora do limite de preço definido na coleta |
| Sem bairro, sem rua ou sem cidade | Localização incompleta |
""")

code("""
razao_cond = df_raw[CONDOMINIO] / df_raw[PRECO]
razao_iptu_anual = df_raw[IPTU] * 12 / df_raw[PRECO]

suspeitos = pd.DataFrame({
    "preço ≤ 0": df_raw[PRECO] <= 0,
    "metragem ≤ 0": df_raw[AREA] <= 0,
    "metragem entre 1 e 9 m²": df_raw[AREA].between(1, 9),
    "condomínio -1 (exceto casa de rua)": nao_casa & df_raw[CONDOMINIO].eq(-1),
    "condomínio simbólico (R$ 1 a 9)": df_raw[CONDOMINIO].between(1, 9),
    "condomínio > 1% do preço ao mês": razao_cond > 0.01,
    "IPTU -1": df_raw[IPTU].eq(-1),
    "IPTU anual > 2% do preço": razao_iptu_anual > 0.02,
    "preço > R$ 1 milhão": df_raw[PRECO] > LIMITE_PRECO_COLETA,
    "sem bairro": df_raw[BAIRRO].isna(),
    "sem rua": df_raw[RUA].isna(),
    "sem cidade ou cidade diferente do Rio": df_raw["cidade"].ne("Rio de Janeiro"),
})

print(f"Mediana do condomínio / preço (valores > 0): {razao_cond[df_raw[CONDOMINIO] > 0].median():.3%} ao mês")
print(f"Mediana do IPTU anual / preço (valores > 0): {razao_iptu_anual[df_raw[IPTU] > 0].median():.3%} ao ano")
pd.DataFrame({"casos": suspeitos.sum(), "% da base": (suspeitos.mean() * 100).round(2)})
""")

md("""
## 5.2 Preço e metragem iguais ou inferiores a zero

Nenhum preço é zero ou negativo. Já a **metragem tem casos de 0 m² e de 1 a 9 m²**:
""")

code("""
colunas_caso = ["tipo", RUA, BAIRRO, PRECO, AREA, "quartos", "banheiros", CONDOMINIO, IPTU]
df_raw.loc[df_raw[AREA] < 10, colunas_caso].sort_values(AREA)
""")

md("""
São apartamentos e casas com quartos e banheiros, e não há imóvel habitável com 0, 1 ou 2 m². É um **erro de cadastro**: o anunciante deixou o campo em branco (zero) ou digitou um número incompleto. As quitinetes de 12 a 14 m² ficam fora dessa regra, porque são raras, mas possíveis.

## 5.3 Condomínio incompatível com o preço
""")

code("""
df_raw.loc[suspeitos["condomínio > 1% do preço ao mês"], colunas_caso] \\
    .assign(**{"condomínio / preço": razao_cond}) \\
    .sort_values(CONDOMINIO, ascending=False).head(10)
""")

md("""
No topo, imóveis de algumas centenas de milhares de reais com **condomínio maior que o próprio preço do imóvel**. São erros de cadastro: o valor provavelmente foi digitado em centavos (sem a vírgula), somado ao ano inteiro ou confundido com outro campo.

Os **condomínios simbólicos** (R$ 1 a R$ 9) são outro tipo de erro: nenhum condomínio real custa isso. São, quase certamente, um "não informado" digitado com qualquer valor para passar pela validação do formulário.

## 5.4 IPTU muito elevado ou muito baixo
""")

code("""
print("IPTU anual acima de 2% do preço:")
df_raw.loc[suspeitos["IPTU anual > 2% do preço"], colunas_caso] \\
    .assign(**{"IPTU anual / preço": razao_iptu_anual}) \\
    .sort_values(IPTU, ascending=False).head(10)
""")

code("""
print("Distribuição dos IPTUs mais baixos (R$/mês):")
print(df_raw.loc[df_raw[IPTU].between(-1, 9), IPTU].value_counts().sort_index().to_string())
""")

md("""
- **Muito elevados:** IPTUs de dezenas ou centenas de milhares de reais **por mês** são impossíveis para imóveis de até R$ 1 milhão. O mais provável é que o valor tenha sido digitado em centavos ou que o anunciante tenha informado o valor venal, ou o preço, no campo do IPTU.
- **Muito baixos:** além do −1 e do zero (seção 2), há centenas de IPTUs entre R$ 1 e R$ 9 por mês. Ao contrário do condomínio, eles **podem ser reais**: imóveis de baixo valor venal pagam IPTUs anuais pequenos, e R$ 5 por mês equivale a R$ 60 por ano. O pico em **exatamente R$ 1** é mais suspeito e parece outro "não informado" disfarçado, mas não há como provar. Por isso esses valores são **mantidos** e não entram como erro.

## 5.5 Imóveis fora do limite de preço definido na coleta
""")

code("""
print(f"Menor preço: R$ {df_raw[PRECO].min():,.0f}")
print(f"Maior preço: R$ {df_raw[PRECO].max():,.0f}")
print(f"Anúncios acima de R$ 1 milhão: {suspeitos['preço > R$ 1 milhão'].sum()}")
print(f"Anúncios exatamente no limite: {(df_raw[PRECO] == LIMITE_PRECO_COLETA).sum()}")
""")

md("""
**Nenhum anúncio está acima do limite**, e há anúncios exatamente em R$ 1.000.000, o que mostra que o filtro da coleta é inclusivo e foi respeitado. Esse resultado confirma que o filtro de preço enviado à API (seção 3 da etapa 1) funcionou.

## 5.6 Concentração suspeita de preços em exatamente R$ 200 mil

Na ponta de baixo da faixa de preços aparece um padrão estranho, também visível no histograma da seção 9.3:
""")

code("""
PRECO_PISO = 200_000
no_piso = df_raw[PRECO] == PRECO_PISO

print("Os preços mais frequentes da base:")
print(df_raw[PRECO].value_counts().head(5).to_string())
print(f"\\nAnúncios a exatamente R$ 200 mil: {no_piso.sum():,} ({no_piso.mean():.1%})")
print(f"Anúncios abaixo de R$ 200 mil   : {(df_raw[PRECO] < PRECO_PISO).sum()}")

vizinhos = df_raw[PRECO].between(180_000, 230_000) & ~no_piso
pd.DataFrame({
    "exatamente R$ 200 mil": df_raw.loc[no_piso, [AREA, "quartos", CONDOMINIO, IPTU]].median(),
    "R$ 180 a 230 mil (exceto 200)": df_raw.loc[vizinhos, [AREA, "quartos", CONDOMINIO, IPTU]].median(),
}).rename_axis("mediana de")
""")

md("""
**Mais de 10% dos anúncios custam exatamente R$ 200 mil**, várias vezes mais que o segundo preço mais frequente, e quase não há anúncios abaixo desse valor. O perfil desses imóveis é parecido com o dos vizinhos de preço: mesmo número de quartos, com área e condomínio um pouco menores, o que é coerente com imóveis que valeriam um pouco menos.

A explicação mais provável é um **piso de preço da plataforma**: o QuintoAndar não aceita (ou não exibe) preços abaixo de R$ 200 mil nos anúncios próprios, e imóveis mais baratos acabam anunciados exatamente no piso. A base de referência do professor, coletada em outra data, mostra o mesmo fenômeno. Cada valor isolado é plausível, então não há um erro detectável linha a linha. Mas o efeito agregado é uma **distorção da cauda inferior**: parte desses imóveis provavelmente vale menos de R$ 200 mil, e uma análise do segmento mais barato do mercado ficaria enviesada.

**Tratamento:** manter os anúncios e sinalizá-los (coluna `preco_no_piso` na base tratada).

## 5.7 Registros sem bairro ou endereço
""")

code("""
df_raw.loc[suspeitos[["sem bairro", "sem rua", "sem cidade ou cidade diferente do Rio"]].any(axis=1),
           ["id", RUA, BAIRRO, "regiao_qa", "cidade", PRECO]]
""")

md("""
Todos os anúncios têm rua, e só um não tem bairro. Os anúncios **sem cidade** têm rua e bairro do Rio, assim como o anúncio com a cidade **"Lins"**, que é na verdade um bairro (Lins de Vasconcelos) digitado no campo de cidade. A localização desses registros pode ser recuperada pela rua e pelo bairro.

## 5.8 Visão geral dos suspeitos
""")

code("""
n_problemas = suspeitos.sum(axis=1)
print(f"Anúncios com ao menos um problema: {(n_problemas > 0).sum():,} ({(n_problemas > 0).mean():.2%})")
print(f"Anúncios com mais de um problema : {(n_problemas > 1).sum()}")
""")

md("""
Cerca de um em cada dez anúncios tem algum valor inconsistente, mas quase sempre em **um único campo**, e na grande maioria das vezes em condomínio ou IPTU (custos opcionais no cadastro). Preço, metragem e localização, os campos centrais, quase não têm problemas. Esse padrão favorece **corrigir o campo** em vez de descartar o anúncio inteiro.
""")

# =====================================================================
# 6. Outliers
# =====================================================================
md("""
# 6. Análise de *outliers*

## 6.1 Técnicas utilizadas

Usamos três técnicas objetivas e comparamos os resultados:

1. **IQR na escala original:** é *outlier* o valor abaixo de Q1 − 1,5·IQR ou acima de Q3 + 1,5·IQR, em que IQR = Q3 − Q1. É a mesma regra que o boxplot usa para desenhar os pontos isolados.
2. **IQR na escala logarítmica:** a mesma regra aplicada a `log(x)`. Áreas, condomínios e IPTUs são muito assimétricos, e na escala original o IQR marca como *outlier* muitos valores apenas altos. No log, a distribuição fica mais simétrica, e a regra marca só o que é extremo de verdade.
3. **Percentis 1% e 99%:** um corte simples, que mostra o que está nas pontas.

Condomínio e IPTU são analisados só nos valores **positivos**: o −1 e o zero já foram tratados nas seções 2 e 5, e o log não existe para zero ou negativos. Pelo mesmo motivo, a metragem é analisada só acima de zero.
""")

code("""
def limites_iqr(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


VARIAVEIS_OUTLIER = {"Preço": PRECO, "Condomínio (> 0)": CONDOMINIO, "IPTU (> 0)": IPTU, "Metragem (> 0)": AREA}

linhas, outlier_log = [], {}
for nome, coluna in VARIAVEIS_OUTLIER.items():
    serie = df_raw.loc[df_raw[coluna] > 0, coluna]
    inf, sup = limites_iqr(serie)
    inf_log, sup_log = np.exp(limites_iqr(np.log(serie)))
    p1, p99 = serie.quantile([0.01, 0.99])
    outlier_log[coluna] = (serie < inf_log) | (serie > sup_log)
    linhas.append({
        "variável": nome,
        "IQR: limites": f"{max(inf, 0):,.0f} a {sup:,.0f}",
        "IQR: outliers": ((serie < inf) | (serie > sup)).sum(),
        "IQR log: limites": f"{inf_log:,.0f} a {sup_log:,.0f}",
        "IQR log: outliers": outlier_log[coluna].sum(),
        "P1 a P99": f"{p1:,.0f} a {p99:,.0f}",
    })
pd.DataFrame(linhas).set_index("variável")
""")

md("""
**Leitura do resultado:**

- **Preço: nenhum *outlier*** por nenhuma das técnicas. Como a coleta foi limitada a R$ 1 milhão (e, na prática, começa no piso de R$ 200 mil), não sobra cauda longa: o próprio desenho da coleta "cortou" os extremos.
- **Metragem:** o IQR na escala original marca mais de 1.600 anúncios, quase todos imóveis apenas maiores que a média (acima de 208 m²). Na escala log, sobram cerca de 500, entre áreas muito pequenas e muito grandes.
- **Condomínio e IPTU:** na escala log, o IQR marca **mais** anúncios que na original, porque passa a enxergar também a ponta de **baixo**: os condomínios abaixo de cerca de R$ 120 e os IPTUs abaixo de R$ 12 por mês, valores pequenos demais. As duas pontas concentram os problemas já vistos na seção 5.
- Os **percentis** dão a faixa onde está o "miolo" da base, e as pontas (acima do P99) concentram os valores impossíveis já vistos na seção 5.

## 6.2 Registros extremos e a sua natureza

Para cada variável, olhamos os valores das pontas e o **preço por m²**, que ajuda a separar um imóvel extremo legítimo (área grande com preço compatível) de um erro (área e preço incompatíveis).
""")

code("""
COLUNAS_CASO = ["tipo", BAIRRO, PRECO, AREA, "preco_m2", "quartos", "banheiros", CONDOMINIO, IPTU]
df_m2 = df_raw.assign(preco_m2=(df_raw[PRECO] / df_raw[AREA].where(df_raw[AREA] > 0)).round(0))

print("METRAGEM: as 10 maiores áreas")
df_m2.nlargest(10, AREA)[COLUNAS_CASO]
""")

md("""
Duas situações diferentes:

- **Casas com áreas muito grandes** em bairros de terrenos grandes, como Vargem Grande e Ilha de Guaratiba, são plausíveis. Muitas vezes o anunciante informa a **área do terreno** em vez da área construída. São imóveis legítimos, mas com uma ambiguidade de cadastro. Já casas de 1.250 a 1.600 m² em bairros urbanos e densos, como Engenho Novo e Engenho de Dentro, são mais suspeitas.
- **Áreas de milhares de m² com preço por m² de poucas centenas de reais em apartamentos** são **erros de cadastro**: o apartamento de 2.164 m² em Copacabana por R$ 850 mil (R$ 393/m², quando o m² do bairro passa de R$ 10 mil) é um zero a mais ou a área do prédio inteiro. Mesmo entre as casas, um m² de R$ 250 a R$ 700 é muito baixo para o Rio, o que reforça a hipótese de que se trata da área do terreno.
""")

code("""
print("CONDOMÍNIO: os 8 maiores valores")
df_m2.nlargest(8, CONDOMINIO)[COLUNAS_CASO]
""")

code("""
print("IPTU: os 8 maiores valores")
df_m2.nlargest(8, IPTU)[COLUNAS_CASO]
""")

md("""
**Condomínio e IPTU extremos são erros de cadastro, sem dúvida.** São valores mensais da ordem de grandeza do próprio preço do imóvel, impossíveis em qualquer cenário.
""")

code("""
print("PREÇO: os 5 mais caros e os 5 mais baratos")
pd.concat([df_m2.nlargest(5, PRECO), df_m2.nsmallest(5, PRECO)])[COLUNAS_CASO]
""")

md("""
**Os preços das pontas são legítimos:** no topo, imóveis de R$ 1 milhão com áreas e bairros compatíveis; na base, apartamentos pequenos na Zona Norte (Irajá, Vaz Lobo, Piedade), de R$ 125 mil a R$ 135 mil.

## 6.3 Classificação dos extremos

| Variável | Tipo de extremo | Diagnóstico | Tratamento proposto |
|---|---|---|---|
| Preço | Imóveis próximos de R$ 1 milhão ou de R$ 125 mil | **Legítimos**: o teto da coleta impede preços extremos | Manter |
| Metragem | Casas grandes em bairros de terrenos grandes | **Legítimos**, com possível área de terreno em vez de área útil | Manter e sinalizar |
| Metragem | Áreas enormes com m² irrisório, áreas de 0 a 9 m² | **Erro de cadastro** | Sinalizar. Área < 10 m² vira ausente |
| Condomínio | Acima de 1% do preço ao mês | **Erro de cadastro** | Transformar em ausente, mantendo o anúncio |
| IPTU | Acima de 2% do preço ao ano | **Erro de cadastro** | Transformar em ausente, mantendo o anúncio |
""")

# =====================================================================
# 7. Categóricas
# =====================================================================
md("""
# 7. Consistência das variáveis categóricas

## 7.1 Bairro

### Diferenças de escrita, espaços extras e maiúsculas/minúsculas
""")

code("""
def normalizar(texto):
    \"\"\"Chave de comparação: sem espaços extras, sem acentos e em minúsculas.\"\"\"
    texto = re.sub(r"\\s+", " ", str(texto).strip().lower())
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


bairros = df_raw[BAIRRO].dropna()
print(f"Bairros distintos (como estão)                    : {bairros.nunique()}")
print(f"Com espaços no início ou no fim                   : {(bairros != bairros.str.strip()).sum()} anúncios")
print(f"Com espaços duplos no meio                        : {bairros.str.contains('  ').sum()} anúncios")
print(f"Distintos após strip()                            : {bairros.str.strip().nunique()}")
print(f"Distintos após strip() + minúsculas               : {bairros.str.strip().str.lower().nunique()}")
print(f"Distintos após strip() + minúsculas + sem acentos : {bairros.map(normalizar).nunique()}")
""")

md("""
O número de bairros distintos **cai a cada etapa de normalização**. Cada queda é um conjunto de grafias diferentes para o mesmo bairro. Os grupos abaixo mostram todas as formas encontradas para cada bairro (o texto entre aspas deixa visíveis os espaços extras):
""")

code("""
grupos = bairros.groupby(bairros.map(normalizar)).unique()
grupos = grupos[grupos.str.len() > 1]
print(f"{len(grupos)} bairros escritos de mais de uma forma:\\n")
for formas in grupos:
    print(" | ".join(f"'{f}'" for f in formas))
""")

md("""
Aparecem os três problemas do enunciado:

- **espaços extras** no início ou no fim ("Botafogo ", " São Cristóvão");
- **diferenças de maiúsculas e minúsculas** ("barra olimpica", "Todos Os Santos", "Freguesia (jacarepaguá)");
- **acentos ausentes** ("Meier", "Jacarepagua", "Grajau", "Praca da Bandeira").

O caso mais extremo é **Barra Olímpica, escrito de cinco formas**. Para um computador, cada uma é um bairro diferente: uma análise de preço por bairro dividiria os anúncios de Barra Olímpica em cinco grupos.

### Valores aparentemente incorretos

Os bairros raros, que aparecem em até 3 anúncios depois da normalização acima, concentram os erros que não são só de espaço, caixa ou acento:
""")

code("""
chave = df_raw[BAIRRO].map(normalizar, na_action="ignore")
forma_mais_comum = df_raw[BAIRRO].str.strip().groupby(chave).agg(lambda s: s.value_counts().index[0])
contagem = chave.map(forma_mais_comum).value_counts()
print(sorted(contagem[contagem <= 3].index))
""")

md("""
| Problema | Exemplos |
|---|---|
| **Abreviação** | "R. Bandeirantes", "Rcr Bandeirantes", "Rec dos Bandeirante", "I. Governador", "L Vasconcelos", "V Carvalho", "São Fco Xavier" |
| **Erro de digitação** | "Bota Fogo", "Vila Izabel", "Recreio Bandeirante" |
| **Lixo grudado no nome** | "Jacarepaguá1", "Campo Grande NUMERO 270", "Colégio Rj", "Tomas Coelho Rj" |
| **Bairro + complemento** | "Taquara - Jacarepaguá", "Pechincha (jacarepaguá)", "Tijuca - Praça Saes Peña", "Jardim Letícia - Campo Grande" |
| **Grafias alternativas** | "Recreio" e "Recreio dos Bandeirantes"; "Freguesia", "Freguesia de Jacarepaguá" e "Freguesia (Jacarepaguá)"; "Osvaldo Cruz" e "Oswaldo Cruz"; "Braz de Pina" e "Brás de Pina" |
| **Tudo em maiúsculas** | "PACIÊNCIA" |
| **Valor inválido** | "800" |
| **Localidade não reconhecida** | "Argentino", "Ciro Aranha", "Condomínio Capim", "Posse", "Vila Centenário": podem ser sub-bairros, condomínios ou localidades de outro município |

### Proposta de padronização

1. **Limpeza mecânica:** remover espaços extras e escolher, para cada grupo de grafias equivalentes (mesma chave sem acento e em minúsculas), a **forma mais frequente**, que costuma ser a correta.
2. **Dicionário de correção** para os casos que a limpeza mecânica não resolve (abreviações, digitação, complementos, grafias alternativas).
3. **Valores inválidos** (só números) passam a ser **ausentes**.
4. **Localidades não reconhecidas** são mantidas como estão, porque corrigi-las exigiria adivinhar.
""")

code("""
CORRECAO_BAIRRO = {
    "Freguesia (jacarepaguá)": "Freguesia (Jacarepaguá)",
    "Freguesia de Jacarepaguá": "Freguesia (Jacarepaguá)",
    "Freguesia": "Freguesia (Jacarepaguá)",
    "Freguesia (jacarepaguá)pechincha": "Freguesia (Jacarepaguá)",
    "Freguesia (Ilha)": "Freguesia (Ilha do Governador)",
    "Todos Os Santos": "Todos os Santos",
    "Recreio": "Recreio dos Bandeirantes",
    "R. Bandeirantes": "Recreio dos Bandeirantes",
    "R. dos Bandeirantes": "Recreio dos Bandeirantes",
    "Rcr Bandeirantes": "Recreio dos Bandeirantes",
    "Rec dos Bandeirante": "Recreio dos Bandeirantes",
    "Recreio Bandeirante": "Recreio dos Bandeirantes",
    "I. Governador": "Ilha do Governador",
    "I. do Governador": "Ilha do Governador",
    "Ilha Governador": "Ilha do Governador",
    "L Vasconcelos": "Lins de Vasconcelos",
    "V Carvalho": "Vicente de Carvalho",
    "São Fco Xavier": "São Francisco Xavier",
    "Bota Fogo": "Botafogo",
    "Vila Izabel": "Vila Isabel",
    "Braz de Pina": "Brás de Pina",
    "Osvaldo Cruz": "Oswaldo Cruz",
    "PACIÊNCIA": "Paciência",
    "Jacarepaguá1": "Jacarepaguá",
    "Jacarepaguá - Rio de Janeiro": "Jacarepaguá",
    "Campo Grande NUMERO 270": "Campo Grande",
    "Rio da Prata, Campo Grande": "Campo Grande",
    "Jardim Letícia - Campo Grande": "Campo Grande",
    "Colégio Rj": "Colégio",
    "Tomas Coelho Rj": "Tomás Coelho",
    "Taquara - Jacarepaguá": "Taquara",
    "Taquara (jacarepaguá)": "Taquara",
    "Pechincha (jacarepaguá)": "Pechincha",
    "Tijuca - Praça Saes Peña": "Tijuca",
    "Moneró (ilha)": "Moneró",
    "Bairro Riachuelo": "Riachuelo",
}


def padronizar_bairro(serie):
    chave = serie.map(normalizar, na_action="ignore")
    mais_comum = serie.str.strip().groupby(chave).agg(lambda s: s.value_counts().index[0])
    resultado = chave.map(mais_comum).replace(CORRECAO_BAIRRO)
    return resultado.mask(~resultado.str.contains(r"[A-Za-zÀ-ú]", na=False))  # só números -> ausente


bairro_padronizado = padronizar_bairro(df_raw[BAIRRO])
alterado = df_raw[BAIRRO].ne(bairro_padronizado) & df_raw[BAIRRO].notna()

print(f"Bairros distintos: {df_raw[BAIRRO].nunique()} -> {bairro_padronizado.nunique()}")
print(f"Anúncios com o bairro alterado: {alterado.sum()} ({alterado.mean():.1%})\\n")
pd.DataFrame({"antes": df_raw.loc[alterado, BAIRRO].map(repr), "depois": bairro_padronizado[alterado]}) \\
    .value_counts().rename("anúncios").reset_index().sort_values("depois").head(25)
""")

md("""
A padronização reduz os bairros distintos de 215 para 144 e altera cerca de 5% dos anúncios. A maior correção é "Freguesia (jacarepaguá)", com o "j" minúsculo: é a forma mais comum no site, e por isso a limpeza mecânica a escolheria, mas o dicionário a corrige. Sem a padronização, qualquer contagem ou mediana por bairro estaria fragmentada.

## 7.2 Rua

A rua tem os mesmos problemas do bairro e um problema próprio: **o campo mistura formatos**.
""")

code("""
ruas = df_raw[RUA].dropna()
com_numero = ruas.str.contains(r"\\d")
endereco_completo = ruas.str.contains(r" - .*Rio de Janeiro|Cep|\\d{5}-\\d{3}", case=False)
abreviada = ruas.str.match(r"^(R\\.|R |Av\\.|Av |Estr\\.|Est\\.|Trav\\.)")

print(f"Ruas distintas (como estão)                 : {ruas.nunique():,}")
print(f"Distintas após normalização (espaço/acento) : {ruas.map(normalizar).nunique():,}")
print(f"Com algum número no nome                    : {com_numero.sum()}")
print(f"Com endereço completo (bairro, cidade, CEP) : {endereco_completo.sum()}")
print(f"Com tipo de logradouro abreviado            : {abreviada.sum()}")
print("\\nExemplos de endereço completo no campo de rua:")
for texto in ruas[endereco_completo].sample(6, random_state=1):
    print("  ", texto)
print("\\nExemplos de abreviação:", ruas[abreviada].drop_duplicates().head(6).tolist())
""")

md("""
A maioria das ruas vem só com o nome do logradouro, como esperado. Mas uma parte traz o **endereço completo**, com número, bairro, cidade, estado, CEP e país, outra usa **abreviações** ("R.", "Av.", "Estr.", "Mal."), e há casos com o **nome do condomínio antes da rua** ("Condomínio do Edifício ... - Rua ...") ou até com **bloco e apartamento** no fim. Como na mesma base aparecem "Rua X", "R. X" e "R. X 106 - Vila Isabel Rio de Janeiro - Rj 20551-140 Brasil", a mesma rua vira vários valores diferentes, e o número do imóvel aparece só em parte dos registros.

**Proposta de padronização:** manter só o logradouro, que é o trecho (entre os separados por " - ") que começa com um tipo de logradouro, como Rua, Avenida ou Estrada. Em seguida, remover número, CEP, bloco e apartamento e os espaços extras, e expandir as abreviações do tipo de logradouro. Abreviações no meio do nome ("Mal.", "Sen.", "Nossa Sra.") ficam como estão.

As **variações de acento** (cerca de 120 ruas, como "Avenida das Américas" e "Avenida das Americas") **não** são unificadas automaticamente. No bairro, a forma mais frequente é quase sempre a correta, porque cada bairro tem centenas de anúncios. Na rua, com poucos anúncios por logradouro, a forma errada pode ser a mais frequente: num teste, a regra trocou "Rua Campina Grande" por "Rua Campína Grande". Unificar exigiria uma lista oficial de logradouros, que a base não tem.
""")

code("""
ABREVIACOES = {r"^R\\.?\\s": "Rua ", r"^Av\\.?\\s": "Avenida ", r"^Estr?\\.\\s": "Estrada ", r"^Trav\\.?\\s": "Travessa "}
LOGRADOURO = re.compile(r"^(Rua|R\\.?|Avenida|Av\\.?|Estrada|Estr?\\.|Travessa|Trav\\.?|Praça|Largo|Ladeira|"
                        r"Alameda|Boulevard|Via|Caminho|Praia|Rodovia|Vila|Beco)\\s", re.IGNORECASE)


def escolher_trecho(texto):
    \"\"\"Em 'Condomínio X - Rua Y 10 - Bairro ...', fica com o trecho que começa por um tipo de logradouro.\"\"\"
    trechos = [t.strip() for t in texto.split(" - ")]
    return next((t for t in trechos if LOGRADOURO.match(t)), trechos[0])


def padronizar_rua(serie):
    rua = serie.str.strip().str.replace(r"\\s+", " ", regex=True)
    rua = rua.map(escolher_trecho, na_action="ignore")
    rua = rua.str.replace(r"\\s+\\d+\\s+Cep.*$", "", regex=True, case=False)        # "Rua X 73 Cep ..."
    rua = rua.str.replace(r"\\s+(Bloco|Bl\\.?|Apto?\\.?)\\s.*$", "", regex=True, case=False)  # bloco e apartamento
    rua = rua.str.replace(r"\\s+\\d+\\w?$", "", regex=True)                       # número no final ("215b")
    for padrao, completo in ABREVIACOES.items():
        rua = rua.str.replace(padrao, completo, regex=True)
    return rua


rua_padronizada = padronizar_rua(df_raw[RUA])
mudou_rua = df_raw[RUA].ne(rua_padronizada)
print(f"Ruas distintas: {df_raw[RUA].nunique():,} -> {rua_padronizada.nunique():,}")
print(f"Anúncios com a rua alterada: {mudou_rua.sum()} ({mudou_rua.mean():.1%})\\n")
pd.DataFrame({"antes": df_raw.loc[mudou_rua, RUA], "depois": rua_padronizada[mudou_rua]}) \\
    .drop_duplicates().sample(10, random_state=3)
""")

md("""
A padronização transforma endereços completos, abreviados ou precedidos do nome do condomínio no nome do logradouro, que passa a ser comparável entre anúncios. Ruas com letra no próprio nome, como "Rua Projetada C", são preservadas, porque só se remove o número que vem **no final**. Há um efeito colateral conhecido: logradouros cujo nome **termina** em número (como "Rua Projetada 04") perdem o número. Esses casos são raros, mas mostram que regras automáticas de texto sempre têm exceções.

Remover bloco e apartamento também protege a **privacidade**: junto com a rua e o número, eles identificariam o imóvel exato do anunciante.

## 7.3 Demais categóricas
""")

code("""
print(f"tipo  : {df_raw['tipo'].value_counts().to_dict()}")
print(f"cidade: {df_raw['cidade'].value_counts(dropna=False).to_dict()}")
""")

md("""
`tipo` vem de uma lista fechada da plataforma e não tem variações de escrita. Em `cidade`, além dos ausentes, aparece o valor "Lins", que é um bairro (seção 5.7). Os problemas de padronização se concentram nos campos de **texto livre**: bairro e rua.
""")

# =====================================================================
# 8. Indicadores
# =====================================================================
md("""
# 8. Indicadores de qualidade

## 8.1 Base tratada

Para medir o efeito dos tratamentos, montamos uma cópia da base com as correções propostas. **Nenhuma linha é removida.** Os valores errados viram ausentes, e os casos duvidosos ganham uma coluna de sinalização.

| Tratamento | Seção |
|---|---|
| `condominio` −1 em casa de rua → 0 (não tem condomínio) | 2.2 |
| `condominio` −1 nos demais tipos, simbólico (R$ 1 a 9) ou acima de 1% do preço → ausente | 2.2 e 5.3 |
| `iptu` −1 ou com valor anual acima de 2% do preço → ausente | 2.2 e 5.4 |
| `area_m2` abaixo de 10 m² → ausente | 5.2 |
| `bairro` e `rua` padronizados | 7 |
| `cidade` ausente ou "Lins" → "Rio de Janeiro" (rua e bairro são do Rio) | 5.7 |
| `condominio_iptu` recalculado a partir dos valores tratados | 1.2 |
| Colunas `possivel_duplicata`, `preco_no_piso`, `outlier_area` | 3.2, 5.6 e 6 |
""")

code("""
df_tratado = df_raw.copy()

casa_sem_condominio = (df_raw["tipo"] == "Casa") & df_raw[CONDOMINIO].eq(-1)
condominio_invalido = (df_raw[CONDOMINIO].eq(-1) & ~casa_sem_condominio) | df_raw[CONDOMINIO].between(1, 9) | (razao_cond > 0.01)
iptu_invalido = df_raw[IPTU].eq(-1) | (razao_iptu_anual > 0.02)
area_invalida = df_raw[AREA] < 10

df_tratado.loc[casa_sem_condominio, CONDOMINIO] = 0
df_tratado[CONDOMINIO] = df_tratado[CONDOMINIO].mask(condominio_invalido)
df_tratado[IPTU] = df_tratado[IPTU].mask(iptu_invalido)
df_tratado[AREA] = df_tratado[AREA].mask(area_invalida)
df_tratado["condominio_iptu"] = df_tratado[CONDOMINIO] + df_tratado[IPTU]  # ausente se um dos dois for ausente
df_tratado[BAIRRO] = bairro_padronizado
df_tratado[RUA] = rua_padronizada
df_tratado["cidade"] = "Rio de Janeiro"

df_tratado["possivel_duplicata"] = df_raw.duplicated(subset=COLUNAS_IMOVEL, keep=False)
df_tratado["preco_no_piso"] = no_piso
df_tratado["outlier_area"] = outlier_log[AREA].reindex(df_raw.index, fill_value=False)

assert len(df_tratado) == len(df_raw)
assert (df_tratado[[CONDOMINIO, IPTU]].dropna() >= 0).all().all()
print(f"Casas de rua com condomínio -1 -> 0 : {casa_sem_condominio.sum():,}")
print(f"Condomínios anulados                : {condominio_invalido.sum():,}")
print(f"IPTUs anulados                      : {iptu_invalido.sum():,}")
print(f"Áreas anuladas                      : {area_invalida.sum()}")
print(f"Linhas: {len(df_raw):,} -> {len(df_tratado):,} (nenhuma removida)")
""")

md("""
## 8.2 Definição dos indicadores

- **Completude:** percentual de células preenchidas. É calculada (a) sobre **todas as 22 colunas** da base e (b) sobre as **seis variáveis do enunciado**, de duas formas: só com os ausentes declarados, o que o `isna()` enxerga, e contando também os ausentes disfarçados (−1 fora das casas de rua e área zero).
- **Unicidade:** percentual de registros que não são cópia de outro, por três critérios: código do anúncio, todas as características (exceto código e URL) e rua + bairro + preço + metragem.
- **Validade:** percentual de registros que respeitam **todas** as regras de cada conjunto:
  - **regras do enunciado:** preço > 0, preço ≤ R$ 1.000.000, metragem > 0;
  - **regras ampliadas:** as do enunciado **mais** metragem ≥ 10 m², condomínio sem código −1 (exceto casa de rua) e entre R$ 10 e 1% do preço, IPTU sem código −1 e com valor anual ≤ 2% do preço, e bairro preenchido. Um valor **ausente não viola** a regra: ausência é medida pela completude.

Na base original, um valor errado conta como **inválido**. Na base tratada, ele foi transformado em ausente e passa a pesar na **completude**. É a troca esperada: um dado ausente é honesto, um dado errado engana.
""")

code("""
COLUNAS_ENUNCIADO = [PRECO, CONDOMINIO, IPTU, AREA, BAIRRO, RUA]


def validade(base):
    cond, iptu, area = base[CONDOMINIO], base[IPTU], base[AREA]
    casa_rua = base["tipo"] == "Casa"
    regras_enunciado = (base[PRECO] > 0) & (base[PRECO] <= LIMITE_PRECO_COLETA) & (area.isna() | (area > 0))
    cond_ok = cond.isna() | (cond == 0) | (casa_rua & (cond == -1)) | ((cond >= 10) & (cond <= 0.01 * base[PRECO]))
    iptu_ok = iptu.isna() | ((iptu >= 0) & (iptu * 12 <= 0.02 * base[PRECO]))
    regras_ampliadas = regras_enunciado & (area.isna() | (area >= 10)) & cond_ok & iptu_ok & base[BAIRRO].notna()
    return regras_enunciado.mean(), regras_ampliadas.mean()


def indicadores(base, completude_real):
    val_enunciado, val_ampliada = validade(base)
    return {
        "Completude: todas as colunas (isna)": base[df_raw.columns].notna().mean().mean(),
        "Completude: variáveis do enunciado (isna)": base[COLUNAS_ENUNCIADO].notna().mean().mean(),
        "Completude: variáveis do enunciado, com disfarçados": completude_real,
        "Unicidade: código do anúncio": 1 - base["id"].duplicated().mean(),
        "Unicidade: todas as características": 1 - base.duplicated(subset=COLUNAS_IMOVEL).mean(),
        "Unicidade: rua + bairro + preço + metragem": 1 - base.duplicated(subset=[RUA, BAIRRO, PRECO, AREA]).mean(),
        "Validade: regras do enunciado": val_enunciado,
        "Validade: regras ampliadas": val_ampliada,
    }


disfarcados_raw = (nao_casa & df_raw[CONDOMINIO].eq(-1)).sum() + df_raw[IPTU].eq(-1).sum() + df_raw[AREA].eq(0).sum()
completude_real_raw = (df_raw[COLUNAS_ENUNCIADO].notna().sum().sum() - disfarcados_raw) / (len(df_raw) * 6)
completude_real_tratado = df_tratado[COLUNAS_ENUNCIADO].notna().mean().mean()

tabela_ind = pd.DataFrame({
    "base original": indicadores(df_raw, completude_real_raw),
    "base tratada": indicadores(df_tratado, completude_real_tratado),
})
tabela_ind.map("{:.2%}".format)
""")

md("""
**Leitura dos indicadores:**

- **Completude:** pelo `isna()`, a base original parece praticamente completa, tanto nas 22 colunas quanto nas seis do enunciado. Contando os ausentes disfarçados, a completude real das variáveis do enunciado é menor: o −1 esconde cerca de 1.900 condomínios e IPTUs não informados. Na base tratada, a completude cai mais um pouco, porque os valores impossíveis também viraram ausentes.
- **Unicidade:** 100% pelo código do anúncio, porque o *scraper* não repetiu anúncios. Pelas características, a unicidade fica um pouco abaixo: são os anúncios do mesmo imóvel (ou de unidades iguais do mesmo prédio) da seção 3. Na base tratada, a unicidade por rua cai ligeiramente: com a rua padronizada, anúncios do mesmo endereço escritos de formas diferentes passam a ser reconhecidos como iguais.
- **Validade pelas regras do enunciado:** praticamente 100%. Os únicos violadores são as áreas zeradas, porque a coleta respeitou o limite de preço.
- **Validade pelas regras ampliadas:** bem menor na base original, por causa do −1 e dos valores impossíveis de condomínio e IPTU. Depois do tratamento, os valores inválidos viraram ausentes, e a validade se aproxima de 100%.
""")

# =====================================================================
# 9. Visualizações
# =====================================================================
md("""
# 9. Visualizações

## 9.1 Valores ausentes por variável

O gráfico separa os **ausentes declarados** (célula vazia) dos **ausentes disfarçados** (código −1 fora das casas de rua, área zero) nas seis variáveis do enunciado, além de `cidade` e `suites`.
""")

code("""
faltas = pd.DataFrame({
    "declarado (célula vazia)": df_raw[COLUNAS_ENUNCIADO + ["cidade", "suites"]].isna().sum(),
    "disfarçado (-1 ou zero)": pd.Series({
        CONDOMINIO: (nao_casa & df_raw[CONDOMINIO].eq(-1)).sum(),
        IPTU: df_raw[IPTU].eq(-1).sum(),
        AREA: df_raw[AREA].eq(0).sum(),
    }),
}).fillna(0) / len(df_raw) * 100
faltas = faltas.loc[faltas.sum(axis=1).sort_values().index]

fig, ax = plt.subplots(figsize=(10, 4.5))
faltas.plot.barh(stacked=True, ax=ax, color=["#4C72B0", "#DD8452"], width=0.7)
for i, total in enumerate(faltas.sum(axis=1)):
    ax.text(total, i, f" {total:.2f}%", va="center", fontsize=9)
ax.set(title="Valores ausentes por variável", xlabel="% dos anúncios", ylabel="")
ax.set_xlim(0, faltas.sum(axis=1).max() * 1.15)
ax.legend(title="Tipo de ausência", loc="lower right")
plt.tight_layout()
plt.show()
""")

md("""
**Interpretação:** as duas variáveis com mais ausência, IPTU e condomínio, **não têm nenhuma célula vazia**: toda a ausência delas é disfarçada pelo código −1. Uma análise que olhasse apenas `isna()` concluiria, erradamente, que essas colunas estão completas. Nas demais variáveis, a ausência é pontual.

## 9.2 Boxplots do preço e da metragem

Cada boxplot é mostrado na escala original e na escala logarítmica. A metragem é desenhada só acima de zero.
""")

code("""
fig, eixos = plt.subplots(2, 2, figsize=(14, 6))

for linha, (coluna, nome, divisor, unidade, cor) in enumerate([
    (PRECO, "Preço", 1e3, "R$ mil", "#4C72B0"),
    (AREA, "Metragem", 1, "m²", "#55A868"),
]):
    valores = df_raw.loc[df_raw[coluna] > 0, coluna] / divisor
    sns.boxplot(x=valores, ax=eixos[linha, 0], color=cor, flierprops={"markersize": 2, "alpha": 0.3})
    eixos[linha, 0].set(title=f"Boxplot: {nome.lower()} (escala original)", xlabel=f"{nome} ({unidade})")
    sns.boxplot(x=valores, ax=eixos[linha, 1], color=cor, log_scale=True, flierprops={"markersize": 2, "alpha": 0.3})
    eixos[linha, 1].set(title=f"Boxplot: {nome.lower()} (escala log)", xlabel=f"{nome} ({unidade}, escala log)")

plt.tight_layout()
plt.show()
""")

md("""
**Interpretação:**

- **Preço:** a caixa (os 50% centrais) vai de cerca de R$ 255 mil a R$ 650 mil, e os bigodes cobrem toda a faixa da coleta, de R$ 125 mil a R$ 1 milhão, **sem nenhum ponto isolado**. O teto da coleta eliminou os extremos. O boxplot também mostra uma limitação da técnica: o piso de R$ 200 mil, o maior problema do preço, **não aparece** nele. Ele só fica visível no histograma (seção 9.3).
- **Metragem:** na escala original, a caixa fica espremida à esquerda e centenas de pontos aparecem como *outliers*, até alguns milhares de m². Na escala log, a caixa fica centralizada, e os pontos isolados se reduzem às duas pontas: **à esquerda, as áreas de 1 a 9 m² (erros)**; à direita, as casas com terreno grande e os prováveis erros de cadastro.

## 9.3 Distribuições do preço e da metragem
""")

code("""
fig, eixos = plt.subplots(1, 2, figsize=(14, 4.5))

precos = df_raw[PRECO] / 1e3
sns.histplot(precos, bins=88, ax=eixos[0], color="#4C72B0")
eixos[0].axvline(precos.median(), color="#C44E52", ls="--", label=f"mediana = R$ {precos.median():,.0f} mil")
eixos[0].axvline(precos.mean(), color="#333333", ls=":", label=f"média = R$ {precos.mean():,.0f} mil")
eixos[0].set(title="Distribuição: preço", xlabel="Preço (R$ mil)", ylabel="Anúncios")
eixos[0].legend()

areas = df_raw.loc[df_raw[AREA] > 0, AREA]
sns.histplot(areas, bins=80, ax=eixos[1], color="#55A868", log_scale=True)
eixos[1].axvline(areas.median(), color="#C44E52", ls="--", label=f"mediana = {areas.median():,.0f} m²")
eixos[1].set(title="Distribuição: metragem (escala log)", xlabel="Metragem (m², escala log)", ylabel="Anúncios")
eixos[1].legend()

plt.tight_layout()
plt.show()
""")

md("""
**Interpretação:**

- **Preço:** a distribuição é assimétrica à direita (mais imóveis baratos que caros), com a média acima da mediana. O destaque é a **barra isolada em R$ 200 mil**, muito mais alta que todas as outras e com quase nada à esquerda: é o piso de preço da seção 5.6, o principal problema de qualidade visível neste gráfico. Também aparecem picos menores em valores "redondos" (R$ 250 mil, R$ 300 mil, R$ 350 mil...), um arredondamento natural dos preços anunciados. A distribuição termina abruptamente em R$ 1 milhão, o limite da coleta.
- **Metragem:** na escala log, a forma é próxima de um sino, concentrada entre 40 e 150 m². Os "dentes" do histograma mostram que muitos anunciantes **arredondam** a área (50, 60, 70, 80, 100 m²). Os poucos valores na extrema esquerda, abaixo de 10 m², são os erros de cadastro.

## 9.4 Complemento: condomínio e IPTU

As duas variáveis com mais problemas merecem um gráfico próprio. Usamos a escala log para enxergar ao mesmo tempo os valores normais e os absurdos:
""")

code("""
fig, eixos = plt.subplots(1, 2, figsize=(14, 4))
for ax, coluna, nome, cor in [(eixos[0], CONDOMINIO, "Condomínio", "#8172B3"), (eixos[1], IPTU, "IPTU", "#937860")]:
    positivos = df_raw.loc[df_raw[coluna] > 0, coluna]
    sns.histplot(positivos, bins=70, log_scale=True, ax=ax, color=cor)
    ax.axvline(10, color="#C44E52", ls="--", lw=1)
    ax.set(title=f"{nome} mensal (valores > 0, escala log)", xlabel=f"{nome} (R$/mês, escala log)", ylabel="Anúncios")
    ax.text(0.98, 0.95, f"= -1: {df_raw[coluna].eq(-1).sum():,}\\n= 0: {df_raw[coluna].eq(0).sum():,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#CCCCCC"})
plt.tight_layout()
plt.show()
""")

md("""
**Interpretação:** o grosso dos condomínios fica entre algumas centenas e poucos milhares de reais, e o dos IPTUs entre dezenas e poucas centenas, faixas plausíveis para imóveis de até R$ 1 milhão. Mas os dois histogramas têm **caudas impossíveis à direita**, com valores de centenas de milhares de reais por mês, e grupos de valores simbólicos à esquerda da linha tracejada (R$ 10). Os quadros no canto lembram que, além do que aparece no gráfico, milhares de anúncios têm −1 ou zero, que não podem ser desenhados em escala log.
""")

# =====================================================================
# 10. Diagnóstico final
# =====================================================================
md("""
# 10. Diagnóstico final da qualidade dos dados

## Você considera essa base suficientemente confiável para realizar uma análise do mercado imobiliário do Rio de Janeiro?

**Sim, com ressalvas, e depois dos tratamentos propostos.** A base é confiável para analisar **preço e área por bairro** no segmento de **imóveis de até R$ 1 milhão anunciados diretamente pelo QuintoAndar**. Os custos de condomínio e IPTU podem ser usados com cuidado, depois de tratados. A base **não** representa o mercado inteiro: imóveis acima de R$ 1 milhão e anúncios de terceiros (classificados) ficaram de fora por construção da coleta, e o segmento mais barato está distorcido pelo piso de R$ 200 mil.

**Por que sim:**

- **Volume e cobertura:** mais de 18 mil anúncios, com 98,8% de cobertura do que a plataforma informa, distribuídos por cerca de 140 bairros (depois da padronização).
- **Variáveis centrais íntegras:** o preço está 100% preenchido, é sempre positivo e respeita o limite da coleta. Rua e bairro estão praticamente completos.
- **Formato correto:** todos os números chegaram como números, sem símbolos, graças à coleta pela API.
- **Sem anúncios repetidos** pelo código. As possíveis duplicatas de imóvel são poucas e estão sinalizadas.
- **Validação externa:** na etapa 1, os campos coincidiram com a base de referência do professor em quase 100% dos anúncios em comum.

## Principais problemas encontrados

| # | Problema | Variáveis | Seção |
|---|---|---|---|
| 1 | **Código −1** como "não informado" (ou "sem condomínio", nas casas de rua) | condomínio, IPTU | 2.2 |
| 2 | **Concentração de preços em exatamente R$ 200 mil** (provável piso da plataforma) | preço | 5.6 |
| 3 | **Grafias diferentes para o mesmo bairro**: espaços extras, maiúsculas/minúsculas, acentos, abreviações, complementos | bairro | 7.1 |
| 4 | **Campo de rua com formatos misturados**: só o nome, com número, endereço completo com CEP, abreviações | rua | 7.2 |
| 5 | Valores impossíveis: condomínio e IPTU de centenas de milhares de reais por mês, condomínios simbólicos | condomínio, IPTU | 5.3 e 5.4 |
| 6 | Metragem zero ou abaixo de 10 m², e áreas enormes com m² irrisório | metragem | 5.2 e 6.2 |
| 7 | Anúncios idênticos em todas as características com códigos diferentes | todas | 3.2 |
| 8 | IPTU zero ambíguo (isenção ou não informado) | IPTU | 2.2 |
| 9 | Cidade ausente ou preenchida com um bairro ("Lins") | cidade | 5.7 |

## Problemas corrigidos

- **Código −1:** nas casas de rua, convertido em 0 (sem condomínio); nos demais casos, transformado em ausente.
- **Valores impossíveis** de condomínio, IPTU e metragem: transformados em ausentes, preservando os anúncios.
- **Bairros:** espaços, caixa e acentos unificados pela forma mais frequente, e abreviações, erros e complementos corrigidos por dicionário.
- **Ruas:** reduzidas ao nome do logradouro, com espaços extras removidos e abreviações expandidas.
- **Cidade:** preenchida como Rio de Janeiro, confirmada pela rua e pelo bairro.
- **`condominio_iptu`** recalculado com os valores tratados, em vez da soma do site, que tratava −1 como zero.

## Problemas que permaneceram

- **O piso de R$ 200 mil** não pode ser corrigido sem o preço real. Os anúncios foram apenas sinalizados.
- **Possíveis duplicatas e áreas extremas** foram sinalizadas, não removidas. Sem o número do imóvel e o apartamento, não há como confirmar se são o mesmo imóvel.
- **Valores errados, mas plausíveis** (um condomínio de R$ 800 que deveria ser R$ 1.800) não são detectáveis.
- **IPTU zero:** não dá para distinguir isenção de falta de informação.
- **Variações de acento nas ruas** ("Americas" e "Américas"): sem uma lista oficial de logradouros, não há como saber com segurança qual é a forma correta.
- **Localidades não reconhecidas** e áreas arredondadas permanecem.

## Limitações provocadas pelo processo de *scraping*

1. **Cobertura parcial do mercado por decisão da coleta:** só anúncios próprios do QuintoAndar (os classificados são bloqueados pelo `robots.txt` e ficaram de fora) e só até R$ 1 milhão.
2. **Cobertura de 98,8%:** a API informa um total um pouco maior do que é possível recuperar pelas faixas de preço. A paginação instável da API, contornada na etapa 1, mostra que a fonte não é feita para coleta em massa.
3. **Dependência de uma API interna não documentada:** o formato pode mudar sem aviso e quebrar o *scraper*, e os códigos (como o −1) não têm documentação. O significado deles foi deduzido dos dados.
4. **Retrato de um único dia:** os preços mudam (na etapa 1, cerca de 10% dos anúncios mudaram de preço em 13 dias), e não há série histórica.
5. **Preço anunciado, não preço de venda:** o valor fechado costuma ser menor que o pedido.
6. **Dados digitados pelos anunciantes:** o *scraper* copia fielmente o que está no site, inclusive os erros de quem preencheu o cadastro. Os campos de texto livre (bairro e rua) concentram os problemas de padronização, e os campos opcionais (condomínio e IPTU) concentram os problemas de completude.

**Em resumo:** a coleta tem qualidade **alta nas variáveis centrais** (preço, localização) e **problemas sérios, porém tratáveis, nos custos** (condomínio e IPTU) e na **padronização dos textos** (bairro e rua). Com os tratamentos aplicados, ela é adequada para uma análise do mercado carioca de imóveis de até R$ 1 milhão no nível de bairro, desde que os resultados sejam apresentados com as ressalvas acima.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}
nbf.write(nb, ARQUIVO)
print(f"Notebook gerado: {ARQUIVO} ({len(cells)} células)")
