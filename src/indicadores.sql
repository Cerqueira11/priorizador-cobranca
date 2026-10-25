-- Indicador 1: Taxa de pagamento e ticket médio por faixa de atraso
SELECT
    CASE
        WHEN dias_atraso <= 30 THEN '1. Ate 30 dias'
        WHEN dias_atraso <= 60 THEN '2. 31 a 60 dias'
        WHEN dias_atraso <= 90 THEN '3. 61 a 90 dias'
        ELSE '4. Acima de 90 dias'
    END AS faixa_atraso,
    COUNT(*) AS qtd_clientes,
    ROUND(AVG(valor_devido), 2) AS ticket_medio,
    ROUND(AVG(prob_pagamento) * 100, 1) AS prob_media_pct,
    ROUND(AVG(pagou) * 100, 1) AS taxa_pagamento_real_pct
FROM clientes_priorizados
GROUP BY 1
ORDER BY 1;

-- Indicador 2: Valor total em aberto e retorno esperado por faixa de propensão e ação
SELECT
    faixa_propensao,
    acao_sugerida,
    COUNT(*) AS qtd_clientes,
    ROUND(SUM(valor_devido), 2) AS valor_total_aberto,
    ROUND(SUM(score_prioridade), 2) AS valor_esperado_recuperavel,
    ROUND(AVG(pagou) * 100, 1) AS taxa_pagamento_real_pct
FROM clientes_priorizados
GROUP BY faixa_propensao, acao_sugerida
ORDER BY AVG(prob_pagamento) DESC;

-- Indicador 3: Taxa de pagamento por canal preferido
SELECT
    canal_preferido,
    COUNT(*) AS qtd_clientes,
    ROUND(AVG(prob_pagamento) * 100, 1) AS prob_media_pct,
    ROUND(AVG(pagou) * 100, 1) AS taxa_pagamento_real_pct,
    ROUND(SUM(score_prioridade), 2) AS valor_esperado_total
FROM clientes_priorizados
GROUP BY canal_preferido
ORDER BY taxa_pagamento_real_pct DESC;

-- Indicador 4: Top 20 clientes prioritários para contato humano hoje (faixa Média ordenada por retorno esperado)
SELECT
    id_cliente,
    valor_devido,
    dias_atraso,
    canal_preferido,
    ROUND(prob_pagamento * 100, 1) AS prob_pagamento_pct,
    score_prioridade AS valor_esperado_rs,
    acao_sugerida
FROM clientes_priorizados
WHERE acao_sugerida = 'Ligacao'
ORDER BY score_prioridade DESC
LIMIT 20;
