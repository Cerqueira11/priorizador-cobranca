"""Orquestrador fim-a-fim do pipeline de cobrança (dados -> modelo -> priorização e SQL)."""

from src.gerar_dados import main as gerar_dados
from src.priorizar import main as priorizar_clientes
from src.treinar_modelo import treinar_e_avaliar


def main() -> None:
    print(">>> [1/3] Gerando base sintética de clientes inadimplentes...")
    gerar_dados()

    print("\n>>> [2/3] Treinando e avaliando modelos de propensão ao pagamento...")
    treinar_e_avaliar()

    print("\n>>> [3/3] Calculando priorização, régua de acionamento e indicadores SQL...")
    priorizar_clientes()

    print("\n[SUCESSO] Pipeline completo executado com sucesso!")


if __name__ == "__main__":
    main()
