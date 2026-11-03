#Terminal chat. /login and /logout are handled here in code; everything else goes to the agent.
from src import agent, tools

EXIT_WORDS = {"sair", "exit"}
LOGIN_FAILED = "Não foi possível entrar com esse dado. Confira o e-mail ou use o telefone do cadastro."


class Chat:
    """One conversation: who is logged in plus the message history."""

    def __init__(self, bot):
        self.bot = bot
        self.customer_id: int | None = None
        self.history: list = []

    def handle(self, text: str) -> str:
        """Answer one line typed by the customer."""
        command, _, arg = text.strip().partition(" ")
        if command == "/login":
            return self.login(arg)
        if command == "/logout":
            return self.logout()
        # ask() only hands back a new history when the model answered, so a failed call changes nothing.
        reply, self.history = agent.ask(self.bot, self.history, text, self.customer_id)
        return reply

    def login(self, contact: str) -> str:
        if not contact.strip():
            return "Use: /login seu-email-ou-telefone"
        customer = tools.find_customer(contact)
        if customer is None:
            return LOGIN_FAILED
        if customer["customer_id"] != self.customer_id:
            self.history = []  # a new customer must not inherit the previous one's conversation
        self.customer_id = customer["customer_id"]
        # There's no password in the data. A real store would send a one-time code to the e-mail or phone first.
        return f"Olá, {customer['name'].split()[0]}! Você entrou na sua conta."

    def logout(self) -> str:
        self.customer_id = None
        self.history = []  # the conversation may hold personal data
        return "Você saiu da sua conta."


def main() -> None:
    chat = Chat(agent.build_agent())
    print("Empório da Música, atendimento virtual. Digite 'sair' para encerrar.")
    print("Para ver seus pedidos: /login seu-email-ou-telefone   |   /logout para sair da conta.\n")
    while True:
        try:
            text = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            break
        label = "Sistema" if text.startswith("/") else "Melodia"
        try:
            print(f"{label}: {chat.handle(text)}\n")
        except Exception as exc:  # model offline, quota, network: keep the chat open
            print(f"[Não consegui falar com o modelo agora: {exc}]\n")


if __name__ == "__main__":
    main()