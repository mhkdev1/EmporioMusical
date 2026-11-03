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
        self.pending: str | None = None  # first contact typed, waiting for the second one

    def handle(self, text: str) -> str:
        """Answer one line typed by the customer."""
        command, _, arg = text.strip().partition(" ")
        if command == "/login":
            return self.login(arg)
        if command == "/logout":
            return self.logout()
        if self.pending:  # this line answers "which record is yours?"
            return self.login(self.pending, confirm=text)
        # ask() only hands back a new history when the model answered, so a failed call changes nothing.
        reply, self.history = agent.ask(self.bot, self.history, text, self.customer_id)
        return reply

    def login(self, contact: str, confirm: str | None = None) -> str:
        if not contact.strip():
            return "Use: /login seu-email-ou-telefone"
        result = tools.find_customer(contact, confirm)
        self.pending = None
        if result["status"] == "need_more":
            self.pending = contact
            other = "telefone" if result["ask_for"] == "phone" else "e-mail"
            return f"Encontrei mais de um cadastro com esse dado. Para confirmar que é você, digite o {other} do seu cadastro."
        if result["status"] == "not_found":
            return LOGIN_FAILED
        if result["customer_id"] != self.customer_id:
            self.history = []  # a new customer must not inherit the previous one's conversation
        self.customer_id = result["customer_id"]
        # There's no password in the data. A real store would send a one-time code to the e-mail or phone first.
        return f"Olá, {result['name'].split()[0]}! Você entrou na sua conta."
    
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
        label = "Sistema" if text.startswith("/") or chat.pending else "Melodia"
        try:
            print(f"{label}: {chat.handle(text)}\n")
        except Exception as exc:  # model offline, quota, network: keep the chat open
            print(f"[Não consegui falar com o modelo agora: {exc}]\n")




if __name__ == "__main__":
    main()