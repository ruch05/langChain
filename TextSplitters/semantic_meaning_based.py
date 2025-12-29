from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text="""Cricket is more than a sport; it is a cultural institution that blends athletic skill, strategic depth, and a strong sense of tradition. Played and followed by millions across continents, cricket has evolved over centuries while retaining a distinctive character that sets it apart from other games. From dusty village grounds to vast international stadiums, cricket continues to unite people through shared passion and competition.
At its core, cricket is a contest between bat and ball, but its true complexity lies in its balance between individual brilliance and collective effort. A single player can change the course of a match with a dazzling century or a devastating spell of bowling, yet success ultimately depends on teamwork, discipline, and planning. Captains must think several moves ahead, adjusting field placements and tactics in response to changing conditions, player form, and the state of the game.
Farmers sustain societies by nurturing the land, producing food, and supporting livelihoods through patience and hard work.


Terrorists spread fear and destruction through violence, undermining peace, security, and innocent lives."""

splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation", breakpoint_threshold_amount=0.5
)

result = splitter.create_documents([text])
print(len(result))
print(result)