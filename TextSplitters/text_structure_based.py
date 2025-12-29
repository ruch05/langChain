from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """Cricket: A Game of Skill, Strategy, and Spirit
Cricket is more than a sport; it is a cultural institution that blends athletic skill, strategic depth, and a strong sense of tradition. Played and followed by millions across continents, cricket has evolved over centuries while retaining a distinctive character that sets it apart from other games. From dusty village grounds to vast international stadiums, cricket continues to unite people through shared passion and competition.
At its core, cricket is a contest between bat and ball, but its true complexity lies in its balance between individual brilliance and collective effort. A single player can change the course of a match with a dazzling century or a devastating spell of bowling, yet success ultimately depends on teamwork, discipline, and planning. Captains must think several moves ahead, adjusting field placements and tactics in response to changing conditions, player form, and the state of the game.
One of cricket's most remarkable features is its variety of formats. Traditional Test matches, played over five days, reward patience, endurance, and technical mastery. Shorter formats, such as One Day Internationals and Twenty20 matches, emphasize speed, innovation, and entertainment. This adaptability has allowed cricket to remain relevant in a fast-paced world while preserving a form that honors its historical roots.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)
print(len(chunks))