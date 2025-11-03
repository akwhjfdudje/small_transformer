import os, random

OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "fiction_corpus.txt")
NUM_LINES = 3000
SEED = 456
random.seed(SEED)

templates = [
    "{character} walked into the {place}, noticing {event}, and felt {adjective}, while {other_character} {action}.",
    "\"{dialogue},\" whispered {character}, as {other_event} unfolded in the {place}.",
    "It was a {adjective} day; {character} decided to {action} and {other_character} followed, unaware of {event}.",
    "{character} thought about {idea}, {other_character} interrupted with {dialogue}, and the {event} outside {verb}.",
    "In the distance, {event} signaled the beginning of {adventure}, and {character} knew {truth}.",
    "Suddenly, {event} disrupted the calm; {character} and {other_character} exchanged glances, unsure of {consequence}.",
]

characters = ["Alice","Bob","Eve","John","Clara","Marcus","Luna","Oliver"]
places = ["forest","castle","village","market","laboratory","space station","mountain pass","desert"]
events = ["a storm","a sudden scream","strange lights","an earthquake","a shadowy figure","an unexpected letter","a distant roar"]
adjectives = ["bright","gloomy","mysterious","chilly","serene","haunting","eerie","vivid","forbidding"]
actions = ["explore the cave","open the book","escape the room","decode the message","climb the tower","follow the path"]
objects = ["amulet","mirror","sword","device","key","journal","crystal","map"]
dialogues = ["We must leave now","I can't believe this","Everything will be fine","What was that noise?","Follow me quickly","This changes everything"]
ideas = ["freedom","the secret plan","their destiny","a hidden truth","the prophecy"]
other_events = ["the sun set","the wind howled","shadows stretched across the walls","footsteps echoed"]
adventures = ["a new journey","the quest for the lost artifact","an unexpected escape","the unraveling of the mystery"]
truths = ["nothing would be the same again","their lives were about to change","the secret was revealed"]
verbs = ["echoed through the valley","shook the walls","lit the night sky","warned of danger"]
consequences = ["what was about to happen","the fate of the village","the coming storm","the unfolding drama"]

lines=[]
for _ in range(NUM_LINES):
    tmpl = random.choice(templates)
    line = tmpl.format(
        character=random.choice(characters),
        other_character=random.choice(characters),
        place=random.choice(places),
        event=random.choice(events),
        other_event=random.choice(other_events),
        adjective=random.choice(adjectives),
        action=random.choice(actions),
        object=random.choice(objects),
        dialogue=random.choice(dialogues),
        idea=random.choice(ideas),
        adventure=random.choice(adventures),
        truth=random.choice(truths),
        verb=random.choice(verbs),
        consequence=random.choice(consequences)
    )
    lines.append(line)

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_FILE,"w",encoding="utf-8") as f:
    for ln in lines:
        f.write(ln+"\n")

print(f"Wrote {NUM_LINES} dense fiction lines to {OUT_FILE}")
