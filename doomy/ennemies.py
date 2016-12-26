ENNEMIES = [
    "Arachnotron", "Archvile", "BaronOfHell", "HellKnight", "Cacodemon", "Cyberdemon",
    "Demon", "Spectre", "ChaingunGuy", "DoomImp", "Fatso", "LostSoul", "PainElemental", "Revenant",
    "ShotgunGuy", "SpiderMastermind", "WolfensteinSS", "ZombieMan", "MarineBFG", "MarineBerserk",
    "MarineChaingun", "MarineChainsaw", "MarineFist", "MarinePistol", "MarinePlasma", "MarineRailgun",
    "MarineRocket", "MarineSSG", "MarineShotgun", "ScriptedMarine", "StealthArachnotron",
    "StealthArchvile", "StealthBaron", "StealthHellKnight", "StealthCacodemon", "StealthDemon",
    "StealthChaingunGuy", "StealthDoomImp", "StealthFatso", "StealthRevenant", "StealthShotgunGuy",
    "StealthZombieMan"
]


def get_cone_ennemies(state):
    return [x for x in state.labels if x.object_name in ENNEMIES]


def get_visible_ennemies(state, walls):
    cone = get_cone_ennemies(state)
    player = (state.game_variables[0], state.game_variables[1])

    visible = []
    for entity in cone:
        if all([is_visible(player, wall, entity) for wall in walls]):
            visible.append(entity)

    return visible


def is_visible(player, wall, entity):
    a = player
    b = (entity.object_position_x, entity.object_position_y)

    return not does_intersect((a, b), wall)


def does_intersect(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return False

    return True
