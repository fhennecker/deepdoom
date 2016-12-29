ENNEMIES = [
    "Arachnotron", "Archvile", "BaronOfHell", "HellKnight", "Cacodemon", "Cyberdemon",
    "Demon", "Spectre", "ChaingunGuy", "DoomImp", "Fatso", "LostSoul", "PainElemental", "Revenant",
    "ShotgunGuy", "SpiderMastermind", "WolfensteinSS", "ZombieMan", "MarineBFG", "MarineBerserk",
    "MarineChaingun", "MarineChainsaw", "MarineFist", "MarinePistol", "MarinePlasma", "MarineRailgun",
    "MarineRocket", "MarineSSG", "MarineShotgun", "ScriptedMarine", "StealthArachnotron",
    "StealthArchvile", "StealthBaron", "StealthHellKnight", "StealthCacodemon", "StealthDemon",
    "StealthChaingunGuy", "StealthDoomImp", "StealthFatso", "StealthRevenant", "StealthShotgunGuy",
    "StealthZombieMan", "Zombieman",
]

PICKUPS = [
    "Allmap", "ArmorBonus", "Backpack", "Berserk", "BFG9000", "BlueArmor", "BlueCard",
    "BlueSkull", "BlurSphere", "Cell", "CellPack", "Chaingun", "Chainsaw", "Clip", "ClipBox",
    "Fist", "GreenArmor", "HealthBonus", "Infrared", "InvulnerabilitySphere", "Medikit",
    "Megasphere", "Pistol", "PlasmaRifle", "RadSuit", "RedCard", "RedSkull", "RocketAmmo",
    "RocketBox", "RocketLauncher", "Shell", "ShellBox", "Shotgun", "Soulsphere", "Stimpack",
    "SuperShotgun", "YellowCard", "YellowSkull"
]

BLASTS = [
    "BulletPuff", "Blood", "BaronBall"
]

IGNORABLE = [
    "TeleportFog", "DoomPlayer"
]


def get_cone_entities(state, entity_type):
    return [x for x in state.labels if x.object_name in entity_type]


def has_visible(state, walls, entity_type):
    cone = get_cone_entities(state, entity_type)
    player = (state.game_variables[0], state.game_variables[1])
    for entity in cone:
        if all([is_visible(player, wall, entity) for wall in walls]):
            return True
    return False


def has_visible_entities(state, wall):
    types = ENNEMIES, PICKUPS, BLASTS
    return [has_visible(state, wall, x) for x in types]


def is_visible(player, wall, entity):
    a = player
    b = (entity.object_position_x, entity.object_position_y)
    c, d = wall

    return not does_intersect(a, b, c, d)


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def does_intersect(a, b, c, d):
    "return true if line segments ab and cd intersect"
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
