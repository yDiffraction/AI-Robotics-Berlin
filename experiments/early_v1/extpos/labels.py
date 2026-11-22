"""Label policy for external sources (one place, used by every fetch / match step).

Task: fire on a throw *motion* (real throw or feint: the arm winds up and swings like a throw),
stay silent on everything else.

roles
  pos  throw motion: an arm winds up and swings an object away from the body
  neg  clearly not a throw motion (arm actions that do not look like a throw, idle-type actions)
  amb  overhand / throw-like swings whose label is debatable under the "feint = positive" rule
       (racket smash, hitting with an object, tossing up, catching). Never used as negatives:
       an overhand swing without release IS a feint by our definition, so calling these
       negatives would contradict our own feint positives. Kept as a separate bank so they can
       be evaluated (or used as positives) on purpose.
"""

# DailyDVS-200 action id -> (name, role). Ids/names from resource/action_description.csv.
DD_CLASSES = {
    # throw motion
    147: ("throwing_ball", "pos"),          # throw the ball to a distance
    37: ("throw_things", "pos"),            # throw the thing in hand to the ground (overhand)
    187: ("pitch", "pos"),                  # throw the ball into the basket
    # debatable
    69: ("throw_hats_in_the_air", "amb"),
    70: ("throw_rubbish", "amb"),
    171: ("bounce_volleyball_on_hand", "amb"),
    146: ("swing_badminton_racket", "amb"),
    49: ("hit_people_with_things", "amb"),
    # arm actions that are not throws
    127: ("punch", "neg"), 13: ("wave_hands", "neg"), 14: ("point_to_something", "neg"),
    84: ("rotate_arm", "neg"), 54: ("stretch", "neg"), 21: ("surrender", "neg"),
    56: ("salute", "neg"), 176: ("bounce_ball", "neg"),
    # idle-type
    0: ("yawn", "neg"), 4: ("put_hands_in_pockets", "neg"), 15: ("cross_arms", "neg"),
    31: ("use_mobile_phone", "neg"), 51: ("nod", "neg"), 52: ("shake_head", "neg"),
    90: ("check_time", "neg"), 107: ("walk", "neg"), 122: ("touch_nose", "neg"),
    165: ("stand_up", "neg"),
}

# HMDB51 class -> role. Only classes with a standing / moving person in view are used.
HMDB_CLASSES = {
    "throw": "pos",
    "catch": "amb", "shoot_ball": "amb", "swing_baseball": "amb", "golf": "amb", "hit": "amb",
    "fencing": "amb", "sword": "amb", "sword_exercise": "amb", "draw_sword": "amb",
    "wave": "neg", "punch": "neg", "clap": "neg", "walk": "neg", "stand": "neg", "turn": "neg",
    "pick": "neg", "shake_hands": "neg", "kick_ball": "neg", "kick": "neg", "jump": "neg",
    "run": "neg", "brush_hair": "neg", "drink": "neg", "talk": "neg", "smoke": "neg",
    "push": "neg", "climb_stairs": "neg",
}
