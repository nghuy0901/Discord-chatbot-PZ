---
title: "Condition"
category: Player
type: Condition
source_url: https://pzwiki.net/wiki/Condition
scraped_at: 2026-02-24 16:24:29
method: archived_wiki_markdown
---

# Condition

Player Moodle • Skill • Trait • Occupation • Inventory • Nutrition • Sleep • Condition • Health • Knox Infection

Condition (also known as durability) is a property of various items which tells how much the item has been used. In case of weapons and clothing , condition tells how much more the item can be used before it will break permanently; it will no longer be usable by the player. A player with the correct skill will be able to repair some weapons. When the green bar of a drainable item such as painkillers is gone completely, the item disappears. The item's condition is portrayed by the length of the green bar.

## Overview

Weapons deteriorate and eventually become unusable with repeated use.

The likelihood of dropping a level of condition depends on the particular weapon (for example, a spiked baseball bat is more likely to become damaged with each hit than a regular baseball bat ). When the final level of condition in a weapon is reached, the weapon will be unequipped. The weapon will show a red cross next to its icon to denote that it is broken and can no longer be used.

Some weapons which have lost some or all of their condition can be repaired. By possessing the correct items in the main inventory, and/or by having certain skills at a certain level or higher, players can attempt to repair their weapons

## Chances

The odds of a weapon losing condition depends on three factors: the weapon's ConditionLowerChanceOneIn value, the player's maintenance skill, and the player's proficiency with the particular weapon typo. The calculation is as follows:

lossChance = 1 / (ConditionLowerChanceOneIn + floor(floor(MaintenanceLevel + (WeaponLevel/2))/2)*2)

### Example

For example, a baseball bat , with a ConditionLowerChanceOneIn value of 20 would have the following odds:

### 0
- **Maintenance level:** 0
- **Weapon level:** 0
- **Condition lower chance:** 1/20 (5%)
### 0
- **Maintenance level:** 0
- **Weapon level:** 4
- **Condition lower chance:** 1/22 (5%)
### 0
- **Maintenance level:** 0
- **Weapon level:** 8
- **Condition lower chance:** 1/24 (4%)
### 4
- **Maintenance level:** 4
- **Weapon level:** 0
- **Condition lower chance:** 1/24 (4%)
### 8
- **Maintenance level:** 8
- **Weapon level:** 0
- **Condition lower chance:** 1/28 (4%)
### 4
- **Maintenance level:** 4
- **Weapon level:** 4
- **Condition lower chance:** 1/26 (4%)
### 4
- **Maintenance level:** 4
- **Weapon level:** 8
- **Condition lower chance:** 1/28 (4%)
### 8
- **Maintenance level:** 8
- **Weapon level:** 8
- **Condition lower chance:** 1/32 (3%)
### 10
- **Maintenance level:** 10
- **Weapon level:** 10
- **Condition lower chance:** 1/34 (3%)

## Repairing

Some items can be repaired, such as weapons and vehicle parts , by right-clicking them in the inventory , and selecting "Repair..." in the contextual menu. This will display a list of items that can be used to repair it. Hovering over a repair item reveals how much it "Potentially Repairs" and the "Chance of success".

- Potentially repairs : The percentage of the item's max condition that can be repaired.
- Chance of success : The likelihood of repairing successfully. Failure will result in damaging the item further.

After each repair (shown by the "Repaired" multiplier in the item's tooltip), the "Potentially repairs" value will decrease for all repair items. This makes repairing less effective over time compared to finding a new, undamaged item. Only successful repairs will count towards the number of times an item has been repaired.

### Melee weapons

- Repaired using various materials .
- Most repair items usually have several units to one item (shown in the repair item tooltip)
- Some repair items require no skills, making them useful in emergencies, as they potentially repair less, making them less effective long-term.

### Ranged weapons

- The player must have two of the same or a variant of a firearm, such as JS-2000 Shotgun or Sawed-off JS-2000 Shotgun .
- One of the firearms is consumed, and the other is returned with an extra 50% of its max condition.
- It's recommended to use two weapons that have less than 25% condition to maximize the benefit.

### Factors affecting repair

Factors that affect the effectiveness of a repair are:

- The repair item being used (see respective item pages for the repair chance modifiers).
- The player's skill level for an item that requires a skill, such as wood glue requiring carpentry , or a firearm requiring aiming .
- Whether a player has the lucky or unlucky trait .
- How many times the item has already been repaired.

#### Success chance

The chance of successfully repairing an item is determined by the following:

- Decreased by 30% for each level below the required level (no effect if no skill is required).
- Increased by 5% for each level above the required level (no effect if no skill is required).
- Decreased by 2% for each time it has been repaired.
- Increased by 5% for the lucky trait
- Decreased by 5% for the unlucky trait

#### Potentially repaired

The amount that an item is repaired is based on the following:

Repair item order: Some items are more effective at repairing certain items, which is determined by its position in the list of repair items.

- First item: 50%
- Second item: 20%
- Other items: 10%

Times repaired: This is then divided by the number of times it's been repaired plus 1 (rounded up). So, for an item that has been repaired twice:

- First item: 17% (50 / 3)
- Second item: 7% (20 / 3)
- Other items: 4% (10 / 3)

Skill level: Repair items that have a required skill affect the amount repaired.

- Increased by 5% for each level above the required, with a maximum increase of 25%.
- Decreased by 15% for each level below the required.

Vehicle parts: Vehicle parts have an extra "condition modifier" which is based on the method of repairing, and is multiplied by the condition to be repaired.

- 1.2 when repaired with a propane torch
- 0.5 when repaired with screws

## Average condition

Average condition is the average number of hits it will take for a weapon to break at level 0 weapon and maintenance skill. This value is not used in the game, and is only used throughout the wiki to provide a clear comparison between each weapon. The average condition is the result of multiplying ConditionMax and ConditionLowerChanceOneIn .

## References
