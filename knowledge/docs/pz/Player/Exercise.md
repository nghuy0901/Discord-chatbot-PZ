---
title: "Exercise"
category: Player
type: Exercise
source_url: https://pzwiki.net/wiki/Exercise
scraped_at: 2026-02-24 16:23:50
method: archived_wiki_markdown
---

# Exercise

Game mechanics Zombie • Health • Crafting • Building • Fire • AI • Noise • Weather • Erosion • Electricity • Metagame • Helicopter

Exercise is a gameplay mechanic in which the player can raise their passive skills, fitness , and strength . Players open the Exercise menu in the Health Tab to choose what exercises to do. Exercising will exert the player which will reduce endurance .

## Exercises

### Squats

Improves Fitness when performed regularly.

**Stiffness location:** Legs.

**Benefit:** Provides 4 fitness XP per squat.

### Push-ups

Improves strength when performed regularly.

**Stiffness location:** Arms and Chest.

**Benefit:** Provides 6 strength XP per push-up.

### Sit-ups

Improves fitness when performed regularly.

**Stiffness location:** Abs.

**Benefit:** Provides 2 fitness XP per sit-up.

### Burpees

Improves strength and fitness when performed regularly. Will drain endurance a lot.

**Stiffness location:** Legs, Arms, and Chest.

**Benefit:** Provides 3.2 fitness and 4.8 strength XP per burpee.

### Barbell curls

Improves strength when performed regularly.

A barbell is required to perform this exercise.

**Stiffness location:** Arms and Chest.

**Benefit:** Provides 7.2 strength XP per barbell curl.

### Dumbbell presses

Improves strength when performed regularly.

A dumbbell is required to perform this exercise.

**Stiffness location:** Arms.

**Benefit:** Provides 7.2 strength XP per dumbbell press.

### Bicep curls

A dumbbell is required to perform this exercise.

**Stiffness location:** Arms.

**Benefit:** Provides 7.2 strength XP per bicep curl.

### Summary

This table shows approximate experience per hour given by each exercise. These values were obtained in-game by performing each exercise for 30 in-game minutes (10 for dumbbell presses and bicep curls) uninterrupted, recording the total difference in experience, then doubling that difference (sextupling for dumbbell presses and bicep curls) and dividing by experience per rep; could use verification via source code analysis.

### Squats
- **Exercise:** Squats
- **Fitness per repetition:** 4.0
- **Strength per repetition:** 0.0
- **Repetitions per hour:** 52
- **Fitness per hour:** 208
- **Strength per hour:** 0
### Push-ups
- **Exercise:** Push-ups
- **Fitness per repetition:** 0.0
- **Strength per repetition:** 6.0
- **Repetitions per hour:** 96
- **Fitness per hour:** 0
- **Strength per hour:** 576
### Sit-ups
- **Exercise:** Sit-ups
- **Fitness per repetition:** 2.0
- **Strength per repetition:** 0.0
- **Repetitions per hour:** 72
- **Fitness per hour:** 144
- **Strength per hour:** 0
### Burpees
- **Exercise:** Burpees
- **Fitness per repetition:** 3.2
- **Strength per repetition:** 4.8
- **Repetitions per hour:** 60
- **Fitness per hour:** 192
- **Strength per hour:** 288
### Barbell curls
- **Exercise:** Barbell curls
- **Fitness per repetition:** 0.0
- **Strength per repetition:** 7.2
- **Repetitions per hour:** ?
- **Fitness per hour:** ?
- **Strength per hour:** ?
### Dumbbell presses
- **Exercise:** Dumbbell presses
- **Fitness per repetition:** 0.0
- **Strength per repetition:** 7.2
- **Repetitions per hour:** 138
- **Fitness per hour:** 0
- **Strength per hour:** 994
### Bicep curls
- **Exercise:** Bicep curls
- **Fitness per repetition:** 0.0
- **Strength per repetition:** 7.2
- **Repetitions per hour:** 108
- **Fitness per hour:** 0
- **Strength per hour:** 778

The time it takes to make one repetition will increase as exertion level goes up. Anything higher than moderate exertion will add time to a repetition.

## Stiffness (Exercise Fatigue)

Doing exercises will build up Stiffness, which eventually causes Exercise Fatigue. Exercise Fatigue will inflict pain. It will occur on some body parts, which depend on the exercise the player has done.  The severity of the pain depend on a few things:

- Time spent doing exercises.
- Regularity of the exercise.

Having Exercise Fatigue on the arms will lower Melee Damage, and Attack Speed.
Having Exercise Fatigue on the legs will make the player clumsier and slower.

### Regularity

In the exercise menu, there's a bar called "Regularity" under every exercise, which increases as the player does more of that kind of exercise. Having high regularity will decrease the severity of muscle fatigue caused by that exercise.

- Some professions start with a boost to regularity. fitness instructors start with between 40-59% regularity for each exercise, fire officers start with 10-19%, and security guards start with 7-11%.

## Code

This section contains source code from Project Zomboid Source: ProjectZomboid\media\lua\shared\Definitions\ FitnessExercise.lua Retrieved : Build 41.78.16 squats = { type = "squats" , name = getText ( "IGUI_Squats" ), tooltip = getText ( "IGUI_Squats_Tooltip" ), stiffness = "legs" , -- where we gonna build stiffness (can be a list separated by "," can be legs, arms or abs) metabolics = Metabolics . Fitness , xpMod = 1 , }; pushups = { type = "pushups" , name = getText ( "IGUI_PushUps" ), tooltip = getText ( "IGUI_PushUps_Tooltip" ), stiffness = "arms,chest" , metabolics = Metabolics . Fitness , xpMod = 1 , }; situp = { type = "situp" , name = getText ( "IGUI_SitUps" ), tooltip = getText ( "IGUI_SitUps_Tooltip" ), stiffness = "abs" , metabolics = Metabolics . Fitness , xpMod = 1 , }; burpees = { type = "burpees" , name = getText ( "IGUI_Burpees" ), tooltip = getText ( "IGUI_Burpees_Tooltip" ), stiffness = "legs,arms,chest" , -- where we gonna build stiffness (can be a list separated by "," can be legs, arms or abs) metabolics = Metabolics . FitnessHeavy , xpMod = 0.8 , -- few less xp as it gives xp for 3 body parts }; barbellcurl = { type = "barbellcurl" , name = getText ( "IGUI_BarbellCurl" ), tooltip = getText ( "IGUI_BarbellCurl_Tooltip" ), item = "Base.BarBell" , prop = "twohands" , -- prop is where we gonna put our item, 2 hand, primary or switch (one hand, then the other every X times) stiffness = "arms,chest" , metabolics = Metabolics . FitnessHeavy , xpMod = 1.2 , }; dumbbellpress = { type = "dumbbellpress" , name = getText ( "IGUI_DumbbellPress" ), tooltip = getText ( "IGUI_PushUps_Tooltip" ), item = "Base.DumbBell" , prop = "switch" , stiffness = "arms" , metabolics = Metabolics . FitnessHeavy , xpMod = 1.8 , }; bicepscurl = { type = "bicepscurl" , name = getText ( "IGUI_BicepsCurl" ), tooltip = getText ( "IGUI_PushUps_Tooltip" ), item = "Base.DumbBell" , prop = "switch" , -- switch is special, as i have 2 anim, one for left hand and one for right, i'll switch every X repeat the hand used stiffness = "arms" , metabolics = Metabolics . FitnessHeavy , xpMod = 1.8 , };
